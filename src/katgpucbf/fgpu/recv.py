################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Recv module."""

import functools
import logging
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import aiokatcp
import numba
import numpy as np
import spead2.recv.asyncio
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS, N_POLS
from ..recv import BaseLayout, Chunk, StatsCollector
from ..recv import make_stream as make_base_stream
from ..recv import user_data_type
from ..spead import DIGITISER_STATUS_ID, DIGITISER_STATUS_SATURATION_COUNT_SHIFT, TIMESTAMP_ID
from ..utils import TimeConverter, TimeoutSensorStatusObserver
from . import METRIC_NAMESPACE

#: Number of partial chunks to allow at a time. Using 1 would reject any out-of-order
#: heaps (which can happen with a multi-path network). 2 is sufficient provided heaps
#: are not delayed by a whole chunk.
MAX_CHUNKS = 2

logger = logging.getLogger(__name__)

heaps_counter = Counter("input_heaps", "number of heaps received", ["pol"], namespace=METRIC_NAMESPACE)
chunks_counter = Counter("input_chunks", "number of chunks received", ["pol"], namespace=METRIC_NAMESPACE)
samples_counter = Counter("input_samples", "number of digitiser samples received", ["pol"], namespace=METRIC_NAMESPACE)
bytes_counter = Counter(
    "input_bytes", "number of bytes of digitiser samples received", ["pol"], namespace=METRIC_NAMESPACE
)
missing_heaps_counter = Counter(
    "input_missing_heaps", "number of heaps dropped on the input", ["pol"], namespace=METRIC_NAMESPACE
)
dig_clip_counter = Counter(
    "input_clipped_samples", "number of ADC samples that clipped", ["pol"], namespace=METRIC_NAMESPACE
)
_PER_POL_COUNTERS = [
    heaps_counter,
    chunks_counter,
    samples_counter,
    bytes_counter,
    missing_heaps_counter,
    dig_clip_counter,
]

stats_collector = StatsCollector(
    {
        "too_old_heaps": ("input_too_old_heaps", "number of heaps that arrived too late to be processed"),
        "katgpucbf.metadata_heaps": ("input_metadata_heaps", "number of heaps not containing payload"),
        "katgpucbf.bad_timestamp_heaps": (
            "input_bad_timestamp_heaps",
            "timestamp not a multiple of samples per packet",
        ),
    },
    labelnames=["pol"],
    namespace=METRIC_NAMESPACE,
)


class _Statistic(IntEnum):
    """Custom statistics for the SPEAD receiver."""

    # Note: the values are important and must match the registration order
    # of the statistics.
    METADATA_HEAPS = 0
    BAD_TIMESTAMP_HEAPS = 1


@dataclass(frozen=True)
class Layout(BaseLayout):
    """Parameters controlling the sizes of heaps and chunks."""

    sample_bits: int
    heap_samples: int
    chunk_samples: int
    mask_timestamp: bool

    @property
    def heap_bytes(self) -> int:  # noqa: D401
        """Number of payload bytes per heap."""
        return self.heap_samples * self.sample_bits // BYTE_BITS

    @property
    def chunk_heaps(self) -> int:  # noqa: D401
        """Number of heaps per chunk."""
        return self.chunk_samples // self.heap_samples

    @property
    def timestamp_mask(self) -> np.uint64:  # noqa: D401
        """Mask to AND with incoming timestamps."""
        return ~np.uint64(self.heap_samples - 1 if self.mask_timestamp else 0)

    @functools.cached_property
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        """Low-level code for placing heaps in chunks."""
        heap_samples = self.heap_samples
        heap_bytes = self.heap_bytes
        chunk_heaps = self.chunk_heaps
        chunk_samples = self.chunk_samples
        timestamp_mask = self.timestamp_mask
        n_statistics = len(_Statistic)

        # numba.types doesn't have a size_t, so assume it is the same as uintptr_t
        @numba.cfunc(
            types.void(types.CPointer(chunk_place_data), types.uintp, types.CPointer(user_data_type)),
            nopython=True,
        )
        def chunk_place_impl(data_ptr, data_size, user_data_ptr):
            data = numba.carray(data_ptr, 1)
            items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
            timestamp = items[0]
            payload_size = items[1]
            status = items[2]
            user_data = numba.carray(user_data_ptr, 1)
            batch_stats = numba.carray(
                intp_to_voidptr(data[0].batch_stats),
                user_data[0].stats_base + n_statistics,
                dtype=np.uint64,
            )
            if payload_size != heap_bytes or timestamp < 0:
                # It's something unexpected - maybe it has descriptors or a stream
                # control item. Ignore it.
                batch_stats[user_data[0].stats_base + _Statistic.METADATA_HEAPS] += 1
                return
            timestamp &= timestamp_mask
            if timestamp % heap_samples != 0:
                batch_stats[user_data[0].stats_base + _Statistic.BAD_TIMESTAMP_HEAPS] += 1
                return
            data[0].chunk_id = timestamp // chunk_samples
            data[0].heap_index = timestamp // heap_samples % chunk_heaps
            data[0].heap_offset = data[0].heap_index * heap_bytes

            extra = numba.carray(intp_to_voidptr(data[0].extra), 1, dtype=np.uint16)
            data[0].extra_offset = data[0].heap_index * 2  # 2 is sizeof(uint16)
            data[0].extra_size = 2
            extra[0] = status >> DIGITISER_STATUS_SATURATION_COUNT_SHIFT

        return chunk_place_impl


def make_streams(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffers: Sequence[spead2.recv.ChunkRingbuffer],
    src_affinity: Sequence[int],
) -> list[spead2.recv.ChunkRingStream]:
    """Create SPEAD receiver streams.

    Small helper function with F-engine-specific logic in it. Returns a stream
    for each polarisation.

    Parameters
    ----------
    layout
        Heap size and chunking parameters.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent.
    free_ringbuffers
        Ringbuffers for holding chunks for recycling once they've been used
        (one per pol).
    src_affinity
        CPU core affinity for the worker threads ([-1, -1] for no affinity).
    """
    # Reference counters to make the labels exist before the first scrape
    for pol in range(N_POLS):
        for counter in _PER_POL_COUNTERS:
            counter.labels(pol)

    streams = [
        make_base_stream(
            layout=layout,
            spead_items=[TIMESTAMP_ID, spead2.HEAP_LENGTH_ID, DIGITISER_STATUS_ID],
            max_active_chunks=MAX_CHUNKS,
            max_heap_extra=np.dtype(np.uint16).itemsize,
            data_ringbuffer=data_ringbuffer,
            free_ringbuffer=free_ringbuffers[pol],
            affinity=src_affinity[pol],
            max_heaps=1,  # Digitiser heaps are single-packet, so no need for more
            stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps"],
            stream_id=pol,
        )
        for pol in range(N_POLS)
    ]
    for pol, stream in enumerate(streams):
        stats_collector.add_stream(stream, [str(pol)])
    return streams


def make_sensors(sensor_timeout: float) -> aiokatcp.SensorSet:
    """Create the sensors needed to hold receiver statistics.

    Parameters
    ----------
    sensor_timeout
        Time (in seconds) without updates before sensors for received data go
        into error and sensors for missing data become nominal.
    """
    sensors = aiokatcp.SensorSet()
    for pol in range(N_POLS):
        timestamp_sensors: list[aiokatcp.Sensor] = [
            aiokatcp.Sensor(
                int,
                f"input{pol}-rx-timestamp",
                "The timestamp (in samples) of the last chunk of data received from the digitiser",
                default=-1,
                initial_status=aiokatcp.Sensor.Status.ERROR,
            ),
            aiokatcp.Sensor(
                float,
                f"input{pol}-rx-unixtime",
                "The timestamp (in UNIX time) of the last chunk of data received from the digitiser",
                default=-1.0,
                initial_status=aiokatcp.Sensor.Status.ERROR,
            ),
        ]
        for sensor in timestamp_sensors:
            TimeoutSensorStatusObserver(sensor, sensor_timeout, aiokatcp.Sensor.Status.ERROR)
            sensors.add(sensor)
        missing_sensors: list[aiokatcp.Sensor] = [
            aiokatcp.Sensor(
                float,
                f"input{pol}-rx-missing-unixtime",
                "The timestamp (in UNIX time) of the last chunk of received data with missing packets",
                default=-1.0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        ]
        for sensor in missing_sensors:
            TimeoutSensorStatusObserver(sensor, sensor_timeout, aiokatcp.Sensor.Status.NOMINAL)
            sensors.add(sensor)
    return sensors


async def chunk_sets(
    streams: list[spead2.recv.ChunkRingStream],
    layout: Layout,
    sensors: aiokatcp.SensorSet,
    time_converter: TimeConverter,
) -> AsyncGenerator[list[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    This code receives chunks of data from the C++-domain Ringbuffer, matches
    them by timestamp, and ``yield`` to the caller.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.

    Parameters
    ----------
    streams
        A list of stream objects - there should be only two of them, because
        each represents a polarisation.
    layout
        Structure of the streams
    sensors
        Sensor set containing at least the sensors created by
        :func:`make_sensors`.
    time_converter
        Converter to turn data timestamps into sensor timestamps.
    """
    n_pol = len(streams)
    # Working buffer to match up pairs of chunks from both pols. There is
    # a deque for each pol, ordered by time
    buf: list[deque[Chunk]] = [deque() for _ in streams]
    ring = cast(spead2.recv.asyncio.ChunkRingbuffer, streams[0].data_ringbuffer)
    lost = 0

    first_timestamp = -1  # Updated to the actual first timestamp on the first chunk
    # These duplicate the Prometheus counters, because prometheus_client
    # doesn't provide an efficient way to get the current value
    # (REGISTRY.get_sample_value is documented as being intended only for unit
    # tests).
    n_heaps = [0] * n_pol
    n_missing_heaps = [0] * n_pol

    # `try`/`finally` block acting as a quick-and-dirty context manager,
    # to ensure that we clean up nicely after ourselves if we are stopped.
    try:
        async for chunk in ring:
            assert isinstance(chunk, Chunk)
            # Inspect the chunk we have just received.
            chunk.timestamp = chunk.chunk_id * layout.chunk_samples
            pol = chunk.stream_id
            good = np.sum(chunk.present)
            if not good:
                chunk.recycle()
                continue
            if first_timestamp == -1:
                # TODO: use chunk.present to determine the actual first timestamp
                first_timestamp = chunk.timestamp
            lost += layout.chunk_heaps - good
            logger.debug(
                "Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)",
                chunk.timestamp,
                pol,
                good,
                layout.chunk_heaps,
                lost,
            )
            unix_time = time_converter.adc_to_unix(chunk.timestamp)
            sensors[f"input{pol}-rx-timestamp"].set_value(chunk.timestamp, timestamp=unix_time)
            sensors[f"input{pol}-rx-unixtime"].set_value(unix_time, timestamp=unix_time)

            buf[pol].append(chunk)

            # Age out old chunks that will never match. This happens if the
            # chunk is older than the newest chunk for every pol.
            min_newest = min((b[-1].chunk_id if b else -1) for b in buf)
            for pol, b in enumerate(buf):
                while b and b[0].chunk_id < min_newest:
                    logger.warning("Chunk not matched: timestamp=%#x pol=%d", b[0].chunk_id * layout.chunk_samples, pol)
                    # Chunk was passed by without getting used. Return to the pool.
                    b.popleft().recycle()

            # If we have a matching pair of chunks, then we can yield.
            if all(b and b[0].chunk_id == chunk.chunk_id for b in buf):
                expected_heaps = (chunk.timestamp - first_timestamp + layout.chunk_samples) // layout.heap_samples
                out = []
                for b in buf:
                    c = b.popleft()
                    out.append(c)
                    pol = c.stream_id
                    # The cast is to force numpy ints to Python ints.
                    buf_good = int(np.sum(c.present))
                    heaps_counter.labels(pol).inc(buf_good)
                    chunks_counter.labels(pol).inc()
                    samples_counter.labels(pol).inc(buf_good * layout.heap_samples)
                    bytes_counter.labels(pol).inc(buf_good * layout.heap_bytes)
                    # Zero out saturation count for heaps that were never received
                    # (otherwise the value is undefined memory).
                    assert c.extra is not None
                    c.extra[c.present == 0] = 0
                    dig_clip_counter.labels(pol).inc(int(np.sum(c.extra, dtype=np.uint64)))
                    # Determine how many heaps we expected to have seen by
                    # now, and subtract from it the number actually seen to
                    # determine the number missing. This accounts for both
                    # heaps lost within chunks and lost chunks.
                    n_heaps[c.stream_id] += buf_good
                    new_missing = expected_heaps - n_heaps[pol]
                    if new_missing > n_missing_heaps[pol]:
                        missing_heaps_counter.labels(pol).inc(new_missing - n_missing_heaps[pol])
                        n_missing_heaps[pol] = new_missing
                        sensors[f"input{pol}-rx-missing-unixtime"].set_value(
                            unix_time, timestamp=unix_time, status=aiokatcp.Sensor.Status.ERROR
                        )

                yield out
    finally:
        stats_collector.update()  # Ensure final stats updates are captured
        for b in buf:
            for c in b:
                c.recycle()


__all__ = ["Chunk", "Layout", "chunk_sets"]
