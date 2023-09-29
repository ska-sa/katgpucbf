################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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
import math
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from enum import IntEnum

import aiokatcp
import numba
import numpy as np
import spead2.recv.asyncio
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS, MIN_SENSOR_UPDATE_PERIOD, N_POLS
from .. import recv as base_recv
from ..recv import BaseLayout, Chunk, StatsCollector
from ..spead import DIGITISER_ID_ID, DIGITISER_STATUS_ID, DIGITISER_STATUS_SATURATION_COUNT_SHIFT, TIMESTAMP_ID
from ..utils import DeviceStatusSensor, TimeConverter, TimeoutSensorStatusObserver
from . import METRIC_NAMESPACE

#: Number of partial chunks to allow at a time. Using 1 would reject any out-of-order
#: heaps (which can happen with a multi-path network). 2 is sufficient provided heaps
#: are not delayed by a whole chunk.
MAX_CHUNKS = 2

logger = logging.getLogger(__name__)

heaps_counter = Counter("input_heaps", "number of heaps received", ["pol"], namespace=METRIC_NAMESPACE)
chunks_counter = Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE)
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
    namespace=METRIC_NAMESPACE,
)

user_data_type = types.Record.make_c_struct(
    [
        ("stats_base", types.size_t),  # Index for first custom statistic
        ("stride", types.size_t),  # Bytes between polarisations in payload array
    ]
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
        """Number of heaps per chunk, on time axis."""
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
            pol = items[3] & 1  # Polarisation is LSB of digitiser ID field
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
            heap_index = timestamp // heap_samples % chunk_heaps
            data[0].heap_index = heap_index + pol * chunk_heaps
            data[0].heap_offset = heap_index * heap_bytes + pol * user_data.stride

            extra = numba.carray(intp_to_voidptr(data[0].extra), 1, dtype=np.uint16)
            data[0].extra_offset = data[0].heap_index * 2  # 2 is sizeof(uint16)
            data[0].extra_size = 2
            extra[0] = status >> DIGITISER_STATUS_SATURATION_COUNT_SHIFT

        return chunk_place_impl


def make_stream_group(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    src_affinity: Sequence[int],
    stride: int,
) -> spead2.recv.ChunkStreamRingGroup:
    """Create SPEAD receiver streams.

    Small helper function with F-engine-specific logic in it. Returns a stream
    for each polarisation.

    Parameters
    ----------
    layout
        Heap size and chunking parameters.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent.
    free_ringbuffer
        Ringbuffer for holding chunks for recycling once they've been used.
    src_affinity
        CPU core affinity for the worker threads (one per thread).
        Use -1 to indicate no affinity for a thread.
    stride
        Bytes between polarisations in chunk payload array
    """
    # Reference counters to make the labels exist before the first scrape
    for pol in range(N_POLS):
        for counter in _PER_POL_COUNTERS:
            counter.labels(pol)

    user_data = np.zeros(1, dtype=user_data_type.dtype)
    user_data["stride"] = stride
    group = base_recv.make_stream_group(
        layout=layout,
        spead_items=[TIMESTAMP_ID, spead2.HEAP_LENGTH_ID, DIGITISER_STATUS_ID, DIGITISER_ID_ID],
        max_active_chunks=MAX_CHUNKS,
        max_heap_extra=np.dtype(np.uint16).itemsize,
        data_ringbuffer=data_ringbuffer,
        free_ringbuffer=free_ringbuffer,
        affinity=src_affinity,
        max_heaps=1,  # Digitiser heaps are single-packet, so no need for more
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps"],
        user_data=user_data,
        explicit_start=True,
    )
    for stream in group:
        stats_collector.add_stream(stream)
    return group


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
                f"input{pol}.rx.timestamp",
                "The timestamp (in samples) of the last chunk of data received from the digitiser",
                default=-1,
                initial_status=aiokatcp.Sensor.Status.ERROR,
                auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
                auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
            ),
            aiokatcp.Sensor(
                aiokatcp.core.Timestamp,
                f"input{pol}.rx.unixtime",
                "The timestamp (in UNIX time) of the last chunk of data received from the digitiser",
                default=aiokatcp.core.Timestamp(-1.0),
                initial_status=aiokatcp.Sensor.Status.ERROR,
                auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
                auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
            ),
        ]
        for sensor in timestamp_sensors:
            TimeoutSensorStatusObserver(sensor, sensor_timeout, aiokatcp.Sensor.Status.ERROR)
            sensors.add(sensor)

        missing_sensors: list[aiokatcp.Sensor] = [
            aiokatcp.Sensor(
                aiokatcp.core.Timestamp,
                f"input{pol}.rx.missing-unixtime",
                "The timestamp (in UNIX time) when missing data was last detected",
                default=aiokatcp.core.Timestamp(-1.0),
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        ]
        for sensor in missing_sensors:
            TimeoutSensorStatusObserver(sensor, sensor_timeout, aiokatcp.Sensor.Status.NOMINAL)
            sensors.add(sensor)

    sensors.add(DeviceStatusSensor(sensors, "rx.device-status", "F-engine is receiving a good, clean digitiser stream"))

    return sensors


async def iter_chunks(
    ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    layout: Layout,
    sensors: aiokatcp.SensorSet,
    time_converter: TimeConverter,
) -> AsyncGenerator[Chunk, None]:
    """Iterate over the chunks and update sensors.

    It also populates the chunk timestamp.

    Parameters
    ----------
    ringbuffer
        Source of chunks.
    layout
        Structure of the streams.
    sensors
        Sensor set containing at least the sensors created by
        :func:`make_sensors`.
    time_converter
        Converter to turn data timestamps into sensor timestamps.
    """
    lost = 0
    first_timestamp = -1  # Updated to the actual first timestamp on the first chunk
    # These duplicate the Prometheus counters, because prometheus_client
    # doesn't provide an efficient way to get the current value
    # (REGISTRY.get_sample_value is documented as being intended only for unit
    # tests).
    n_heaps = [0] * N_POLS
    n_missing_heaps = [0] * N_POLS

    # `try`/`finally` block acting as a quick-and-dirty context manager,
    # to ensure that we clean up nicely after ourselves if we are stopped.
    try:
        async for chunk in ringbuffer:
            assert isinstance(chunk, Chunk)
            # Inspect the chunk we have just received.
            chunk.timestamp = chunk.chunk_id * layout.chunk_samples
            good = np.sum(chunk.present)
            if not good:
                # Dummy chunk created by spead2
                chunk.recycle()
                continue
            if first_timestamp == -1:
                # TODO: use chunk.present to determine the actual first timestamp
                first_timestamp = chunk.timestamp
            lost += chunk.present.size - good
            logger.debug(
                "Received chunk: timestamp=%#x (%d/%d, lost %d)",
                chunk.timestamp,
                good,
                chunk.present.size,
                lost,
            )
            unix_time = time_converter.adc_to_unix(chunk.timestamp)
            for pol in range(N_POLS):
                sensors[f"input{pol}.rx.timestamp"].set_value(chunk.timestamp, timestamp=unix_time)
                sensors[f"input{pol}.rx.unixtime"].set_value(aiokatcp.core.Timestamp(unix_time), timestamp=unix_time)

            pol_expected_heaps = (chunk.timestamp - first_timestamp + layout.chunk_samples) // layout.heap_samples
            chunks_counter.inc()
            # Zero out saturation count for heaps that were never received
            # (otherwise the value is undefined memory).
            assert chunk.extra is not None
            chunk.extra[chunk.present == 0] = 0
            for pol in range(N_POLS):
                # The cast is to force numpy ints to Python ints.
                buf_good = int(np.sum(chunk.present[pol]))
                heaps_counter.labels(pol).inc(buf_good)
                samples_counter.labels(pol).inc(buf_good * layout.heap_samples)
                bytes_counter.labels(pol).inc(buf_good * layout.heap_bytes)
                dig_clip_counter.labels(pol).inc(int(np.sum(chunk.extra[pol], dtype=np.uint64)))
                # Determine how many heaps we expected to have seen by
                # now, and subtract from it the number actually seen to
                # determine the number missing. This accounts for both
                # heaps lost within chunks and lost chunks.
                n_heaps[pol] += buf_good
                new_missing = pol_expected_heaps - n_heaps[pol]
                if new_missing > n_missing_heaps[pol]:
                    missing_heaps_counter.labels(pol).inc(new_missing - n_missing_heaps[pol])
                    n_missing_heaps[pol] = new_missing
                    sensors[f"input{pol}.rx.missing-unixtime"].set_value(
                        aiokatcp.core.Timestamp(unix_time), timestamp=unix_time, status=aiokatcp.Sensor.Status.ERROR
                    )
            yield chunk
    finally:
        stats_collector.update()  # Ensure final stats updates are captured


__all__ = ["Chunk", "Layout", "iter_chunks"]
