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

"""SPEAD receiver utilities."""
import functools
import logging
import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import IntEnum

import numba
import numpy as np
import spead2.recv.asyncio
import spead2.send.asyncio
from aiokatcp import Sensor, SensorSampler, SensorSet
from aiokatcp.core import Timestamp
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS, COMPLEX, MIN_SENSOR_UPDATE_PERIOD, N_POLS
from ..recv import BaseLayout, Chunk, StatsCollector
from ..recv import make_stream as make_base_stream
from ..recv import user_data_type
from ..spead import FENG_ID_ID, TIMESTAMP_ID
from ..utils import DeviceStatusSensor, TimeConverter, TimeoutSensorStatusObserver
from . import METRIC_NAMESPACE

logger = logging.getLogger(__name__)

heaps_counter = Counter("input_heaps", "number of heaps received", namespace=METRIC_NAMESPACE)
chunks_counter = Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE)
bytes_counter = Counter("input_bytes", "number of bytes of input data received", namespace=METRIC_NAMESPACE)
missing_heaps_counter = Counter(
    "input_missing_heaps", "number of heaps dropped on the input", namespace=METRIC_NAMESPACE
)

stats_collector = StatsCollector(
    {
        "incomplete_heaps_evicted": ("input_incomplete_heaps", "number of heaps only partially received"),
        "too_old_heaps": ("input_too_old_heaps", "number of heaps that arrived too late to be processed"),
        "katgpucbf.metadata_heaps": ("input_metadata_heaps", "number of heaps not containing payload"),
        "katgpucbf.bad_timestamp_heaps": (
            "input_bad_timestamp_heaps",
            "timestamp is not a multiple of expected step size",
        ),
        "katgpucbf.bad_feng_id_heaps": ("input_bad_feng_id_heaps", "fengine ID is out of range"),
    },
    namespace=METRIC_NAMESPACE,
)


class _Statistic(IntEnum):
    """Custom statistics for the SPEAD receiver."""

    # NOTE: the values are important and must match the registration order
    # of the statistics.
    METADATA_HEAPS = 0
    BAD_TIMESTAMP_HEAPS = 1
    BAD_FENG_ID_HEAPS = 2


@dataclass(frozen=True)
class Layout(BaseLayout):
    """Parameters controlling the sizes of heaps and chunks.

    Parameters
    ----------
    n_ants
        The number of antennas that data will be received from
    n_channels_per_stream
        The number of frequency channels contained in the stream.
    n_spectra_per_heap
        The number of time samples received per frequency channel.
    timestamp_step
        Each heap contains a timestamp. The timestamp between consecutive heaps
        changes depending on the FFT size and the number of time samples per
        channel. This parameter defines the difference in timestamp values
        between consecutive heaps. This parameter can be calculated from the
        array configuration parameters for power-of-two array sizes, but is
        configurable to allow for greater flexibility during testing.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    heaps_per_fengine_per_chunk
        Each chunk out of the SPEAD2 receiver will contain multiple heaps from
        each antenna. This parameter specifies the number of heaps per antenna
        that each chunk will contain.
    """

    n_ants: int
    n_channels_per_stream: int
    n_spectra_per_heap: int
    timestamp_step: int
    sample_bits: int
    heaps_per_fengine_per_chunk: int

    @property
    def heap_bytes(self):
        """Calculate number of bytes in a heap based on layout parameters."""
        return self.n_channels_per_stream * self.n_spectra_per_heap * N_POLS * COMPLEX * self.sample_bits // BYTE_BITS

    @property
    def chunk_heaps(self) -> int:  # noqa: D401
        """Number of heaps per chunk."""
        return self.heaps_per_fengine_per_chunk * self.n_ants

    @functools.cached_property
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        n_ants = self.n_ants
        timestamp_step = self.timestamp_step
        heaps_per_fengine_per_chunk = self.heaps_per_fengine_per_chunk
        heap_bytes = self.heap_bytes
        n_statistics = len(_Statistic)

        @numba.cfunc(
            types.void(types.CPointer(chunk_place_data), types.uintp, types.CPointer(user_data_type)), nopython=True
        )
        def chunk_place_impl(data_ptr, data_size, user_data_ptr):
            data = numba.carray(data_ptr, 1)
            user_data = numba.carray(user_data_ptr, 1)
            batch_stats = numba.carray(
                intp_to_voidptr(data[0].batch_stats),
                user_data[0].stats_base + n_statistics,
                dtype=np.uint64,
            )
            items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
            timestamp = items[0]
            fengine_id = items[1]
            payload_size = items[2]
            if payload_size != heap_bytes or timestamp < 0 or fengine_id < 0:
                # It's something unexpected - possibly descriptors. Ignore it.
                batch_stats[user_data[0].stats_base + _Statistic.METADATA_HEAPS] += 1
                return
            if timestamp % timestamp_step != 0:
                # Invalid timestamp
                batch_stats[user_data[0].stats_base + _Statistic.BAD_TIMESTAMP_HEAPS] += 1
                return
            if fengine_id >= n_ants:
                # Invalid F-engine ID
                batch_stats[user_data[0].stats_base + _Statistic.BAD_FENG_ID_HEAPS] += 1
                return
            # Compute position of this heap on the time axis, starting from
            # timestamp 0
            heap_time_abs = timestamp // timestamp_step
            data[0].chunk_id = heap_time_abs // heaps_per_fengine_per_chunk
            # Position of this heap on the time axis, from the start of the chunk
            heap_time = heap_time_abs % heaps_per_fengine_per_chunk
            data[0].heap_index = heap_time * n_ants + fengine_id
            data[0].heap_offset = data[0].heap_index * heap_bytes

        return chunk_place_impl


def make_stream(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    src_affinity: int,
    max_active_chunks: int,
) -> spead2.recv.ChunkRingStream:
    """Create a SPEAD receiver stream.

    Helper function with XB-engine-specific logic in it.

    Parameters
    ----------
    layout
        Heap size and chunking parameters.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent.
    free_ringbuffer
        Ringbuffer for holding chunks for recycling once they've been used.
    src_affinity
        CPU core affinity for the worker thread.
    max_active_chunks
        Maximum number of chunks under construction.
    """
    user_data = np.zeros(1, dtype=user_data_type.dtype)
    stream = make_base_stream(
        layout=layout,
        spead_items=[TIMESTAMP_ID, FENG_ID_ID, spead2.HEAP_LENGTH_ID],
        max_active_chunks=max_active_chunks,
        data_ringbuffer=data_ringbuffer,
        free_ringbuffer=free_ringbuffer,
        affinity=src_affinity,
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps", "katgpucbf.bad_feng_id_heaps"],
        substreams=layout.n_ants,
        stop_on_stop_item=False,  # By default, a heap containing a stream control stop item will terminate the stream
        user_data=user_data,
    )
    stats_collector.add_stream(stream)
    return stream


def make_sensors(sensor_timeout: float) -> SensorSet:
    """Create the sensors needed to hold receiver statistics.

    Parameters
    ----------
    sensor_timeout
        Time (in seconds) without updates before sensors for received data go
        into error and sensors for missing data becoming nominal.
    """
    sensors = SensorSet()
    timestamp_sensors: list[Sensor] = [
        Sensor(
            int,
            "rx.timestamp",
            "The timestamp (in samples) of the last chunk of data received from an F-engine",
            default=-1,
            initial_status=Sensor.Status.ERROR,
            auto_strategy=SensorSampler.Strategy.EVENT_RATE,
            auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
        ),
        Sensor(
            Timestamp,
            "rx.unixtime",
            "The timestamp (in UNIX time) of the last chunk of data received from an F-engine",
            default=Timestamp(-1.0),
            initial_status=Sensor.Status.ERROR,
            auto_strategy=SensorSampler.Strategy.EVENT_RATE,
            auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
        ),
    ]
    for sensor in timestamp_sensors:
        TimeoutSensorStatusObserver(sensor, sensor_timeout, Sensor.Status.ERROR)
        sensors.add(sensor)

    missing_sensors: list[Sensor] = [
        Sensor(
            Timestamp,
            "rx.missing-unixtime",
            "The timestamp (in UNIX time) when missing data was last detected",
            default=Timestamp(-1.0),
            initial_status=Sensor.Status.NOMINAL,
        )
    ]
    for sensor in missing_sensors:
        TimeoutSensorStatusObserver(sensor, sensor_timeout, Sensor.Status.NOMINAL)
        sensors.add(sensor)

    sensors.add(DeviceStatusSensor(sensors, "rx.device-status", "XB-engine is receiving a good, clean F-engine stream"))

    return sensors


async def recv_chunks(
    stream: spead2.recv.ChunkRingStream,
    layout: Layout,
    sensors: SensorSet,
    time_converter: TimeConverter,
) -> AsyncGenerator[Chunk, None]:
    """Retrieve chunks from the ringbuffer, updating metrics as they are received.

    The returned chunks are yielded from this asynchronous generator.

    Parameters
    ----------
    stream
        Stream object handling reception of F-engine data.
    layout
        Structure of the stream.
    sensors
        Sensor set containing at least the sensors created by
        :func:`make_sensors`.
    time_converter
        Converter to turn data timestamps into sensor timestamps.
    """
    ringbuffer = stream.data_ringbuffer
    prev_chunk_id = -1
    valid_chunk_received = False
    assert isinstance(ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
    async for chunk in ringbuffer:
        assert isinstance(chunk, Chunk)

        # Compute metrics
        expected_heaps = len(chunk.present)
        received_heaps = int(np.sum(chunk.present))
        dropped_heaps = expected_heaps - received_heaps

        if received_heaps == 0:
            # It's not impossible for there to be a completely empty chunk
            # during normal operation (caused by spead2's windowing
            # algorithm). Return the chunk to the stream since we are not
            # going to yield it.
            chunk.recycle()
            continue
        elif not valid_chunk_received:
            # Need to check which is the first "proper" Chunk
            valid_chunk_received = True
            prev_chunk_id = chunk.chunk_id - 1

        # TODO: Perhaps make this 'chunk timestamp step' a property of the Layout?
        chunk.timestamp = chunk.chunk_id * layout.timestamp_step * layout.heaps_per_fengine_per_chunk
        unix_time = time_converter.adc_to_unix(chunk.timestamp)
        sensors["rx.timestamp"].set_value(chunk.timestamp, timestamp=unix_time)
        sensors["rx.unixtime"].set_value(Timestamp(unix_time), timestamp=unix_time)

        # Check if we've missed any chunks
        expected_chunk_id = prev_chunk_id + 1
        if chunk.chunk_id != expected_chunk_id:
            missed_chunks = chunk.chunk_id - expected_chunk_id
            logger.warning(
                "Receiver missed %d chunks. Expected ID: %d, received ID: %d.",
                missed_chunks,
                expected_chunk_id,
                chunk.chunk_id,
            )
            dropped_heaps += missed_chunks * expected_heaps

        if dropped_heaps > 0:
            sensors["rx.missing-unixtime"].set_value(Timestamp(unix_time), Sensor.Status.ERROR, timestamp=unix_time)

        # Increment Prometheus counters
        missing_heaps_counter.inc(dropped_heaps)
        heaps_counter.inc(received_heaps)
        chunks_counter.inc(1)
        bytes_counter.inc(chunk.data.nbytes * received_heaps // expected_heaps)
        prev_chunk_id = chunk.chunk_id

        yield chunk
    stats_collector.update()  # Ensure final stats updates are captured
