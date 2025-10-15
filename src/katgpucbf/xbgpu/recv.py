################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import IntEnum

import numba
import numpy as np
import spead2.recv.asyncio
import spead2.send.asyncio
from aiokatcp import SensorSet
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS, COMPLEX, N_POLS
from .. import recv as base_recv
from ..recv import Chunk, Counters, LayoutMixin, StatsCollector, user_data_type
from ..spead import FENG_ID_ID, TIMESTAMP_ID
from ..utils import TimeConverter
from . import METRIC_NAMESPACE

logger = logging.getLogger(__name__)

counters = Counters(
    heaps=Counter("input_heaps", "number of heaps received", namespace=METRIC_NAMESPACE),
    chunks=Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE),
    bytes=Counter("input_bytes", "number of bytes of input data received", namespace=METRIC_NAMESPACE),
    missing_heaps=Counter("input_missing_heaps", "number of heaps dropped on the input", namespace=METRIC_NAMESPACE),
    samples=Counter("input_samples", "number of complex samples received", namespace=METRIC_NAMESPACE),
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
class Layout(LayoutMixin):
    """Parameters controlling the sizes of heaps and chunks.

    Parameters
    ----------
    n_ants
        The number of antennas that data will be received from
    n_channels_per_substream
        The number of frequency channels contained in the stream.
    n_spectra_per_heap
        The number of time samples received per frequency channel.
    heap_timestamp_step
        Each heap contains a timestamp. The timestamp between consecutive heaps
        changes depending on the FFT size and the number of time samples per
        channel. This parameter defines the difference in timestamp values
        between consecutive heaps. This parameter can be calculated from the
        array configuration parameters for power-of-two array sizes, but is
        configurable to allow for greater flexibility during testing.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    chunk_batches
        Each chunk out of the SPEAD2 receiver will contain multiple heaps from
        each antenna. This parameter specifies the number of heaps per antenna
        that each chunk will contain.
    """

    n_ants: int
    n_channels_per_substream: int
    n_spectra_per_heap: int
    heap_timestamp_step: int
    sample_bits: int
    chunk_batches: int

    @property
    def heap_bytes(self):  # noqa: D102
        return (
            self.n_channels_per_substream * self.n_spectra_per_heap * N_POLS * COMPLEX * self.sample_bits // BYTE_BITS
        )

    @property
    def batch_heaps(self) -> int:  # noqa: D102
        return self.n_ants

    @property
    def heap_samples(self) -> int:  # noqa: D102
        return self.n_channels_per_substream * self.n_spectra_per_heap

    @property
    def chunk_timestamp_step(self) -> int:  # noqa: D102
        return self.heap_timestamp_step * self.chunk_batches

    @functools.cached_property
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        n_ants = self.n_ants
        heap_timestamp_step = self.heap_timestamp_step
        chunk_batches = self.chunk_batches
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
            if timestamp % heap_timestamp_step != 0:
                # Invalid timestamp
                batch_stats[user_data[0].stats_base + _Statistic.BAD_TIMESTAMP_HEAPS] += 1
                return
            if fengine_id >= n_ants:
                # Invalid F-engine ID
                batch_stats[user_data[0].stats_base + _Statistic.BAD_FENG_ID_HEAPS] += 1
                return
            # Compute position of this heap on the time axis, starting from
            # timestamp 0
            heap_time_abs = timestamp // heap_timestamp_step
            data[0].chunk_id = heap_time_abs // chunk_batches
            # Position of this heap on the time axis, from the start of the chunk
            heap_time = heap_time_abs % chunk_batches
            data[0].heap_index = heap_time * n_ants + fengine_id
            data[0].heap_offset = data[0].heap_index * heap_bytes

        return chunk_place_impl


def make_stream(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    recv_affinity: int,
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
    recv_affinity
        CPU core affinity for the worker thread.
    max_active_chunks
        Maximum number of chunks under construction.
    """
    user_data = np.zeros(1, dtype=user_data_type.dtype)
    stream = base_recv.make_stream(
        layout=layout,
        spead_items=[TIMESTAMP_ID, FENG_ID_ID, spead2.HEAP_LENGTH_ID],
        max_active_chunks=max_active_chunks,
        data_ringbuffer=data_ringbuffer,
        free_ringbuffer=free_ringbuffer,
        affinity=recv_affinity,
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps", "katgpucbf.bad_feng_id_heaps"],
        substreams=layout.n_ants,
        stop_on_stop_item=False,  # By default, a heap containing a stream control stop item will terminate the stream
        user_data=user_data,
    )
    stats_collector.add_stream(stream)
    return stream


def iter_chunks(
    ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    layout: Layout,
    sensors: SensorSet,
    time_converter: TimeConverter,
) -> AsyncGenerator[Chunk, None]:
    """Retrieve chunks from the ringbuffer, updating metrics as they are received.

    The returned chunks are yielded from this asynchronous generator.

    Parameters
    ----------
    ringbuffer
        Source of chunks.
    layout
        Structure of the stream.
    sensors
        Sensor set containing at least the sensors created by
        :func:`.make_sensors`.
    time_converter
        Converter to turn data timestamps into sensor timestamps.
    """
    return base_recv.iter_chunks(
        ringbuffer,
        layout,
        sensors,
        time_converter,
        None,
        counters,
        stats_collector,
    )
