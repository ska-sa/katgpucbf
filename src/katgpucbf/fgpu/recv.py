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

"""Recv module."""

import functools
import logging
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

from .. import BYTE_BITS, N_POLS
from .. import recv as base_recv
from ..recv import Chunk, Counters, LayoutMixin, StatsCollector
from ..spead import DIGITISER_ID_ID, DIGITISER_STATUS_ID, DIGITISER_STATUS_SATURATION_COUNT_SHIFT, TIMESTAMP_ID
from ..utils import TimeConverter
from . import METRIC_NAMESPACE

#: Number of partial chunks to allow at a time. Using 1 would reject any out-of-order
#: heaps (which can happen with a multi-path network). 2 is sufficient provided heaps
#: are not delayed by a whole chunk.
MAX_CHUNKS = 2

logger = logging.getLogger(__name__)

counters = Counters(
    heaps=Counter("input_heaps", "number of heaps received", ["pol"], namespace=METRIC_NAMESPACE),
    chunks=Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE),
    samples=Counter("input_samples", "number of digitiser samples received", ["pol"], namespace=METRIC_NAMESPACE),
    bytes=Counter("input_bytes", "number of bytes of digitiser samples received", ["pol"], namespace=METRIC_NAMESPACE),
    missing_heaps=Counter(
        "input_missing_heaps", "number of heaps dropped on the input", ["pol"], namespace=METRIC_NAMESPACE
    ),
    clipped_samples=Counter(
        "input_clipped_samples", "number of ADC samples that clipped", ["pol"], namespace=METRIC_NAMESPACE
    ),
)

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
class Layout(LayoutMixin):
    """Parameters controlling the sizes of heaps and chunks."""

    sample_bits: int
    heap_samples: int
    chunk_timestamp_step: int
    mask_timestamp: bool

    @property
    def heap_bytes(self) -> int:  # noqa: D102
        return self.heap_samples * self.sample_bits // BYTE_BITS

    @property
    def chunk_batches(self) -> int:  # noqa: D102
        return self.chunk_timestamp_step // self.heap_samples

    @property
    def batch_heaps(self) -> int:  # noqa: D102
        return N_POLS

    @property
    def timestamp_mask(self) -> np.uint64:
        """Mask to AND with incoming timestamps."""
        return ~np.uint64(self.heap_samples - 1 if self.mask_timestamp else 0)

    @property
    def pol_chunk_bytes(self) -> int:
        """Number of bytes for the data in one polarisation of a chunk."""
        return self.chunk_timestamp_step * self.sample_bits // BYTE_BITS

    @functools.cached_property
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        """Low-level code for placing heaps in chunks."""
        heap_samples = self.heap_samples
        heap_bytes = self.heap_bytes
        chunk_batches = self.chunk_batches
        chunk_timestamp_step = self.chunk_timestamp_step
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
            data[0].chunk_id = timestamp // chunk_timestamp_step
            heap_index = timestamp // heap_samples % chunk_batches
            data[0].heap_index = heap_index + pol * chunk_batches
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
    recv_affinity: Sequence[int],
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
    recv_affinity
        CPU core affinity for the worker threads (one per thread).
        Use -1 to indicate no affinity for a thread.
    stride
        Bytes between polarisations in chunk payload array
    """
    # Reference counters to make the labels exist before the first scrape
    for pol in range(N_POLS):
        counters.labels(str(pol))

    user_data = np.zeros(len(recv_affinity), dtype=user_data_type.dtype)
    user_data["stride"] = stride
    group = base_recv.make_stream_group(
        layout=layout,
        spead_items=[TIMESTAMP_ID, spead2.HEAP_LENGTH_ID, DIGITISER_STATUS_ID, DIGITISER_ID_ID],
        max_active_chunks=MAX_CHUNKS,
        max_heap_extra=np.dtype(np.uint16).itemsize,
        data_ringbuffer=data_ringbuffer,
        free_ringbuffer=free_ringbuffer,
        affinity=recv_affinity,
        max_heaps=1,  # Digitiser heaps are single-packet, so no need for more
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps"],
        user_data=user_data,
        explicit_start=True,
    )
    stats_collector.add_stream_group(group)
    return group


def iter_chunks(
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
        :func:`.make_sensors`.
    time_converter
        Converter to turn data timestamps into sensor timestamps.
    """
    return base_recv.iter_chunks(
        ringbuffer,
        layout,
        sensors,
        time_converter,
        [(str(i), f"input{i}") for i in range(N_POLS)],
        counters,
        stats_collector,
    )


__all__ = ["Chunk", "Layout", "iter_chunks"]
