################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import AsyncGenerator

import numba
import numpy as np
import spead2.recv.asyncio
import spead2.send
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS, COMPLEX, N_POLS
from ..recv import BaseLayout, Chunk, StatsToCounters
from ..recv import make_stream as make_base_stream
from ..recv import user_data_type
from ..spead import FENG_ID_ID, TIMESTAMP_ID
from . import METRIC_NAMESPACE

heaps_counter = Counter("input_heaps", "number of heaps received", namespace=METRIC_NAMESPACE)
chunks_counter = Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE)
bytes_counter = Counter("input_bytes", "number of bytes of input data received", namespace=METRIC_NAMESPACE)
incomplete_heaps_counter = Counter(
    "input_incomplete_heaps", "number of heaps only partially received", namespace=METRIC_NAMESPACE
)
too_old_heaps_counter = Counter(
    "input_too_old_heaps", "number of heaps that arrived too late to be processed", namespace=METRIC_NAMESPACE
)
missing_heaps_counter = Counter(
    "input_missing_heaps", "number of heaps dropped on the input", namespace=METRIC_NAMESPACE
)
metadata_heaps_counter = Counter(
    "input_metadata_heaps", "number of heaps not containing payload", namespace=METRIC_NAMESPACE
)
bad_timestamp_heaps_counter = Counter(
    "input_bad_timestamp_heaps", "timestamp is not a multiple of expected step size", namespace=METRIC_NAMESPACE
)
bad_feng_id_heaps_counter = Counter("input_bad_feng_id_heaps", "fengine ID is out of range", namespace=METRIC_NAMESPACE)


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
    n_channels
        The total number of frequency channels out of the F-Engine.
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
            fengine = items[1]
            payload_size = items[2]
            if payload_size != heap_bytes or timestamp < 0 or fengine < 0:
                # It's something unexpected - possibly descriptors. Ignore it.
                batch_stats[user_data[0].stats_base + _Statistic.METADATA_HEAPS] += 1
                return
            if timestamp % timestamp_step != 0:
                # Invalid timestamp
                batch_stats[user_data[0].stats_base + _Statistic.BAD_TIMESTAMP_HEAPS] += 1
                return
            if fengine >= n_ants:
                # Invalid F-engine ID
                batch_stats[user_data[0].stats_base + _Statistic.BAD_FENG_ID_HEAPS] += 1
                return
            # Compute position of this heap on the time axis, starting from
            # timestamp 0
            heap_time_abs = timestamp // timestamp_step
            data[0].chunk_id = heap_time_abs // heaps_per_fengine_per_chunk
            # Position of this heap on the time axis, from the start of the chunk
            heap_time = heap_time_abs % heaps_per_fengine_per_chunk
            data[0].heap_index = heap_time * n_ants + fengine
            data[0].heap_offset = data[0].heap_index * heap_bytes

        return chunk_place_impl


def make_stream(
    layout: Layout, data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer, src_affinity: int, rx_reorder_tol: int
) -> spead2.recv.ChunkRingStream:
    """Create a SPEAD receiver stream.

    Helper function with XB-engine-specific logic in it.

    Parameters
    ----------
    layout
        Heap size and chunking parameters.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent.
    src_affinity
        CPU core affinity for the worker thread.
    """
    rx_heap_timestamp_step = layout.n_channels_per_stream * layout.n_ants * 2 * layout.n_spectra_per_heap
    return make_base_stream(
        layout=layout,
        spead_items=[TIMESTAMP_ID, FENG_ID_ID, spead2.HEAP_LENGTH_ID],
        max_active_chunks=math.ceil(rx_reorder_tol / rx_heap_timestamp_step / layout.heaps_per_fengine_per_chunk) + 1,
        data_ringbuffer=data_ringbuffer,
        affinity=src_affinity,
        # max_heaps is set quite high because timing jitter/bursting means there
        # could be multiple heaps from one F-Engine during the time it takes
        # another to transmit (NGC-471).
        # TODO: find a cleaner solution.
        max_heaps=layout.n_ants * (spead2.send.StreamConfig.DEFAULT_BURST_SIZE // layout.heap_bytes + 1) * 128,
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps", "katgpucbf.bad_feng_id_heaps"],
    )


async def recv_chunks(stream: spead2.recv.ChunkRingStream) -> AsyncGenerator[Chunk, None]:
    """Retrieve chunks from the ringbuffer, updating metrics as they are received.

    The returned chunks are yielded from this asynchronous generator.
    """
    counter_map = {
        "incomplete_heaps_evicted": incomplete_heaps_counter,
        "too_old_heaps": too_old_heaps_counter,
        "katgpucbf.metadata_heaps": metadata_heaps_counter,
        "katgpucbf.bad_timestamp_heaps": bad_timestamp_heaps_counter,
        "katgpucbf.bad_feng_id_heaps": bad_feng_id_heaps_counter,
    }
    stats_to_counters = StatsToCounters(counter_map, stream.config)
    ringbuffer = stream.data_ringbuffer
    assert isinstance(ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
    async for chunk in ringbuffer:
        assert isinstance(chunk, Chunk)
        # Update metrics
        expected_heaps = len(chunk.present)
        received_heaps = int(np.sum(chunk.present))
        dropped_heaps = expected_heaps - received_heaps

        missing_heaps_counter.inc(dropped_heaps)
        heaps_counter.inc(expected_heaps)
        chunks_counter.inc(1)
        bytes_counter.inc(chunk.data.nbytes * received_heaps // expected_heaps)
        # NOTE: this won't be synchronised with the chunk, as more heaps may
        # have arrived after the chunk was pushed to the ringbuffer. But on
        # the time scales at which Prometheus scrapes it will be close.
        stats_to_counters.update(stream.stats)

        yield chunk
