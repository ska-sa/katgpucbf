################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Handle receiving tied-array-channelised-voltage data."""

import functools
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

from .. import BYTE_BITS, COMPLEX, N_POLS
from .. import recv as base_recv
from ..recv import Chunk, Counters, LayoutMixin, StatsCollector
from ..spead import BEAM_ANTS_ID, FREQUENCY_ID, TIMESTAMP_ID
from ..utils import TimeConverter
from . import METRIC_NAMESPACE

#: Number of chunks to allow to be under construction
MAX_CHUNKS = 2  # TODO: may need to increase to tolerate reordering

counters = Counters(
    heaps=Counter("input_heaps", "number of heaps received", ["pol"], namespace=METRIC_NAMESPACE),
    chunks=Counter("input_chunks", "number of chunks received", namespace=METRIC_NAMESPACE),
    samples=Counter("input_samples", "number of complex voltage samples received", ["pol"], namespace=METRIC_NAMESPACE),
    bytes=Counter("input_bytes", "number of bytes of input data received", ["pol"], namespace=METRIC_NAMESPACE),
    missing_heaps=Counter(
        "input_missing_heaps", "number of heaps dropped on the input", ["pol"], namespace=METRIC_NAMESPACE
    ),
)

stats_collector = StatsCollector(
    {
        "incomplete_heaps_evicted": ("input_incomplete_heaps", "number of heaps only partially received"),
        "too_old_heaps": ("input_too_old_heaps", "number of heaps that arrived too late to be processed"),
        "katgpucbf.metadata_heaps": ("input_metadata_heaps", "number of heaps not containing payload"),
        "katgpucbf.bad_timestamp_heaps": (
            "input_bad_timestamp_heaps",
            "timestamp not a multiple of samples per packet",
        ),
        "katgpucbf.bad_frequency_heaps": (
            "input_bad_frequency_heaps",
            "channel is not a multiple of channels per substream or is out of range",
        ),
    },
    namespace=METRIC_NAMESPACE,
)


user_data_type = types.Record.make_c_struct(
    [
        ("stats_base", types.size_t),  # Index for first custom statistic
        ("pol", types.size_t),  # Which beam this is
    ]
)


class _Statistic(IntEnum):
    """Custom statistics for the SPEAD receiver."""

    # Note: the values are important and must match the registration order
    # of the statistics.
    METADATA_HEAPS = 0
    BAD_TIMESTAMP_HEAPS = 1
    BAD_FREQUENCY_HEAPS = 2


@dataclass(frozen=True)
class Layout(LayoutMixin):
    """Parameters controlling the sizes of heaps and chunks.

    Parameters
    ----------
    sample_bits
        Bits per sample (for each of real and imaginary).
    n_channels
        Total number of channels.
    n_channels_per_substream
        The number of frequency channels in each beam substream.
    n_spectra_per_heap
        The number of samples on the time axis in each heap.
    chunk_batches
        Number of batches per chunk.
    heap_timestamp_step
        Increase in timestamp between successive heaps. Timestamps
        must also be a multiple of this value.
    """

    sample_bits: int
    n_channels: int
    n_channels_per_substream: int
    n_spectra_per_heap: int
    chunk_batches: int
    heap_timestamp_step: int

    def __post_init__(self) -> None:
        # Could probably be relaxed in future, but would need to be investigated.
        if self.sample_bits % BYTE_BITS != 0:
            raise ValueError(f"sample_bits must be a multiple of {BYTE_BITS}")
        if self.n_channels % self.n_channels_per_substream != 0:
            raise ValueError(
                f"n_channels ({self.n_channels}) is not a multiple of "
                f"n_channels_per_substream ({self.n_channels_per_substream})"
            )

    @property
    def heap_bytes(self) -> int:  # noqa: D102
        return self.n_channels_per_substream * self.n_spectra_per_heap * self.sample_bits * COMPLEX // BYTE_BITS

    @property
    def n_pol_substreams(self) -> int:
        """Number of substreams in each polarisation."""
        return self.n_channels // self.n_channels_per_substream

    @property
    def batch_heaps(self) -> int:  # noqa: D102
        return self.n_pol_substreams * N_POLS

    @property
    def chunk_timestamp_step(self) -> int:  # noqa: D102
        return self.heap_timestamp_step * self.chunk_batches

    @property
    def heap_samples(self) -> int:  # noqa: D102
        return self.n_spectra_per_heap * self.n_channels_per_substream

    @functools.cached_property
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        heap_bytes = self.heap_bytes
        heap_timestamp_step = self.heap_timestamp_step
        n_channels = self.n_channels
        n_channels_per_substream = self.n_channels_per_substream
        chunk_batches = self.chunk_batches
        n_pol_substreams = self.n_pol_substreams
        n_heaps_per_pol = chunk_batches * n_pol_substreams
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
            items = numba.carray(intp_to_voidptr(data[0].items), 4, dtype=np.int64)
            timestamp = items[0]
            frequency = items[1]
            beam_ants = items[2]  # TODO should this be used to flag bad data?
            payload_size = items[3]
            pol = user_data[0].pol
            if payload_size != heap_bytes or timestamp < 0 or frequency < 0 or beam_ants < 0:
                # Probably a metadata heap - ignore it
                batch_stats[user_data[0].stats_base + _Statistic.METADATA_HEAPS] += 1
                return
            if timestamp % heap_timestamp_step != 0:
                # Invalid timestamp
                batch_stats[user_data[0].stats_base + _Statistic.BAD_TIMESTAMP_HEAPS] += 1
                return
            if frequency % n_channels_per_substream != 0 or frequency >= n_channels:
                # Invalid frequency
                batch_stats[user_data[0].stats_base + _Statistic.BAD_FREQUENCY_HEAPS] += 1
                return
            # Heap index on the time axis, from timestamp 0
            heap_time_abs = timestamp // heap_timestamp_step
            data[0].chunk_id = heap_time_abs // chunk_batches
            # Position of this heap on the time axis, from the start of the chunk
            heap_time = heap_time_abs % chunk_batches
            # Position of this heap on the frequency axis
            heap_freq = frequency // n_channels_per_substream
            data[0].heap_index = pol * n_heaps_per_pol + heap_time * n_pol_substreams + heap_freq
            data[0].heap_offset = data[0].heap_index * heap_bytes

        return chunk_place_impl


def make_stream_group(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    recv_affinity: int,
    pol_labels: Sequence[str],
) -> spead2.recv.ChunkStreamRingGroup:
    """Create a stream group for receiving dual-polarised beam data.

    The readers are not added to the streams.

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
        Use -1 to indicate no affinity.
    pol_labels
        Prometheus labels to apply to the polarisations (must have length 2).
    """
    # Reference counters to make the labels exist before the first scrape
    assert len(pol_labels) == N_POLS
    for pol in pol_labels:
        counters.labels(pol)

    user_data = np.zeros(N_POLS, dtype=user_data_type.dtype)
    user_data["pol"] = np.arange(N_POLS)
    group = base_recv.make_stream_group(
        layout=layout,
        spead_items=[TIMESTAMP_ID, FREQUENCY_ID, BEAM_ANTS_ID, spead2.HEAP_LENGTH_ID],
        max_active_chunks=MAX_CHUNKS,
        data_ringbuffer=data_ringbuffer,
        free_ringbuffer=free_ringbuffer,
        affinity=[recv_affinity] * N_POLS,
        stream_stats=["katgpucbf.metadata_heaps", "katgpucbf.bad_timestamp_heaps", "katgpucbf.bad_frequency_heaps"],
        user_data=user_data,
        substreams=layout.n_pol_substreams,
        explicit_start=True,
    )
    stats_collector.add_stream_group(group)
    return group


def iter_chunks(
    ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    layout: Layout,
    sensors: aiokatcp.SensorSet,
    time_converter: TimeConverter,
    pol_labels: Sequence[str],
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
    pol_labels
        Input polarisation labels (must match those passed to :func:`make_stream_group`).
    """
    return base_recv.iter_chunks(
        ringbuffer,
        layout,
        sensors,
        time_converter,
        [(pol, pol) for pol in pol_labels],
        counters,
        stats_collector,
    )
