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

"""Recv module."""

import functools
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import AsyncGenerator, List, Optional, cast

import numba
import numpy as np
import spead2.recv.asyncio
from numba import types
from prometheus_client import Counter
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .. import BYTE_BITS
from ..recv import BaseLayout, Chunk, StatsToCounters, user_data_type
from . import METRIC_NAMESPACE

logger = logging.getLogger(__name__)

heaps_counter = Counter("input_heaps", "number of heaps received", ["pol"], namespace=METRIC_NAMESPACE)
chunks_counter = Counter("input_chunks", "number of chunks received", ["pol"], namespace=METRIC_NAMESPACE)
bytes_counter = Counter(
    "input_bytes", "number of bytes of digitiser samples received", ["pol"], namespace=METRIC_NAMESPACE
)
too_old_heaps_counter = Counter(
    "input_too_old_heaps", "number of heaps that arrived too late to be processed", ["pol"], namespace=METRIC_NAMESPACE
)
missing_heaps_counter = Counter(
    "input_missing_heaps", "number of heaps dropped on the input", ["pol"], namespace=METRIC_NAMESPACE
)
metadata_heaps_counter = Counter(
    "input_metadata_heaps", "number of heaps not containing payload", ["pol"], namespace=METRIC_NAMESPACE
)
bad_timestamp_heaps_counter = Counter(
    "input_bad_timestamp_heaps", "timestamp not a multiple of samples per packet", ["pol"], namespace=METRIC_NAMESPACE
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
    def chunk_bytes(self) -> int:  # noqa: D401
        """Number of bytes per chunk."""
        return self.chunk_samples * self.sample_bits // BYTE_BITS

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
        def chunk_place_impl(data_ptr, data_size, user_data_ptr):  # pragma: nocover
            data = numba.carray(data_ptr, 1)
            items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
            timestamp = items[0]
            payload_size = items[1]
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

        return chunk_place_impl


async def chunk_sets(
    streams: List[spead2.recv.ChunkRingStream],
    layout: Layout,
) -> AsyncGenerator[List[Chunk], None]:
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
    """
    n_pol = len(streams)
    # Working buffer to match up pairs of chunks from both pols.
    buf: List[Optional[Chunk]] = [None] * n_pol
    ring = cast(spead2.recv.asyncio.ChunkRingbuffer, streams[0].data_ringbuffer)
    lost = 0
    stats_to_counters = [
        StatsToCounters(
            {
                "too_old_heaps": too_old_heaps_counter.labels(pol),
                "katgpucbf.metadata_heaps": metadata_heaps_counter.labels(pol),
                "katgpucbf.bad_timestamp_heaps": bad_timestamp_heaps_counter.labels(pol),
            },
            stream.config,
        )
        for pol, stream in enumerate(streams)
    ]
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
            if first_timestamp == -1:
                # TODO: use chunk.present to determine the actual first timestamp
                first_timestamp = chunk.timestamp
            good = np.sum(chunk.present)
            lost += layout.chunk_heaps - good
            logger.debug(
                "Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)",
                chunk.timestamp,
                pol,
                good,
                layout.chunk_heaps,
                lost,
            )

            # Update stream statistics. Note that these are not necessarily
            # synchronised with the chunk.
            for updater, stream in zip(stats_to_counters, streams):
                updater.update(stream.stats)

            # Check whether we have a chunk already for this pol.
            old = buf[pol]
            if old is not None:
                logger.warning("Chunk not matched: timestamp=%#x pol=%d", old.chunk_id * layout.chunk_samples, pol)
                # Chunk was passed by without getting used. Return to the pool.
                streams[pol].add_free_chunk(old)
                buf[pol] = None

            # Stick the chunk in the buffer.
            buf[pol] = chunk

            # If we have both chunks and they match up, then we can yield.
            if all(c is not None and c.chunk_id == chunk.chunk_id for c in buf):
                expected_heaps = (chunk.timestamp - first_timestamp + layout.chunk_samples) // layout.heap_samples
                # mypy isn't smart enough to see that the list can't have Nones
                # in it at this point.
                for c in cast(List[Chunk], buf):
                    pol = c.stream_id
                    # The cast is to force numpy ints to Python ints.
                    buf_good = int(np.sum(c.present))
                    heaps_counter.labels(pol).inc(buf_good)
                    chunks_counter.labels(pol).inc()
                    bytes_counter.labels(pol).inc(buf_good * layout.heap_bytes)
                    # Determine how many heaps we expected to have seen by
                    # now, and subtract from it the number actually seen to
                    # determine the number missing. This accounts for both
                    # heaps lost within chunks and lost chunks.
                    n_heaps[c.stream_id] += buf_good
                    new_missing = expected_heaps - n_heaps[pol]
                    if new_missing > n_missing_heaps[pol]:
                        missing_heaps_counter.labels(pol).inc(new_missing - n_missing_heaps[pol])
                        n_missing_heaps[pol] = new_missing

                # mypy isn't smart enough to see that the list can't have Nones
                # in it at this point.
                yield buf  # type: ignore
                # Empty the buffer again for next use.
                buf = [None] * n_pol
    finally:
        for c2 in buf:
            if c2 is not None:
                streams[c2.stream_id].add_free_chunk(c2)


__all__ = ["Chunk", "Layout", "chunk_sets"]
