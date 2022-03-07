################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Shared utilities for receiving SPEAD data."""

import ctypes
from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, Tuple, Union

import numba
import numpy as np
import scipy
import spead2.recv
import spead2.recv.asyncio
from numba import types
from prometheus_client import Counter

user_data_type = types.Record.make_c_struct(
    [
        ("stats_base", types.uintp),  # Index for first custom statistic
    ]
)


class Chunk(spead2.recv.Chunk):
    """Collection of heaps passed to the GPU at one time.

    It extends the spead2 base class to store a timestamp (computed from
    the chunk ID when the chunk is received), and optionally store a
    gdrcopy device array.
    """

    # Refine the types used in the base class
    present: np.ndarray
    data: np.ndarray
    # New fields
    device: object
    timestamp: int

    def __init__(self, *args, device: object = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.timestamp = 0  # Actual value filled in when chunk received


class StatsToCounters:
    """Reflect stream statistics as Prometheus Counters.

    This class tracks the last-known value of each statistic for the
    corresponding stream.

    Parameters
    ----------
    counter_map
        Dictionary that maps a stream statistic (by name) to a counter.
    config
        Configuration of the stream for which the statistics will be retrieved.
    """

    def __init__(self, counter_map: Mapping[str, Counter], config: spead2.recv.StreamConfig) -> None:
        self._updates = [(config.get_stat_index(name), counter) for name, counter in counter_map.items()]
        self._current = [0] * len(self._updates)

    def update(self, stats: spead2.recv.StreamStats) -> None:
        """Update the counters based on the current stream statistics."""
        for i, (index, counter) in enumerate(self._updates):
            new = stats[index]
            counter.inc(new - self._current[i])
            self._current[i] = new


class BaseLayout(ABC):
    """Abstract base class for chunk layouts to derive from."""

    @property
    @abstractmethod
    def heap_bytes(self) -> int:  # noqa: D401
        """Number of payload bytes per heap."""
        ...

    @property
    @abstractmethod
    def chunk_heaps(self) -> int:  # noqa: D401
        """Number of heaps per chunk."""
        ...

    @property
    def chunk_bytes(self) -> int:  # noqa: D401
        """Number of bytes per chunk."""
        return self.heap_bytes * self.chunk_heaps

    @property
    @abstractmethod
    def _chunk_place(self) -> numba.core.ccallback.CFunc:
        ...

    def chunk_place(self, stats_base: int) -> scipy.LowLevelCallable:
        """Generate low-level code for placing heaps in chunks.

        Parameters
        ----------
        stats_base
            Index of first custom statistic
        """
        user_data = np.zeros(1, dtype=user_data_type.dtype)
        user_data["stats_base"] = stats_base
        return scipy.LowLevelCallable(
            self._chunk_place.ctypes,
            user_data=user_data.ctypes.data_as(ctypes.c_void_p),
            signature="void (void *, size_t, void *)",
        )


def make_stream(
    layout: BaseLayout,
    spead_items: List[int],
    max_active_chunks: int,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    affinity: int,
    max_heaps: int,
    stream_stats: List[str],
    *,
    stream_id: int = 0,
) -> spead2.recv.ChunkRingStream:
    """Create a SPEAD receiver stream.

    Parameters
    ----------
    layout
        Heap size and chunking parameters
    spead_items
        List of SPEAD item IDs to be expected in the heap headers.
    max_active_chunks
        Maximum number of chunks under construction.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent
    affinity
        CPU core affinity for the worker thread (negative to not set an affinity)
    max_heaps
        Maximum number of heaps to have open at once, increase to account for
        packets from multiple heaps arriving in a disorderly fashion (likely due
        to multiple senders sending to the multicast endpoint being received).
    stream_stats
        Stats to hook up to prometheus
    stream_id
        Stream ID parameter to pass through to the stream config. Canonical use-
        case is the polarisation index in the F-engine.
    """
    stream_config = spead2.recv.StreamConfig(
        max_heaps=max_heaps,
        memcpy=spead2.MEMCPY_NONTEMPORAL,
        stream_id=stream_id,
    )
    stats_base = stream_config.next_stat_index()
    for stat in stream_stats:
        stream_config.add_stat(stat)

    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=spead_items,
        max_chunks=max_active_chunks,
        place=layout.chunk_place(stats_base),
    )
    # Ringbuffer size is largely arbitrary: just needs to be big enough to
    # never fill up.
    free_ringbuffer = spead2.recv.ChunkRingbuffer(128)
    return spead2.recv.ChunkRingStream(
        spead2.ThreadPool(1, [] if affinity < 0 else [affinity]),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )


def add_reader(
    stream: spead2.recv.ChunkRingStream,
    *,
    src: Union[str, List[Tuple[str, int]]],
    interface: Optional[str],
    ibv: bool,
    comp_vector: int,
    buffer: int,
) -> None:
    """Connect a stream to an underlying transport.

    See the documentation for :class:`.Engine` for an explanation of the parameters.
    """
    if isinstance(src, str):
        stream.add_udp_pcap_file_reader(src)
    elif ibv:
        if interface is None:
            raise ValueError("--src-interface is required with --src-ibv")
        ibv_config = spead2.recv.UdpIbvConfig(
            endpoints=src,
            interface_address=interface,
            buffer_size=buffer,
            comp_vector=comp_vector,
        )
        stream.add_udp_ibv_reader(ibv_config)
    else:
        buffer_size = buffer // len(src)  # split it across the endpoints
        for endpoint in src:
            stream.add_udp_reader(
                endpoint[0],
                endpoint[1],
                buffer_size=buffer_size,
                interface_address=interface or "",
            )
