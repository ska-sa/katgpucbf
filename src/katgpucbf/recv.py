################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numba.core.ccallback
import numpy as np
import scipy
import spead2.recv
import spead2.recv.asyncio
from numba import types
from prometheus_client import REGISTRY, CollectorRegistry, Metric
from prometheus_client.core import CounterMetricFamily
from prometheus_client.registry import Collector

user_data_type = types.Record.make_c_struct(
    [
        ("stats_base", types.uintp),  # Index for first custom statistic
    ]
)


class Chunk(spead2.recv.Chunk):
    """Collection of heaps passed to the GPU at one time.

    It extends the spead2 base class to store a timestamp (computed from
    the chunk ID when the chunk is received), and optionally store a
    vkgdr device array.
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


class StatsCollector(Collector):
    """Collect statistics from spead2 streams as Prometheus metrics."""

    @dataclass
    class _StreamInfo:
        """Information about a single registered stream."""

        stream: "weakref.ReferenceType[spead2.recv.ChunkRingStream]"
        indices: List[int]  # Indices of counters, in the order given by counter_map
        prev: List[int]  # Amounts already counted

    class _LabelSet:
        """Information shared by all streams with the same set of labels."""

        labels: Tuple[str, ...]
        totals: Dict[str, int]  # sum over all streams, indexed by stat name
        created: float  # time of creation
        streams: List["StatsCollector._StreamInfo"]

        def __init__(self, labels: Tuple[str, ...], stat_names: Iterable[str]) -> None:
            self.labels = labels
            self.totals = {stat_name: 0 for stat_name in stat_names}
            self.created = time.time()
            self.streams = []

        def add_stream(self, stream: spead2.recv.ChunkRingStream) -> None:
            """Register a new stream."""
            config = stream.config
            indices = [config.get_stat_index(name) for name in self.totals.keys()]
            # Get the current statistics and immediately update with them
            stats = stream.stats
            prev = []
            for i, stat_name in enumerate(self.totals.keys()):
                cur = stats[indices[i]]
                self.totals[stat_name] += cur
                prev.append(cur)
            self.streams.append(StatsCollector._StreamInfo(weakref.ref(stream), indices, prev))

        def update(self) -> None:
            """Fetch statistics from all streams and update totals."""
            # We build a new copy of the streams list which excludes any that
            # were garbage collected.
            new_streams = []
            for stream_info in self.streams:
                stream = stream_info.stream()
                if stream is not None:
                    new_streams.append(stream_info)
                    stats = stream.stats
                    for i, stat_name in enumerate(self.totals.keys()):
                        cur = stats[stream_info.indices[i]]
                        self.totals[stat_name] += cur - stream_info.prev[i]
                        stream_info.prev[i] = cur
            self.streams = new_streams

    @staticmethod
    def _build_full_name(namespace: str, name: str) -> str:
        """Combine the (optional) namespace with the name."""
        full_name = ""
        if namespace:
            full_name += namespace + "_"
        full_name += name
        return full_name

    def __init__(
        self,
        counter_map: Mapping[str, Tuple[str, str]],
        labelnames: Iterable[str] = (),
        namespace: str = "",
        registry: CollectorRegistry = REGISTRY,
    ) -> None:
        self._counter_map = {
            stat_name: (self._build_full_name(namespace, name), description)
            for stat_name, (name, description) in counter_map.items()
        }
        self._labelnames = tuple(labelnames)
        self._label_sets: Dict[Tuple[str, ...], "StatsCollector._LabelSet"] = {}
        if registry:
            registry.register(self)

    def update(self) -> None:
        """Update the internal totals from the streams.

        This is done automatically by :meth:`collect`, but it can also be
        called explicitly. This may be useful to do just before a stream
        goes out of scope, to ensure that counter updates since the last
        scrape are not lost when the stream is garbage collected.
        """
        for label_set in self._label_sets.values():
            label_set.update()

    def add_stream(self, stream: spead2.recv.ChunkRingStream, labels: Iterable[str] = ()) -> None:
        """Register a new stream.

        If the collector was constructed with a non-empty ``labelnames``, then
        ``labels`` must contain the same number of elements to provide the
        labels for the metrics that this stream will update.

        .. warning::

           Calling this more than once with the same stream will cause that
           stream's statistics to be counted multiple times.
        """
        labels_tuple = tuple(labels)
        if len(labels_tuple) != len(self._labelnames):
            raise ValueError("labels must have the same length as labelnames")
        if labels_tuple not in self._label_sets:
            self._label_sets[labels_tuple] = self._LabelSet(labels_tuple, self._counter_map.keys())
        self._label_sets[labels_tuple].add_stream(stream)

    def collect(self) -> Iterable[Metric]:
        """Implement Prometheus' Collector interface."""
        self.update()
        for stat_name, (counter_name, counter_help) in self._counter_map.items():
            metric = CounterMetricFamily(counter_name, counter_help, labels=self._labelnames)
            for labels, label_set in self._label_sets.items():
                metric.add_metric(labels, label_set.totals[stat_name], created=label_set.created)
            yield metric


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
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    affinity: int,
    stream_stats: List[str],
    **kwargs: Any,
) -> spead2.recv.ChunkRingStream:
    """Create a SPEAD receiver stream.

    Parameters
    ----------
    layout
        Heap size and chunking parameters.
    spead_items
        List of SPEAD item IDs to be expected in the heap headers.
    max_active_chunks
        Maximum number of chunks under construction.
    data_ringbuffer
        Output ringbuffer to which chunks will be sent.
    free_ringbuffer
        Ringbuffer for holding chunks for recycling once they've been used.
    affinity
        CPU core affinity for the worker thread (negative to not set an affinity).
    stream_stats
        Stats to hook up to prometheus.
    kwargs
        Other keyword arguments are passed to :class:`spead2.recv.StreamConfig`.
    """
    stream_config = spead2.recv.StreamConfig(memcpy=spead2.MEMCPY_NONTEMPORAL, **kwargs)
    stats_base = stream_config.next_stat_index()
    for stat in stream_stats:
        stream_config.add_stat(stat)

    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=spead_items,
        max_chunks=max_active_chunks,
        place=layout.chunk_place(stats_base),
    )

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
