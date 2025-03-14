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

"""Network transmission handling."""

import asyncio
import functools
from collections.abc import Callable, Sequence
from typing import Required, TypedDict, Unpack

import numpy as np
import spead2.send
import spead2.send.asyncio
from aiokatcp import SensorSet
from katsdptelstate.endpoint import Endpoint
from prometheus_client import Counter

from .. import COMPLEX, N_POLS
from ..send import send_rate
from ..spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_DTYPE, IMMEDIATE_FORMAT, TIMESTAMP_ID
from ..utils import TimeConverter
from . import METRIC_NAMESPACE

#: Number of non-payload bytes per packet (header, 8 items pointers)
PREAMBLE_SIZE = 72
output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", ["stream"], namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter(
    "output_bytes", "number of payload bytes transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_samples_counter = Counter(
    "output_samples", "number of complex samples transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
skipped_heaps_counter = Counter(
    "output_skipped_heaps", "heaps not sent because input data was incomplete", ["stream"], namespace=METRIC_NAMESPACE
)
output_clip_counter = Counter(
    "output_clipped_samples", "number of samples that were saturated", ["stream", "pol"], namespace=METRIC_NAMESPACE
)


class _FengRawKwargs(TypedDict, total=False):
    """Helper class for type annotations."""

    shape: Required[tuple[int, ...]]
    dtype: np.dtype
    format: list[tuple[str, int]]


def make_item_group(**feng_raw_kwargs: Unpack[_FengRawKwargs]) -> spead2.send.ItemGroup:
    """Create an item group (without values).

    The `feng_raw_kwargs` must specify the shape and dtype/format for the
    ``feng_raw`` item.
    """
    item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
    item_group.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        FENG_ID_ID,
        "feng_id",
        "Uniquely identifies the F-Engine source for the data.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        FREQUENCY_ID,
        "frequency",
        "Identifies the first channel in the band of frequencies in the SPEAD heap.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        FENG_RAW_ID,
        "feng_raw",
        "Channelised complex data from both polarisations of digitiser associated with F-Engine.",
        **feng_raw_kwargs,
    )
    return item_group


class Batch:
    """Holds all the heaps for a single timestamp.

    It does not own its memory - the backing store is in :class:`Chunk`.

    Parameters
    ----------
    timestamp
        Zero-dimensional array of dtype ``>u8`` holding the timestamp.
    data
        Payload data for the batch, of shape (channels, spectra_per_heap, N_POLS).
    saturated
        Saturation data for the batch, of shape (N_POLS,)
    feng_id
        Value to put in ``feng_id`` SPEAD item
    n_substreams
        Number of substreams into which the channels are divided
    """

    def __init__(
        self, timestamp: np.ndarray, data: np.ndarray, saturated: np.ndarray, *, n_substreams: int, feng_id: int
    ) -> None:
        n_channels = data.shape[0]
        assert n_channels % n_substreams == 0
        n_channels_per_substream = n_channels // n_substreams
        self.heaps = []
        self.data = data
        self.saturated = saturated

        item_group = make_item_group(shape=(n_channels_per_substream,) + data.shape[1:], dtype=data.dtype)
        item_group[TIMESTAMP_ID].value = timestamp
        item_group[FENG_ID_ID].value = feng_id
        for i in range(n_substreams):
            start_channel = i * n_channels_per_substream
            heap_data = data[start_channel : start_channel + n_channels_per_substream]
            assert heap_data.flags.c_contiguous, "Heap data must be contiguous"
            item_group[FREQUENCY_ID].value = start_channel
            item_group[FENG_RAW_ID].value = heap_data
            heap = item_group.get_heap(data="all", descriptors="none")
            heap.repeat_pointers = True
            self.heaps.append(spead2.send.HeapReference(heap, substream_index=i))


def _multi_send(
    streams: list["spead2.send.asyncio.AsyncStream"], heaps: list[spead2.send.HeapReference]
) -> asyncio.Future:
    """Send a list of heaps across several streams.

    The list of heaps is broken into contiguous blocks, with each block sent to
    one stream.

    This returns a future, rather than being a coroutine. Thus, it should not
    be wrapped with :func:`asyncio.create_task`.
    """
    if len(streams) == 1:  # Most common case
        return streams[0].async_send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
    else:
        futures = []
        for i, stream in enumerate(streams):
            first = i * len(heaps) // len(streams)
            last = (i + 1) * len(heaps) // len(streams)
            futures.append(stream.async_send_heaps(heaps[first:last], spead2.send.GroupMode.ROUND_ROBIN))
        return asyncio.gather(*futures)


class Chunk:
    """An array of batches, spanning multiple timestamps.

    Parameters
    ----------
    data
        Storage for voltage data, with shape (n_batches, n_channels,
        n_spectra_per_heap, N_POLS) and a dtype returned by
        :func:`.gaussian_dtype`.
    saturated
        Storage for saturation counts, with shape (n_batches, N_POLS)
        and dtype uint32.
    n_substreams
        Number of substreams over which the data will be divided
        (must divide evenly into the number of channels).
    feng_id
        F-Engine ID to place in the SPEAD heaps
    spectra_samples
        Difference in timestamps between successive batches
    """

    def __init__(
        self,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        n_substreams: int,
        feng_id: int,
        spectra_samples: int,
    ) -> None:
        n_batches = data.shape[0]
        n_channels = data.shape[1]
        n_spectra_per_heap = data.shape[2]
        if n_channels % n_substreams != 0:
            raise ValueError("n_substreams must divide into n_channels")
        self.data = data
        self.saturated = saturated
        #: Whether each batch has valid data
        self.present = np.zeros(n_batches, dtype=bool)
        #: Timestamp of the first heap
        self._timestamp = 0
        #: Callback to return the chunk to the appropriate queue
        self.cleanup: Callable[[], None] | None = None
        self._timestamp_step = n_spectra_per_heap * spectra_samples
        #: Storage for timestamps in the SPEAD heaps.
        self._timestamps = (np.arange(n_batches) * self._timestamp_step).astype(IMMEDIATE_DTYPE)
        # The ... in indexing causes numpy to give a 0d array view, rather than
        # a scalar.
        self._batches = [
            Batch(self._timestamps[i, ...], data[i], saturated[i], feng_id=feng_id, n_substreams=n_substreams)
            for i in range(n_batches)
        ]

    @property
    def timestamp(self) -> int:
        """Timestamp of the first heap.

        Setting this property updates the timestamps stored in all the heaps.
        This should not be done while a previous call to :meth:`send` is still
        in progress.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        delta = value - self._timestamp
        self._timestamps += delta
        self._timestamp = value

    @staticmethod
    def _inc_counters(batch: Batch, output_name: str, future: asyncio.Future) -> None:
        if not future.cancelled() and future.exception() is None:
            output_heaps_counter.labels(output_name).inc(len(batch.heaps))
            output_bytes_counter.labels(output_name).inc(batch.data.nbytes)
            output_samples_counter.labels(output_name).inc(batch.data.size)
            for pol in range(N_POLS):
                output_clip_counter.labels(output_name, pol).inc(batch.saturated[pol])

    async def send(
        self,
        streams: list["spead2.send.asyncio.AsyncStream"],
        batches: int,
        time_converter: TimeConverter,
        sensors: SensorSet,
        output_name: str,
    ) -> None:
        """Transmit heaps over SPEAD streams.

        Batches from 0 to `batches` - 1 are sent asynchronously. The contents of
        each batch are distributed over the streams. If the number of streams
        does not divide into the number of destination endpoints, there will be
        imbalances, because the partitioning is the same for every batch.
        """
        futures = []
        saturated = [0] * N_POLS
        for present, batch in zip(self.present[:batches], self._batches[:batches], strict=True):
            if present:
                futures.append(_multi_send(streams, batch.heaps))
                futures[-1].add_done_callback(functools.partial(self._inc_counters, batch, output_name))
                for pol in range(N_POLS):
                    # Cast np.uint32 to Python int to avoid overflow
                    saturated[pol] += int(batch.saturated[pol])
            else:
                skipped_heaps_counter.labels(output_name).inc(len(batch.heaps))
        if futures:
            await asyncio.gather(*futures)
        end_timestamp = self._timestamp + self._timestamp_step * len(self._batches)
        end_time = time_converter.adc_to_unix(end_timestamp)
        for pol in range(N_POLS):
            sensor = sensors[f"{output_name}.input{pol}.feng-clip-cnt"]
            sensor.set_value(sensor.value + saturated[pol], timestamp=end_time)


def make_streams(
    *,
    output_name: str,
    thread_pool: spead2.ThreadPool,
    endpoints: list[Endpoint],
    interfaces: list[str],
    ttl: int,
    ibv: bool,
    packet_payload: int,
    comp_vector: int,
    buffer: int,
    bandwidth: float,
    send_rate_factor: float,
    feng_id: int,
    n_ants: int,
    n_data_heaps: int,
    chunks: Sequence[Chunk],
) -> list["spead2.send.asyncio.AsyncStream"]:
    """Create asynchronous SPEAD streams for transmission.

    Each stream is configured with substreams for all the end-points. They
    differ only in the network interface used (there is one per interface).
    Thus, they can be used interchangeably for load-balancing purposes.
    """
    dtype = chunks[0].data.dtype  # Type for each complex value
    memory_regions: list[object] = [chunk.data for chunk in chunks]
    heap_payload = sum(chunk.data.nbytes for chunk in chunks) // n_data_heaps
    # Work backwards from payload byte rate to get heap payload rate
    heap_interval = heap_payload / (N_POLS * bandwidth * dtype.itemsize)
    config = spead2.send.StreamConfig(
        rate=send_rate(
            packet_header=PREAMBLE_SIZE,
            packet_payload=packet_payload,
            heap_payload=heap_payload,
            heap_interval=heap_interval,
            send_rate_factor=send_rate_factor,
        )
        / len(interfaces),
        max_packet_size=packet_payload + PREAMBLE_SIZE,
        # Adding len(endpoints) to accommodate descriptors sent for each substream
        max_heaps=n_data_heaps + len(endpoints),
    )
    streams: list[spead2.send.asyncio.AsyncStream]
    if ibv:
        ibv_configs = [
            spead2.send.UdpIbvConfig(
                endpoints=[(ep.host, ep.port) for ep in endpoints],
                interface_address=interface,
                ttl=ttl,
                comp_vector=comp_vector,
                memory_regions=memory_regions,
                buffer_size=buffer // len(interfaces),
            )
            for interface in interfaces
        ]
        streams = [spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config) for ibv_config in ibv_configs]
    else:
        streams = [
            spead2.send.asyncio.UdpStream(
                thread_pool,
                [(ep.host, ep.port) for ep in endpoints],
                config,
                ttl=ttl,
                interface_address=interface,
                buffer_size=buffer // len(interfaces),
            )
            for interface in interfaces
        ]
    for i, stream in enumerate(streams):
        # Ensure that streams do not interfere with each other or with those of
        # other F-engines. This assumes that there are at most 256
        # interfaces. IDs may get reused after 2^40/n_ants heaps, which should
        # be much larger than a receiver's window.
        stream.set_cnt_sequence((i << 40) + feng_id, n_ants)
    # Referencing the labels causes them to be created, in advance of data
    # actually being transmitted.
    output_heaps_counter.labels(output_name)
    output_bytes_counter.labels(output_name)
    output_samples_counter.labels(output_name)
    skipped_heaps_counter.labels(output_name)
    for pol in range(N_POLS):
        output_clip_counter.labels(output_name, pol)
    return streams


def make_descriptor_heap(
    *,
    channels_per_substream: int,
    spectra_per_heap: int,
    sample_bits: int,
) -> spead2.send.Heap:
    """Create a descriptor heap for output F-Engine data."""
    raw_kwargs: _FengRawKwargs = {"shape": (channels_per_substream, spectra_per_heap, N_POLS, COMPLEX)}
    try:
        raw_kwargs["dtype"] = np.dtype(f"int{sample_bits}")
    except TypeError:
        # The number of bits doesn't neatly fit a numpy dtype
        raw_kwargs["format"] = [("i", sample_bits)]

    item_group = make_item_group(**raw_kwargs)
    return item_group.get_heap(descriptors="all", data="none")
