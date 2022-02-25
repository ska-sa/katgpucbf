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

"""Network transmission handling."""

import asyncio
import functools
from typing import Iterable, List, Optional, Sequence

import numpy as np
import spead2.send
import spead2.send.asyncio
from katsdpsigproc.accel import DeviceArray
from katsdptelstate.endpoint import Endpoint
from prometheus_client import Counter

from .. import N_POLS
from ..spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, TIMESTAMP_ID
from . import METRIC_NAMESPACE

#: Number of non-payload bytes per packet (header, 8 items pointers)
PREAMBLE_SIZE = 72
output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter("output_bytes", "number of payload bytes transmitted", namespace=METRIC_NAMESPACE)


class Frame:
    """Holds all the heaps for a single timestamp.

    It does not own its memory - the backing store is in :class:`Chunk`.

    Parameters
    ----------
    timestamp
        Zero-dimensional array of dtype ``>u8`` holding the timestamp.
    data
        Payload data for the frame, of shape (channels, spectra_per_heap, N_POLS, COMPLEX).
    feng_id
        Value to put in ``feng_id`` SPEAD item
    substreams
        Number of substreams into which the channels are divided
    """

    def __init__(self, timestamp: np.ndarray, data: np.ndarray, *, substreams: int, feng_id: int) -> None:
        channels = data.shape[0]
        assert channels % substreams == 0
        channels_per_substream = channels // substreams
        self.descriptor_heaps = []
        self.heaps = []

        imm_format = [("u", FLAVOUR.heap_address_bits)]
        self.data = data

        # We don't necessarily need a new ItemGroup for each substream, just
        # one that has updated Frequency (start_channel) and data fields.
        new_ig = spead2.send.ItemGroup(flavour=FLAVOUR)
        new_ig.add_item(
            TIMESTAMP_ID,
            "timestamp",
            "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
            shape=(),
            format=imm_format,
            value=timestamp,
        )
        new_ig.add_item(
            FENG_ID_ID,
            "feng_id",
            "Uniquely identifies the F-Engine source for the data.",
            shape=(),
            format=imm_format,
            value=feng_id,
        )
        new_ig.add_item(
            FREQUENCY_ID,
            "frequency",
            "Identifies the first channel in the band of frequencies in the SPEAD heap.",
            shape=(),
            format=imm_format,
        )
        new_ig.add_item(
            FENG_RAW_ID,
            "feng_raw",
            "Channelised complex data from both polarisations of digitiser associated with F-Engine.",
            shape=data[:channels_per_substream].shape,
            dtype=data.dtype,
        )

        for i in range(substreams):
            start_channel = i * channels_per_substream
            heap_data = data[start_channel : start_channel + channels_per_substream]
            assert heap_data.flags.c_contiguous, "Heap data must be contiguous"

            new_ig["frequency"].value = start_channel
            new_ig["feng_raw"].value = heap_data

            descriptor_heap = new_ig.get_heap(descriptors="all", data="none")
            # TODO: Change this to an ndarray, so we can send all descriptors
            #       of the same substream at the same time.
            self.descriptor_heaps.append(spead2.send.HeapReference(descriptor_heap, substream_index=i))

            heap_to_send = new_ig.get_heap(descriptors="none", data="all")
            heap_to_send.repeat_pointers = True
            self.heaps.append(spead2.send.HeapReference(heap_to_send, substream_index=i))


class Chunk:
    """An array of frames, spanning multiple timestamps."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        device: Optional[DeviceArray] = None,
        substreams: int,
        feng_id: int,
    ) -> None:
        n_frames = data.shape[0]
        channels = data.shape[1]
        spectra_per_heap = data.shape[2]
        if channels % substreams != 0:
            raise ValueError("substreams must divide into channels")
        self.data = data
        self.device = device
        #: Timestamp of the first heap
        self._timestamp = 0
        timestamp_step = spectra_per_heap * channels * 2
        #: Storage for timestamps in the SPEAD heaps.
        self._timestamps = (np.arange(n_frames) * timestamp_step).astype(">u8")
        # The ... in indexing causes numpy to give a 0d array view, rather than
        # a scalar.
        self._frames = [
            Frame(self._timestamps[i, ...], data[i], feng_id=feng_id, substreams=substreams) for i in range(n_frames)
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

    @property
    def descriptor_heaps(self) -> List[spead2.send.HeapReference]:
        """Descriptor heaps of all Frames within the Chunk.

        This removes the need to reach into each :class:`Frame` to grab the
        generated descriptor heaps for each substream.
        """
        return [descriptor_heap for frame in self._frames for descriptor_heap in frame.descriptor_heaps]

    @staticmethod
    def _inc_counters(frame: Frame, future: asyncio.Future) -> None:
        if not future.cancelled() and future.exception() is None:
            output_heaps_counter.inc(len(frame.heaps))
            output_bytes_counter.inc(frame.data.nbytes)

    async def send(self, stream: "spead2.send.asyncio.AsyncStream", frames: int) -> None:
        """Transmit heaps on a SPEAD stream.

        Frames from 0 to `frames` - 1 are sent asynchronously.
        """
        futures = []
        for frame in self._frames[:frames]:
            futures.append(stream.async_send_heaps(frame.heaps, spead2.send.GroupMode.ROUND_ROBIN))
            futures[-1].add_done_callback(functools.partial(self._inc_counters, frame))
        await asyncio.gather(*futures)


def make_stream(
    *,
    endpoints: List[Endpoint],
    interface: str,
    ttl: int,
    ibv: bool,
    packet_payload: int,
    affinity: int,
    comp_vector: int,
    adc_sample_rate: float,
    send_rate_factor: float,
    feng_id: int,
    num_ants: int,
    spectra: int,
    spectra_per_heap: int,
    channels: int,
    chunks: Sequence[Chunk],
    extra_memory_regions: Optional[Iterable[object]],
) -> "spead2.send.asyncio.AsyncStream":
    """Create an asynchronous SPEAD stream for transmission."""
    dtype = chunks[0].data.dtype
    rate = N_POLS * adc_sample_rate * dtype.itemsize * send_rate_factor
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    memory_regions: List[object] = [chunk.data for chunk in chunks]
    if extra_memory_regions:
        memory_regions.extend(extra_memory_regions)
    # Send a bit faster than nominal rate to account for header overheads
    rate = N_POLS * adc_sample_rate * dtype.itemsize * send_rate_factor
    config = spead2.send.StreamConfig(
        rate=rate,
        max_packet_size=packet_payload + PREAMBLE_SIZE,
        max_heaps=len(chunks) * spectra // spectra_per_heap * len(endpoints),
    )
    stream: "spead2.send.asyncio.AsyncStream"
    if ibv:
        ibv_config = spead2.send.UdpIbvConfig(
            endpoints=[(ep.host, ep.port) for ep in endpoints],
            interface_address=interface,
            ttl=ttl,
            comp_vector=comp_vector,
            memory_regions=memory_regions,
        )
        stream = spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config)
    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool, [(ep.host, ep.port) for ep in endpoints], config, ttl=ttl, interface_address=interface
        )
    stream.set_cnt_sequence(feng_id, num_ants)
    return stream
