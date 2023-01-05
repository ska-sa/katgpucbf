################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

import numpy as np
import spead2.send
import spead2.send.asyncio
from aiokatcp import SensorSet
from katsdptelstate.endpoint import Endpoint
from prometheus_client import Counter

from .. import COMPLEX, N_POLS
from ..spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID, make_immediate
from ..utils import TimeConverter
from . import METRIC_NAMESPACE

#: Number of non-payload bytes per packet (header, 8 items pointers)
PREAMBLE_SIZE = 72
#: Data type of the output payload
SEND_DTYPE = np.dtype(np.int8)
output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter("output_bytes", "number of payload bytes transmitted", namespace=METRIC_NAMESPACE)
output_samples_counter = Counter("output_samples", "number of samples transmitted", namespace=METRIC_NAMESPACE)
skipped_heaps_counter = Counter(
    "output_skipped_heaps", "heaps not sent because input data was incomplete", namespace=METRIC_NAMESPACE
)
output_clip_counter = Counter(
    "output_clipped_samples", "number of samples that were saturated", ["pol"], namespace=METRIC_NAMESPACE
)


class Frame:
    """Holds all the heaps for a single timestamp.

    It does not own its memory - the backing store is in :class:`Chunk`.

    Parameters
    ----------
    timestamp
        Zero-dimensional array of dtype ``>u8`` holding the timestamp.
    data
        Payload data for the frame, of shape (channels, spectra_per_heap, N_POLS, COMPLEX).
    saturated
        Saturation data for the frame, of shape (N_POLS,)
    feng_id
        Value to put in ``feng_id`` SPEAD item
    substreams
        Number of substreams into which the channels are divided
    """

    def __init__(
        self, timestamp: np.ndarray, data: np.ndarray, saturated: np.ndarray, *, substreams: int, feng_id: int
    ) -> None:
        channels = data.shape[0]
        assert channels % substreams == 0
        channels_per_substream = channels // substreams
        self.heaps = []
        self.data = data
        self.saturated = saturated
        for i in range(substreams):
            start_channel = i * channels_per_substream
            heap = spead2.send.Heap(FLAVOUR)
            heap.repeat_pointers = True
            heap.add_item(make_immediate(TIMESTAMP_ID, timestamp))
            heap.add_item(make_immediate(FENG_ID_ID, feng_id))
            heap.add_item(make_immediate(FREQUENCY_ID, start_channel))
            heap_data = data[start_channel : start_channel + channels_per_substream]
            assert heap_data.flags.c_contiguous, "Heap data must be contiguous"
            heap.add_item(
                spead2.Item(FENG_RAW_ID, "", "", shape=heap_data.shape, dtype=heap_data.dtype, value=heap_data)
            )
            self.heaps.append(spead2.send.HeapReference(heap, substream_index=i))


class Chunk:
    """An array of frames, spanning multiple timestamps."""

    def __init__(
        self,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        substreams: int,
        feng_id: int,
    ) -> None:
        n_frames = data.shape[0]
        channels = data.shape[1]
        spectra_per_heap = data.shape[2]
        if channels % substreams != 0:
            raise ValueError("substreams must divide into channels")
        self.data = data
        self.saturated = saturated
        #: Whether each frame has valid data
        self.present = np.zeros(n_frames, dtype=bool)
        #: Timestamp of the first heap
        self._timestamp = 0
        #: Callback to return the chunk to the appropriate queue
        self.cleanup: Callable[[], None] | None = None
        self._timestamp_step = spectra_per_heap * channels * 2
        #: Storage for timestamps in the SPEAD heaps.
        self._timestamps = (np.arange(n_frames) * self._timestamp_step).astype(">u8")
        # The ... in indexing causes numpy to give a 0d array view, rather than
        # a scalar.
        self._frames = [
            Frame(self._timestamps[i, ...], data[i], saturated[i], feng_id=feng_id, substreams=substreams)
            for i in range(n_frames)
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
    def _inc_counters(frame: Frame, future: asyncio.Future) -> None:
        if not future.cancelled() and future.exception() is None:
            output_heaps_counter.inc(len(frame.heaps))
            output_bytes_counter.inc(frame.data.nbytes)
            output_samples_counter.inc(frame.data.size // COMPLEX)
            for pol in range(N_POLS):
                output_clip_counter.labels(pol).inc(frame.saturated[pol])

    async def send(
        self, stream: "spead2.send.asyncio.AsyncStream", frames: int, time_converter: TimeConverter, sensors: SensorSet
    ) -> None:
        """Transmit heaps on a SPEAD stream.

        Frames from 0 to `frames` - 1 are sent asynchronously.
        """
        futures = []
        saturated = [0] * N_POLS
        for present, frame in zip(self.present[:frames], self._frames[:frames]):
            if present:
                futures.append(stream.async_send_heaps(frame.heaps, spead2.send.GroupMode.ROUND_ROBIN))
                futures[-1].add_done_callback(functools.partial(self._inc_counters, frame))
                for pol in range(N_POLS):
                    saturated[pol] += frame.saturated[pol]
            else:
                skipped_heaps_counter.inc(len(frame.heaps))
        if futures:
            await asyncio.gather(*futures)
        end_timestamp = self._timestamp + self._timestamp_step * len(self._frames)
        end_time = time_converter.adc_to_unix(end_timestamp)
        for pol in range(N_POLS):
            sensor = sensors[f"input{pol}.feng-clip-cnt"]
            sensor.set_value(sensor.value + saturated[pol], timestamp=end_time)


def make_stream(
    *,
    endpoints: list[Endpoint],
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
) -> "spead2.send.asyncio.AsyncStream":
    """Create an asynchronous SPEAD stream for transmission."""
    dtype = chunks[0].data.dtype
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    memory_regions: list[object] = [chunk.data for chunk in chunks]
    # Send a bit faster than nominal rate to account for header overheads
    rate = N_POLS * adc_sample_rate * dtype.itemsize * send_rate_factor
    config = spead2.send.StreamConfig(
        rate=rate,
        max_packet_size=packet_payload + PREAMBLE_SIZE,
        # Adding len(endpoints) to accommodate descriptors sent for each substream
        max_heaps=(len(chunks) * spectra // spectra_per_heap * len(endpoints)) + len(endpoints),
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


def make_descriptor_heap(
    *,
    channels_per_substream: int,
    spectra_per_heap: int,
) -> "spead2.send.Heap":
    """Create a descriptor heap for output F-Engine data."""
    heap_data_shape = (channels_per_substream, spectra_per_heap, N_POLS, COMPLEX)

    ig = spead2.send.ItemGroup(flavour=FLAVOUR)
    ig.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    ig.add_item(
        FENG_ID_ID,
        "feng_id",
        "Uniquely identifies the F-Engine source for the data.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    ig.add_item(
        FREQUENCY_ID,
        "frequency",
        "Identifies the first channel in the band of frequencies in the SPEAD heap.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    ig.add_item(
        FENG_RAW_ID,
        "feng_raw",
        "Channelised complex data from both polarisations of digitiser associated with F-Engine.",
        shape=heap_data_shape,
        dtype=SEND_DTYPE,
    )

    return ig.get_heap(descriptors="all", data="none")
