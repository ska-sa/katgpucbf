################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

"""Module for sending tied array channelised voltage products onto the network."""

import asyncio
from typing import Callable, Final, Sequence

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from aiokatcp import SensorSet
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from .. import COMPLEX, DEFAULT_PACKET_PAYLOAD_BYTES
from ..spead import BF_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID, make_immediate
from ..utils import TimeConverter
from .output import BOutput

# NOTE: ICD suggests `beng_out_bits_per_sample`,
# MK correlator doesn't make this configurable.
SEND_DTYPE = np.dtype(np.int8)


class Frame:
    """Hold all data for heaps with a single timestamp.

    It does not own its memory - the backing store is in :class:`Chunk`.

    Parameters
    ----------
    timestamp
        Zero-dimensional array of dtype ``>u8`` holding the timestamp
    data
        Payload data for the frame with shape (n_channels_per_substream, spectra_per_heap, COMPLEX)
    saturated
        Total number of complex samples that saturated during requantisation
    channel_offset
        The first frequency channel processed.
    n_substreams
        # TODO Re-word this
        Number of beams requiring data to be transmitted
    """

    def __init__(
        self,
        timestamp: np.ndarray,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        channel_offset: int,
        n_substreams: int,
    ) -> None:
        self.heaps = []
        self.data = data
        self.saturated = saturated
        for i in range(n_substreams):
            heap = spead2.send.Heap(flavour=FLAVOUR)
            heap.repeat_pointers = True
            heap.add_item(make_immediate(FREQUENCY_ID, channel_offset))
            heap.add_item(make_immediate(TIMESTAMP_ID, timestamp))
            # TODO: Update `self.data` to slice off beam data for each substream
            heap.add_item(
                spead2.Item(
                    BF_RAW_ID,
                    "bf_raw",
                    "",
                    shape=data.shape,
                    dtype=data.dtype,
                    value=data,
                )
            )
            self.heaps.append(spead2.send.HeapReference(heap, substream_index=i))


class Chunk:
    """An array of :class:`Heaps`."""

    def __init__(
        self,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        channel_offset: int,
        timestamp_step: int,
        n_substreams: int,
    ) -> None:
        # data.shape[0] should be some outer element indicating n_frames
        # to fill a Chunk.
        # NOTE: I think this value == heaps-per-feng-per-chunk????
        n_frames = data.shape[0]
        self.data = data  # data should have all the dimensions required
        self.saturated = saturated

        # TODO: Timestamp step is the same as XBEngine's rx_heap_timestamp_step
        self._timestamp = 0
        self._timestamp_step = timestamp_step
        self._timestamps = (np.arange(n_frames) * self._timestamp_step).astype(">u8")

        # Need a future for each heap to be sent
        self.futures: list[asyncio.Future] = [asyncio.get_running_loop().create_future() for _ in range(n_frames)]
        for future in self.futures:
            future.set_result(None)

        self._frames = [
            Frame(
                self._timestamps[i, ...],
                data[i],
                saturated[i],
                channel_offset=channel_offset,
                n_substreams=n_substreams,
            )
            for i in range(n_frames)
        ]

    @property
    def timestamp(self) -> int:
        """
        Timestamp of the first heap.

        Setting this property updates the timestamps stored in all the heaps.
        This should not be done while a previous call to :meth:`send` is still
        in progress.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        delta = value - self.timestamp
        self._timestamps += delta
        self._timestamp = value


class BSend:
    """
    Class for turning tied array channelised voltage products into SPEAD heaps.

    This is some text.

    Parameters
    ----------
    output
        The BOutput containing the beam name and destination address.
    heaps_per_fengine_per_chunk
        Number of SPEAD heaps from one F-engine in a single received Chunk.
    n_tx_items
        # TODO: Reword
        Number of Chunks to create in order to download data off the GPU.
    n_channels_per_substream, spectra_per_heap, channel_offset
        Engine configuration parameters.
    timestamp_step
        The timestamp step between successive heaps, as dictated by the XBEngine.
    send_rate_factor
        Factor dictating how fast the send-stream should transmit data.
    context
        Device context to create buffers.
    stream_factory
        Callback function to create the spead2 send stream. It is passed the
        stream ocnfiguration and memory buffers.
    packet_payload
        Size, in bytes, for the output packets (tied array channelised voltage
        payload only, headers and padding are added to this).
    tx_enabled
        Enable/Disable transmission. Defaults to starting enabled.
    """

    descriptor_heap: spead2.send.Heap
    header_size: Final[int] = 64

    def __init__(
        self,
        output: BOutput,
        heaps_per_fengine_per_chunk: int,
        n_tx_items: int,
        n_channels_per_substream: int,
        spectra_per_heap: int,
        timestamp_step: int,
        send_rate_factor: float,
        channel_offset: int,
        context: AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        packet_payload: int = DEFAULT_PACKET_PAYLOAD_BYTES,
        tx_enabled: bool = False,
    ) -> None:
        # Now that we've moved away from *multiple* BSend objects, towards
        # a single send_stream with multiple substreams, the BSend object
        # probably still needs to exist
        # - The `send` function will need to keep track of "which substream
        #   is enabled/disabled" Re: capture-{start, stop} <stream_name>
        self.enabled_stream_ids: list[int] = []
        self.tx_enabled = tx_enabled

        self._chunks_queue: asyncio.Queue[Chunk] = asyncio.Queue()
        buffers: list[np.ndarray] = []

        # `n_heaps_to_send` is actually used to dictate the amount of buffers (in XSend)
        # So perhaps we need to change the number of buffers to be range(send_free_queue.maxsize)
        # n_heaps_to_send = len(buffers) // spectra_per_heap

        send_shape = (heaps_per_fengine_per_chunk, n_channels_per_substream, spectra_per_heap, COMPLEX)
        for _ in range(n_tx_items):
            chunk = Chunk(
                accel.HostArray(send_shape, SEND_DTYPE, context=context),
                accel.HostArray(
                    (heaps_per_fengine_per_chunk,),
                    np.uint32,
                ),
                channel_offset=channel_offset,
                timestamp_step=timestamp_step,
                n_substreams=1,  # TODO: Update once single-beam is working
            )
            self._chunks_queue.put_nowait(chunk)
            buffers.append(chunk.data)

        # Multicast stream parameters
        self.heap_payload_size_bytes = n_channels_per_substream * spectra_per_heap * COMPLEX * SEND_DTYPE.itemsize
        # Transport-agnostic stream information
        # Used in XSend to calculate `send_rate_bytes_per_second`, do we need it here?
        # TODO: Scope to move this calculation into a helper in utils
        # packets_per_heap = math.ceil(self.heap_payload_size_bytes / packet_payload)
        # packet_header_overhead_bytes = packets_per_heap * BSend.header_size

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + BSend.header_size,
            max_heaps=n_tx_items * heaps_per_fengine_per_chunk + 1,  # TODO: Update this to be proper
            rate_method=spead2.send.RateMethod.AUTO,
            rate=0.0,  # TODO: Update to use `send_rate_bytes_per_second`, this sends as fast as possible
        )
        self.stream = stream_factory(stream_config, buffers)

        item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
        item_group.add_item(
            FREQUENCY_ID,
            "frequency",  # Misleading name, but it's what the ICD specifies
            "Value of the first channel in collections stored here.",
            shape=[],
            format=IMMEDIATE_FORMAT,
        )
        item_group.add_item(
            TIMESTAMP_ID,
            "timestamp",
            "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
            shape=[],
            format=IMMEDIATE_FORMAT,
        )
        item_group.add_item(
            BF_RAW_ID,
            "bf_raw",
            "",  # TODO: What to even say here? ICD says "Channelised complex data"
            shape=buffers[0].shape[1:],
            dtype=buffers[0].dtype,
        )

        self.descriptor_heap = item_group.get_heap(descriptors="all", data="none")

    def send_chunk(
        self,
        chunk: Chunk,
        time_converter: TimeConverter | None = None,
        sensors: SensorSet | None = None,
    ) -> None:
        """
        Transmit a Chunk's heaps.

        .. todo::
            Also update its relevant counters and sensor values.

            This might need to be moved outside BSend because it looks
            a bit awkward. Stream already owns the Chunks.
        """
        if self.tx_enabled:
            chunk.futures = [
                self.stream.async_send_heaps(frame.heaps, mode=spead2.send.GroupMode.ROUND_ROBIN)
                for frame in chunk._frames
            ]
            # await asyncio.gather(*chunk.futures)
            self._chunks_queue.put_nowait(chunk)

    def enable_substream(self, stream_id: int, enable: bool = True) -> None:
        """Enable/Disable a substream's data transmission.

        :class:`.BSend` operates as a large single stream with multiple
        substreams. Each substream is its own data product and is required
        to be enabled/disabled independently.

        Parameters
        ----------
        stream_id
            ID of the substream, corresponds to the <beam-id><pol>
            convention, e.g. stream_id 3 has a stream_name ending in <1x>.
        enable
            Boolean indicating whether the `stream_id` should be enabled or
            disabled.
        """
        pass

    async def get_free_chunk(self) -> Chunk:
        """Return a Chunk once it has completed its send futures."""
        chunk = await self._chunks_queue.get()
        # TODO: Do we need to do this if we gather the futures in `send_chunk`?
        await asyncio.wait(chunk.futures)
        return chunk

    async def send_stop_heap(self) -> None:
        """Send a Stop Heap over the spead2 transport."""
        stop_heap = spead2.send.Heap(FLAVOUR)
        stop_heap.add_end()
        # Flush just to ensure that we don't overflow the stream's queue.
        # It's a heavy-handed approach, but we don't care about performance
        # during shutdown.
        await self.stream.async_flush()
        # TODO: Send across all substreams once multiple beams are supported
        await self.stream.async_send_heap(stop_heap)


def make_stream(
    *,
    endpoints: list[Endpoint],
    interface: str,
    ttl: int,
    use_ibv: bool,
    affinity: int,
    comp_vector: int,
    stream_config: spead2.send.StreamConfig,
    buffers: Sequence[np.ndarray],
) -> "spead2.send.asyncio.AsyncStream":
    """Create asynchronous SPEAD stream for transmission.

    This is architected to be a single send stream with multiple substreams,
    each corresponding to a tied-array-channelised-voltage output data product.
    The `endpoints` need not be a contiguous list of multicast addresses.
    """
    stream: spead2.send.asyncio.AsyncStream
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])

    if use_ibv:
        stream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            stream_config,
            spead2.send.UdpIbvConfig(
                endpoints=[(ep.host, ep.port) for ep in endpoints],
                interface_address=interface,
                ttl=ttl,
                comp_vector=comp_vector,
                memory_regions=list(buffers),
            ),
        )
    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool,
            [(ep.host, ep.port) for ep in endpoints],
            stream_config,
            interface_address=interface,
            ttl=ttl,
        )

    return stream
