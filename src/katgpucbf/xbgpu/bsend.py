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
        Payload data for the frame with shape (n_beams,
        n_channels_per_substream, spectra_per_heap, COMPLEX).
    saturated
        Total number of complex samples that saturated during requantisation,
        with shape (n_beams,).
    channel_offset
        The first frequency channel processed.
    """

    def __init__(
        self,
        timestamp: np.ndarray,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        channel_offset: int,
    ) -> None:
        self.heaps: list[spead2.send.HeapReference] = []
        self.data = data
        self.saturated = saturated
        n_substreams = saturated.shape[0]
        for i in range(n_substreams):
            heap = spead2.send.Heap(flavour=FLAVOUR)
            heap.repeat_pointers = True
            heap.add_item(make_immediate(FREQUENCY_ID, channel_offset))
            heap.add_item(make_immediate(TIMESTAMP_ID, timestamp))
            heap.add_item(
                spead2.Item(
                    BF_RAW_ID,
                    "bf_raw",
                    "",
                    shape=data.shape[1:],  # Get rid of the 'beam' dimension
                    dtype=data.dtype,
                    value=self.data[i, ...],
                )
            )
            self.heaps.append(spead2.send.HeapReference(heap, substream_index=i))


class Chunk:
    """
    An array of :class:`Heaps`.

    Parameters
    ----------
    data
        Storage for tied-array-channelised-voltage data, with shape (n_frames,
        n_beams, n_channels_per_substream, n_spectra_per_heap, COMPLEX) and
        dtype :const:`SEND_DTYPE`.
    saturated
        Storage for saturation counts, with shape (n_frames, n_beams) and dtype
        uint32.
    channel_offset
        The first frequency channel processed.
    timestamp_step
        Timestamp step between successive :class:`Frame` in a chunk.
    """

    def __init__(
        self,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        channel_offset: int,
        timestamp_step: int,
    ) -> None:
        n_frames = data.shape[0]
        self.data = data  # data should have all the dimensions required
        self.saturated = saturated

        self._timestamp = 0
        self._timestamp_step = timestamp_step
        self._timestamps = (np.arange(n_frames) * self._timestamp_step).astype(">u8")

        self._frames = [
            Frame(
                self._timestamps[i, ...],
                data[i],
                saturated[i],
                channel_offset=channel_offset,
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

    def send(
        self,
        send_stream: "BSend",
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> asyncio.Future:
        """
        Transmit a chunk's heaps over a SPEAD stream.

        The chunk holds a reference to the gathered future, which needs to be
        awaited before using again.

        .. todo::

            Also update its relevant counters and sensor values.
        """
        if any(send_stream.tx_enabled):
            send_futures = []
            for frame in self._frames:
                heaps_to_send = [heap for heap, enabled in zip(frame.heaps, send_stream.tx_enabled) if enabled]
                send_futures.append(
                    send_stream.stream.async_send_heaps(heaps_to_send, mode=spead2.send.GroupMode.ROUND_ROBIN)
                )
            # TODO: Update counters and sensor with chunk.saturation
            return asyncio.gather(*send_futures)
        else:
            # TODO: Is it necessary to handle this case?
            return asyncio.create_task(send_stream.stream.async_flush())


class BSend:
    """
    Class for turning tied array channelised voltage products into SPEAD heaps.

    This class creates a queue of chunks that can be sent out onto the network.
    To obtain a chunk, call :meth:`get_free_chunk` - which will return a
    :class:`Chunk`. This object will create a limited number of transmit
    buffers and keep recycling them, avoiding any memory allocation at runtime.

    The transmission of a chunk's data is abstracted by :meth:`send_chunk`, which
    invokes, then waits on transmission before putting it back on the queue for
    reuse.

    This object keeps track of each tied-array-channelised-voltage data stream by
    means of a substreams in :class:`spead2.send.asyncio.AsyncStream`, allowing
    for individual enabling and disabling of the data product.

    To allow this class to be used with multiple transports, the constructor
    takes a factory function to create the stream.

    Parameters
    ----------
    outputs
        List of :class:`.output.BOutput`.
    heaps_per_fengine_per_chunk
        Number of SPEAD heaps from one F-engine in a single received Chunk.
    n_tx_items
        TODO: There must be a better way to say this
        Number of Chunks to create in order to download data off the GPU.
    n_channels_per_substream, spectra_per_heap, channel_offset
        See :class:`.XBEngine` for further information.
    timestamp_step
        The timestamp step between successive heaps, as dictated by the XBEngine.
    send_rate_factor
        Factor dictating how fast the send-stream should transmit data.
    context
        Device context to create buffers.
    stream_factory
        Callback function to create the spead2 send stream. It is passed the
        stream configuration and memory buffers.
    packet_payload
        Size, in bytes, for the output packets (tied array channelised voltage
        payload only, headers and padding are added to this).
    tx_enabled
        Enable/Disable transmission.
    """

    descriptor_heap: spead2.send.Heap
    header_size: Final[int] = 64

    def __init__(
        self,
        outputs: Sequence[BOutput],
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
        self.tx_enabled = [tx_enabled] * len(outputs)
        self.n_beams = len(outputs)

        self._free_chunks_queue: asyncio.Queue[Chunk] = asyncio.Queue()
        buffers: list[np.ndarray] = []

        send_shape = (heaps_per_fengine_per_chunk, self.n_beams, n_channels_per_substream, spectra_per_heap, COMPLEX)
        for _ in range(n_tx_items):
            chunk = Chunk(
                accel.HostArray(send_shape, SEND_DTYPE, context=context),
                accel.HostArray(
                    (heaps_per_fengine_per_chunk, self.n_beams),
                    np.uint32,
                    context=context,
                ),
                channel_offset=channel_offset,
                timestamp_step=timestamp_step,
            )
            self._free_chunks_queue.put_nowait(chunk)
            buffers.append(chunk.data)

        # TODO: Follow suit with other Engines to calculate `rate`

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + BSend.header_size,
            max_heaps=n_tx_items * heaps_per_fengine_per_chunk * self.n_beams,
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
            "Beamformer output for frequency-domain beam.",
            shape=buffers[0].shape[2:],
            dtype=buffers[0].dtype,
        )

        self.descriptor_heap = item_group.get_heap(descriptors="all", data="none")

    def enable_substream(self, stream_id: int, enable: bool = True) -> None:
        """Enable/Disable a substream's data transmission.

        :class:`.BSend` operates as a large single stream with multiple
        substreams. Each substream is its own data product and is required
        to be enabled/disabled independently.

        Parameters
        ----------
        stream_id
            Index of the substream's data product.
        enable
            Boolean indicating whether the `stream_id` should be enabled or
            disabled.
        """
        self.tx_enabled[stream_id] = enable

    async def get_free_chunk(self) -> Chunk:
        """
        Obtain a :class:`.Chunk` for transmission.

        .. todo::

            Is this entirely necessary? Left in because it felt a bit strange
            to dig into the :class:`.BSend` from the :class:`.XBEngine`.
        """
        chunk = await self._free_chunks_queue.get()
        return chunk

    async def send_chunk(self, chunk: Chunk, time_converter: TimeConverter, sensors: SensorSet) -> None:
        """Send a chunk's data and put it on the queue."""
        try:
            await chunk.send(self, time_converter, sensors)
        except asyncio.CancelledError:
            pass
        except Exception as ex:
            raise ex
        finally:
            self._free_chunks_queue.put_nowait(chunk)

    async def send_stop_heap(self) -> None:
        """Send a Stop Heap over the spead2 transport."""
        stop_heap = spead2.send.Heap(FLAVOUR)
        stop_heap.add_end()
        # Flush just to ensure that we don't overflow the stream's queue.
        # It's a heavy-handed approach, but we don't care about performance
        # during shutdown.
        await self.stream.async_flush()
        for i in range(self.n_beams):
            await self.stream.async_send_heap(stop_heap, substream_index=i)


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
