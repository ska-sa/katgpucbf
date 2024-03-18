################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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
import functools
import logging
from math import ceil
from typing import Callable, Final, Sequence

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from aiokatcp import SensorSet
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint
from prometheus_client import Counter

from .. import COMPLEX, DEFAULT_PACKET_PAYLOAD_BYTES
from ..spead import BF_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID, make_immediate
from ..utils import TimeConverter
from . import METRIC_NAMESPACE
from .output import BOutput

output_heaps_counter = Counter(
    "output_b_heaps", "number of B-engine heaps transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_bytes_counter = Counter(
    "output_b_bytes", "number of B-engine payload bytes transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_samples_counter = Counter(
    "output_b_samples", "number of complex beam samples transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_clip_counter = Counter(
    "output_b_clipped_samples", "number of beam samples that were saturated", ["stream"], namespace=METRIC_NAMESPACE
)

logger = logging.getLogger(__name__)
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
        self.heaps: list[spead2.send.Heap] = []
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
            self.heaps.append(heap)


class Chunk:
    r"""
    An array of :class:`Frame`\ s.

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
        Timestamp step between successive :class:`Frame`\ s in a chunk.
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
        self.data = data
        self.saturated = saturated

        self._timestamp = 0
        self._timestamp_step = timestamp_step
        self._timestamps = (np.arange(n_frames) * self._timestamp_step).astype(">u8")

        # NOTE: The future indicates when it is safe to modify the chunk,
        # i.e. it is not being transmitted. At construction there is nothing to
        # wait for, so we mark it ready.
        self.future = asyncio.get_running_loop().create_future()
        self.future.set_result(None)

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
        This should only be done when :attr:`future` is done.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        delta = value - self.timestamp
        self._timestamps += delta
        self._timestamp = value

    @staticmethod
    def _inc_counters(
        data_shape: tuple[int, int, int], beam_dtype: np.dtype, beam_names: Sequence[str], future: asyncio.Future
    ) -> None:
        """Increment beam stream Prometheus counters.

        Parameters
        ----------
        data_shape
            The shape of the beam data being transmitted. Expected in the
            format of (n_channels_per_substream, samples_per_spectra, COMPLEX).
        beam_names
            List of beam stream names that are enabled for transmission for
            this `future`.
        future
            Future returned by the spead2 stream's `async_send_heaps`.

        .. todo::

            (NGC-1172) Handle missing data transmission for Prometheus counters.
        """
        if not future.cancelled() and future.exception() is None:
            for beam_name in beam_names:
                output_heaps_counter.labels(beam_name).inc(1)
                # Each beam data sample is 8-bit
                # - Multiply across dimensions to get total bytes
                output_bytes_counter.labels(beam_name).inc(np.prod(data_shape) * beam_dtype.itemsize)
                # - Multiply across the first two dimensions to get complex sample count
                output_samples_counter.labels(beam_name).inc(np.prod(data_shape[:-1]))

    def send(
        self,
        send_stream: "BSend",
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> asyncio.Future:
        """
        Transmit a chunk's heaps over a SPEAD stream.

        This method returns immediately and sends the data asynchronously. Before
        modifying the chunk, first await :attr:`future`.

        .. todo::

            (NGC-1173): Update counter and sensor with chunk.saturation
        """
        n_enabled = sum(send_stream.tx_enabled)
        rate = send_stream.bytes_per_second_per_beam * n_enabled
        if n_enabled > 0:
            send_futures: list[asyncio.Future] = []
            enabled_stream_names = [
                output_name for output_name, enabled in zip(send_stream.beam_names, send_stream.tx_enabled) if enabled
            ]
            for frame in self._frames:
                # TODO (NGC-1232): building this list every time may be too expensive.
                # Consider caching it and invalidating when streams are enabled/disabled.
                heaps_to_send = [
                    spead2.send.HeapReference(heap, substream_index=i, rate=rate)
                    for i, (heap, enabled) in enumerate(zip(frame.heaps, send_stream.tx_enabled))
                    if enabled
                ]
                send_futures.append(
                    send_stream.stream.async_send_heaps(heaps_to_send, mode=spead2.send.GroupMode.ROUND_ROBIN)
                )
                send_futures[-1].add_done_callback(
                    functools.partial(
                        self._inc_counters,
                        frame.data.shape[1:],  # Get rid of 'beam' dimension
                        frame.data.dtype,
                        enabled_stream_names,
                    )
                )

            self.future = asyncio.gather(*send_futures)
        else:
            # TODO: Is it necessary to handle this case?
            self.future = asyncio.create_task(send_stream.stream.async_flush())
        return self.future


class BSend:
    r"""
    Class for turning tied array channelised voltage products into SPEAD heaps.

    This class creates a queue of chunks that can be sent out onto the network.
    To obtain a chunk, call :meth:`get_free_chunk` - which will return a
    :class:`Chunk`. This object will create a limited number of transmit
    buffers and keep recycling them, avoiding any memory allocation at runtime.

    The transmission of a chunk's data is abstracted by :meth:`send_chunk`. This
    invokes transmission and immediately returns the :class:`Chunk` back to the
    queue for reuse.

    This object keeps track of each tied-array-channelised-voltage data stream by
    means of a substreams in :class:`spead2.send.asyncio.AsyncStream`, allowing
    for individual enabling and disabling of the data product.

    To allow this class to be used with multiple transports, the constructor
    takes a factory function to create the stream.

    Parameters
    ----------
    outputs
        Sequence of :class:`.output.BOutput`.
    frames_per_chunk
        Number of :class:`Frame`\ s in each transmitted :class:`Chunk`.
    n_tx_items
        Number of :class:`Chunk` to create.
    adc_sample_rate, n_channels, n_channels_per_substream, spectra_per_heap, channel_offset
        See :class:`.XBEngine` for further information.
    timestamp_step
        The timestamp step between successive heaps.
    send_rate_factor
        Factor dictating how fast the send-stream should transmit data.
    context
        Device context to create buffers.
    stream_factory
        Callback function to create the spead2 send stream. It is passed the
        stream configuration and memory buffers.
    packet_payload
        Size, in bytes, for the output packets (tied array channelised voltage
        payload only; headers and padding are added to this).
    tx_enabled
        Enable/Disable transmission.
    """

    descriptor_heap: spead2.send.Heap
    header_size: Final[int] = 64

    def __init__(
        self,
        outputs: Sequence[BOutput],
        frames_per_chunk: int,
        n_tx_items: int,
        n_channels: int,
        n_channels_per_substream: int,
        spectra_per_heap: int,
        adc_sample_rate: float,
        timestamp_step: int,
        send_rate_factor: float,
        channel_offset: int,
        context: AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        packet_payload: int = DEFAULT_PACKET_PAYLOAD_BYTES,
        tx_enabled: bool = False,
    ) -> None:
        if n_channels % n_channels_per_substream != 0:
            raise ValueError("n_channels must be an integer multiple of n_channels_per_substream")
        if channel_offset % n_channels_per_substream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_substream")

        self.tx_enabled = [tx_enabled] * len(outputs)
        n_beams = len(outputs)
        self.beam_names = [output.name for output in outputs]

        self._chunks_queue: asyncio.Queue[Chunk] = asyncio.Queue()
        buffers: list[np.ndarray] = []

        send_shape = (frames_per_chunk, n_beams, n_channels_per_substream, spectra_per_heap, COMPLEX)
        for _ in range(n_tx_items):
            chunk = Chunk(
                accel.HostArray(send_shape, SEND_DTYPE, context=context),
                accel.HostArray(
                    (frames_per_chunk, n_beams),
                    np.uint32,
                    context=context,
                ),
                channel_offset=channel_offset,
                timestamp_step=timestamp_step,
            )
            self._chunks_queue.put_nowait(chunk)
            buffers.append(chunk.data)

        heap_payload_size_bytes = n_channels_per_substream * spectra_per_heap * COMPLEX * SEND_DTYPE.itemsize

        # Transport-agnostic stream information
        packets_per_heap = ceil(heap_payload_size_bytes / packet_payload)
        packet_header_overhead_bytes = packets_per_heap * BSend.header_size

        heap_interval = timestamp_step / adc_sample_rate
        self.bytes_per_second_per_beam = (
            (heap_payload_size_bytes + packet_header_overhead_bytes) / heap_interval * send_rate_factor
        )

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + BSend.header_size,
            # + 1 below for the descriptor per beam
            max_heaps=(n_tx_items * frames_per_chunk + 1) * n_beams,
            rate_method=spead2.send.RateMethod.AUTO,
        )
        self.stream = stream_factory(stream_config, buffers)
        # Set heap count sequence to allow a receiver to ingest multiple
        # B-engine outputs, if they should so choose.
        self.stream.set_cnt_sequence(
            channel_offset // n_channels_per_substream,
            n_channels // n_channels_per_substream,
        )

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
        """Obtain a :class:`.Chunk` for transmission.

        We await the chunk's :attr:`future` to be sure we are not overwriting
        data that is still being transmitted. If sending failed, it is no
        longer being transmitted, and therefore safe to return the chunk.

        Raises
        ------
        asyncio.CancelledError
            If the chunk's send future is cancelled.
        """
        chunk = await self._chunks_queue.get()
        try:
            await chunk.future
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error sending chunk")
        return chunk

    def send_chunk(self, chunk: Chunk, time_converter: TimeConverter, sensors: SensorSet) -> None:
        """Send a chunk's data and put it on the :attr:`_chunks_queue`."""
        chunk.send(self, time_converter, sensors)
        self._chunks_queue.put_nowait(chunk)

    async def send_stop_heap(self) -> None:
        """Send a Stop Heap over the spead2 transport."""
        stop_heap = spead2.send.Heap(FLAVOUR)
        stop_heap.add_end()
        # Flush just to ensure that we don't overflow the stream's queue.
        # It's a heavy-handed approach, but we don't care about performance
        # during shutdown.
        await self.stream.async_flush()
        for i in range(len(self.beam_names)):
            await self.stream.async_send_heap(stop_heap, substream_index=i)


def make_stream(
    *,
    beam_names: list[str],
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
    # Referencing the labels causes them to be created, in advance of data
    # actually being transmitted.
    for beam_name in beam_names:
        output_heaps_counter.labels(beam_name)
        output_bytes_counter.labels(beam_name)
        output_samples_counter.labels(beam_name)
        output_clip_counter.labels(beam_name)

    return stream
