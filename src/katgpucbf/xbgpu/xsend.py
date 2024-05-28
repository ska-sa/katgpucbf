################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

"""Module for sending baseline correlation products onto the network."""

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Final

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext
from prometheus_client import Counter

from .. import COMPLEX, DEFAULT_PACKET_PAYLOAD_BYTES
from ..spead import FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID, XENG_RAW_ID
from . import METRIC_NAMESPACE

output_heaps_counter = Counter(
    "output_x_heaps", "number of X-engine heaps transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_bytes_counter = Counter(
    "output_x_bytes", "number of X-engine payload bytes transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_visibilities_counter = Counter(
    "output_x_visibilities", "number of scalar visibilities", ["stream"], namespace=METRIC_NAMESPACE
)
output_clipped_visibilities_counter = Counter(
    "output_x_clipped_visibilities",
    "number of scalar visibilities that saturated",
    ["stream"],
    namespace=METRIC_NAMESPACE,
)
skipped_accum_counter = Counter(
    "output_x_skipped_accs",
    "skipped output accumulations because input data was entirely incomplete",
    ["stream"],
    namespace=METRIC_NAMESPACE,
)
incomplete_accum_counter = Counter(
    "output_x_incomplete_accs",
    "incomplete output accumulations because input data was partially incomplete",
    ["stream"],
    namespace=METRIC_NAMESPACE,
)
SEND_DTYPE = np.dtype(np.int32)


def make_item_group(xeng_raw_shape: tuple[int, ...]) -> spead2.send.ItemGroup:
    """Create an item group (with no values)."""
    item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
    item_group.add_item(
        FREQUENCY_ID,
        "frequency",  # Misleading name, but it's what the ICD specifies
        "Value of first channel in collections stored here.",
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
        XENG_RAW_ID,
        "xeng_raw",
        "Integrated baseline correlation products.",
        shape=xeng_raw_shape,
        dtype=SEND_DTYPE,
    )
    return item_group


class Heap:
    """Hold all the data for a heap.

    The content of the heap can change, but the class is frozen.
    """

    def __init__(
        self, context: AbstractContext, n_channels_per_substream: int, n_baselines: int, channel_offset: int
    ) -> None:
        self.buffer: Final = accel.HostArray(
            (n_channels_per_substream, n_baselines, COMPLEX), SEND_DTYPE, context=context
        )
        self.saturated: Final = accel.HostArray((), np.uint32, context=context)
        self._timestamp: Final = np.zeros((), dtype=">u8")  # Big-endian to be used in-place by the heap
        self.future = asyncio.get_running_loop().create_future()
        self.future.set_result(None)

        item_group = make_item_group(self.buffer.shape)
        item_group[TIMESTAMP_ID].value = self._timestamp
        item_group[FREQUENCY_ID].value = channel_offset
        item_group[XENG_RAW_ID].value = self.buffer
        self.heap: Final = item_group.get_heap(descriptors="none", data="all")
        self.heap.repeat_pointers = True

    @property
    def timestamp(self) -> int:  # noqa: D102
        return int(self._timestamp[()])

    @timestamp.setter
    def timestamp(self, value: int) -> None:  # noqa: D102
        self._timestamp[()] = value


def make_stream(
    *,
    output_name: str,
    dest_ip: str,
    dest_port: int,
    interface_ip: str,
    ttl: int,
    use_ibv: bool,
    affinity: int,
    comp_vector: int,
    stream_config: spead2.send.StreamConfig,
    buffers: Sequence[np.ndarray],
) -> "spead2.send.asyncio.AsyncStream":
    """Produce a UDP spead2 stream used for transmission."""
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    stream: spead2.send.asyncio.AsyncStream
    if use_ibv:
        stream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            stream_config,
            spead2.send.UdpIbvConfig(
                endpoints=[(dest_ip, dest_port)],
                interface_address=interface_ip,
                ttl=ttl,
                comp_vector=comp_vector,
                memory_regions=list(buffers),
            ),
        )

    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool,
            [(dest_ip, dest_port)],
            stream_config,
            interface_address=interface_ip,
            ttl=ttl,
        )

    # Reference the labels causing them to be created in
    # advance of any data being transmitted.
    output_heaps_counter.labels(output_name)
    output_bytes_counter.labels(output_name)
    output_visibilities_counter.labels(output_name)
    output_clipped_visibilities_counter.labels(output_name)
    skipped_accum_counter.labels(output_name)
    incomplete_accum_counter.labels(output_name)
    return stream


class XSend:
    """
    Class for turning baseline correlation products into SPEAD heaps and transmitting them.

    This class creates a queue of buffers that can be sent out onto the
    network. To get one of these buffers call :meth:`get_free_heap` - it will
    return a buffer. Once the necessary data has been copied to the buffer and
    it is ready to be sent onto the network, pass it back to this object using
    :meth:`send_heap`. This object will create a limited number of buffers and
    keep recycling them - avoiding any memory allocation at runtime.

    This has been designed to run in an asyncio loop, and :meth:`get_free_heap`
    function makes sure that the next buffer in the queue is not in flight
    before returning.

    To allow this class to be used with multiple transports, the constructor
    takes a factory function to create the stream.

    Parameters
    ----------
    n_ants
        The number of antennas that have been correlated.
    n_channels
        The total number of channels across all X-Engines. Must be a multiple
        of `n_channels_per_substream`.
    n_channels_per_substream
        The number of frequency channels contained per substream.
    dump_interval_s
        A new heap is transmitted every `dump_interval_s` seconds. Set to zero
        to send as fast as possible.
    send_rate_factor
        Configure the spead2 sender with a rate proportional to this factor.
        This value is intended to dictate a data transmission rate slightly
        higher/faster than the ADC rate.

        .. note::

           A factor of zero (0) tells the sender to transmit as fast as
           possible.
    channel_offset
        Fixed value to be included in the SPEAD heap indicating the lowest
        channel value transmitted by this heap.  Must be a multiple of
        `n_channels_per_substream`.
    context
        All buffers to be transmitted will be created from this context.
    stream_factory
        Callback function that will create the spead2 stream. It is passed the
        stream configuration and the memory buffers.
    n_send_heaps_in_flight
        Number of buffers that will be queued at any one time. I don't see any
        need for this to be configurable, the data rates are likely too low for
        it to be an issue. I have put it here more to be explicit than anything
        else. This argument is optional.
    packet_payload
        Size in bytes for output packets (baseline correlation products
        payload only, headers and padding are then added to this).
    tx_enabled
        Start with output transmission enabled.
    """

    # Class static constants
    header_size: Final[int] = 64

    def __init__(
        self,
        output_name: str,
        n_ants: int,
        n_channels: int,
        n_channels_per_substream: int,
        dump_interval_s: float,
        send_rate_factor: float,
        channel_offset: int,
        context: AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        n_send_heaps_in_flight: int = 5,
        packet_payload: int = DEFAULT_PACKET_PAYLOAD_BYTES,
        tx_enabled: bool = False,
    ) -> None:
        if dump_interval_s < 0:
            raise ValueError("Dump interval must be 0 or greater.")

        if n_channels % n_channels_per_substream != 0:
            raise ValueError("n_channels must be an integer multiple of n_channels_per_substream")
        if channel_offset % n_channels_per_substream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_substream")

        self.output_name = output_name
        self.tx_enabled = tx_enabled

        # Array Configuration Parameters
        self.n_ants: Final[int] = n_ants
        self.n_channels_per_substream: Final[int] = n_channels_per_substream
        n_baselines: Final[int] = (self.n_ants + 1) * (self.n_ants) * 2

        # Multicast Stream Parameters
        self.heap_payload_size_bytes = self.n_channels_per_substream * n_baselines * COMPLEX * SEND_DTYPE.itemsize

        self._heaps_queue: asyncio.Queue[Heap] = asyncio.Queue()
        buffers: list[accel.HostArray] = []

        for _ in range(n_send_heaps_in_flight):
            heap = Heap(context, n_channels_per_substream, n_baselines, channel_offset)
            self._heaps_queue.put_nowait(heap)
            buffers.append(heap.buffer)

        # Transport-agnostic stream information
        packets_per_heap = math.ceil(self.heap_payload_size_bytes / packet_payload)
        packet_header_overhead_bytes = packets_per_heap * XSend.header_size

        if dump_interval_s != 0:
            send_rate_bytes_per_second = (
                (self.heap_payload_size_bytes + packet_header_overhead_bytes) / dump_interval_s * send_rate_factor
            )  # * send_rate_factor adds a buffer to the rate to compensate for any unexpected jitter
        else:
            # Pass zero to stream_config to send as fast as possible.
            send_rate_bytes_per_second = 0

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + XSend.header_size,
            max_heaps=n_send_heaps_in_flight + 1,  # + 1 to allow for descriptors
            rate_method=spead2.send.RateMethod.AUTO,
            rate=send_rate_bytes_per_second,
        )
        self.stream = stream_factory(stream_config, buffers)
        # Set heap count sequence to allow a receiver to ingest multiple
        # X-engine outputs, if they should so choose.
        self.stream.set_cnt_sequence(
            channel_offset // n_channels_per_substream,
            n_channels // n_channels_per_substream,
        )

        item_group = make_item_group(buffers[0].shape)
        self.descriptor_heap = item_group.get_heap(descriptors="all", data="none")

    def send_heap(self, heap: Heap) -> None:
        """Take in a buffer and send it as a SPEAD heap.

        This function is non-blocking. There is no guarantee that a heap has
        been sent by the time the function completes.

        Parameters
        ----------
        heap
            Heap to send
        """
        if self.tx_enabled:
            saturated = int(heap.saturated)  # Save a copy before giving away the heap
            heap.future = self.stream.async_send_heap(heap.heap)
            self._heaps_queue.put_nowait(heap)
            # NOTE: It's not strictly true to say that the data has been sent at
            # this point; it's only been queued for sending. But it should be close
            # enough for monitoring data rates at the granularity that this is
            # typically done.
            output_heaps_counter.labels(self.output_name).inc(1)
            output_bytes_counter.labels(self.output_name).inc(heap.buffer.nbytes)
            output_visibilities_counter.labels(self.output_name).inc(heap.buffer.shape[0] * heap.buffer.shape[1])
            output_clipped_visibilities_counter.labels(self.output_name).inc(saturated)
        else:
            # :meth:`get_free_heap` still needs to await some Future before
            # returning a buffer.
            heap.future = asyncio.create_task(self.stream.async_flush())
            self._heaps_queue.put_nowait(heap)

    async def get_free_heap(self) -> Heap:
        """
        Return a heap from the internal fifo queue when one is available.

        There are a limited number of heaps in existence and
        they are all stored with a future object. If the future is complete,
        the buffer is not being used for sending and it will return the heap
        immediately. If the future is still busy, this function will wait
        asynchronously for the future to be done.

        This function is compatible with asyncio.

        Returns
        -------
        heap
            Free heap
        """
        heap = await self._heaps_queue.get()
        await asyncio.wait([heap.future])
        return heap

    async def send_stop_heap(self) -> None:
        """Send a Stop Heap over the spead2 transport."""
        stop_heap = spead2.send.Heap(FLAVOUR)
        stop_heap.add_end()
        # Flush just to ensure that we don't overflow the stream's queue.
        # It's a heavy-handed approach, but we don't care about performance
        # during shutdown.
        await self.stream.async_flush()
        await self.stream.async_send_heap(stop_heap)
