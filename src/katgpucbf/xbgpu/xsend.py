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

"""
Module for sending baselines produced by the GPU X-Engine onto the network.

This module has been designed to work with asyncio.

The data sent onto the network conforms to the SPEAD protocol. This module
takes the baseline data, turns it into a SPEAD heap and then transmits that
heap out onto the network using the spead2 Python module. The high-performance
ibverbs implementation of spead2 is recommended even though the data rates out
are very low. This is due to the ibverbs implementation using far fewer system
resources. The format of the packets transmitted by SPEAD2 can be found here:
- :ref:`baseline-correlation-products-data-packet-format`.

The XSend class creates its own buffers and data in those buffers will be
encapsulated into SPEAD heaps and sent onto the network. The user can request
the buffers from the object, populate them and then give them back to the object
for transmission. In using ibverbs, the memory regions of the XSend-generated
buffers have been registered with ibverbs to enable zero copy transmission -
using other buffers will force an extra copy.  Zero copy transmission means
that the data to be transmitted can sent from its current memory location
directly to the NIC without having to be copied to an intermediary memory
location in the process, thereby halving the memory bandwidth required to send.
"""

import asyncio
import math
import queue
from typing import Callable, Final, List, Sequence, Tuple

import katsdpsigproc
import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from prometheus_client import Counter

from .. import COMPLEX
from ..spead import FLAVOUR, FREQUENCY_ID, TIMESTAMP_ID, XENG_RAW_ID
from . import METRIC_NAMESPACE

output_heaps_counter = Counter("output_heaps_x", "number of X-engine heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter(
    "output_bytes_x", "number of X-engine payload bytes transmitted", namespace=METRIC_NAMESPACE
)


def make_stream(
    *,
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
    if use_ibv:
        return spead2.send.asyncio.UdpIbvStream(
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
        return spead2.send.asyncio.UdpStream(
            thread_pool,
            [(dest_ip, dest_port)],
            stream_config,
            interface_address=interface_ip,
            ttl=ttl,
        )


class BufferWrapper:
    """
    Holds a buffer object that has been configured so that it can be zero-copy transferred onto the network.

    In order to preserve the zero-copy properties of the buffer, data can
    only be copied to the buffer within this class, the buffer handle
    cannot be overwritten. For example: ``buffer_wrapper.buffer = new_array``
    will fail as it attempts to assign a new object to the buffer variable
    while ``buffer_wrapper.buffer[:] = new_array`` will succeed as it
    overwrites the values in the buffer, not the buffer itself.

    There may be a better way to hold these buffer objects and prevent them
    being overwritten other than wrapping them in an BufferWrapper but I could
    not think of any at the time of writing.

    Parameters
    ----------
    buffer
        The array configured for zero-copy transfers.
    """

    def __init__(self, buffer: np.ndarray) -> None:
        self._buffer: np.ndarray = buffer

    @property
    def buffer(self) -> np.ndarray:
        """Return the buffer associated with this class."""
        return self._buffer

    @buffer.setter
    def buffer(self, buffer) -> None:
        """
        Overwrite the buffer's handle - this functionality has been disabled.

        Instead of overwriting the handle, copy to the buffer using array
        indexing [:] instead.
        """
        raise AttributeError("Don't overwrite the buffer's handle! Copy to it instead using array indexing [:].")


class XSend:
    """
    Class for turning baseline correlation products into SPEAD heaps and transmitting them.

    This class creates a queue of buffers that can be sent out onto the
    network. To get one of these buffers call :meth:`get_free_heap` - it will
    return a buffer wrapped in a :class:`BufferWrapper` object. Once the
    necessary data has been copied to the buffer and it is ready to be sent
    onto the network, pass it back to this object using :meth:`send_heap`. This
    object will create a limited number of buffers and keep recycling them -
    avoiding any memory allocation at runtime.

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
        of `n_channels_per_stream`.
    n_channels_per_stream
        The number of frequency channels contained per stream.
    dump_interval_s
        A new heap is transmitted every `dump_interval_s` seconds. Set to zero
        to send as fast as possible.
    send_rate_factor
        Configure the SPEAD2 sender with a rate proportional to this factor.
        This value is intended to dictate a data transmission rate slightly
        higher/faster than the ADC rate.

        .. note::

           A factor of zero (0) tells the sender to transmit as fast as
           possible.
    channel_offset
        Fixed value to be included in the SPEAD heap indicating the lowest
        channel value transmitted by this heap.  Must be a multiple of
        `n_channels_per_stream`.
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
    """

    # Class static constants
    max_payload_size: Final[int] = 2048
    header_size: Final[int] = 64
    max_packet_size: Final[int] = max_payload_size + header_size

    # Initialise class including all variables
    def __init__(
        self,
        n_ants: int,
        n_channels: int,
        n_channels_per_stream: int,
        dump_interval_s: float,
        send_rate_factor: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        n_send_heaps_in_flight: int = 5,
    ) -> None:
        # 1. Check that given arguments are sane.
        if dump_interval_s < 0:
            raise ValueError("Dump interval must be 0 or greater.")

        if n_channels % n_channels_per_stream != 0:
            raise ValueError("n_channels must be an integer multiple of n_channels_per_stream")
        if channel_offset % n_channels_per_stream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_stream")

        # 2. Array Configuration Parameters
        self.n_ants: Final[int] = n_ants
        self.n_channels_per_stream: Final[int] = n_channels_per_stream
        self.n_baselines: Final[int] = (self.n_ants + 1) * (self.n_ants) * 2
        self.dump_interval_s: Final[float] = dump_interval_s
        self.send_rate_factor: Final[float] = send_rate_factor
        self._sample_bits: Final[int] = 32

        # 3. Multicast Stream Parameters
        self.channel_offset: Final[int] = channel_offset

        self.heap_payload_size_bytes: Final[int] = (
            self.n_channels_per_stream * self.n_baselines * COMPLEX * self._sample_bits // 8
        )
        self.heap_shape: Final[Tuple] = (self.n_channels_per_stream, self.n_baselines, COMPLEX)
        self._n_send_heaps_in_flight: Final[int] = n_send_heaps_in_flight

        # 4. Allocate memory buffers
        self.context: Final[katsdpsigproc.abc.AbstractContext] = context

        # 4.1 There may be scope to use asyncio queues here instead - need to figure it out
        self._heaps_queue: queue.Queue[Tuple[asyncio.Future, BufferWrapper]] = queue.Queue(
            maxsize=self._n_send_heaps_in_flight
        )
        self.buffers: List[accel.HostArray] = []

        # 4.2 Create buffers once-off to be reused for sending data.
        for _ in range(self._n_send_heaps_in_flight):
            # 4.2.1 Create a buffer from the accel context.
            # TODO: I'm not too happy about this hardcoded int32 here, but I don't
            # have an object close by that I can get the dtype from.
            buffer = accel.HostArray(self.heap_shape, np.int32, context=self.context)

            # 4.2.2 Create a dummy future object that is already marked as
            # "done" Each buffer is paired with a future so these dummy onces
            # are necessary for initial start up.
            dummy_future: asyncio.Future = asyncio.Future()
            dummy_future.set_result("")

            # 4.2.3. Wrap buffer in BufferWrapper, join it together
            # with its future as a tuple and put it on the heaps queue
            self._heaps_queue.put((dummy_future, BufferWrapper(buffer)))

            # 4.2.4 Store buffer in array so that it can be assigned to ibverbs.
            # memory regions by the stream_factory.
            self.buffers.append(buffer)

        # 5. Generate all required stream information that is not specific to transports defined in the child classes
        packets_per_heap = math.ceil(self.heap_payload_size_bytes / XSend.max_payload_size)
        packet_header_overhead_bytes = packets_per_heap * XSend.header_size

        # 5.1 If the dump_interval is set to zero, pass zero to stream_config to send as fast as possible.
        if self.dump_interval_s != 0:
            send_rate_bytes_per_second = (
                (self.heap_payload_size_bytes + packet_header_overhead_bytes)
                / self.dump_interval_s
                * self.send_rate_factor
            )  # * send_rate_factor adds a buffer to the rate to compensate for any unexpected jitter
        else:
            send_rate_bytes_per_second = 0

        stream_config = spead2.send.StreamConfig(
            max_packet_size=self.max_packet_size,
            max_heaps=self._n_send_heaps_in_flight,
            rate_method=spead2.send.RateMethod.AUTO,
            rate=send_rate_bytes_per_second,
        )
        # This class is currently marked as _private in the spead2 stub files,
        # in a future revision it may be changed to public.
        self.source_stream = stream_factory(stream_config, self.buffers)
        self.source_stream.set_cnt_sequence(
            channel_offset // n_channels_per_stream,
            n_channels // n_channels_per_stream,
        )

        # 6. Create item group - This is the SPEAD2 object that stores all heap format information.
        self.item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
        self.item_group.add_item(
            FREQUENCY_ID,
            "frequency",  # Misleading name, but it's what the ICD specifies
            "Value of first channel in collections stored here.",
            shape=[],
            format=[("u", FLAVOUR.heap_address_bits)],
        )
        self.item_group.add_item(
            TIMESTAMP_ID,
            "timestamp",
            "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
            shape=[],
            format=[("u", FLAVOUR.heap_address_bits)],
        )
        self.item_group.add_item(
            XENG_RAW_ID,
            "xeng_raw",
            "Integrated baseline correlation products.",
            shape=self.heap_shape,
            dtype=np.int32,
        )

        # 6.1 The first heap is the SPEAD descriptor - store it for transmission when required
        self.descriptor_heap = self.item_group.get_heap(descriptors="all", data="none")

    def send_heap(self, timestamp: int, buffer_wrapper: BufferWrapper) -> None:
        """Take in a :class:`BufferWrapper` and send it as a SPEAD heap.

        This function is non-blocking. There is no guarantee that a packet has
        been sent by the time the function completes.

        Parameters
        ----------
        timestamp
            The timestamp that will be assigned to the buffer when it is
            encapsulated in a SPEAD heap.
        buffer_wrapper
            Wrapped buffer to sent as a SPEAD heap.
        """
        self.item_group["timestamp"].value = timestamp
        self.item_group["frequency"].value = self.channel_offset
        self.item_group["xeng_raw"].value = buffer_wrapper.buffer

        heap_to_send = self.item_group.get_heap(descriptors="none", data="all")
        # This flag forces the heap to include all item_group pointers in every
        # packet belonging to a single heap instead of just in the first
        # packet. This is done to duplicate the format of the packets out of
        # the MeerKAT SKARABs.
        heap_to_send.repeat_pointers = True

        future = self.source_stream.async_send_heap(heap_to_send)
        self._heaps_queue.put((future, buffer_wrapper))
        # Note: it's not strictly true to say that the data has been sent at
        # this point; it's only been queued for sending. But it should be close
        # enough for monitoring data rates at the granularity that this is
        # typically done.
        output_heaps_counter.inc(1)
        output_bytes_counter.inc(buffer_wrapper.buffer.nbytes)

    async def get_free_heap(self) -> BufferWrapper:
        """
        Return a :class:BufferWrapper` object from the internal fifo queue when one is available.

        There are a limited number of BufferWrapper in existence and
        they are all stored with a future object. If the future is complete,
        the buffer is not being used for sending and it will return the buffer
        immediately. If the future is still busy, this function will wait
        asynchronously for the future to be done.

        This function is compatible with asyncio.

        Returns
        -------
        buffer_wrapper
            Free buffer wrapped in a :class:`BufferWrapper`.
        """
        future, buffer_wrapper = self._heaps_queue.get()
        await asyncio.wait([future])
        return buffer_wrapper

    def send_descriptor_heap(self) -> None:
        """
        Send the SPEAD descriptor over the SPEAD2 transport.

        This function transmits the descriptor heap created at the start of
        transmission. I am unsure if this is the correct or best way to do
        this. At this stage in development descriptors have not been considered
        deeply.

        This function has no associated unit test - it will likely need to be
        revisited later as its need and function become clear.
        """
        self.source_stream.async_send_heap(self.descriptor_heap)
