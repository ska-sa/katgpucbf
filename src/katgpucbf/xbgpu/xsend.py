"""
Module for sending baselines produced by the GPU X-Engine onto the network.

This module has been designed to work with asyncio.

The data sent onto the network conforms to the SPEAD protocol. This module takes the baseline data, turns it into a
SPEAD heap and then transmits that heap out onto the network using the SPEAD2 python module. The high-performance
ibverbs implementation of SPEAD2 will be used even though the data rates out are very low due to the ibverbs
implementation using far fewer system resources. The format of the packets transmitted by SPEAD2 can be found here:
https://docs.google.com/drawings/d/1d3CMrMl8wTQfVlyX5NXztMGGHhak-37mt4IoK7Idc_I/edit

This module defines three seperate classes - an abstract XEngineSPEADAbstractSend base class and, the
XEngineSPEADIbvSend and XEngineSPEADInprocSend child classes. The main object that will be used is the
XEngineSPEADIbvSend class as this actually creates the SPEAD2 ibverbs transport. However in order to implement unit
tests without having a network, the SPEAD2 inproc transport must be used. This led to the creation of an
XEngineSPEADInprocSend class that implements the inproc transport. Most of this module's logic is in the
XEngineSPEADAbstractSend base class, the child classes only implement the the different transports.

The XEngineSPEADAbstractSend class creates its own buffers and data in those buffers will be encapsulated into SPEAD
heaps and sent onto the network. The user can request the buffers from the object, populate them and then give them back
to the class for transmission. The user must not give other buffers to the class as while this will work, it will be
much slower. The memory regions of the XEngineSPEADAbstractSend generated buffers have been registered with ibverbs to
enable zero copy transmission - using other buffers will force an extra copy. Zero copy transmission means that the
data to be transmitted can sent from its current memory location directly to the NIC without having to be copied to
an intermediary memory location in the process, thereby halving the memory bandwidth required to send.
"""

import spead2
import spead2.send.asyncio
import katsdpsigproc
import katsdpsigproc.accel as accel
import numpy as np
import asyncio
from typing_extensions import Final  # type: ignore # This should change from "typing_extensions" to  "typing" in Python 3.8
import typing
import queue
import math
from abc import ABC


class XEngineSPEADAbstractSend(ABC):
    """
    Base class for turning baseline correlation products into SPEAD heaps and transmitting them.

    This class creates a queue of buffers that can be sent out onto the network. To get one of these buffers call the
    buffer_wrapper = thisObject.get_free_heap() object function - it will return a buffer wrapped in a
    XEngineHeapBufferWrapper object. Once the necessary data has been copied to the buffer and it is ready to be sent
    onto the network, pass it back to this object using the thisObject.send_heap(buffer_wrapper) command. This object
    will create a limited number of buffers and keep recycling them - avoiding any memory allocation at runtime.

    This has been designed to run in an asyncio loop, and the get_free_heap() function makes sure that the next buffer
    in the queue is not in flight before returning.

    This base class is missing the SPEAD2 transports for transmitting data. The child classes of this class are
    responsible for implementing the specific transports.

    While this base class is meant to be abstract, it has no abstract functions and so it can be constructed
    without generating an error (the self.sourceStream: spead2.send.asyncio.AbstractStream class member is the
    abstract part). An error will only be thrown once an attempt is made to access the self.context object.
    """

    class XEngineHeapBufferWrapper:
        """
        Holds a buffer object that has been configured so that it can be zero-copy transferred onto the network.

        In order to preserve the zero-copy properties of th buffer, data can only be copied to the buffer within this
        class, the buffer handle cannot be overwritten. For example: "buffer_wrapper.buffer = newArray" will fail as
        it attempts to assign a new object to the buffer variable while "buffer_wrapper.buffer[:] = newArray" will
        succeed as it overwrites the values in the buffer, not the buffer itself.

        There may be a better way to hold these buffer objects and prevent them being overwritten other than
        wrapping them in an XEngineHeapBufferWrapper class but I could not think of any at the time of writing.
        """

        def __init__(self, buffer: np.ndarray) -> None:
            """
            Initialise the XEngineHeapBufferWrapper object.

            Parameters
            ----------
            buffer: np.ndarray
                The buffer object configured for zero-copy transfers.
            """
            self._buffer: np.ndarray = buffer

        @property
        def buffer(self) -> np.ndarray:
            """Return the buffer associated with this class."""
            return self._buffer

        @buffer.setter
        def buffer(self, buffer):
            """
            Overwrite the buffer's handle - this functionality has been disabled.

            Instead of overwriting the handle, copy to the buffer using array indexing [:] instead.
            """
            raise AttributeError("Don't overwrite the buffer's handle! Copy to it instead using array indexing [:].")

    # SPEAD static constants
    TIMESTAMP_ID: Final[int] = 0x1600
    CHANNEL_OFFSET: Final[int] = 0x4103
    DATA_ID: Final[int] = 0x1800
    default_spead_flavour: Final[dict] = {
        "version": 4,
        "item_pointer_bits": 64,
        "heap_address_bits": 48,
        "bug_compat": 0,
    }

    # Class static constants
    max_payload_size: Final[int] = 2048
    header_size: Final[int] = 64
    max_packet_size: Final[int] = max_payload_size + header_size
    complexity: Final[int] = 2

    # Initialise class includng all variables
    def __init__(
        self,
        n_ants: int,
        n_channels_per_stream: int,
        n_pols: int,
        dump_interval_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        n_send_heaps_in_flight: int = 5,
    ) -> None:
        """
        Construct an XEngineSPEADAbstractSend object.

        Parameters
        ----------
        n_ants: int
            The number of antennas that have been correlated.
        n_channels_per_stream: int
            The number of frequency channels contained per stream.
        n_pols: int
            The number of pols per antenna. Expected to always be 2.
        dump_interval_s: float
            A new heap is transmitted every dump_interval_s seconds. Set to zero to send as fast as possible.
        channel_offset: int
            Fixed value to be included in the SPEAD heap indicating the lowest channel value transmitted by this heap.
            Must be a multiple of n_channels_per_stream.
        context: katsdpsigproc.abc.AbstractContext
            All buffers to be transmitted will be created from this context.
        n_send_heaps_in_flight: int
            Number of buffers that will be queued at any one time. I don't see any need for this to be configurable, the
            data rates are likely too low for it to be an issue. I have put it here more to be explicit than anything
            else. This argument is optional
        """
        # 1. Check that given arguments are sane.
        if n_pols != 2:
            raise ValueError("n_pols must equal 2 - no other modes supported at the moment.")

        if dump_interval_s < 0:
            raise ValueError("Dump interval must be 0 or greater.")

        if channel_offset % n_channels_per_stream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_stream")

        # 2. Array Configuration Parameters
        self.n_ants: Final[int] = n_ants
        self.n_channels_per_stream: Final[int] = n_channels_per_stream
        self.n_pols: Final[int] = n_pols
        self.n_baselines: Final[int] = (self.n_ants + 1) * (self.n_ants) // 2
        self.dump_interval_s: Final[float] = dump_interval_s
        self._sample_bits: Final[int] = 32

        # 3. Multicast Stream Parameters
        self.channel_offset: Final[int] = channel_offset
        self.heap_payload_size_bytes: Final[int] = (
            self.n_channels_per_stream
            * self.n_baselines
            * self.n_pols
            * self.n_pols
            * XEngineSPEADAbstractSend.complexity
            * self._sample_bits
            // 8
        )
        self.heap_shape: Final[typing.Tuple] = (self.n_channels_per_stream, self.n_baselines, self.n_pols, self.n_pols)
        self._n_send_heaps_in_flight: Final[int] = n_send_heaps_in_flight

        # 4. Allocate memory buffers
        self.context: Final[katsdpsigproc.abc.AbstractContext] = context

        # 4.1 There may be scope to use asynio queues here instead - need to figure it out
        self._heaps_queue: queue.Queue[
            typing.Tuple[asyncio.Future, XEngineSPEADAbstractSend.XEngineHeapBufferWrapper]
        ] = queue.Queue(maxsize=self._n_send_heaps_in_flight)
        self.buffers: typing.List[accel.HostArray] = []

        # 4.2 Create buffers once-off to be reused for sending data.
        for i in range(self._n_send_heaps_in_flight):
            # 4.2.1 Create a buffer from the accel context.
            buffer = accel.HostArray(self.heap_shape, np.int64, context=self.context)

            # 4.2.2 Create a dummy future object that is already marked as "done" Each buffer is paired with a future
            # so these dummy onces are necessary for initial start up.
            dummyFuture: asyncio.Future = asyncio.Future()
            dummyFuture.set_result("")

            # 4.2.3. Wrap buffer in XEngineHeapBufferWrapper, join it together with its future as a tuple and put it
            # on the heaps queue
            self._heaps_queue.put((dummyFuture, XEngineSPEADAbstractSend.XEngineHeapBufferWrapper(buffer)))

            # 4.2.4 Store buffer in array so that it can be assigned to ibverbs memory regions in the
            # XEngineSPEADIbvSend stream.
            self.buffers.append(buffer)

        # 5. Generate all required stream information that is not specific to transports defined in the child classes
        packets_per_heap = math.ceil(self.heap_payload_size_bytes / XEngineSPEADAbstractSend.max_payload_size)
        packet_header_overhead_bytes = packets_per_heap * XEngineSPEADAbstractSend.header_size

        # 5.1 If the dump_interval is set to zero, pass zero to streamConfig to send as fast as possible.
        if self.dump_interval_s != 0:
            send_rate_Bps = (
                (self.heap_payload_size_bytes + packet_header_overhead_bytes) / self.dump_interval_s * 1.1
            )  # *1.1 adds a 10 percent buffer to the rate to compensate for any unexpected jitter
        else:
            send_rate_Bps = 0

        self.streamConfig = spead2.send.StreamConfig(
            max_packet_size=self.max_packet_size,
            max_heaps=self._n_send_heaps_in_flight,
            rate_method=spead2.send.RateMethod.AUTO,
            rate=send_rate_Bps,
        )
        self.sourceStream: spead2.send.asyncio.AbstractStream  # Left unassigned to remain abstract.

        # 6. Create item group - This is the SPEAD2 object that stores all heap format information.
        self.item_group = spead2.send.ItemGroup(
            flavour=spead2.Flavour(**XEngineSPEADAbstractSend.default_spead_flavour)
        )
        self.item_group.add_item(
            XEngineSPEADAbstractSend.CHANNEL_OFFSET,
            "channel offset",
            "Value of first channel in collections stored here",
            shape=[],
            format=[("u", XEngineSPEADAbstractSend.default_spead_flavour["heap_address_bits"])],
        )
        self.item_group.add_item(
            XEngineSPEADAbstractSend.TIMESTAMP_ID,
            "timestamp",
            "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
            shape=[],
            format=[("u", XEngineSPEADAbstractSend.default_spead_flavour["heap_address_bits"])],
        )
        self.item_group.add_item(
            XEngineSPEADAbstractSend.DATA_ID,
            "xeng_raw",
            "Integrated baseline correlation products",
            shape=self.heap_shape,
            dtype=np.int64,
        )

        # 6.1 The first heap is the SPEAD descriptor - store it for transmission when required
        self.descriptor_heap = self.item_group.get_heap(descriptors="all", data="none")

    def send_heap(self, timestamp: int, bufferWrapper: XEngineHeapBufferWrapper) -> None:
        """
        Take in an XEngineHeapBufferWrapper object and send it onto the network as a SPEAD heap.

        This funtion is non-blocking. There is no guarentee that a packet has been sent by the time the function
        completes.

        Parameters
        ----------
        timestamp: int
            The timestamp that will be assigned to the buffer when it is encapsulated in a SPEAD heap.
        bufferWrapper: XEngineHeapBufferWrapper
            Wrapped buffer to sent as a SPEAD heap.
        """
        self.item_group["timestamp"].value = timestamp
        self.item_group["channel offset"].value = self.channel_offset
        self.item_group["xeng_raw"].value = bufferWrapper.buffer

        heap_to_send = self.item_group.get_heap(descriptors="none", data="all")
        # This flag forces the heap to include all item_group pointers in every packet belonging to a single heap
        # instead of just in the first packet. This is done to duplicate the format of the packets out of the MeerKAT
        # SKARABs.
        heap_to_send.repeat_pointers = True

        future = self.sourceStream.async_send_heap(heap_to_send)
        self._heaps_queue.put((future, bufferWrapper))

    async def get_free_heap(self) -> XEngineHeapBufferWrapper:
        """
        Return an XEngineHeapBufferWrapper object from the internal fifo queue when one is avaiable.

        There are a limited number of XEngineHeapBufferWrapper in existence and they are all stored with a future
        object. If the future is complete, the buffer is not being used for sending and it will return the buffer
        immediatly. If the future is still busy, this function will wait asynchronously for the future to be done.

        This function is compatible with asyncio.

        Parameters
        ----------
        timestamp: int
            The timestamp that will be assigned to the buffer when it is encapsulated in a SPEAD heap.
        bufferWrapper: XEngineHeapBufferWrapper
            Wrapped buffer to sent as a SPEAD heap.

        Returns
        -------
        bufferWrapper: XEngineHeapBufferWrapper
            Free buffer wrapped in an XEngineHeapBufferWrapper object.
        """
        future, bufferWrapper = self._heaps_queue.get()
        await asyncio.wait([future])
        return bufferWrapper

    def send_descriptor_heap(self):
        """
        Send the SPEAD descriptor over the SPEAD2 transport.

        This function transmits the descriptor heap created at the start of transmission. I am unsure if this is the
        correct or best way to do this. At this stage in development descriptors have not been considered deeply.

        This function has no associated unit test - it will likely need to be revisited later as its need and function
        become clear.
        """
        self.sourceStream.async_send_heap(self.descriptor_heap)


class XEngineSPEADIbvSend(XEngineSPEADAbstractSend):
    """
    Child class of XEngineSPEADAbstractSend that implementing SPEAD2 ibverbs transport.

    The ibverbs transport enables high-performance, low-overhead network transmission using Mellanox NICs.
    """

    def __init__(
        self,
        n_ants: int,
        n_channels_per_stream: int,
        n_pols: int,
        dump_interval_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        endpoint: typing.Tuple[str, int],
        interface_address: str,
        thread_affinity: int,
    ) -> None:
        """
        Construct an XEngineSPEADIbvSend object.

        This is a derived class of XEngineSPEADAbstractSend configured to use the ibverbs SPEAD2 transport.

        Parameters
        ----------
        n_ants: int
            The number of antennas that have been correlated.
        n_channels_per_stream: int
            The number of frequency channels contained in the stream.
        n_pols: int
            The number of pols per antenna. Expected to always be 2 at the moment.
        dump_interval_s: float
            A new heap is transmitted every dump_interval_s seconds. Set to zero to send as fast as possible.
        channel_offset: int
            Fixed value to be included in the SPEAD heap indicating the lowest channel value transmitted by this heap.
            Must be a multiple of n_channels_per_stream.
        context: katsdpsigproc.abc.AbstractContext
            All buffers to be transmitted will be created from this context.
        endpoint: typing.Tuple[str, int]
            Multicast address and port to transport data on. Expects "("x.x.x.x", port_num)" format
        interface_address: str
            IP address of the server interface that will transmit the multicast data.
        thread_affinity: int
            CPU core that will be used by the SPEAD2 ibverbs transport for all processing.
        """
        # 1. Initialise base class
        super().__init__(
            n_ants=n_ants,
            n_channels_per_stream=n_channels_per_stream,
            n_pols=n_pols,
            dump_interval_s=dump_interval_s,
            channel_offset=channel_offset,
            context=context,
        )

        # 2. Assign simple member variables
        self.endpoint: Final[typing.Tuple[str, int]] = endpoint

        # 3. Create SPEAD2 stream using ibverbs transport for sending data onto a network
        thread_pool = spead2.ThreadPool()
        self.sourceStream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            self.streamConfig,
            spead2.send.UdpIbvConfig(
                endpoints=[self.endpoint],
                interface_address=interface_address,
                ttl=4,
                comp_vector=thread_affinity,
                memory_regions=self.buffers,
            ),
        )


class XEngineSPEADInprocSend(XEngineSPEADAbstractSend):
    """
    Child class of XEngineSPEADAbstractSend that implements the SPEAD2 inproc transport.

    This allows for in process unit testing without needing to be connected to a network.
    """

    def __init__(
        self,
        n_ants: int,
        n_channels_per_stream: int,
        n_pols: int,
        dump_interval_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        queue: spead2.InprocQueue,
    ) -> None:
        """
        Construct an XEngineSPEADInprocSend object.

        This is a child class of XEngineSPEADAbstractSend configured to use the in process SPEAD2 transport for unit
        testing without a network.

        Parameters
        ----------
        n_ants: int
            The number of antennas that have been correlated.
        n_channels_per_stream: int
            The number of frequency channels contained in the stream.
        n_pols: int
            The number of pols per antenna. Expected to always be 2 at the moment.
        dump_interval_s: float
            A new heap is transmitted every dump_interval_s seconds. For the inproc transport this rate is respected
            but is not very useful. Set to zero to send as fast as possible.
        channel_offset: int
            Fixed value to be included in the SPEAD heap indicating the lowest channel value transmitted by this heap.
            Must be a multiple of n_channels_per_stream.
        context: katsdpsigproc.abc.AbstractContext
            All buffers to be transmitted will be created from this context.
        queue: spead2.InprocQueue
            SPEAD2 inproc queue to send heaps to.
        """
        super().__init__(
            n_ants=n_ants,
            n_channels_per_stream=n_channels_per_stream,
            n_pols=n_pols,
            dump_interval_s=dump_interval_s,
            channel_offset=channel_offset,
            context=context,
        )
        self.queue: spead2.InprocQueue = queue
        thread_pool = spead2.ThreadPool()
        self.sourceStream = spead2.send.asyncio.InprocStream(
            thread_pool,
            [self.queue],
            self.streamConfig,
        )
