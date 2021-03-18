"""
Module for sending baselines produced by the GPU X-Engine onto the network.

This module has been designed to work with asyncio.

The data sent onto the network conforms to the SPEAD protocol. This module takes the baseline data, turns it into a
SPEAD heap and then transmits that heap out onto the network using the SPEAD2 python module. The high-performance
ibverbs implementation of SPEAD2 will be used even though the data rates out are very low due to the ibverbs
implementation using far fewer system resources.

This module defines three seperate classes - an abstract XEngineSPEADSend base class and, the XEngineSPEADIbvSend and
XEngineSPEADInprocSend child classes. The main object that will be used is the XEngineSPEADIbvSend class as this
actually creates the SPEAD2 ibverbs transport. However in order to implement unit tests without having a network, the
SPEAD2 inproc transport must be used. This led to the creation of an XEngineSPEADInprocSend class that implements the
inproc transport. Most of this module's logic is in the XEngineSPEADSend base class, the child classes only implement the
the different transports.

The XEngineSPEADSend class creates its own buffers and data in those buffers will be encapsulated into SPEAD heaps and
sent onto the network. The user can request the buffers from the object, populate them and then give them back to the
class for transmission. The user must not give other buffers to the class as while this will work, it will be much
slower. The memory regions of the XEngineSPEADSend generated buffers have been registered with ibverbs to enable zero
copy transmission - other buffers will create an extra copy.
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


class XEngineSPEADSend(ABC):
    """Base class that m."""

    class XEngineHeapBufferWrapper:
        """TODO: Write this docstring."""

        def __init__(self, buffer: np.ndarray) -> None:
            """TODO: Write this docstring."""
            self.buffer: np.ndarray = buffer

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
        dump_rate_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        n_send_heaps_in_flight: int = 5,  # I dont see any need for this to be configurable, the data rates are likely
        # too low for it to be an issue. I have put it here more to be explicit than anything else.
    ) -> None:
        """TODO: Write this docstring."""
        # 1. Array Configuration Parameters
        self.n_ants: Final[int] = n_ants
        self.n_channels_per_stream: Final[int] = n_channels_per_stream
        self.n_pols: Final[int] = n_pols
        self.n_baselines: Final[int] = (self.n_pols * self.n_ants + 1) * (self.n_ants * self.n_pols) // 2
        self.dump_rate_s: Final[float] = dump_rate_s
        self._sample_bits: Final[int] = 32

        # 2. Multicast Stream Parameters
        self.channel_offset: Final[int] = channel_offset
        self.heap_size_bytes: Final[int] = (
            self.n_channels_per_stream * self.n_baselines * XEngineSPEADSend.complexity * self._sample_bits // 8
        )
        self._n_send_heaps_in_flight: Final[int] = n_send_heaps_in_flight

        # 3. Allocate memory buffers
        self.context: Final[katsdpsigproc.abc.AbstractContext] = context

        # 3.1 There may be scope to use asynio queues here instead - need to figure it out
        self._heaps_queue: queue.Queue[
            typing.Tuple[asyncio.Future, XEngineSPEADSend.XEngineHeapBufferWrapper]
        ] = queue.Queue(maxsize=self._n_send_heaps_in_flight)
        self.buffers: typing.List[accel.HostArray] = []  # say if this is the best way to do things

        # 3.2 Create buffers once-off to be reused for sending data.
        for i in range(self._n_send_heaps_in_flight):
            # 3.2.1 Create a buffer from the accel context.
            buffer = accel.HostArray((self.heap_size_bytes,), np.uint8, context=self.context)

            # 3.2.2 Create a dummy future object that is already marked as "done" Each buffer is paired with a future
            # so these dummy onces are necessary for initial start up.
            dummyFuture: asyncio.Future = asyncio.Future()
            dummyFuture.set_result("")

            # 3.2.3. Wrap buffer in XEngineHeapBufferWrapper, join it together with its future as a tuple and put it
            # on the heaps queue
            self._heaps_queue.put((dummyFuture, XEngineSPEADSend.XEngineHeapBufferWrapper(buffer)))

            # 3.2.4 Store buffer in array so that it can be assigned to ibverbs memory regions in the
            # XEngineSPEADIbvSend stream.
            self.buffers.append(buffer)

        # 4. Generate all required stream information that is not specific to transports defined in the child classes
        packets_per_heap = math.ceil(self.heap_size_bytes / XEngineSPEADSend.max_payload_size)
        packet_header_overhead_bytes = packets_per_heap * XEngineSPEADSend.header_size
        send_rate_Bps = (
            (self.heap_size_bytes + packet_header_overhead_bytes) / self.dump_rate_s * 1.1
        )  # *1.1 adds a 10 percent buffer to the rate to compensate for any unexpected jitter

        self.streamConfig = spead2.send.StreamConfig(
            max_packet_size=self.max_packet_size,
            max_heaps=self._n_send_heaps_in_flight,
            rate_method=spead2.send.RateMethod.AUTO,
            rate=send_rate_Bps,
        )
        self.sourceStream: spead2.send.asyncio.AbstractStream

        # 5. Create item group - This is the SPEAD2 object that stores all heap format information.
        self.item_group = spead2.send.ItemGroup(flavour=spead2.Flavour(**XEngineSPEADSend.default_spead_flavour))
        self.item_group.add_item(
            XEngineSPEADSend.CHANNEL_OFFSET,
            "channel offset",
            "Value of first channel in collections stored here",
            shape=[],
            format=[("u", XEngineSPEADSend.default_spead_flavour["heap_address_bits"])],
        )
        self.item_group.add_item(
            XEngineSPEADSend.TIMESTAMP_ID,
            "timestamp",
            "timestamp description",
            shape=[],
            format=[("u", XEngineSPEADSend.default_spead_flavour["heap_address_bits"])],
        )
        self.item_group.add_item(
            XEngineSPEADSend.DATA_ID,
            "xeng_raw",
            "Integrated baseline correlation products",
            shape=(self.heap_size_bytes,),
            dtype=np.int8,
        )

        # 5.1 Throw away first heap - need to get this as it contains a bunch of descriptor information that we dont want
        # for the purposes of this test.
        self.item_group.get_heap()

    def send_heap(self, timestamp: int, bufferWrapper: XEngineHeapBufferWrapper) -> None:
        """TODO: Write this docstring."""
        self.item_group["timestamp"].value = timestamp
        self.item_group["channel offset"].value = self.channel_offset
        self.item_group["xeng_raw"].value = bufferWrapper.buffer

        heap_to_send = self.item_group.get_heap()
        # Say why this flag is needed
        heap_to_send.repeat_pointers = True

        future = self.sourceStream.async_send_heap(heap_to_send)
        self._heaps_queue.put((future, bufferWrapper))

    async def get_free_heap(self) -> XEngineHeapBufferWrapper:
        """TODO: Write this docstring."""
        future, bufferWrapper = self._heaps_queue.get()
        await asyncio.wait([future])
        return bufferWrapper


class XEngineSPEADIbvSend(XEngineSPEADSend):
    """TODO: Write this docstring."""

    def __init__(
        self,
        n_ants: int,
        n_channels_per_stream: int,
        n_pols: int,
        dump_rate_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        endpoint: typing.Tuple[str, int],
        interface_address: str,
        thread_affinity: int,
    ) -> None:  # Pass endpoint here
        """TODO: Write this docstring."""
        # 1. Initialise base class
        XEngineSPEADSend.__init__(
            self,
            n_ants=n_ants,
            n_channels_per_stream=n_channels_per_stream,
            n_pols=n_pols,
            dump_rate_s=dump_rate_s,
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
        del thread_pool  # This line is copied from the SPEAD2 examples.


class XEngineSPEADInprocSend(XEngineSPEADSend):
    """TODO: Write this docstring."""

    def __init__(
        self,
        n_ants: int,
        n_channels_per_stream: int,
        n_pols: int,
        dump_rate_s: float,
        channel_offset: int,
        context: katsdpsigproc.abc.AbstractContext,
        queue: spead2.InprocQueue,
    ) -> None:
        """TODO: Write this docstring."""
        XEngineSPEADSend.__init__(
            self,
            n_ants=n_ants,
            n_channels_per_stream=n_channels_per_stream,
            n_pols=n_pols,
            dump_rate_s=dump_rate_s,
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
        del thread_pool  # This line is copied from the SPEAD2 examples.
