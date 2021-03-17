"""TODO: Write this."""

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
    """TODO: Write this docstring."""

    class XEngineHeapBufferWrapper:
        """TODO: Write this docstring."""

        def __init__(self, buffer: np.ndarray) -> None:
            """TODO: Write this docstring."""
            self.buffer: np.ndarray = buffer

    # SPEAD static constants
    TIMESTAMP_ID: Final = 0x1600
    CHANNEL_OFFSET: Final = 0x4103
    DATA_ID: Final = 0x1800
    default_spead_flavour: Final = {"version": 4, "item_pointer_bits": 64, "heap_address_bits": 48, "bug_compat": 0}

    # Class static constants
    max_payload_size: Final = 2048
    header_size: Final = 64
    max_packet_size: Final = max_payload_size + header_size
    complexity: Final = 2

    # Initialise class includng all variables
    def __init__(self) -> None:
        """TODO: Write this docstring."""
        self.n_ants: int = 64
        self.n_channels_per_stream: int = 128
        self.n_pols: int = 2
        self.n_baselines: int = (self.n_pols * self.n_ants + 1) * (self.n_ants * self.n_pols) // 2
        self.sample_bits: int = 32
        self.dump_rate_s = 0.4
        self.heap_size_bytes: int = (
            self.n_channels_per_stream * self.n_baselines * XEngineSPEADSend.complexity * self.sample_bits // 8
        )
        self.context: katsdpsigproc.katsdpsigproc.abc.AbstractContext = accel.create_some_context(
            device_filter=lambda x: x.is_cuda
        )
        self.n_send_heaps_in_flight: int = 5
        self.channel_offset = 128

        # THere may be scope to use asynio queues here instead - need to figure it out
        self._heaps_queue: queue.Queue[
            typing.Tuple[asyncio.Future, XEngineSPEADSend.XEngineHeapBufferWrapper]
        ] = queue.Queue(maxsize=self.n_send_heaps_in_flight)
        self.buffers = []  # say if this is the best way to do things

        packets_per_heap = math.ceil(self.heap_size_bytes / XEngineSPEADSend.max_payload_size)
        packet_header_overhead_bytes = packets_per_heap * XEngineSPEADSend.header_size
        rate_Bps = (
            (self.heap_size_bytes + packet_header_overhead_bytes) / self.dump_rate_s * 1.1
        )  # 1.1 adds a 10 percent buffer
        print(self.n_baselines, self.heap_size_bytes / 1024 / 1024 * 8, rate_Bps / 1024 / 1024 / 1.1 * 8)
        self.streamConfig = spead2.send.StreamConfig(
            max_packet_size=self.max_packet_size,
            max_heaps=self.n_send_heaps_in_flight,
            rate_method=spead2.send.RateMethod.AUTO,
            rate=rate_Bps,
        )
        self.sourceStream: spead2.send.asyncio.AbstractStream

        for i in range(self.n_send_heaps_in_flight):
            # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
            buffer = accel.HostArray((self.heap_size_bytes,), np.uint8, context=self.context)
            # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
            dummyFuture: asyncio.Future = asyncio.Future()
            dummyFuture.set_result("")

            self._heaps_queue.put((dummyFuture, XEngineSPEADSend.XEngineHeapBufferWrapper(buffer)))
            self.buffers.append(buffer)

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

        # 3.2 Throw away first heap - need to get this as it contains a bunch of descriptor information that we dont want
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

    def __init__(self) -> None:  # Pass endpoint here
        """TODO: Write this docstring."""
        XEngineSPEADSend.__init__(self)
        self.endpoint: typing.Tuple[str, int] = ("239.10.10.11", 7149)
        thread_pool = spead2.ThreadPool()
        self.sourceStream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            self.streamConfig,
            spead2.send.UdpIbvConfig(
                endpoints=[self.endpoint],
                interface_address="10.100.44.1",
                ttl=4,
                comp_vector=2,
                memory_regions=self.buffers,
            ),
        )
        del thread_pool  # This line is copied from the SPEAD2 examples.


class XEngineSPEADInprocSend(XEngineSPEADSend):
    """TODO: Write this docstring."""

    def __init__(self) -> None:  # Pass endpoint here
        """TODO: Write this docstring."""
        XEngineSPEADSend.__init__(self)
        self.queue: spead2.InprocQueue = spead2.InprocQueue()
        thread_pool = spead2.ThreadPool()
        self.sourceStream = spead2.send.asyncio.InprocStream(
            thread_pool,
            [self.queue],
            self.streamConfig,
        )
        del thread_pool  # This line is copied from the SPEAD2 examples.
