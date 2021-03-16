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


class XEngineSPEADSend:
    """TODO: Write this docstring."""

    class XEngineHeapBufferWrapper:
        """TODO: Write this docstring."""

        def __init__(self, buf: np.ndarray) -> None:
            """TODO: Write this docstring."""
            self.buf: np.ndarray = buf

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
        self.heap_size_bytes: int = (
            self.n_channels_per_stream * self.n_baselines * XEngineSPEADSend.complexity * self.sample_bits // 8
        )
        self.context: katsdpsigproc.katsdpsigproc.abc.AbstractContext = accel.create_some_context(
            device_filter=lambda x: x.is_cuda
        )
        self.n_send_heaps_in_flight: int = 5
        self.endpoint: typing.Tuple[str, int] = ("239.10.10.11", 7149)
        self.channel_offset = 128

        # THere may be scope to use asynio queues here instead - need to figure it out
        self._free_heaps_queue: queue.Queue[XEngineSPEADSend.XEngineHeapBufferWrapper] = queue.Queue(
            maxsize=self.n_send_heaps_in_flight
        )
        self._in_flight_heaps_queue: queue.Queue[
            typing.Tuple[asyncio.Future, XEngineSPEADSend.XEngineHeapBufferWrapper]
        ] = queue.Queue(maxsize=self.n_send_heaps_in_flight)
        bufs = []

        for i in range(self.n_send_heaps_in_flight):
            # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
            buf = accel.HostArray((self.heap_size_bytes,), np.uint8, context=self.context)
            # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
            self._free_heaps_queue.put(XEngineSPEADSend.XEngineHeapBufferWrapper(buf))
            bufs.append(buf)

        thread_pool = spead2.ThreadPool()
        self.sourceStream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            spead2.send.StreamConfig(
                max_packet_size=self.max_packet_size,
                max_heaps=self.n_send_heaps_in_flight,
                rate_method=spead2.send.RateMethod.AUTO,
                rate=10e6,  # TODO Calculate a reasonable rate
            ),
            spead2.send.UdpIbvConfig(
                endpoints=[self.endpoint], interface_address="10.100.44.1", ttl=4, comp_vector=2, memory_regions=bufs
            ),
        )
        del thread_pool  # This line is copied from the SPEAD2 examples.

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

    def send_heap(self, timestamp: int, heap_buffer: XEngineHeapBufferWrapper) -> None:
        """TODO: Write this docstring."""
        self.item_group["timestamp"].value = timestamp
        self.item_group["channel offset"].value = self.channel_offset
        self.item_group["xeng_raw"].value = heap_buffer.buf

        heap_to_send = self.item_group.get_heap()
        # Say why this flag is needed
        heap_to_send.repeat_pointers = True

        future = self.sourceStream.async_send_heap(heap_to_send)
        self._in_flight_heaps_queue.put((future, heap_buffer))

    async def get_free_heap(self) -> XEngineHeapBufferWrapper:
        """TODO: Write this docstring."""
        future, bufWrap = self._in_flight_heaps_queue.get()
        await asyncio.wait([future])
        return bufWrap


# print(XEngineSPEADSend.max_packet_size)

# x = XEngineSPEADSend()
# print(x.endpoint)

# x.send_heap(0x1, x._free_heaps_queue.get())
# x.send_heap(0x2, x._free_heaps_queue.get())
# x.send_heap(0x3, x._free_heaps_queue.get())
# x.send_heap(0x4, x._free_heaps_queue.get())
# x.send_heap(0x4, x._free_heaps_queue.get())

# loop = asyncio.get_event_loop()
# loop.run_until_complete(x.get_free_heap())
# loop.close()


# for i in range(n_send_chunks):
#     ig["timestamp"].value = (i + 1) * 0x1000
#     ig["channel offset"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
#     ig["xeng_raw"].value = bufs[i]
#     print("Sending", time.time())

#     heap_to_send = ig.get_heap()
#     # Say why needed
#     heap_to_send.repeat_pointers = True
#     futures = [sourceStream.async_send_heap(heap_to_send)]
#     asyncio.get_event_loop().run_until_complete(asyncio.wait(futures))
#     print("Sent   ", time.time())
#     print()
