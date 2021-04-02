"""TODO: Write this."""

import asyncio
from typing import List

import katxgpu.monitor

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource


class XBEngineProcessingLoop:
    """TODO: Write this."""

    class CommunicationItem:
        """
        TODO: Write this.

        Facilitates processing between different loops
        Describe that this could become seperate items like katxgpu
        Could be its on class
        """

        timestamp: int
        events: List[katsdpsigproc.abc.AbstractEvent]
        buffer: katsdpsigproc.accel.DeviceArray

        def __init__(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.reset(timestamp)

        def reset(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.timestamp = timestamp
            self.events = []
            self.buffer = []

        def add_event(self, event: katsdpsigproc.abc.AbstractEvent):
            """TODO: Write this."""
            self.events.append(event)

        async def async_wait_for_events(self):
            """TODO: Write this."""
            await katsdpsigproc.resource.async_wait_for_events(self.events)

        # def enqueue_wait(self, command_queue: katsdpsigproc.abc.AbstractCommandQueue) -> None:
        #     """
        #     TODO: Write this.

        #     Describe what a "wait for event" is
        #     The name is a bit misleading
        #     This adds the created events to the
        #     """
        #     print("Enquing Wait")
        #     print()
        #     command_queue.enqueue_wait_for_events(self.events)
        #     print("Enquing Wait done")

    def __init__(self):
        """TODO: Write this."""
        print("Created Processing Loop Object")

        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        n_rx_items = 3
        n_tx_items = 1

        use_file_monitor = False
        if use_file_monitor:
            monitor: katxgpu.monitor.Monitor = katxgpu.monitor.FileMonitor("temp_file.log")
        else:
            monitor = katxgpu.monitor.NullMonitor()
        self.monitor = monitor

        # These queues are CUDA streams
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        # Inter asyncio communication process - monitor make queue is a wrapper of asyncio queue
        self._rx_item_queue = monitor.make_queue(
            "rx_item_queue", n_rx_items
        )  # type: asyncio.Queue[XBEngineProcessingLoop.CommunicationItem]
        self._rx_free_item_queue = monitor.make_queue(
            "rx_free_item_queue", n_rx_items
        )  # type: asyncio.Queue[XBEngineProcessingLoop.CommunicationItem]
        self._tx_item_queue = monitor.make_queue(
            "tx_item_queue", n_tx_items
        )  # type: asyncio.Queue[XBEngineProcessingLoop.CommunicationItem]
        self._tx_free_item_queue = monitor.make_queue(
            "tx_free_item_queue", n_tx_items
        )  # type: asyncio.Queue[XBEngineProcessingLoop.CommunicationItem]

        for i in range(n_rx_items):
            self._rx_free_item_queue.put_nowait(XBEngineProcessingLoop.CommunicationItem())
        for i in range(n_tx_items):
            self._tx_free_item_queue.put_nowait(XBEngineProcessingLoop.CommunicationItem())

    async def _receiver_loop(self):
        """TODO: Write this."""
        print("Receiver Loop Start")
        while True:
            await asyncio.sleep(1)
            item = await self._rx_free_item_queue.get()
            item.timestamp += 1
            print(f"1. Timestamp: {item.timestamp}")
            await self._rx_item_queue.put(item)  # Dont think it needs to be async

    async def _gpu_proc_loop(self):
        """TODO: Write this."""
        print("GPU Proc Loop Start")
        while True:
            itemRx = await self._rx_item_queue.get()
            itemTx = await self._tx_free_item_queue.get()
            itemTx.timestamp = itemTx.timestamp + itemRx.timestamp + 1
            print(f"2. Timestamp: {itemTx.timestamp}")
            await self._tx_item_queue.put(itemTx)
            itemRx.reset()
            await self._rx_free_item_queue.put(itemRx)

    async def _sender_loop(self):
        """TODO: Write this."""
        print("Sender Loop Start")
        while True:
            item = await self._tx_item_queue.get()
            print(f"3. Timestamp: {item.timestamp + 1}")
            print()
            item.reset()
            await self._tx_free_item_queue.put(item)

    async def run(self):
        """TODO: Write this."""
        # NOTE: Put in todo about upgrading this to python 3.8
        loop = asyncio.get_event_loop()
        receiver_task = loop.create_task(self._receiver_loop())
        gpu_proc_task = loop.create_task(self._gpu_proc_loop())
        sender_task = loop.create_task(self._sender_loop())
        await receiver_task
        await gpu_proc_task
        await sender_task
