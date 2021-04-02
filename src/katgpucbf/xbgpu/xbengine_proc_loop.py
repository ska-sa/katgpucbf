"""TODO: Write this."""

import asyncio
import numpy as np
from typing import List

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource

import katxgpu.monitor
import katxgpu._katxgpu.recv as recv


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

        n_ants = 64
        n_channels_total = 32768
        n_channels_per_stream = 128
        n_samples_per_channel = 256
        n_pols = 2
        sample_bits = 8

        ############################

        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        n_rx_items = 3  # To high means to much GPU memory gets allocated
        n_tx_items = 1

        self.rx_running = True

        use_file_monitor = False
        if use_file_monitor:
            monitor: katxgpu.monitor.Monitor = katxgpu.monitor.FileMonitor("temp_file.log")
        else:
            monitor = katxgpu.monitor.NullMonitor()
        self.monitor = monitor

        # ########################
        # This step represents the difference in timestamp between two consecutive heaps received from the same F-Engine. We
        # multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.
        # While we can workout the timestamp_step from other parameters that configure the receiver, we pass it as a seperate
        # argument to the reciever for cases where the n_channels_per_stream changes across streams (likely for non-power-of-
        # two array sizes).
        self.chunk_timestamp_step = n_channels_total * 2 * n_samples_per_channel
        heaps_per_fengine_per_chunk = 10
        rx_thread_affinity = 2
        ringbuffer_capacity = 8
        self.ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
        self.receiverStream = recv.Stream(
            n_ants=n_ants,
            n_channels=n_channels_per_stream,
            n_samples_per_channel=n_samples_per_channel,
            n_pols=n_pols,
            sample_bits=sample_bits,
            timestamp_step=self.chunk_timestamp_step,
            heaps_per_fengine_per_chunk=heaps_per_fengine_per_chunk,
            ringbuffer=self.ringbuffer,
            thread_affinity=rx_thread_affinity,
            use_gdrcopy=False,
            monitor=monitor,
        )

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
            rx_item = XBEngineProcessingLoop.CommunicationItem()
            rx_item.buffer = katsdpsigproc.accel.DeviceArray(self.context, (self.receiverStream.chunk_bytes,), np.uint8)
            self._rx_free_item_queue.put_nowait(rx_item)
        for i in range(n_tx_items):
            tx_item = XBEngineProcessingLoop.CommunicationItem()
            tx_item.buffer = katsdpsigproc.accel.DeviceArray(self.context, (self.receiverStream.chunk_bytes,), np.uint8)
            self._tx_free_item_queue.put_nowait(tx_item)

    async def _receiver_loop(self):
        """TODO: Write this."""
        print("Receiver Loop Start")
        for i in range(10):
            await asyncio.sleep(1)
            item = await self._rx_free_item_queue.get()
            item.timestamp += i * 10
            print(f"1. Timestamp: {item.timestamp}")
            host_buffer = katsdpsigproc.accel.HostArray(
                (self.receiverStream.chunk_bytes,), np.uint8, (self.receiverStream.chunk_bytes,), context=self.context
            )  # not necessary
            host_buffer[:] = i
            print(host_buffer)
            item.buffer.set_async(self._upload_command_queue, host_buffer)
            item.add_event(self._upload_command_queue.enqueue_marker())
            await self._rx_item_queue.put(item)  # Dont think it needs to be async
        self.rx_running = False

    async def _gpu_proc_loop(self):
        """TODO: Write this."""
        print("GPU Proc Loop Start")
        while self.rx_running is True:
            itemRx = await self._rx_item_queue.get()
            itemTx = await self._tx_free_item_queue.get()
            await itemRx.async_wait_for_events()
            itemRx.buffer.copy_region(
                self._proc_command_queue,
                itemRx.buffer,
                np.s_[0 :: self.receiverStream.chunk_bytes],
                np.s_[0 :: self.receiverStream.chunk_bytes],
            )
            itemTx.add_event(self._proc_command_queue.enqueue_marker())
            itemTx.timestamp = itemTx.timestamp + itemRx.timestamp + 1
            print(f"2. Timestamp: {itemTx.timestamp}")
            await self._tx_item_queue.put(itemTx)
            itemRx.reset()
            await self._rx_free_item_queue.put(itemRx)

    async def _sender_loop(self):
        """TODO: Write this."""
        print("Sender Loop Start")
        while self.rx_running is True:
            host_buffer = katsdpsigproc.accel.HostArray(
                (self.receiverStream.chunk_bytes,), np.uint8, (self.receiverStream.chunk_bytes,), context=self.context
            )  # not necessary
            item = await self._tx_item_queue.get()
            await item.async_wait_for_events()
            item.buffer.get_async(self._download_command_queue, host_buffer)
            print(f"3. Timestamp: {item.timestamp + 1}")
            print(host_buffer)
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
