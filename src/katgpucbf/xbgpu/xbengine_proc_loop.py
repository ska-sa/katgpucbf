"""
TODO: Write this.

TODO:
1. Autoresync logic
2. Sender to inproc transport/ibverbs
3. Receiver to buffer/ibverbs/pcap transport
4. Figure out what to do with the comp vector
5. Make Two communication items
"""

import asyncio
import numpy as np
from typing import List

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource

import katxgpu.monitor
import katxgpu.tensorcore_xengine_core
import katxgpu.precorrelation_reorder
import katxgpu._katxgpu.recv as recv
import katxgpu.xsend
import katxgpu.ringbuffer


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
        buffer_device: katsdpsigproc.accel.DeviceArray
        chunk: katxgpu._katxgpu.recv.Chunk

        def __init__(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.reset(timestamp)

        def reset(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.timestamp = timestamp
            self.events = []
            # Need to reset chunk

        def add_event(self, event: katsdpsigproc.abc.AbstractEvent):
            """TODO: Write this."""
            self.events.append(event)

        async def async_wait_for_events(self):
            """TODO: Write this."""
            await katsdpsigproc.resource.async_wait_for_events(self.events)

    def __init__(self):
        """TODO: Write this."""
        print("Created Processing Loop Object")

        self.n_ants = 64
        self.n_channels_total = 32768
        self.n_channels_per_stream = 128
        self.n_samples_per_channel = 256
        self.n_pols = 2
        self.sample_bits = 8
        complexity = 2

        ############################

        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        n_rx_items = 3  # To high means to much GPU memory gets allocated
        n_tx_items = 2

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
        self.rx_heap_timestamp_step = self.n_channels_total * 2 * self.n_samples_per_channel
        self.heaps_per_fengine_per_chunk = 10
        rx_thread_affinity = 2
        ringbuffer_capacity = 8
        self.ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
        self.receiverStream = recv.Stream(
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_pols=self.n_pols,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.heaps_per_fengine_per_chunk,
            ringbuffer=self.ringbuffer,
            thread_affinity=rx_thread_affinity,
            use_gdrcopy=False,
            monitor=monitor,
        )
        self.rx_bytes_per_heap = (
            self.n_ants * self.n_channels_per_stream * self.n_samples_per_channel * self.n_pols * complexity
        )

        # Sender stuff
        self.tx_thread_affinity = 3
        self.dump_rate_s = 0.5
        self.sendStream = katxgpu.xsend.XEngineSPEADIbvSend(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_pols=self.n_pols,
            dump_rate_s=self.dump_rate_s,
            channel_offset=self.n_channels_per_stream * 4,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            endpoint=("239.10.10.11", 7149),
            interface_address="10.100.44.1",
            thread_affinity=self.tx_thread_affinity,
        )

        # These queues are CUDA streams
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        # Set up GPU Kernels
        template = katxgpu.tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
        )
        self.tensorCoreXEngineCoreOperation = template.instantiate(self._proc_command_queue)

        template = katxgpu.precorrelation_reorder.PreCorrelationReorderTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_batches=self.heaps_per_fengine_per_chunk,
        )
        self.preCorrelationReorderOperation = template.instantiate(self._proc_command_queue)

        self.reordered_buffer_device = katsdpsigproc.accel.DeviceArray(
            self.context, self.preCorrelationReorderOperation.template.outputDataShape, np.int16
        )
        self.preCorrelationReorderOperation.bind(outReordered=self.reordered_buffer_device)

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
            rx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context, self.preCorrelationReorderOperation.template.inputDataShape, np.int16
            )
            self._rx_free_item_queue.put_nowait(rx_item)
        for i in range(n_tx_items):
            tx_item = XBEngineProcessingLoop.CommunicationItem()
            tx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context, self.tensorCoreXEngineCoreOperation.template.outputDataShape, np.int64
            )
            self._tx_free_item_queue.put_nowait(tx_item)

        for i in range(8):
            # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
            buf = katsdpsigproc.accel.HostArray(
                self.preCorrelationReorderOperation.template.inputDataShape, np.int16, context=self.context
            )
            # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
            chunk = recv.Chunk(buf)
            # 6.3 Give the chunk to the receiver - once this is done we no longer need to track the chunk object.
            self.receiverStream.add_chunk(chunk)

    def add_udp_ibv_transport(self):
        """TODO: Write this."""
        self.receiverStream.add_udp_ibv_reader([("239.10.10.10", 7149)], "10.100.44.1", 10000000, 0)
        print("Added reader")

    async def _receiver_loop(self):
        """TODO: Write this."""
        print("Receiver Loop Start")
        asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
            self.receiverStream.ringbuffer, self.monitor, "recv_ringbuffer", "get_chunks"
        )
        recieved_chunks = 0
        i = 0
        received = 0
        dropped = 0
        async for chunk in asyncRingbuffer:
            received += len(chunk.present)
            dropped += len(chunk.present) - sum(chunk.present)
            print(
                f"Chunk: {i:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}. {len(chunk.base)}"
            )
            i += 1
            # 8.3 Once we are done with the chunk, give it back to the receiver so that the receiver has access to more
            # chunks without having to allocate more memory.

            item = await self._rx_free_item_queue.get()
            item.timestamp += chunk.timestamp
            item.chunk = chunk

            item.buffer_device.set_async(self._upload_command_queue, chunk.base)
            item.add_event(self._upload_command_queue.enqueue_marker())
            await self._rx_item_queue.put(item)  # Dont think it needs to be async

            recieved_chunks += 1
            if recieved_chunks == 30:
                self.rx_running = False
                self.receiverStream.stop()

        # async for chunk in asyncRingbuffer:
        #     recieved_chunks+=1
        #     print(f"1. Timestamp: {item.timestamp}")
        #     # host_buffer = katsdpsigproc.accel.HostArray(
        #     #     self.preCorrelationReorderOperation.template.inputDataShape,
        #     #     np.int16,
        #     #     self.preCorrelationReorderOperation.template.inputDataShape,
        #     #     context=self.context,
        #     # )  # not necessary
        #     # initial_val = self.heaps_per_fengine_per_chunk * i + 1
        #     # for j in range(self.heaps_per_fengine_per_chunk):
        #     #     host_buffer[j] = initial_val + j
        #     import time
        #     #print(time.time(), chunk.base[0][0][0][0][0], "...", chunk.base[-1][-1][-1][-1][-1], chunk.base.dtype, self.receiverStream.chunk_bytes, chunk.base.shape)
        #     #print(chunk.base[0][0][0][0][0], chunk.base[0][0][0][0][1], chunk.base[0][0][1][0][1], chunk.base[5][-1][-1][-1][0])
        #     item.buffer_device.set_async(self._upload_command_queue, chunk.base)
        #     item.add_event(self._upload_command_queue.enqueue_marker())
        #     await self._rx_item_queue.put(item)  # Dont think it needs to be async

        # if(recieved_chunks == 4):
        #     self.rx_running = False
        #     break

    async def _gpu_proc_loop(self):
        """
        TODO: Write this.

        Make this work correctly - currently it does not sync on timestamps and specific numbers of accumulations.
        """
        print("GPU Proc Loop Start")
        while self.rx_running is True:
            # 1. Get all items
            rx_item = await self._rx_item_queue.get()
            tx_item = await self._tx_free_item_queue.get()
            await rx_item.async_wait_for_events()
            self.receiverStream.add_chunk(rx_item.chunk)

            # 2. Process data in items
            self.tensorCoreXEngineCoreOperation.bind(outVisibilities=tx_item.buffer_device)
            self.preCorrelationReorderOperation.bind(inSamples=rx_item.buffer_device)
            self.preCorrelationReorderOperation()

            # TODO: Make this less clunky
            # TODO: Add auto resync logic
            self.tensorCoreXEngineCoreOperation.zero_visibilities()
            for i in range(self.heaps_per_fengine_per_chunk):
                buffer_slice = katsdpsigproc.accel.DeviceArray(
                    self.context,
                    self.tensorCoreXEngineCoreOperation.template.inputDataShape,
                    np.int16,
                    raw=self.reordered_buffer_device.buffer.ptr + self.rx_bytes_per_heap * i,
                )
                self.tensorCoreXEngineCoreOperation.bind(inSamples=buffer_slice)
                self.tensorCoreXEngineCoreOperation()

            # 3 Pass all items on to sender
            tx_item.add_event(self._proc_command_queue.enqueue_marker())
            tx_item.timestamp = tx_item.timestamp + rx_item.timestamp + 1
            print(f"2. Timestamp: {tx_item.timestamp}")
            await self._tx_item_queue.put(tx_item)
            rx_item.reset()
            await self._rx_free_item_queue.put(rx_item)
        print("Done running GPU Proc")

    async def _sender_loop(self):
        """TODO: Write this."""
        print("Sender Loop Start")
        while self.rx_running is True:
            # 1. Get the item to transfer
            item = await self._tx_item_queue.get()
            await item.async_wait_for_events()

            # 2. Get a free heap buffer to copy the GPU data to
            buffer_wrapper = await self.sendStream.get_free_heap()

            # 3. Transfer GPU buffer in item to free buffer.
            item.buffer_device.get_async(self._download_command_queue, buffer_wrapper.buffer)
            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            # 4. Tell sender to transmit heap buffer on network.
            self.sendStream.send_heap(item.timestamp, buffer_wrapper)

            print(f"3. Timestamp: {item.timestamp + 1}")
            # print(
            #     np.sum(buffer_wrapper.buffer),
            #     hex(buffer_wrapper.buffer[0][0][0][0]),
            #     buffer_wrapper.buffer[-1][-1][-1][-1],
            # )
            # print()

            # 5. Reset item and put it back on the the _tx_free_item_queue
            item.reset()
            await self._tx_free_item_queue.put(item)
        print("Done running sender")

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
