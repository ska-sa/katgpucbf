"""
TODO: Write this.

TODO:
1. Autoresync logic - proper accumulation time and dump rate
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

        self.adc_sample_rate = 1712e6
        self.heap_accumulation_threshold = 52  # 256
        self.n_ants = 64
        self.n_channels_total = 32768
        self.n_channels_per_stream = self.n_channels_total // 256
        self.n_samples_per_channel = 256
        self.n_pols = 2
        self.sample_bits = 8
        complexity = 2

        ############################

        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        n_rx_items = 5  # To high means to much GPU memory gets allocated
        n_tx_items = 3

        self.rx_transport_added = False
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
        self.tx_thread_affinity = 2

        # Care needs to be taken when setting this value - document why
        self.timestamp_increment_per_accumulation = self.heap_accumulation_threshold * self.rx_heap_timestamp_step
        self.dump_rate_s = self.timestamp_increment_per_accumulation / self.adc_sample_rate

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

        for i in range(20):
            # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
            buf = katsdpsigproc.accel.HostArray(
                self.preCorrelationReorderOperation.template.inputDataShape, np.int16, context=self.context
            )
            # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
            chunk = recv.Chunk(buf)
            # 6.3 Give the chunk to the receiver - once this is done we no longer need to track the chunk object.
            self.receiverStream.add_chunk(chunk)

    def add_udp_ibv_receiver_transport(self):
        """TODO: Write this - add command line arguments."""
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_udp_ibv_reader([("239.10.10.10", 7149)], "10.100.44.1", 10000000, 2)

    def add_buffer_receiver_transport(self, buffer: bytes):
        """TODO: Write this."""
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_buffer_reader(buffer)

    def add_pcap_receiver_transport(self, pcap_file_name: str):
        """TODO: Write this."""
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_udp_pcap_file_reader(pcap_file_name)

    async def _receiver_loop(self):
        """TODO: Write this."""
        print("Receiver Loop Start")
        asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
            self.receiverStream.ringbuffer, self.monitor, "recv_ringbuffer", "get_chunks"
        )
        recieved_chunks = 0
        chunk_index = 0
        received = 0
        dropped = 0

        async for chunk in asyncRingbuffer:
            received += len(chunk.present)
            dropped += len(chunk.present) - sum(chunk.present)
            # Rework this
            # print(
            #   f"f{hex(chunk.timestamp)} Chunk: {chunk_index:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}."
            # )
            chunk_index += 1
            # 8.3 Once we are done with the chunk, give it back to the receiver so that the receiver has access to more
            # chunks without having to allocate more memory.

            item = await self._rx_free_item_queue.get()
            item.timestamp += chunk.timestamp
            item.chunk = chunk

            item.buffer_device.set_async(self._upload_command_queue, chunk.base)
            item.add_event(self._upload_command_queue.enqueue_marker())
            await self._rx_item_queue.put(item)  # Dont think it needs to be async

            recieved_chunks += 1
            # if recieved_chunks == 30:
            #     self.rx_running = False
            #     self.receiverStream.stop()

    async def _gpu_proc_loop(self):
        """
        TODO: Write this.

        Make this work correctly - currently it does not sync on timestamps and specific numbers of accumulations.
        """
        print("GPU Proc Loop Start")

        # import time

        # old_time = time.time()
        # old_ts = 0

        tx_item = await self._tx_free_item_queue.get()
        # The very first heap sent out the X-Engine will have a timestamp of zero, every other heap will have the
        # correct timestamp
        tx_item.timestamp = 0
        self.tensorCoreXEngineCoreOperation.bind(outVisibilities=tx_item.buffer_device)
        self.tensorCoreXEngineCoreOperation.zero_visibilities()

        while self.rx_running is True:
            # 1. Get all items
            rx_item = await self._rx_item_queue.get()
            await rx_item.async_wait_for_events()
            current_timestamp = rx_item.timestamp
            self.receiverStream.add_chunk(rx_item.chunk)

            # 2. Process data in items
            self.preCorrelationReorderOperation.bind(inSamples=rx_item.buffer_device)
            self.preCorrelationReorderOperation()

            # TODO: Make this less clunky

            for i in range(self.heaps_per_fengine_per_chunk):
                buffer_slice = katsdpsigproc.accel.DeviceArray(
                    self.context,
                    self.tensorCoreXEngineCoreOperation.template.inputDataShape,
                    np.int16,
                    raw=self.reordered_buffer_device.buffer.ptr + self.rx_bytes_per_heap * i,
                )
                self.tensorCoreXEngineCoreOperation.bind(inSamples=buffer_slice)
                self.tensorCoreXEngineCoreOperation()

                next_heap_timestamp = current_timestamp + self.rx_heap_timestamp_step
                if next_heap_timestamp % self.timestamp_increment_per_accumulation == 0:

                    # new_time = time.time()
                    # print(
                    #     round(new_time - old_time, 2),
                    #     hex(tx_item.timestamp),
                    #     hex(tx_item.timestamp - old_ts),
                    #     self.dump_rate_s,
                    #     self.rx_running,
                    # )
                    # old_time = new_time
                    # old_ts = tx_item.timestamp

                    tx_item.add_event(self._proc_command_queue.enqueue_marker())
                    await self._tx_item_queue.put(tx_item)

                    tx_item = await self._tx_free_item_queue.get()
                    tx_item.timestamp = next_heap_timestamp
                    self.tensorCoreXEngineCoreOperation.bind(outVisibilities=tx_item.buffer_device)
                    self.tensorCoreXEngineCoreOperation.zero_visibilities()

                current_timestamp += self.rx_heap_timestamp_step

            rx_item.reset()
            await self._rx_free_item_queue.put(rx_item)

        # Used for clossing off the queue
        await self._tx_item_queue.put(tx_item)

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

            # 5. Reset item and put it back on the the _tx_free_item_queue
            item.reset()
            await self._tx_free_item_queue.put(item)
        print("Done running sender")

    async def _send_descriptors_loop(self):
        """TODO: Write this."""
        while self.rx_running is True:
            self.sendStream.send_descriptor_heap()
            await asyncio.sleep(5)

    async def run(self):
        """TODO: Write this."""
        if self.rx_transport_added is not False:
            raise AttributeError("Transport for receiving data has not yet been set.")

        # NOTE: Put in todo about upgrading this to python 3.8
        loop = asyncio.get_event_loop()

        # Check here that everythin is inited properly

        receiver_task = loop.create_task(self._receiver_loop())
        gpu_proc_task = loop.create_task(self._gpu_proc_loop())
        sender_task = loop.create_task(self._sender_loop())
        descriptor_task = loop.create_task(self._send_descriptors_loop())
        await receiver_task
        await gpu_proc_task
        await sender_task
        await descriptor_task
