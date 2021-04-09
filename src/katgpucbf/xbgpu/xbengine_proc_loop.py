"""
TODO: Write this.

TODO:
Close _receiver_loop properly - set running equal to false, if data is not being received, receiver loop will not stop
Talk about accumulation epochs
Define a batch
Decide what to do with file monitor
B-Engine
"""

# General Imports
import time
import asyncio
import numpy as np
from typing import List

# SARAO Developed Package Imports
import spead2
import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource

# Internal katxgpu Package Imports
import katxgpu.monitor
import katxgpu.tensorcore_xengine_core
import katxgpu.precorrelation_reorder
import katxgpu._katxgpu.recv as recv
import katxgpu.xsend
import katxgpu.ringbuffer


class QueueItem:
    """
    A storage container to facilitates communication between different functions in the XBEngineProcessingLoop object.

    This queue item contains a buffer of preallocated GPU memory. This memory is reused many times in the processing
    loops to prevent unecessary allocations. The item also contains a list of events. Before accessing the data in the
    buffer, the user needs to ensure that the events have all been completed.
    """

    timestamp: int
    events: List[katsdpsigproc.abc.AbstractEvent]
    buffer_device: katsdpsigproc.accel.DeviceArray

    def __init__(self, timestamp: int = 0) -> None:
        """Initialise the queue item."""
        self.reset(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp and events in the QueueItem object."""
        self.timestamp = timestamp
        self.events = []
        # Need to reset chunk

    def add_event(self, event: katsdpsigproc.abc.AbstractEvent):
        """Add an event to the list of events in the QueueItem."""
        self.events.append(event)

    async def async_wait_for_events(self):
        """Wait for all events on the list of events to be comlete."""
        await katsdpsigproc.resource.async_wait_for_events(self.events)


class RxQueueItem(QueueItem):
    """
    Extension of the QueueItem to also store a chunk reference.

    The RxQueueItem between the sender and the gpu proc loops need to also store a reference to the chunk that data in
    the GPU buffer was copied from. This allows the gpu proc loop to hand the chunk back to the receiver once the copy
    is complete to reuse resources.
    """

    chunk: katxgpu._katxgpu.recv.Chunk

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp, events and chunk in the QueueItem object."""
        super().reset(timestamp=timestamp)
        self.chunk = None


class XBEngineProcessingLoop:
    """
    Class that creates an entire GPU XB-Engine pipeline.

    Currently the B-Engine functionality has not been added. This class currently only creates an X-Engine pipeline.

    This pipeline encompasses receiving SPEAD heaps from F-Engine. Sending them to the GPU for processing and then
    sending them back out on the network.

    The X-Engine processing is performed across three different async_methods. Data is passed between these items
    using asyncio.Queues. The three processing loops are as follows:
    1. _receiver_loop
    2. _gpu_proc_loop
    3. _sender_loop
    There is also a seperate loop for sending descriptors onto the network.

    This class allows for different types of transports to be used for the sender and receiver code. These transports
    allow for in process unit tests to be created that do not require access to the network.
    """

    # 1. Array Configuration Parameters - Parameters used to configure the entire array
    adc_sample_rate_Hz: int
    heap_accumulation_threshold: int  # Specify a number of heaps to accumulate per accumulation epoch.
    n_ants: int
    n_channels_total: int
    n_channels_per_stream: int
    n_samples_per_channel: int
    n_pols: int
    sample_bits: int

    # 2. Derived Parameters - Parameters specific to the X-Engine derived from the array configuration parameters
    rx_heap_timestamp_step: int  # Change in timestamp between consecutive received heaps.
    timestamp_increment_per_accumulation: int
    rx_bytes_per_heap_batch: int  # Number of bytes in a batch of received heaps with a specific timestamp.

    # 3. Engine Parameters - Parameters not used in the array but needed for this engine
    batches_per_chunk: int  # Sets the number of batches of heaps to store per chunk.
    channel_offset_value: int  # Used in the heap to indicate the first channel in the sequence of channels in the stream

    # 4. Flags used at some point in the program
    rx_transport_added: bool  # False if no rx transport has been added, true otherwise
    tx_transport_added: bool  # False if no tx transport has been added, true otherwise
    running: bool  # Remains true until the process must close - then set to false and all the loops will stop

    # 5. Monitor for tracking the number of chunks queued in the receiver and items in the queues
    monitor: katxgpu.monitor.Monitor

    # 6. Queues for passing items between different asyncio functions. The _rx_item_queue passes items from the
    # _receiver_loop function to the _gpu_proc_loop function and the _tx_item_queue passes items from the
    # _gpu_proc_loop to the _sender_loop function. Once an item has been used, it will be passed back on the
    # corresponding _free_item_queue to ensure that all allocated buffers go back into circulation.
    _rx_item_queue: "asyncio.Queue[RxQueueItem]"
    _rx_free_item_queue: "asyncio.Queue[RxQueueItem]"
    _tx_item_queue: "asyncio.Queue[QueueItem]"
    _tx_free_item_queue: "asyncio.Queue[QueueItem]"

    # 7. Objects for sending and receiving data
    ringbuffer: recv.Ringbuffer  # Ringbuffer passed to stream where all completed chunks wait.
    receiverStream: recv.Stream
    sendStream: katxgpu.xsend.XEngineSPEADAbstractSend

    # 8. GPU Kernels and GPU Context
    context: katsdpsigproc.abc.AbstractContext  # Implements either a CUDA or OpenCL context.
    tensorCoreXEngineCoreOperation: katxgpu.tensorcore_xengine_core.TensorCoreXEngineCore
    preCorrelationReorderOperation: katxgpu.precorrelation_reorder.PreCorrelationReorder
    reordered_buffer_device: katsdpsigproc.accel.DeviceArray  # Buffer linking reorder kernel to correlation kernel

    # 9. Command queues for syncing different operations on the GPU - a command queue is the OpenCL name for a CUDA
    # stream. An abstract command queue can either be implemented as an OpenCL command queue or a Cuda stream depending
    # on the context.
    _upload_command_queue: katsdpsigproc.abc.AbstractCommandQueue
    _proc_command_queue: katsdpsigproc.abc.AbstractCommandQueue
    _download_command_queue: katsdpsigproc.abc.AbstractCommandQueue

    def __init__(
        self,
        adc_sample_rate_Hz: int,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_stream: int,
        n_samples_per_channel: int,
        n_pols: int,
        sample_bits: int,
        heap_accumulation_threshold: int,
        channel_offset_value: int,
        rx_thread_affinity: int,
        batches_per_chunk: int,  # Used for GPU memory tuning
    ):
        """
        Construct an XBEngineProcessingLoop object.

        This constructor allocates all memory buffers to be used in the lifetime of the project.

        It does not specify the transports to be used. These need to be specified by the add_*_receiver_transport() and
        the add_*_sender_transport() functions provided in this class.

        Parameters
        ----------
        adc_sample_rate_Hz: int
            Sample rate of the digitisers in the current array. This value is required to calculate the packet spacing
            of the output heaps. If it is set incorrectly, the packet spacing could be too large causing the pipeline to
            stall as heaps queue at the sender faster than they are sent.
        n_ants: int
            The number of antennas to be correlated.
        n_channels_total: int
            The total number of frequency channels out of the F-Engine.
        n_channels_per_stream: int
            The number of frequency channels contained per stream.
        n_pols: int
            The number of pols per antenna. Expected to always be 2.
        n_samples_per_channel: int
            The number of time samples received per frequency channel.
        sample_bits: int
            The number of bits per sample. Only 8 bits is supported at the moment.
        heap_accumulation_threshold: int
            The number of consecutive heaps to accumulate. Used to determine the sync epoch.
        channel_offset_value: int
            Fixed value to be included in the SPEAD heap indicating the lowest channel value transmitted by this heap.
            Must be a multiple of n_channels_per_stream.
        rx_thread_affinity: int
            Specifc CPU core to assign the RX stream processing thread to.
        batches_per_chunk: int
            A batch is a collection of heaps from different antennas with the same timestamp. This parameter specifies
            the number of consecutive batches to store in the same chunk. The higher this value is, the more GPU and
            system RAM is allocated, the lower this value is, the more work the python processing thread is requried to
            do.
        """
        # 1. Assign configuration variables.
        # 1.1 Ensure that constructor arguments conform to
        if n_pols != 2:
            raise ValueError("n_pols must equal 2 - no other values supported at the moment.")

        if sample_bits != 8:
            raise ValueError("sample_bits must equal 2 - no other values supported at the moment.")

        if channel_offset_value % n_channels_per_stream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_stream")

        # 1.2 Assign array configuration variables
        self.adc_sample_rate_Hz = adc_sample_rate_Hz
        self.heap_accumulation_threshold = heap_accumulation_threshold
        self.n_ants = n_ants
        self.n_channels_total = n_channels_total
        self.n_channels_per_stream = n_channels_per_stream
        self.n_samples_per_channel = n_samples_per_channel
        self.n_pols = n_pols
        self.sample_bits = sample_bits
        complexity = 2

        # 1.3 Calculate derived parameters.
        # This step represents the difference in timestamp between two consecutive heaps received from the same F-Engine. We
        # multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.
        # While we can workout the timestamp_step from other parameters that configure the receiver, we pass it as a seperate
        # argument to the reciever for cases where the n_channels_per_stream changes across streams (likely for non-power-of-
        # two array sizes).
        self.rx_heap_timestamp_step = self.n_channels_total * 2 * self.n_samples_per_channel
        # This is the number of bytes for a single batch of F-Engines. A chunk consists of multiple batches.
        self.rx_bytes_per_heap_batch = (
            self.n_ants * self.n_channels_per_stream * self.n_samples_per_channel * self.n_pols * complexity
        )
        # This is how much the timestamp increments by between succesive accumulation epochs
        self.timestamp_increment_per_accumulation = self.heap_accumulation_threshold * self.rx_heap_timestamp_step

        # 1.4 Assign engine configuration parameters
        self.batches_per_chunk = batches_per_chunk
        self.channel_offset_value = channel_offset_value

        # 1.5 Set runtime flags to their initial states
        self.tx_transport_added = False
        self.rx_transport_added = False
        self.running = True

        # 2. Set up file monitor for tracking the state of the reciever chunks and the queues. This monitor is hardcoded
        # to not write data to a file. If debugging of the queues is needed, setting the use_file_monitor to true should
        # aid in this debugging.
        # TODO: Decide how to configure and manage the monitor.
        use_file_monitor = False
        if use_file_monitor:
            self.monitor = katxgpu.monitor.FileMonitor("temp_file.log")
        else:
            self.monitor = katxgpu.monitor.NullMonitor()

        # 3. Create the receiverStream object. This object has no attached transport yet and will not function until
        # one of the add_*_receiver_transport() functions has been called.
        ringbuffer_capacity = 15
        self.ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
        self.receiverStream = recv.Stream(
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_pols=self.n_pols,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.batches_per_chunk,
            ringbuffer=self.ringbuffer,
            thread_affinity=rx_thread_affinity,
            use_gdrcopy=False,
            monitor=self.monitor,
        )

        # 4. Create GPU specific objects.
        # 4.1 Create a GPU context, the x.is_cuda flag forces CUDA to be used instead of OpenCL.
        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        # 4.2 Create various command queues (or CUDA streams) to queue GPU functions on.
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        # 4.3 Create reorder and correlation operations and create buffer linking the two operations.
        tensorCoreTemplate = katxgpu.tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
        )
        self.tensorCoreXEngineCoreOperation = tensorCoreTemplate.instantiate(self._proc_command_queue)

        reorderTemplate = katxgpu.precorrelation_reorder.PreCorrelationReorderTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_batches=self.batches_per_chunk,
        )
        self.preCorrelationReorderOperation = reorderTemplate.instantiate(self._proc_command_queue)

        self.reordered_buffer_device = katsdpsigproc.accel.DeviceArray(
            self.context, self.preCorrelationReorderOperation.template.outputDataShape, np.int16
        )
        self.preCorrelationReorderOperation.bind(outReordered=self.reordered_buffer_device)

        # 5. Create various buffers and assign them to the correct queues or objects.
        # 5.1 Define the number of items on each of these queues. The n_rx_items and n_tx_items each wrap a GPU buffer.
        # setting these values too high results in too much GPU memory being consumed. There just need to be enough
        # of them that the different processing loops do not get starved waiting for items. The low single digits is
        # suitable. n_free_chunks wraps buffer in system ram. This can be set quite high as there is much more system
        # RAM than GPU RAM.
        # These values are not configurable as they have been acceptable for most tests cases up until now. If the
        # pipeline starts bottlenecking, then maybe look at increasing these values.
        n_rx_items = 3  # To high means to much GPU memory gets allocated
        n_tx_items = 2
        n_free_chunks = 20

        # 5.2 Create various queues for communication between async funtions. These queues are extended in the monitor
        # class, allowing for the monitor to track the number of items on each queue.
        self._rx_item_queue = self.monitor.make_queue("rx_item_queue", n_rx_items)  # type: asyncio.Queue[RxQueueItem]
        self._rx_free_item_queue = self.monitor.make_queue("rx_free_item_queue", n_rx_items)
        self._tx_item_queue = self.monitor.make_queue("tx_item_queue", n_tx_items)
        self._tx_free_item_queue = self.monitor.make_queue("tx_free_item_queue", n_tx_items)

        # 5.3 Create buffers and assign them correctly.
        # 5.3.1 Create items that will store received chunks that have been transferred to the GPU.
        for i in range(n_rx_items):
            rx_item = RxQueueItem()
            rx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context, self.preCorrelationReorderOperation.template.inputDataShape, np.int16
            )
            self._rx_free_item_queue.put_nowait(rx_item)

        # 5.3.2 Create items that will store correlated data in GPU memory, ready for transferring back to system RAM.
        for i in range(n_tx_items):
            tx_item = QueueItem()
            tx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context, self.tensorCoreXEngineCoreOperation.template.outputDataShape, np.int64
            )
            self._tx_free_item_queue.put_nowait(tx_item)

        # 5.3.1 Create empty chunks and give them to the receiver to use to assemble heaps.
        for i in range(n_free_chunks):
            buf = katsdpsigproc.accel.HostArray(
                self.preCorrelationReorderOperation.template.inputDataShape, np.int16, context=self.context
            )
            chunk = recv.Chunk(buf)
            self.receiverStream.add_chunk(chunk)

    def add_udp_ibv_receiver_transport(self, src_ip: str, src_port: int, interface_ip: str, comp_vector_affinity: int):
        """
        Add the ibv_udp transport to the receiver.

        The receiver will read udp packets off of the specified interface using the ibverbs library to offload
        processing from the CPU.

        This transport is intended to be the transport used in production.

        Parameters
        ----------
        src_ip: str
            multicast IP address of source data.
        src_port: int
            Port of source data
        interface_ip: str
            IP address of interface to listen for data on.
        comp_vector_affinity: int
            Received packets will generate interrupts from the NIC. These interrupts can be assigned to a specific CPU
            core. This parameters determines which core to assign these interrupts to.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_udp_ibv_reader(
            [(src_ip, src_port)], interface_ip, buffer_size=10000000, comp_vector=comp_vector_affinity
        )

    def add_buffer_receiver_transport(self, buffer: bytes):
        """
        Add the buffer transport to the receiver.

        The receiver will read packet data python ByteArray generated by a spead2.send.BytesStream object. The sender
        does not support the inproc transport and as such the buffer transport must be used instead.

        This transport is intended to be used for testing purposes.

        Parameters
        ----------
        buffer: bytes
            Buffer containing simulated packet data.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_buffer_reader(buffer)

    def add_pcap_receiver_transport(self, pcap_file_name: str):
        """
        Add the pcap transport to the receiver. The receiver will read packet data from a pcap file.

        This transport is intended to be used for testing purposes.

        Parameters
        ----------
        filename: string
            Name of PCAP file to open.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiverStream.add_udp_pcap_file_reader(pcap_file_name)

    def add_udp_ibv_sender_transport(self, dest_ip: str, dest_port: int, interface_ip: str, thread_affinity: int):
        """
        Add the ibv_udp transport to the sender.

        The sender will transmit udp packets out of the specified interface using the ibverbs library to offload
        processing from the CPU.

        This transport is intended to be the transport used in production.

        Parameters
        ----------
        dest_ip: str
            multicast IP address of destination data
        dest_port: int
            Port of transmitted data
        interface_ip: str
            IP address of interface to trasnmit data on.
        thread_affinity: int
            The receiver creates its own thread to run in the background transmitting data. It is bound to the CPU
            core specified here.
        """
        if self.tx_transport_added is True:
            raise AttributeError("Transport for sending data has already been set.")
        self.tx_transport_added = True

        # This value staggers the send so that packets within a heap are transmitted onto the network across the entire
        # time between dumps. intervaleCare needs to be taken to ensure that this rate is not set too high. If it is
        # too high, the entire pipeline will stall needlessly waiting for packets to be transmitted too slowly.
        dump_rate_s = self.timestamp_increment_per_accumulation / self.adc_sample_rate_Hz

        self.sendStream = katxgpu.xsend.XEngineSPEADIbvSend(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_pols=self.n_pols,
            dump_rate_s=dump_rate_s,
            channel_offset=self.channel_offset_value,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            endpoint=(dest_ip, dest_port),
            interface_address=interface_ip,
            thread_affinity=thread_affinity,
        )

    def add_inproc_sender_transport(self, queue: spead2.InprocQueue):
        """
        Add the inproc transport to the sender. The sender will send heaps out on an InprocQueue.

        This transport is intended to be used for testing purposes.

        Parameters
        ----------
        queue: spead2.InprocQueue
            SPEAD2 inproc queue to send heaps to.
        """
        if self.tx_transport_added is True:
            raise AttributeError("Transport for sending data has already been set.")
        self.tx_transport_added = True
        # For the inproc transport this value is set very low as the dump rate does affect performanc for an inproc
        # queue and a high dump rate just makes the unit tests take very long to run.
        dump_rate_s = 0.05

        self.sendStream = katxgpu.xsend.XEngineSPEADInprocSend(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_pols=self.n_pols,
            dump_rate_s=dump_rate_s,
            channel_offset=self.channel_offset_value,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            queue=queue,
        )

    async def _receiver_loop(self):
        """
        Receive heaps off of the network in a continuous loop.

        This function does the following:
        1. Wait for a chunk to be assembled on the receiver.
        2. Get a free rx item off of the _rx_free_item_queue.
        3. Initiate the transfer of the chunk from system memory to the buffer in GPU RAM that belongs to the rx_item.
        4. Place the rx_item on _rx_item_queue so that it can be processed downstream.

        The above steps are performed in a loop until the running flag is set to false.

        TODO: If no data is being streamed and the running flag is set to false, this loop will be stuck waiting for
        the next chunk. Try find a way to exit cleanly without adding to much additional logic to this function.
        """
        # 1. Set up initial conditions
        asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
            self.receiverStream.ringbuffer, self.monitor, "recv_ringbuffer", "get_chunks"
        )
        chunk_index = 0
        received_total = 0
        dropped_total = 0

        # 2. Get complete chunks from the ringbuffer.
        async for chunk in asyncRingbuffer:
            # 2.1 Update metrics and log warning if dropped heap is detected within the chunk.
            received_heaps = len(chunk.present)
            dropped_heaps = len(chunk.present) - sum(chunk.present)
            received_total += received_heaps
            dropped_total += dropped_heaps

            # TODO: This must become a proper logging message
            if dropped_heaps != 0:
                print(
                    f"LOG WARNING: Chunk: {chunk_index:>5} Timestamp: {hex(chunk.timestamp)} Received: {sum(chunk.present):>4} of {received_heaps:>4} expected heaps. All time dropped/received heaps: {dropped_total}/{received_total}."
                )

            chunk_index += 1

            # 2.2. Get a free rx_item that will contain the GPU buffer to transfer the received chunk to.
            item = await self._rx_free_item_queue.get()
            item.timestamp += chunk.timestamp
            item.chunk = chunk

            # 2.3. Initiate transfer from recived chunk to rx_item buffer.
            item.buffer_device.set_async(self._upload_command_queue, chunk.base)
            item.add_event(self._upload_command_queue.enqueue_marker())

            # 2.4. Give the rx item to the _gpu_proc_loop.
            await self._rx_item_queue.put(item)  # Dont think it needs to be async

            # 3. If the loop must close, stop the stream.
            if self.running is not True:
                self.receiverStream.stop()

    async def _gpu_proc_loop(self):
        """
        Perform all GPU processing of received data in a continuous loop.

        This function performs the following steps:
        1. Retrieve an rx_item from the _rx_item_queue
        2. Performs the reorder operation on the buffer in the rx item. This gets the buffer data into a format that
        the correlation kernel requires.
        3.1 Apply the correlation kernel to small subsets of the reordered data until all the data has been processed.
        3.2 If sufficient  correlations have occured, transfer the correlated data to a tx_item, pass the tx_item to
        the _tx_item_queue and get a new item from the _tx_free_item_queue.

        The ratio of rx_items to tx_items is not one to one. There are expected to be many more rx_items in for every
        tx_item out.

        The above steps are performed in a loop until the running flag is set to false.

        TODO: Add B-Engine processing in this loop.
        """
        # 1. Set up initial conditions
        old_time = time.time()
        old_ts = 0
        tx_item = await self._tx_free_item_queue.get()
        # The very first heap sent out the X-Engine will have a timestamp of zero which is meaningless, every other
        # heap will have the correct timestamp.
        tx_item.timestamp = 0
        self.tensorCoreXEngineCoreOperation.bind(outVisibilities=tx_item.buffer_device)
        self.tensorCoreXEngineCoreOperation.zero_visibilities()

        while self.running is True:
            # 2. Get item from receiver loop - wait for the HtoD transfers to complete and then give the chunk back to
            # the receiver for reuse.
            rx_item = await self._rx_item_queue.get()
            await rx_item.async_wait_for_events()
            current_timestamp = rx_item.timestamp

            # 2.1 Give the chunk back to the receiver stream - if this is not done, eventually no more data will be
            # received as there will be no available chunks to store it in.
            self.receiverStream.add_chunk(rx_item.chunk)

            # 3. Process the received data.
            # 3.1 Reorder the entire chunk
            self.preCorrelationReorderOperation.bind(inSamples=rx_item.buffer_device)
            self.preCorrelationReorderOperation()

            # 3.2 Perform correlation on reordered data. The correlation kernel does not have the
            # concept of a batch at this stage, so the kernel needs to be run on each different
            # batch in the chunk.
            for i in range(self.batches_per_chunk):
                # 3.2.1 Slice the buffer of reordered data to only select a specific batch. Then run the kernel on this
                # buffer.
                buffer_slice = katsdpsigproc.accel.DeviceArray(
                    self.context,
                    self.tensorCoreXEngineCoreOperation.template.inputDataShape,
                    np.int16,
                    raw=self.reordered_buffer_device.buffer.ptr + self.rx_bytes_per_heap_batch * i,
                )
                self.tensorCoreXEngineCoreOperation.bind(inSamples=buffer_slice)
                self.tensorCoreXEngineCoreOperation()

                # 3.2.2 If the batch timestamp corresponds to the accumulation epoch interval, transfer the correlated
                # data to the sender function. NOTE: The timestamp representing the end of an epoch does not necessarily
                # line up with the chunk timestamp. It will line up with a specific batch within a chunk though, this is
                # why this check has to happen for each batch.
                # This check is the equivilant of the MeerKAT SKARAB X-Engine auto-resync logic.
                next_heap_timestamp = current_timestamp + self.rx_heap_timestamp_step
                if next_heap_timestamp % self.timestamp_increment_per_accumulation == 0:

                    # 3.2.3 Perform some basic logging - these prints will need to be turned into proper python logging
                    # statements
                    new_time = time.time()
                    # We do not expect the time between dumps to be the same each time as the time.time() function
                    # checks the wall time now, not the time between timestamps. The difference between dump timestamps
                    # is expected to be constant
                    # This print message must become a debug.INFO message
                    print(
                        f"LOG INFO: Current dump timestamp: {hex(tx_item.timestamp)}, difference between dump timestamps: {hex(tx_item.timestamp - old_ts)}, wall time between dumps {round(new_time - old_time, 2)}, "
                    )
                    # Not sure under which conditions that this would occur. Something funny would have to happen at the receiver.
                    # This check is here preemptivly - this issue has not been detected yet.
                    if tx_item.timestamp - old_ts != self.timestamp_increment_per_accumulation:
                        print(
                            f"LOG WARNING: Timestamp between heaps equal to {hex(tx_item.timestamp - old_ts)}, expected {self.timestamp_increment_per_accumulation}"
                        )
                    old_time = new_time
                    old_ts = tx_item.timestamp

                    # 3.2.4 Transfer the TX item to the sender loop
                    tx_item.add_event(self._proc_command_queue.enqueue_marker())
                    await self._tx_item_queue.put(tx_item)

                    # 3.2.5 Get a new tx item, assign its buffer correctly and reset the buffer to zero.
                    tx_item = await self._tx_free_item_queue.get()
                    tx_item.timestamp = next_heap_timestamp
                    self.tensorCoreXEngineCoreOperation.bind(outVisibilities=tx_item.buffer_device)
                    self.tensorCoreXEngineCoreOperation.zero_visibilities()

                # 4. Increment batch timestamp.
                current_timestamp += self.rx_heap_timestamp_step

            # 5. Finished with the RX item - reset it and give it back to the receiver loop.
            rx_item.reset()
            await self._rx_free_item_queue.put(rx_item)

        # 6. When the stream is closed, if the sender loop is waiting for a tx item, it will never exit. This funtion
        # puts the current tx_item on the queue. The sender_loop can then stop waiting upon receiving this and exit.
        await self._tx_item_queue.put(tx_item)

    async def _sender_loop(self):
        """
        Send heaps to the network in a continuous loop.

        This function does the following:
        1. Get an item from the _tx_item_queue.
        2. Wait for all the events on this item to complete.
        3. Wait for an available heap buffer to become available.
        4. Transfer the GPU buffer in the item to the heap buffer in system RAM.
        5. Wait for the transfer to complete.
        6. Transmit data in heap buffer out into the network.
        7. Place the tx_item on _tx_item_free_queue so that it can be reused.

        The above steps are performed in a loop until the running flag is set to false.

        NOTE: The transfer from the GPU to the heap buffer and the sending onto the network could be pipeline a bit
        better, but this is not really required in this loop as this whole process occurs at a much slower pace than
        the rest of the pipeline.
        """
        while self.running is True:
            # 1. Get the item to transfer and wait for all GPU events to finish before continuing
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

            # 5. Reset item and put it back on the the _tx_free_item_queue for resue
            item.reset()
            await self._tx_free_item_queue.put(item)

    async def _send_descriptors_loop(self):
        """Send the Baseline Correlation Products Hardware heaps out to the network every 5 seconds."""
        while self.running is True:
            self.sendStream.send_descriptor_heap()
            await asyncio.sleep(5)

    async def run(self):
        """
        Launch all the different async functions required to run the X-Engine.

        These functions will loop forever and only exit once an exit flag is set.
        """
        if self.rx_transport_added is not True:
            raise AttributeError("Transport for receiving data has not yet been set.")
        if self.tx_transport_added is not True:
            raise AttributeError("Transport for sending data has not yet been set.")

        # NOTE: Put in todo about upgrading this to python 3.8
        loop = asyncio.get_event_loop()
        receiver_task = loop.create_task(self._receiver_loop())
        gpu_proc_task = loop.create_task(self._gpu_proc_loop())
        sender_task = loop.create_task(self._sender_loop())
        descriptor_task = loop.create_task(self._send_descriptors_loop())
        await receiver_task
        await gpu_proc_task
        await sender_task
        await descriptor_task
