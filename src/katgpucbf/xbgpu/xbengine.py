"""
This module defines an XBEngine object that implements an entire GPU XB-Engine pipeline.

Additionally this module defines the QueueItem and RxQueueItem objects that are used in the XBEngine for
passing information between different async processing loops within the object.

.. todo::

    - Close _receiver_loop properly - The receiver loop can potentially hang when trying to close. See the function
      docstring for more information. At the moment, there is no clean way to close the pipeline. The stop() function
      attempts this but needs some work.
    - The B-Engine logic has not been implemented yet - this needs to be added eventually. It is expected that this
      logic will need to go in the _gpu_proc_loop for the B-Engine processing
      and then a seperate sender loop would need to be created for sending B-Engine data.
    - Implement monitoring and control - There is no mechanism to interact with or receive metrics from a running
      pipeline.
    - Catch asyncio exceptions - If one of the running asyncio loops has an exception, it will stop running without
      crashing the program or printing the error trace stack. This is not an issue when things are working, but if we
      could catch those exceptions and crash the program, it would make detecting and debugging heaps much simpler.
    - The asyncio syntax in the run() function uses old syntax, once this repo has been updated to python 3.8, update
      this to use the new asyncio syntax.

"""

import asyncio
import logging
import math
import time
from typing import List, Tuple, TypedDict

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource
import spead2
from aiokatcp import DeviceServer, Sensor, SensorSampler

import katgpucbf.xbgpu._katxbgpu.recv as recv
import katgpucbf.xbgpu.precorrelation_reorder
import katgpucbf.xbgpu.ringbuffer
import katgpucbf.xbgpu.tensorcore_xengine_core
import katgpucbf.xbgpu.xsend

from .. import __version__
from ..monitor import Monitor

logger = logging.getLogger(__name__)


class QueueItem:
    """
    Object to enable communication and synchronisation between different functions in the XBEngine object.

    This queue item contains a buffer of preallocated GPU memory. This memory is reused many times in the processing
    functions to prevent unecessary allocations. The item also contains a list of events. Before accessing the data in
    the buffer, the user needs to ensure that the events have all been completed.
    """

    timestamp: int
    events: List[katsdpsigproc.abc.AbstractEvent]
    buffer_device: katsdpsigproc.accel.DeviceArray

    def __init__(self, timestamp: int = 0) -> None:
        """Initialise the queue item."""
        self.reset(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp and events."""
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

    chunk: katgpucbf.xbgpu._katxbgpu.recv.Chunk

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp, events and chunk."""
        super().reset(timestamp=timestamp)
        self.chunk = None


class XBEngine(DeviceServer):
    r"""GPU XB-Engine pipeline.

    Currently the B-Engine functionality has not been added. This class currently
    only creates an X-Engine pipeline.

    This pipeline encompasses receiving SPEAD heaps from F-Engines, sending them
    to the GPU for processing and then sending them back out on the network.

    The X-Engine processing is performed across three different async_methods.
    Data is passed between these items using :class:`asyncio.Queue`\s. The three
    processing functions are as follows:

      1. :func:`_receiver_loop` - Receive chunks from network and initiate
         transfer to GPU.
      2. :func:`_gpu_proc_loop` - Reorder chunk in GPU memory and perform the
         correlation operation on this reordered data.
      3. :func:`_sender_loop` - Transfer correlated data to system RAM and then
         send it out on the network.

    There is also a seperate function for sending descriptors onto the network.

    Items passed between queues may still have GPU operations in progress. Each
    item stores a list of events that can be used to determine if a GPU
    operation is complete.

    In order to reduce the load on the maim thread, received data is collected
    into chunks. A chunk consists of multiple batches of F-Engine heaps where a
    batch is a collection of heaps from all F-Engine with the same timestamp.

    This class allows for different types of transports to be used for the
    sender and receiver code. These transports allow for in-process unit tests
    to be created that do not require access to the network.

    The initialiser allocates all memory buffers to be used during the lifetime
    of the XBEngine object. These buffers are continuously reused to ensure
    memory use remains constrained. It does not specify the transports to be
    used. These need to be specified by the ``add_*_receiver_transport()`` and
    the ``add_*_sender_transport()`` functions provided in this class.

    .. todo::

      A lot of the sensors are common to both the F- and X-engines. It may be
      worth investigating some kind of abstract base class for engines to build
      on top of.

    Parameters
    ----------
    katcp_host
        Hostname or IP on which to listen for KATCP C&M connections.
    katcp_port
        Network port on which to listen for KATCP C&M connections.
    adc_sample_rate_hz
        Sample rate of the digitisers in the current array. This value is required to calculate the packet spacing
        of the output heaps. If it is set incorrectly, the packet spacing could be too large causing the pipeline to
        stall as heaps queue at the sender faster than they are sent.
    n_ants
        The number of antennas to be correlated.
    n_channels_total
        The total number of frequency channels out of the F-Engine.
    n_channels_per_stream
        The number of frequency channels contained per stream.
    n_pols
        The number of pols per antenna. Expected to always be 2.
    n_samples_per_channel
        The number of time samples received per frequency channel.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    heap_accumulation_threshold
        The number of consecutive heaps to accumulate. This value is used to determine the dump rate.
    channel_offset_value
        The index of the first channel in the subset of channels processed by this XB-Engine. Used to set the value
        in the XB-Engine output heaps for spectrum reassembly by the downstream receiver.
    rx_thread_affinity
        Specific CPU core to assign the RX stream processing thread to.
    batches_per_chunk
        A batch is a collection of heaps from different antennas with the same timestamp. This parameter specifies
        the number of consecutive batches to store in the same chunk. The higher this value is, the more GPU and
        system RAM is allocated, the lower this value is, the more work the python processing thread is required to
        do.
    rx_reorder_tol
        Maximum tolerance for jitter between received packets, as a time
        expressed in ADC sample ticks.
    """

    VERSION = "katgpucbf-xbgpu-icd-0.1"
    BUILD_STATE = __version__

    def __init__(
        self,
        *,
        katcp_host: str,
        katcp_port: int,
        adc_sample_rate_hz: float,
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
        rx_reorder_tol: int,
        monitor: Monitor,
    ):
        super(XBEngine, self).__init__(katcp_host, katcp_port)

        # No more than once per second, at least once every 10 seconds.
        # The TypedDict is necessary for mypy to believe the use as
        # kwargs is valid.
        AutoStrategy = TypedDict(  # noqa: N806
            "AutoStrategy",
            {
                "auto_strategy": SensorSampler.Strategy,
                "auto_strategy_parameters": Tuple[float, float],
            },
        )
        auto_strategy = AutoStrategy(
            auto_strategy=SensorSampler.Strategy.EVENT_RATE,
            auto_strategy_parameters=(1.0, 10.0),
        )
        sensors: List[Sensor] = [
            Sensor(
                int,
                "input-heaps-total",
                "number of heaps received (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            Sensor(
                int,
                "input-chunks-total",
                "number of chunks received (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            Sensor(
                int,
                "input-bytes-total",
                "number of bytes of digitiser samples received (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            Sensor(
                int,
                "input-missing-heaps-total",
                "number of heaps dropped on the input (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                # TODO: Think about what status_func should do for the status of the
                # sensor. If it goes into "warning" as soon as a single packet is
                # dropped, then it may not be too useful. Having the information
                # necessary to implement this may involve shifting things between
                # classes.
                **auto_strategy,
            ),
            Sensor(
                int,
                "output-heaps-total",
                "number of heaps transmitted (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            Sensor(
                int,
                "output-bytes-total",
                "number of payload bytes transmitted (prometheus: counter)",
                default=0,
                initial_status=Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
        ]
        for sensor in sensors:
            self.sensors.add(sensor)

        # 1. List object variables and provide type hints - This has no function other than to improve readability.
        # 1.1 Array Configuration Parameters - Parameters used to configure the entire array
        self.adc_sample_rate_hz: float
        self.heap_accumulation_threshold: int  # Specify a number of heaps to accumulate per accumulation.
        self.n_ants: int
        self.n_channels_total: int
        self.n_channels_per_stream: int
        self.n_samples_per_channel: int
        self.n_pols: int
        self.sample_bits: int

        # 1.2 Derived Parameters - Parameters specific to the X-Engine derived from the array configuration parameters
        self.rx_heap_timestamp_step: int  # Change in timestamp between consecutive received heaps.
        self.timestamp_increment_per_accumulation: int  # Time difference between two consecutive accumulations.
        self.rx_bytes_per_heap_batch: int  # Number of bytes in a batch of received heaps with a specific timestamp.
        self.dump_interval_s: float  # Number of seconds between output heaps.

        # 1.3 Engine Parameters - Parameters not used in the array but needed for this engine
        self.batches_per_chunk: int  # Sets the number of batches of heaps to store per chunk.
        self.max_active_chunks: int
        # Used in the heap to indicate the first channel in the sequence of channels in the stream
        self.channel_offset_value: int

        # 1.4 Flags used at some point in this class.
        self.rx_transport_added: bool  # False if no rx transport has been added, true otherwise
        self.tx_transport_added: bool  # False if no tx transport has been added, true otherwise
        # Remains true until the user tells the process to stop - then set to
        # false and close the asyncio functions.
        self.running: bool

        # 1.5 Queues for passing items between different asyncio functions.
        # * The _rx_item_queue passes items from the _receiver_loop function to the _gpu_proc_loop function.
        # * The _tx_item_queue passes items from the _gpu_proc_loop to the _sender_loop function.
        # Once the destination function is finished with an item, it will pass it back to the corresponding
        # _(rx/tx)_free_item_queue to ensure that all allocated buffers are in continuous circulation.
        self._rx_item_queue: asyncio.Queue[RxQueueItem]
        self._rx_free_item_queue: asyncio.Queue[RxQueueItem]
        self._tx_item_queue: asyncio.Queue[QueueItem]
        self._tx_free_item_queue: asyncio.Queue[QueueItem]

        # 1.6 Objects for sending and receiving data
        self.ringbuffer: recv.Ringbuffer  # Ringbuffer passed to stream where all completed chunks wait.
        self.receiver_stream: recv.Stream

        # 1.7 Command queues for syncing different operations on the GPU - a
        # command queue is the OpenCL name for a CUDA stream. An abstract
        # command queue can either be implemented as an OpenCL command queue or
        # a CUDA stream depending on the context.
        self._upload_command_queue: katsdpsigproc.abc.AbstractCommandQueue
        self._proc_command_queue: katsdpsigproc.abc.AbstractCommandQueue
        self._download_command_queue: katsdpsigproc.abc.AbstractCommandQueue

        # 2. Assign configuration variables.
        # 2.1 Ensure that constructor arguments are within the expected range.
        if n_pols != 2:
            raise ValueError("n_pols must equal 2 - no other values supported at the moment.")

        if sample_bits != 8:
            raise ValueError("sample_bits must equal 8 - no other values supported at the moment.")

        if channel_offset_value % n_channels_per_stream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_stream")

        # 2.2 Assign array configuration variables
        self.adc_sample_rate_hz = adc_sample_rate_hz
        self.heap_accumulation_threshold = heap_accumulation_threshold
        self.n_ants = n_ants
        self.n_channels_total = n_channels_total
        self.n_channels_per_stream = n_channels_per_stream
        self.n_samples_per_channel = n_samples_per_channel
        self.n_pols = n_pols
        self.sample_bits = sample_bits
        complexity = 2  # Used to explicitly indicate when a complex number is being allocated.

        # 2.3 Calculate derived parameters.
        # This step represents the difference in timestamp between two
        # consecutive heaps received from the same F-Engine. We multiply step
        # by 2 to account for dropping half of the spectrum due to symmetric
        # properties of the Fourier Transform.  While we can workout the
        # timestamp_step from other parameters that configure the receiver, we
        # pass it as a seperate argument to the reciever for cases where the
        # n_channels_per_stream changes across streams (likely for
        # non-power-of- two array sizes).
        self.rx_heap_timestamp_step = self.n_channels_total * 2 * self.n_samples_per_channel
        # This is the number of bytes for a single batch of F-Engines. A chunk consists of multiple batches.
        self.rx_bytes_per_heap_batch = (
            self.n_ants * self.n_channels_per_stream * self.n_samples_per_channel * self.n_pols * complexity
        )
        # This is how much the timestamp increments by between successive accumulations
        self.timestamp_increment_per_accumulation = self.heap_accumulation_threshold * self.rx_heap_timestamp_step

        # 2.4 Assign engine configuration parameters
        self.batches_per_chunk = batches_per_chunk
        self.max_active_chunks = math.ceil(rx_reorder_tol / self.rx_heap_timestamp_step / self.batches_per_chunk) + 1
        self.channel_offset_value = channel_offset_value

        # 2.5 Set runtime flags to their initial states
        self.tx_transport_added = False
        self.rx_transport_added = False
        self.running = False

        # 3. Declare the Monitor for tracking the state of the reciever chunks and the queues.
        self.monitor = monitor

        # 4. Create the receiver_stream object. This object has no attached transport yet and will not function until
        # one of the add_*_receiver_transport() functions has been called.

        # Ringbuffer capacity is not a command line argument as it is not expected that the user will gain much value
        # by having control over this. It could just cause confusion. The developers should instead set this value once.
        ringbuffer_capacity = 15
        self.ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
        self.receiver_stream = recv.Stream(
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_pols=self.n_pols,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.batches_per_chunk,
            max_active_chunks=self.max_active_chunks,
            ringbuffer=self.ringbuffer,
            thread_affinity=rx_thread_affinity,
            use_gdrcopy=False,
            monitor=self.monitor,
        )

        # 5. Create GPU specific objects.
        # 5.1 Create a GPU context, the x.is_cuda flag forces CUDA to be used instead of OpenCL.
        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        # 5.2 Create various command queues (or CUDA streams) to queue GPU functions on.
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        # 5.3 Create reorder and correlation operations and create buffer linking the two operations.
        tensor_core_template = katgpucbf.xbgpu.tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
        )
        self.tensor_core_x_engine_core = tensor_core_template.instantiate(self._proc_command_queue)

        reorder_template = katgpucbf.xbgpu.precorrelation_reorder.PrecorrelationReorderTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_samples_per_channel=self.n_samples_per_channel,
            n_batches=self.batches_per_chunk,
        )
        self.precorrelation_reorder: katgpucbf.xbgpu.precorrelation_reorder.PrecorrelationReorder = (
            reorder_template.instantiate(self._proc_command_queue)
        )

        self.reordered_buffer_device = katsdpsigproc.accel.DeviceArray(
            self.context,
            self.precorrelation_reorder.slots["out_reordered"].shape,  # type: ignore
            self.precorrelation_reorder.slots["out_reordered"].dtype,  # type: ignore
        )
        self.precorrelation_reorder.bind(out_reordered=self.reordered_buffer_device)

        # 6. Create various buffers and assign them to the correct queues or objects.
        # 6.1 Define the number of items on each of these queues. The n_rx_items and n_tx_items each wrap a GPU buffer.
        # setting these values too high results in too much GPU memory being consumed. There just need to be enough
        # of them that the different processing functions do not get starved waiting for items. The low single digits is
        # suitable. n_free_chunks wraps buffer in system ram. This can be set quite high as there is much more system
        # RAM than GPU RAM. It should be higher than max_active_chunks.
        # These values are not configurable as they have been acceptable for most tests cases up until now. If the
        # pipeline starts bottlenecking, then maybe look at increasing these values.
        n_rx_items = 3  # Too high means too much GPU memory gets allocated
        n_tx_items = 2
        n_free_chunks = self.max_active_chunks + 8

        # 6.2 Create various queues for communication between async funtions. These queues are extended in the monitor
        # class, allowing for the monitor to track the number of items on each queue.
        self._rx_item_queue = self.monitor.make_queue("rx_item_queue", n_rx_items)
        self._rx_free_item_queue = self.monitor.make_queue("rx_free_item_queue", n_rx_items)
        self._tx_item_queue = self.monitor.make_queue("tx_item_queue", n_tx_items)
        self._tx_free_item_queue = self.monitor.make_queue("tx_free_item_queue", n_tx_items)

        # 6.3 Create buffers and assign them correctly.
        # 6.3.1 Create items that will store received chunks that have been transferred to the GPU.
        for _ in range(n_rx_items):
            rx_item = RxQueueItem()
            rx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context,
                self.precorrelation_reorder.slots["in_samples"].shape,  # type: ignore
                self.precorrelation_reorder.slots["in_samples"].dtype,  # type: ignore
            )
            self._rx_free_item_queue.put_nowait(rx_item)

        # 6.3.2 Create items that will store correlated data in GPU memory, ready for transferring back to system RAM.
        for _ in range(n_tx_items):
            tx_item = QueueItem()
            tx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context,
                self.tensor_core_x_engine_core.slots["out_visibilities"].shape,  # type: ignore
                self.tensor_core_x_engine_core.slots["out_visibilities"].dtype,  # type: ignore
            )
            self._tx_free_item_queue.put_nowait(tx_item)

        # 6.3.3 Create empty chunks and give them to the receiver to use to assemble heaps.
        for _ in range(n_free_chunks):
            buf = katsdpsigproc.accel.HostArray(
                self.precorrelation_reorder.slots["in_samples"].shape,  # type: ignore
                self.precorrelation_reorder.slots["in_samples"].dtype,  # type: ignore
                context=self.context,
            )
            chunk = recv.Chunk(buf)
            self.receiver_stream.add_chunk(chunk)

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
            core. This parameter determines which core to assign these interrupts to.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiver_stream.add_udp_ibv_reader(
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
        self.receiver_stream.add_buffer_reader(buffer)

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
        self.receiver_stream.add_udp_pcap_file_reader(pcap_file_name)

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
        # time between dumps. Care needs to be taken to ensure that this rate is not set too high. If it is
        # too high, the entire pipeline will stall needlessly waiting for packets to be transmitted too slowly.
        self.dump_interval_s = self.timestamp_increment_per_accumulation / self.adc_sample_rate_hz

        self.send_stream: katgpucbf.xbgpu.xsend.XEngineSPEADAbstractSend = katgpucbf.xbgpu.xsend.XEngineSPEADIbvSend(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_pols=self.n_pols,
            dump_interval_s=self.dump_interval_s,
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
        self.dump_interval_s = 0

        self.send_stream = katgpucbf.xbgpu.xsend.XEngineSPEADInprocSend(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_pols=self.n_pols,
            dump_interval_s=self.dump_interval_s,
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

        TODO: If no data is being streamed and the running flag is set to false, this function will be stuck waiting for
        the next chunk. Try find a way to exit cleanly without adding to much additional logic to this function.
        """
        # 1. Set up initial conditions
        async_ringbuffer = katgpucbf.xbgpu.ringbuffer.AsyncRingbuffer(
            self.receiver_stream.ringbuffer, self.monitor, "recv_ringbuffer", "get_chunks"
        )
        chunk_index = 0
        expected_heaps_total = 0
        dropped_heaps_total = 0
        # 2. Get complete chunks from the ringbuffer.
        async for chunk in async_ringbuffer:
            # 2.1 Update metrics and log warning if dropped heap is detected within the chunk.
            expected_heaps = len(chunk.present)
            received_heaps = sum(chunk.present)
            dropped_heaps = expected_heaps - received_heaps
            expected_heaps_total += expected_heaps
            dropped_heaps_total += dropped_heaps

            sensor_timestamp = time.time()
            # TODO: This must become a proper logging message; fstrings should
            # be replaced with old-fashioned format strings.
            if dropped_heaps != 0:
                logger.warning(
                    f"Chunk: {chunk_index:>5} Timestamp: {hex(chunk.timestamp)} "
                    f"Received: {received_heaps:>4} of {expected_heaps:>4} expected heaps. "
                    f"All time dropped heaps: {dropped_heaps_total}/{expected_heaps_total}."
                )
                self.sensors["input-missing-heaps-total"].set_value(dropped_heaps_total, timestamp=sensor_timestamp)

            def increment(sensor: Sensor, incr: int):
                sensor.set_value(sensor.value + incr, timestamp=sensor_timestamp)

            increment(self.sensors["input-heaps-total"], expected_heaps)
            increment(self.sensors["input-chunks-total"], 1)
            increment(
                self.sensors["input-bytes-total"], self.receiver_stream.chunk_bytes * received_heaps // expected_heaps
            )

            chunk_index += 1

            # 2.2. Get a free rx_item that will contain the GPU buffer to transfer the received chunk to.
            item = await self._rx_free_item_queue.get()
            item.timestamp += chunk.timestamp
            item.chunk = chunk

            # 2.3. Initiate transfer from recived chunk to rx_item buffer.
            item.buffer_device.set_async(self._upload_command_queue, chunk.base)
            item.add_event(self._upload_command_queue.enqueue_marker())

            # 2.4. Give the rx item to the _gpu_proc_loop function.
            await self._rx_item_queue.put(item)

            # 3. If the function must close, stop the stream.
            if self.running is not True:
                self.receiver_stream.stop()

    async def _gpu_proc_loop(self):
        """
        Perform all GPU processing of received data in a continuous loop.

        This function performs the following steps:
        1. Retrieve an rx_item from the _rx_item_queue
        2. Performs the reorder operation on the buffer in the rx item. This gets the buffer data into a format that
        the correlation kernel requires.
        3.1 Apply the correlation kernel to small subsets of the reordered data until all the data has been processed.
        3.2 If sufficient correlations have occured, transfer the correlated data to a tx_item, pass the tx_item to
        the _tx_item_queue and get a new item from the _tx_free_item_queue.

        The ratio of rx_items to tx_items is not one to one. There are expected to be many more rx_items in for every
        tx_item out.

        The above steps are performed in a loop until the running flag is set to false.

        TODO: Add B-Engine processing in this function.
        """
        # 1. Set up initial conditions
        tx_item = await self._tx_free_item_queue.get()
        # The very first heap sent out the X-Engine will have a timestamp of zero which is meaningless, every other
        # heap will have the correct timestamp.
        tx_item.timestamp = 0
        self.tensor_core_x_engine_core.bind(out_visibilities=tx_item.buffer_device)
        self.tensor_core_x_engine_core.zero_visibilities()

        while self.running:
            # 2. Get item from receiver function - wait for the HtoD transfers to complete and then give the chunk back
            #  to the receiver for reuse.
            rx_item = await self._rx_item_queue.get()
            await rx_item.async_wait_for_events()
            current_timestamp = rx_item.timestamp

            # 2.1 Give the chunk back to the receiver stream - if this is not done, eventually no more data will be
            # received as there will be no available chunks to store it in.
            self.receiver_stream.add_chunk(rx_item.chunk)

            # 3. Process the received data.
            # 3.1 Reorder the entire chunk
            self.precorrelation_reorder.bind(in_samples=rx_item.buffer_device)
            self.precorrelation_reorder()

            # 3.2 Perform correlation on reordered data. The correlation kernel does not have the
            # concept of a batch at this stage, so the kernel needs to be run on each different
            # batch in the chunk.
            for i in range(self.batches_per_chunk):
                # 3.2.1 Slice the buffer of reordered data to only select a specific batch. Then run the kernel on this
                # buffer.
                buffer_slice = katsdpsigproc.accel.DeviceArray(
                    self.context,
                    self.tensor_core_x_engine_core.slots["in_samples"].shape,  # type: ignore
                    self.tensor_core_x_engine_core.slots["in_samples"].dtype,  # type: ignore
                    raw=self.reordered_buffer_device.buffer.ptr + self.rx_bytes_per_heap_batch * i,
                )
                self.tensor_core_x_engine_core.bind(in_samples=buffer_slice)
                self.tensor_core_x_engine_core()

                # 3.2.2 If the batch timestamp corresponds to the accumulation interval, transfer the correlated data to
                # the sender function. NOTE: The timestamp representing the end of an accumulation does not necessarily
                # line up with the chunk timestamp. It will line up with a specific batch within a chunk though, this is
                # why this check has to happen for each batch.
                # This check is the equivilant of the MeerKAT SKARAB X-Engine auto-resync logic.
                next_heap_timestamp = current_timestamp + self.rx_heap_timestamp_step
                if next_heap_timestamp % self.timestamp_increment_per_accumulation == 0:

                    # 3.2.3 Transfer the TX item to the sender function
                    tx_item.add_event(self._proc_command_queue.enqueue_marker())
                    await self._tx_item_queue.put(tx_item)

                    # 3.2.4 Get a new tx item, assign its buffer correctly and reset the buffer to zero.
                    tx_item = await self._tx_free_item_queue.get()
                    tx_item.timestamp = next_heap_timestamp
                    self.tensor_core_x_engine_core.bind(out_visibilities=tx_item.buffer_device)
                    self.tensor_core_x_engine_core.zero_visibilities()

                # 4. Increment batch timestamp.
                current_timestamp += self.rx_heap_timestamp_step

            # 5. Finished with the RX item - reset it and give it back to the receiver loop function.
            rx_item.reset()
            await self._rx_free_item_queue.put(rx_item)

        # 6. When the stream is closed, if the sender loop is waiting for a tx item, it will never exit. This function
        # puts the current tx_item on the queue. The sender_loop can then stop waiting upon receiving this and exit.
        await self._tx_item_queue.put(tx_item)

    async def _sender_loop(self):
        """
        Send heaps to the network in a continuous loop.

        This function does the following:
        1. Get an item from the _tx_item_queue.
        2. Wait for all the events on this item to complete.
        3. Wait for an available heap buffer from the send_stream.
        4. Transfer the GPU buffer in the item to the heap buffer in system RAM.
        5. Wait for the transfer to complete.
        6. Transmit data in heap buffer out into the network.
        7. Place the tx_item on _tx_item_free_queue so that it can be reused.

        The above steps are performed in a loop until the running flag is set to false.

        NOTE: The transfer from the GPU to the heap buffer and the sending onto the network could be pipeline a bit
        better, but this is not really required in this loop as this whole process occurs at a much slower pace than
        the rest of the pipeline.
        """
        old_time_s = time.time()
        old_timestamp = 0

        while self.running:
            # 1. Get the item to transfer and wait for all GPU events to finish before continuing
            item = await self._tx_item_queue.get()
            await item.async_wait_for_events()

            # 2. Get a free heap buffer to copy the GPU data to
            buffer_wrapper = await self.send_stream.get_free_heap()

            # 3 Perform some basic logging.
            # We do not expect the time between dumps to be the same each time as the time.time() function
            # checks the wall time now, not the actual time between timestamps. The difference between dump timestamps
            # is expected to be constant
            new_time_s = time.time()
            time_difference_between_heaps_s = new_time_s - old_time_s

            # 3.1 Log that a heap is about to be sent.
            # TODO: change to an old-fashioned formatted string. fstrings aren't
            # great for logging.
            logger.info(
                f"Current output heap timestamp: {hex(item.timestamp)}, difference between timestamps: "
                f"{hex(item.timestamp - old_timestamp)}, wall time between dumps "
                f"{round(time_difference_between_heaps_s, 2)} s"
            )

            # 3.2. Ensure that the timestamp between output heaps is the value that is expected,
            # Not sure under which conditions that this would occur. Something
            # funny would have to happen at the receiver.
            # This check is here pre-emptivly - this issue has not been detected yet.
            if item.timestamp - old_timestamp != self.timestamp_increment_per_accumulation:
                logger.warning(
                    f"Timestamp between heaps equal to {hex(item.timestamp - old_timestamp)}, expected "
                    f"{hex(self.timestamp_increment_per_accumulation)}"
                )

            # 3.3. Check that items are not being received faster than they are expected to be send.
            # As the output packets are rate limited in such a way to match the dump rate, receiving data too quickly
            # will result in data bottlenecking at the sender, the pipeline eventually stalling and the input buffer
            # overflowing.
            if time_difference_between_heaps_s * 1.05 < self.dump_interval_s:
                logger.warning(
                    f"Time between output heaps: {round(time_difference_between_heaps_s,2)} "
                    f"which is less the expected {round(self.dump_interval_s,2)}. "
                    "If this warning occurs too often, the pipeline will stall "
                    "because the rate limited sender will not keep up with the input rate."
                )

            # 3.4 Update variables used for warning checks.
            old_time_s = new_time_s
            old_timestamp = item.timestamp

            # 4. Transfer GPU buffer in item to free buffer.
            item.buffer_device.get_async(self._download_command_queue, buffer_wrapper.buffer)
            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            # 5. Tell sender to transmit heap buffer on network.
            self.send_stream.send_heap(item.timestamp, buffer_wrapper)

            # 6. Reset item and put it back on the the _tx_free_item_queue for resue
            item.reset()
            await self._tx_free_item_queue.put(item)

    async def run_descriptors_loop(self, interval_s):
        """
        Send the Baseline Correlation Products Hardware heaps out to the network every interval_s seconds.

        This function is not part of the main run function as we do not want it running during the unit tests.
        """
        while self.running:
            self.send_stream.send_descriptor_heap()
            await asyncio.sleep(interval_s)

    async def run(self):
        """
        Launch all the different async functions required to run the X-Engine.

        These functions will loop forever and only exit once an exit flag is set.
        """
        if self.rx_transport_added is not True:
            raise AttributeError("Transport for receiving data has not yet been set.")
        if self.tx_transport_added is not True:
            raise AttributeError("Transport for sending data has not yet been set.")

        self.running = True
        loop = asyncio.get_event_loop()
        await self.start()
        tasks = [
            loop.create_task(self._receiver_loop()),
            loop.create_task(self._gpu_proc_loop()),
            loop.create_task(self._sender_loop()),
        ]
        self.task = asyncio.gather(*tasks)

    def stop(self):
        """
        Stop all the different processing loops launched in the run() function and wind up the receiver stream.

        NOTE 1: This function may not be working correctly. If you have trouble closing the tasks, it may be worth
        re-evaluating this function.
        NOTE 2: The descriptors loop function is not launched by the run() function. It is the user's responsibility to
        stop that function.
        """
        self.receiver_stream.stop()
        self.running = False
        self.task.cancel()
