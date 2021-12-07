################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""
This module defines an XBEngine object that implements an entire GPU XB-Engine pipeline.

Additionally this module defines the QueueItem and RxQueueItem objects that are
used in the XBEngine for passing information between different async processing
loops within the object.

.. todo::

    - The B-Engine logic has not been implemented yet - this needs to be added
      eventually. It is expected that this logic will need to go in the
      _gpu_proc_loop for the B-Engine processing and then a seperate sender
      loop would need to be created for sending B-Engine data.
    - Implement control - There is no mechanism to interact with a running pipeline.
    - Catch asyncio exceptions - If one of the running asyncio loops has an
      exception, it will stop running without crashing the program or printing
      the error trace stack. This is not an issue when things are working, but
      if we could catch those exceptions and crash the program, it would make
      detecting and debugging heaps much simpler.
    - The asyncio syntax in the run() function uses old syntax, once this repo
      has been updated to python 3.8, update this to use the new asyncio
      syntax.

"""

import asyncio
import logging
import math
import time
from typing import List, Optional

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource
import numpy as np
import spead2
from aiokatcp import DeviceServer

from .. import COMPLEX, N_POLS, __version__
from ..monitor import Monitor
from ..ringbuffer import ChunkRingbuffer
from . import recv
from .correlation import CorrelationTemplate
from .precorrelation_reorder import PrecorrelationReorderTemplate
from .xsend import XSend, make_stream

logger = logging.getLogger(__name__)


def done_callback(future: asyncio.Future) -> None:
    """
    Handle cancellation of Processing Loops as a callback.

    Log exceptions as soon as they occur.
    """
    try:
        future.result()  # Evaluate just for exceptions
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Processing failed with exception")


class QueueItem:
    """
    Object to enable communication and synchronisation between different functions in the XBEngine object.

    This queue item contains a buffer of preallocated GPU memory. This memory
    is reused many times in the processing functions to prevent unecessary
    allocations. The item also contains a list of events. Before accessing the
    data in the buffer, the user needs to ensure that the events have all been
    completed.
    """

    timestamp: int
    events: List[katsdpsigproc.abc.AbstractEvent]
    buffer_device: katsdpsigproc.accel.DeviceArray

    def __init__(self, timestamp: int = 0) -> None:
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

    The RxQueueItem between the sender and the gpu proc loops need to also
    store a reference to the chunk that data in the GPU buffer was copied from.
    This allows the gpu proc loop to hand the chunk back to the receiver once
    the copy is complete to reuse resources.
    """

    chunk: Optional[recv.Chunk]

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

    In order to reduce the load on the main thread, received data is collected
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
        Sample rate of the digitisers in the current array. This value is
        required to calculate the packet spacing of the output heaps. If it is
        set incorrectly, the packet spacing could be too large causing the
        pipeline to stall as heaps queue at the sender faster than they are
        sent.
    send_rate_factor
        Configure the SPEAD2 sender with a rate proportional to this factor.
        This value is intended to dictate a data transmission rate slightly
        higher/faster than the ADC rate.
        NOTE:
        - A factor of zero (0) tells the sender to transmit as fast as
          possible.
    n_ants
        The number of antennas to be correlated.
    n_channels_total
        The total number of frequency channels out of the F-Engine.
    n_channels_per_stream
        The number of frequency channels contained per stream.
    n_spectra_per_heap
        The number of time samples received per frequency channel.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    heap_accumulation_threshold
        The number of consecutive heaps to accumulate. This value is used to
        determine the dump rate.
    channel_offset_value
        The index of the first channel in the subset of channels processed by
        this XB-Engine. Used to set the value in the XB-Engine output heaps for
        spectrum reassembly by the downstream receiver.
    src_affinity
        Specific CPU core to assign the RX stream processing thread to.
    chunk_spectra
        A batch is a collection of heaps from different antennas with the same
        timestamp. This parameter specifies the number of consecutive batches
        to store in the same chunk. The higher this value is, the more GPU and
        system RAM is allocated, the lower this value is, the more work the
        python processing thread is required to do.
    rx_reorder_tol
        Maximum tolerance for jitter between received packets, as a time
        expressed in ADC sample ticks.
    monitor
        :class:`Monitor` to use for generating multiple :class:`~asyncio.Queue`
        objects needed to communicate between functions, and handling basic
        reporting for :class:`~asyncio.Queue` sizes and events.
    context
        Device context for katsdpsigproc. It must be a CUDA device.
    """

    VERSION = "katgpucbf-xbgpu-icd-0.1"
    BUILD_STATE = __version__

    def __init__(
        self,
        *,
        katcp_host: str,
        katcp_port: int,
        adc_sample_rate_hz: float,
        send_rate_factor: float,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_stream: int,
        n_spectra_per_heap: int,
        sample_bits: int,
        heap_accumulation_threshold: int,
        channel_offset_value: int,
        src_affinity: int,
        chunk_spectra: int,  # Used for GPU memory tuning
        rx_reorder_tol: int,
        monitor: Monitor,
        context: katsdpsigproc.abc.AbstractContext,
    ):
        super(XBEngine, self).__init__(katcp_host, katcp_port)

        if sample_bits != 8:
            raise ValueError("sample_bits must equal 8 - no other values supported at the moment.")

        if channel_offset_value % n_channels_per_stream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_stream")

        # Array configuration parameters
        self.adc_sample_rate_hz = adc_sample_rate_hz
        self.send_rate_factor = send_rate_factor
        self.heap_accumulation_threshold = heap_accumulation_threshold
        self.n_ants = n_ants
        self.n_channels_total = n_channels_total
        self.n_channels_per_stream = n_channels_per_stream
        self.n_spectra_per_heap = n_spectra_per_heap
        self.sample_bits = sample_bits

        # NOTE: The n_rx_items and n_tx_items each wrap a GPU buffer. Setting
        # these values too high results in too much GPU memory being consumed.
        # There needs to be enough of them that the different processing
        # functions do not get starved waiting for items. The low single digits
        # is suitable. n_free_chunks wraps buffer in system RAM. This can be
        # set quite high as there is much more system RAM than GPU RAM. It
        # should be higher than max_active_chunks.
        # These values are not configurable as they have been acceptable for
        # most tests cases up until now. If the pipeline starts bottlenecking,
        # then maybe look at increasing these values.
        n_rx_items = 3  # Too high means too much GPU memory gets allocated
        n_tx_items = 2

        # Multiply this _step by 2 to account for dropping half of the
        # spectrum due to symmetric properties of the Fourier Transform. While
        # we can workout the timestamp_step from other parameters that
        # configure the receiver, we pass it as a seperate argument to the
        # reciever for cases where the n_channels_per_stream changes across
        # streams (likely for non-power-of-two array sizes).
        self.rx_heap_timestamp_step = self.n_channels_total * 2 * self.n_spectra_per_heap

        # The number of bytes for a single batch of F-Engines. A chunk
        # consists of multiple batches.
        self.rx_bytes_per_heap_batch = (
            self.n_ants * self.n_channels_per_stream * self.n_spectra_per_heap * N_POLS * COMPLEX
        )

        self.timestamp_increment_per_accumulation = self.heap_accumulation_threshold * self.rx_heap_timestamp_step

        # Sets the number of batches of heaps to store per chunk
        self.chunk_spectra = chunk_spectra
        self.max_active_chunks: int = math.ceil(rx_reorder_tol / self.rx_heap_timestamp_step / self.chunk_spectra) + 1
        n_free_chunks: int = self.max_active_chunks + 8  # TODO: Abstract this 'naked' constant
        self.channel_offset_value = channel_offset_value

        # False if no transport has been added, true otherwise.
        self.tx_transport_added = False
        self.rx_transport_added = False

        self.monitor = monitor

        # The receiver_stream object has no attached
        # transport yet and will not function until one of the
        # add_*_receiver_transport() functions has been called.
        self.ringbuffer = ChunkRingbuffer(
            n_free_chunks, name="recv_ringbuffer", task_name="receiver_loop", monitor=monitor
        )
        self.receiver_stream = recv.make_stream(
            n_ants=self.n_ants,
            n_channels_per_stream=self.n_channels_per_stream,
            n_spectra_per_heap=self.n_spectra_per_heap,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.chunk_spectra,
            max_active_chunks=self.max_active_chunks,
            ringbuffer=self.ringbuffer,
            thread_affinity=src_affinity,
        )

        self.context = context

        # A command queue is the OpenCL name for a CUDA stream. An abstract
        # command queue can either be implemented as an OpenCL command queue or
        # a CUDA stream depending on the context.
        self._upload_command_queue: katsdpsigproc.abc.AbstractCommandQueue = self.context.create_command_queue()
        self._proc_command_queue: katsdpsigproc.abc.AbstractCommandQueue = self.context.create_command_queue()
        self._download_command_queue: katsdpsigproc.abc.AbstractCommandQueue = self.context.create_command_queue()

        correlation_template = CorrelationTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_spectra_per_heap=self.n_spectra_per_heap,
        )
        self.correlation = correlation_template.instantiate(self._proc_command_queue)

        reorder_template = PrecorrelationReorderTemplate(
            self.context,
            n_ants=self.n_ants,
            n_channels=self.n_channels_per_stream,
            n_spectra_per_heap=self.n_spectra_per_heap,
            n_batches=self.chunk_spectra,
        )
        self.precorrelation_reorder = reorder_template.instantiate(self._proc_command_queue)

        self.reordered_buffer_device = katsdpsigproc.accel.DeviceArray(
            self.context,
            self.precorrelation_reorder.slots["out_reordered"].shape,  # type: ignore
            self.precorrelation_reorder.slots["out_reordered"].dtype,  # type: ignore
        )
        self.precorrelation_reorder.bind(out_reordered=self.reordered_buffer_device)

        # These queues are extended in the monitor class, allowing for the
        # monitor to track the number of items on each queue.
        # * The _rx_item_queue passes items from the _receiver_loop function to
        #   the _gpu_proc_loop function.
        # * The _tx_item_queue passes items from the _gpu_proc_loop to the
        #   _sender_loop function.
        # Once the destination function is finished with an item, it will pass
        # it back to the corresponding _(rx/tx)_free_item_queue to ensure that
        # all allocated buffers are in continuous circulation.
        self._rx_item_queue: asyncio.Queue[RxQueueItem] = self.monitor.make_queue("rx_item_queue", n_rx_items)
        self._rx_free_item_queue: asyncio.Queue[RxQueueItem] = self.monitor.make_queue("rx_free_item_queue", n_rx_items)
        self._tx_item_queue: asyncio.Queue[QueueItem] = self.monitor.make_queue("tx_item_queue", n_tx_items)
        self._tx_free_item_queue: asyncio.Queue[QueueItem] = self.monitor.make_queue("tx_free_item_queue", n_tx_items)

        for _ in range(n_rx_items):
            rx_item = RxQueueItem()
            rx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context,
                self.precorrelation_reorder.slots["in_samples"].shape,  # type: ignore
                self.precorrelation_reorder.slots["in_samples"].dtype,  # type: ignore
            )
            self._rx_free_item_queue.put_nowait(rx_item)

        for _ in range(n_tx_items):
            tx_item = QueueItem()
            tx_item.buffer_device = katsdpsigproc.accel.DeviceArray(
                self.context,
                self.correlation.slots["out_visibilities"].shape,  # type: ignore
                self.correlation.slots["out_visibilities"].dtype,  # type: ignore
            )
            self._tx_free_item_queue.put_nowait(tx_item)

        for _ in range(n_free_chunks):
            buf = katsdpsigproc.accel.HostArray(
                self.precorrelation_reorder.slots["in_samples"].shape,  # type: ignore
                self.precorrelation_reorder.slots["in_samples"].dtype,  # type: ignore
                context=self.context,
            )
            present = np.zeros(n_ants * self.chunk_spectra, np.uint8)
            chunk = recv.Chunk(data=buf, present=present)
            self.receiver_stream.add_free_chunk(chunk)

    def add_udp_ibv_receiver_transport(
        self, src_ip: str, src_port: int, interface_ip: str, comp_vector: int, buffer_size: int
    ):
        """
        Add the ibv_udp transport to the receiver.

        The receiver will read udp packets off of the specified interface using
        the ibverbs library to offload processing from the CPU.

        This transport is intended to be the transport used in production.

        Parameters
        ----------
        src_ip
            multicast IP address of source data.
        src_port
            Port of source data
        interface_ip
            IP address of interface to listen for data on.
        comp_vector
            Received packets will generate interrupts from the NIC. This value
            selects an interrupt vector, and the OS controls the mapping from
            interrupt vector to CPU core.
        buffer_size
            The size of the network receive buffer.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.receiver_stream.add_udp_ibv_reader(
            [(src_ip, src_port)], interface_ip, buffer_size=buffer_size, comp_vector=comp_vector
        )
        self.rx_transport_added = True

    def add_udp_receiver_transport(self, src_ip: str, src_port: int, interface_ip: str, buffer_size: int):
        """
        Add the 'regular' UDP transport to the receiver.

        Allow the user to run the XBEngine without the use of IBVerbs.

        Parameters
        ----------
        src_ip
            multicast IP address of source data.
        src_port
            Port of source data
        interface_ip
            IP address of interface to listen for data on.
        buffer_size
            The size of the network receive buffer.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")

        self.receiver_stream.add_udp_reader(
            src_ip,
            src_port,
            buffer_size=buffer_size,
            interface_address=interface_ip or "",
        )

        self.rx_transport_added = True

    def add_buffer_receiver_transport(self, buffer: bytes):
        """
        Add the buffer transport to the receiver.

        The receiver will read packet data python ByteArray generated by a
        spead2.send.BytesStream object. The sender does not support the inproc
        transport and as such the buffer transport must be used instead.

        This transport is intended to be used for testing purposes.

        Parameters
        ----------
        buffer
            Buffer containing simulated packet data.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.receiver_stream.add_buffer_reader(buffer)
        self.rx_transport_added = True

    def add_pcap_receiver_transport(self, pcap_filename: str):
        """
        Add the pcap transport to the receiver.

        The receiver will read packet data from a pcap file. This transport is
        intended to be used for testing purposes.

        Parameters
        ----------
        pcap_filename
            Name of PCAP file to open.
        """
        if self.rx_transport_added is True:
            raise AttributeError("Transport for receiving data has already been set.")
        self.rx_transport_added = True
        self.receiver_stream.add_udp_pcap_file_reader(pcap_filename)

    def add_udp_sender_transport(
        self,
        dest_ip: str,
        dest_port: int,
        interface_ip: str,
        ttl: int,
        thread_affinity: int,
        comp_vector: int,
        packet_payload: int,
        use_ibv: bool = True,
    ):
        """
        Add a UDP transport to the sender.

        If indicated (use_ibv), the sender will transmit UDP packets out of the
        specified interface using the ibverbs library to offload processing
        from the CPU.

        The UdpIbvStream is intended for use in production, and the UdpStream
        for local testing on a suitable machine.

        Parameters
        ----------
        dest_ip
            multicast IP address of destination data
        dest_port
            Port of transmitted data
        interface_ip
            IP address of interface to trasnmit data on.
        ttl
            Time to live for the output multicast packets.
        thread_affinity
            The sender creates its own thread to run in the background
            transmitting data. It is bound to the CPU core specified here.
        comp_vector
            Completion vector for transmission, or -1 for polling.
            See :class:`spead2.send.UdpIbvConfig` for further information.
        use_ibv
            Use spead2's ibverbs transport for data transmission.
        packet_payload
            Size in bytes for output packets (baseline correlation products
            payload only, headers and padding are then added to this).
        """
        if self.tx_transport_added is True:
            raise AttributeError("Transport for sending data has already been set.")

        # NOTE: This value staggers the send so that packets within a heap are
        # transmitted onto the network across the entire time between dumps.
        # Care needs to be taken to ensure that this rate is not set too high.
        # If it is too high, the entire pipeline will stall needlessly waiting
        # for packets to be transmitted too slowly.
        self.dump_interval_s: float = self.timestamp_increment_per_accumulation / self.adc_sample_rate_hz

        self.send_stream = XSend(
            n_ants=self.n_ants,
            n_channels=self.n_channels_total,
            n_channels_per_stream=self.n_channels_per_stream,
            dump_interval_s=self.dump_interval_s,
            send_rate_factor=self.send_rate_factor,
            channel_offset=self.channel_offset_value,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            packet_payload=packet_payload,
            stream_factory=lambda stream_config, buffers: make_stream(
                dest_ip=dest_ip,
                dest_port=dest_port,
                interface_ip=interface_ip,
                ttl=ttl,
                use_ibv=use_ibv,
                affinity=thread_affinity,
                comp_vector=comp_vector,
                stream_config=stream_config,
                buffers=buffers,
            ),
        )

        self.tx_transport_added = True

    def add_inproc_sender_transport(self, queue: spead2.InprocQueue):
        """
        Add the inproc transport to the sender.

        The sender will send heaps out on an InprocQueue. This transport is
        intended to be used for testing purposes.

        Parameters
        ----------
        queue
            SPEAD2 inproc queue to send heaps to.
        """
        if self.tx_transport_added is True:
            raise AttributeError("Transport for sending data has already been set.")
        self.tx_transport_added = True

        # For the inproc transport this value is set very low as the dump rate
        # does affect performance for an inproc queue and a high dump rate just
        # makes the unit tests take very long to run.
        self.dump_interval_s = 0

        self.send_stream = XSend(
            n_ants=self.n_ants,
            n_channels=self.n_channels_total,
            n_channels_per_stream=self.n_channels_per_stream,
            dump_interval_s=self.dump_interval_s,
            send_rate_factor=self.send_rate_factor,
            channel_offset=self.channel_offset_value,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            stream_factory=lambda stream_config, buffers: spead2.send.asyncio.InprocStream(
                spead2.ThreadPool(), [queue], stream_config
            ),
        )

    async def _receiver_loop(self):
        """
        Receive heaps off of the network in a continuous loop.

        This function does the following:
        1. Wait for a chunk to be assembled on the receiver.
        2. Get a free rx item off of the _rx_free_item_queue.
        3. Initiate the transfer of the chunk from system memory to the buffer
           in GPU RAM that belongs to the rx_item.
        4. Place the rx_item on _rx_item_queue so that it can be processed downstream.

        The above steps are performed in a loop until there are no more chunks to assembled.
        """
        async for chunk in recv.recv_chunks(self.receiver_stream):
            timestamp = chunk.chunk_id * self.rx_heap_timestamp_step * self.chunk_spectra

            # Get a free rx_item that will contain the GPU buffer to which the
            # received chunk will be transferred.
            item = await self._rx_free_item_queue.get()
            item.timestamp += timestamp
            item.chunk = chunk

            # Initiate transfer from received chunk to rx_item buffer.
            # First wait for asynchronous GPU work on the buffer.
            self._upload_command_queue.enqueue_wait_for_events(item.events)
            item.buffer_device.set_async(self._upload_command_queue, chunk.data)
            item.add_event(self._upload_command_queue.enqueue_marker())

            # Give the received item to the _gpu_proc_loop function.
            await self._rx_item_queue.put(item)

        # spead2 will (eventually) indicate that there are no chunks to async-for through
        logger.debug("_receiver_loop completed")
        self._rx_item_queue.put_nowait(None)

    async def _gpu_proc_loop(self):
        """
        Perform all GPU processing of received data in a continuous loop.

        This function performs the following steps:
        1. Retrieve an rx_item from the _rx_item_queue
        2. Performs the reorder operation on the buffer in the rx item. This
           gets the buffer data into a format that the correlation kernel
           requires.
        3.1 Apply the correlation kernel to small subsets of the reordered data
            until all the data has been processed.
        3.2 If sufficient correlations have occured, transfer the correlated
            data to a tx_item, pass the tx_item to the _tx_item_queue and get a
            new item from the _tx_free_item_queue.

        The ratio of rx_items to tx_items is not one to one. There are expected
        to be many more rx_items in for every tx_item out.

        The above steps are performed in a loop until the running flag is set
        to false.

        TODO: Add B-Engine processing in this function.
        """
        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()

        # NOTE: The very first heap sent out the X-Engine will have a timestamp
        # of zero which is meaningless, every other heap will have the correct
        # timestamp.
        tx_item.timestamp = 0
        self.correlation.bind(out_visibilities=tx_item.buffer_device)
        self.correlation.zero_visibilities()

        while True:
            # Get item from the receiver function.
            # - Wait for the HtoD transfers to complete, then
            # - Give the chunk back to the receiver for reuse.
            rx_item = await self._rx_item_queue.get()
            if rx_item is None:
                break
            await rx_item.async_wait_for_events()
            current_timestamp = rx_item.timestamp

            # NOTE: If this is not done, eventually no more data will be
            # received as there will be no available chunks to store it in.
            self.receiver_stream.add_free_chunk(rx_item.chunk)

            self.precorrelation_reorder.bind(in_samples=rx_item.buffer_device)
            self.precorrelation_reorder()

            reorder_event = self._proc_command_queue.enqueue_marker()
            rx_item.reset()
            rx_item.add_event(reorder_event)
            await self._rx_free_item_queue.put(rx_item)

            # The correlation kernel does not have the concept of a batch at
            # this stage, so the kernel needs to be run on each different batch
            # in the chunk.
            for i in range(self.chunk_spectra):
                buffer_slice = katsdpsigproc.accel.DeviceArray(
                    self.context,
                    self.correlation.slots["in_samples"].shape,  # type: ignore
                    self.correlation.slots["in_samples"].dtype,  # type: ignore
                    raw=self.reordered_buffer_device.buffer.ptr + self.rx_bytes_per_heap_batch * i,
                )
                self.correlation.bind(in_samples=buffer_slice)
                self.correlation()

                # NOTE: The timestamp representing the end of an
                # accumulation does not necessarily line up with the chunk
                # timestamp. It will line up with a specific batch within a
                # chunk though, this is why this check has to happen for each
                # batch. This check is the equivalent of the MeerKAT SKARAB
                # X-Engine auto-resync logic.
                next_heap_timestamp = current_timestamp + self.rx_heap_timestamp_step
                if next_heap_timestamp % self.timestamp_increment_per_accumulation == 0:

                    tx_item.add_event(self._proc_command_queue.enqueue_marker())
                    await self._tx_item_queue.put(tx_item)

                    tx_item = await self._tx_free_item_queue.get()
                    await tx_item.async_wait_for_events()
                    tx_item.timestamp = next_heap_timestamp
                    self.correlation.bind(out_visibilities=tx_item.buffer_device)
                    self.correlation.zero_visibilities()

                current_timestamp += self.rx_heap_timestamp_step

        # When the stream is closed, if the sender loop is waiting for a tx item,
        # it will never exit. Upon receiving this NoneType, the sender_loop can
        # stop waiting and exit.
        logger.debug("_gpu_proc_loop completed")
        self._tx_item_queue.put_nowait(None)

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

        NOTE: The transfer from the GPU to the heap buffer and the sending onto
        the network could be pipeline a bit better, but this is not really
        required in this loop as this whole process occurs at a much slower
        pace than the rest of the pipeline.
        """
        old_time_s = time.time()
        old_timestamp = 0

        while True:
            item = await self._tx_item_queue.get()
            if item is None:
                break
            await item.async_wait_for_events()

            buffer_wrapper = await self.send_stream.get_free_heap()

            # NOTE: We do not expect the time between dumps to be the same each
            # time as the time.time() function checks the wall time now, not
            # the actual time between timestamps. The difference between dump
            # timestamps is expected to be constant.
            new_time_s = time.time()
            time_difference_between_heaps_s = new_time_s - old_time_s

            logger.info(
                "Current output heap timestamp: %#x, difference between timestamps: %#x, "
                "wall time between dumps %.2f s",
                item.timestamp,
                item.timestamp - old_timestamp,
                time_difference_between_heaps_s,
            )

            # NOTE: Not sure under which conditions that this would fail to
            # occur. Something funny would have to happen at the receiver.
            # This check is here pre-emptively - this issue has not been
            # detected (yet).
            if item.timestamp - old_timestamp != self.timestamp_increment_per_accumulation:
                logger.warning(
                    "Timestamp between heaps equal to %#x, expected %#x",
                    item.timestamp - old_timestamp,
                    self.timestamp_increment_per_accumulation,
                )

            # NOTE: As the output packets are rate limited in
            # such a way to match the dump rate, receiving data too quickly
            # will result in data bottlenecking at the sender, the pipeline
            # eventually stalling and the input buffer overflowing.
            if time_difference_between_heaps_s * 1.05 < self.dump_interval_s:
                logger.warning(
                    "Time between output heaps: %.2f which is less the expected %.2f. "
                    "If this warning occurs too often, the pipeline will stall "
                    "because the rate limited sender will not keep up with the input rate.",
                    time_difference_between_heaps_s,
                    self.dump_interval_s,
                )

            old_time_s = new_time_s
            old_timestamp = item.timestamp

            item.buffer_device.get_async(self._download_command_queue, buffer_wrapper.buffer)
            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            self.send_stream.send_heap(item.timestamp, buffer_wrapper)

            item.reset()
            await self._tx_free_item_queue.put(item)

        await self.send_stream.send_stop_heap()
        logger.debug("_sender_loop completed")

    async def run_descriptors_loop(self, interval_s):
        """
        Send the Baseline Correlation Products Hardware heaps out to the network every interval_s seconds.

        This function is not part of the main run function as we do not want it
        running during the unit tests.
        """
        while True:
            self.send_stream.send_descriptor_heap()
            await asyncio.sleep(interval_s)

    async def start(self):
        """
        Launch all the different async functions required to run the X-Engine.

        These functions will loop forever and only exit once the XBEngine receives
        a SIGINT or SIGTERM.
        """
        if self.rx_transport_added is not True:
            raise AttributeError("Transport for receiving data has not yet been set.")
        if self.tx_transport_added is not True:
            raise AttributeError("Transport for sending data has not yet been set.")

        self.task = asyncio.gather(
            asyncio.create_task(self._receiver_loop()),
            asyncio.create_task(self._gpu_proc_loop()),
            asyncio.create_task(self._sender_loop()),
        )

        self.task.add_done_callback(done_callback)

        await super().start()

    async def on_stop(self):
        """
        Shut down processing when the device server is stopped.

        This is called by aiokatcp after closing the listening socket.
        """
        self.receiver_stream.stop()

        try:
            await self.task
        except Exception:
            # Errors get logged by the done_callback in start()
            pass
