################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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
"""

import asyncio
import logging
import math
import time

import aiokatcp
import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.resource
import numpy as np
import spead2.recv
from aiokatcp import DeviceServer
from katsdpsigproc import accel

from .. import (
    DESCRIPTOR_TASK_NAME,
    GPU_PROC_TASK_NAME,
    RECV_TASK_NAME,
    SEND_TASK_NAME,
    SPEAD_DESCRIPTOR_INTERVAL_S,
    __version__,
)
from .. import recv as base_recv
from ..monitor import Monitor
from ..queue_item import QueueItem
from ..recv import RX_SENSOR_TIMEOUT_CHUNKS, RX_SENSOR_TIMEOUT_MIN
from ..ringbuffer import ChunkRingbuffer
from ..send import DescriptorSender
from ..utils import DeviceStatusSensor, TimeConverter, add_time_sync_sensors
from . import recv
from .correlation import Correlation, CorrelationTemplate
from .output import Output, XOutput
from .xsend import XSend, incomplete_accum_counter, make_stream

logger = logging.getLogger(__name__)
MISSING = np.array([-(2**31), 1], dtype=np.int32)


class RxQueueItem(QueueItem):
    """
    Extension of the QueueItem to also store a chunk reference and heap presence.

    The RxQueueItem between the sender and the gpu proc loops need to also
    store a reference to the chunk that data in the GPU buffer was copied from.
    This allows the gpu proc loop to hand the chunk back to the receiver once
    the copy is complete to reuse resources.
    """

    def __init__(self, buffer_device: accel.DeviceArray, present: np.ndarray, timestamp: int = 0) -> None:
        self.buffer_device = buffer_device
        self.present = present
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp, events and chunk."""
        super().reset(timestamp=timestamp)
        self.chunk: recv.Chunk | None = None


class TxQueueItem(QueueItem):
    """
    Extension of the QueueItem to track antennas that have missed data.

    The TxQueueItem between the gpu-proc and sender loops needs to carry a record
    of which antennas have missed data at any point in the accumulation being
    processed. This is used to determine whether any baselines were affected, and have
    their data zeroed accordingly.
    """

    def __init__(
        self,
        buffer_device: accel.DeviceArray,
        saturated: accel.DeviceArray,
        present_ants: np.ndarray,
        timestamp: int = 0,
    ) -> None:
        self.buffer_device = buffer_device
        self.saturated = saturated
        self.present_ants = present_ants
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp, events and present antenna tracker."""
        super().reset(timestamp=timestamp)
        self.present_ants.fill(True)  # Assume they're fine until told otherwise
        self.batches = 0


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
      2. :func:`_gpu_proc_loop` - Perform the correlation operation.
      3. :func:`_sender_loop` - Transfer correlated data to system RAM and then
         send it out on the network.

    There is also a seperate function for sending descriptors onto the network.

    Items passed between queues may still have GPU operations in progress. Each
    item stores a list of events that can be used to determine if a GPU
    operation is complete.

    In order to reduce the load on the main thread, received data is collected
    into chunks. A chunk consists of multiple batches of F-Engine heaps where a
    batch is a collection of heaps from all F-Engine with the same timestamp.

    The initialiser allocates all memory buffers to be used during the lifetime
    of the XBEngine object. These buffers are continuously reused to ensure
    memory use remains constrained.

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
        Configure the spead2 sender with a rate proportional to this factor.
        This value is intended to dictate a data transmission rate slightly
        higher/faster than the ADC rate. A factor of zero (0) tells the sender
        to transmit as fast as possible.
    n_ants
        The number of antennas to be correlated.
    n_channels_total
        The total number of frequency channels out of the F-Engine.
    n_channels_per_stream
        The number of frequency channels contained per stream.
    n_samples_between_spectra
        The number of samples between frequency spectra received.
    n_spectra_per_heap
        The number of time samples received per frequency channel.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    heap_accumulation_threshold
        The number of consecutive heaps to accumulate. This value is used to
        determine the dump rate.
    sync_epoch
        UNIX time corresponding to timestamp zero
    channel_offset_value
        The index of the first channel in the subset of channels processed by
        this XB-Engine. Used to set the value in the XB-Engine output heaps for
        spectrum reassembly by the downstream receiver.
    outputs
        Output streams to generate. Currently this must be a single
        XOutput.
    src
        Endpoint for the incoming data.
    src_interface
        IP address of the network device to use for input.
    src_ibv
        Use ibverbs for input.
    src_affinity
        Specific CPU core to assign the RX stream processing thread to.
    src_comp_vector
        Completion vector for source stream, or -1 for polling.
        See :class:`spead2.recv.UdpIbvConfig` for further information.
    src_buffer
        The size of the network receive buffer.
    heaps_per_fengine_per_chunk
        The number of consecutive batches to store in the same chunk. The higher
        this value is, the more GPU and system RAM is allocated, the lower,
        the more work the Python processing thread is required to do.
    rx_reorder_tol
        Maximum tolerance for jitter between received packets, as a time
        expressed in ADC sample ticks.
    dst
        Destination endpoint for the outgoing data.
    dst_interface
        IP address of the network device to use for output.
    dst_ttl
        TTL for outgoing packets.
    dst_ibv
        Use ibverbs for output.
    dst_packet_payload
        Size for output packets (correlation product payload only, headers and padding are
        added to this).
    dst_affinity
        CPU core for output-handling thread.
    dst_comp_vector
        Completion vector for transmission, or -1 for polling.
        See :class:`spead2.send.UdpIbvConfig` for further information.
    tx_enabled
        Start with correlator output transmission enabled, without having to
        issue a katcp command.
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
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        sample_bits: int,
        heap_accumulation_threshold: int,
        sync_epoch: float,
        channel_offset_value: int,
        outputs: list[Output],
        src: list[tuple[str, int]],  # It's a list but it should be length 1 in xbgpu case.
        src_interface: str,
        src_ibv: bool,
        src_affinity: int,
        src_comp_vector: int,
        src_buffer: int,
        dst_interface: str,
        dst_ttl: int,
        dst_ibv: bool,
        dst_packet_payload: int,
        dst_affinity: int,
        dst_comp_vector: int,
        heaps_per_fengine_per_chunk: int,  # Used for GPU memory tuning
        rx_reorder_tol: int,
        tx_enabled: bool,
        monitor: Monitor,
        context: katsdpsigproc.abc.AbstractContext,
    ):
        super().__init__(katcp_host, katcp_port)
        self._cancel_tasks: list[asyncio.Task] = []  # Tasks that need to be cancelled on shutdown

        # B-engine doesn't work yet
        assert len(outputs) == 1
        assert isinstance(outputs[0], XOutput)

        if sample_bits != 8:
            raise ValueError("sample_bits must equal 8 - no other values supported at the moment.")

        if channel_offset_value % outputs[0].channels_per_substream != 0:
            raise ValueError("channel_offset must be an integer multiple of channels_per_substream")

        # Array configuration parameters
        self.adc_sample_rate_hz = adc_sample_rate_hz
        self.heap_accumulation_threshold = heap_accumulation_threshold
        self.time_converter = TimeConverter(sync_epoch, adc_sample_rate_hz)
        self.n_ants = outputs[0].antennas  # Still needed in sender_loop
        self.sample_bits = sample_bits
        self.channel_offset_value = channel_offset_value

        self._src = src
        self._src_interface = src_interface
        self._src_ibv = src_ibv
        self._src_buffer = src_buffer
        self._src_comp_vector = src_comp_vector

        self._init_tx_enabled = tx_enabled

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
        self.rx_heap_timestamp_step = n_samples_between_spectra * n_spectra_per_heap

        self.populate_sensors(
            self.sensors,
            max(
                RX_SENSOR_TIMEOUT_MIN,
                RX_SENSOR_TIMEOUT_CHUNKS
                * heaps_per_fengine_per_chunk
                * self.rx_heap_timestamp_step
                / adc_sample_rate_hz,
            ),
            (channel_offset_value, channel_offset_value + outputs[0].channels_per_substream),
        )

        self.timestamp_increment_per_accumulation = self.heap_accumulation_threshold * self.rx_heap_timestamp_step

        # Sets the number of batches of heaps to store per chunk
        self.heaps_per_fengine_per_chunk = heaps_per_fengine_per_chunk
        self.max_active_chunks: int = (
            math.ceil(rx_reorder_tol / self.rx_heap_timestamp_step / self.heaps_per_fengine_per_chunk) + 1
        )
        n_free_chunks: int = self.max_active_chunks + 8  # TODO: Abstract this 'naked' constant

        self.monitor = monitor

        data_ringbuffer = ChunkRingbuffer(
            self.max_active_chunks, name="recv_data_ringbuffer", task_name=RECV_TASK_NAME, monitor=monitor
        )
        free_ringbuffer = spead2.recv.ChunkRingbuffer(n_free_chunks)
        self._src_layout = recv.Layout(
            n_ants=outputs[0].antennas,
            n_channels_per_stream=outputs[0].channels_per_substream,
            n_spectra_per_heap=n_spectra_per_heap,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.heaps_per_fengine_per_chunk,
        )
        self.receiver_stream = recv.make_stream(
            layout=self._src_layout,
            data_ringbuffer=data_ringbuffer,
            free_ringbuffer=free_ringbuffer,
            src_affinity=src_affinity,
            max_active_chunks=self.max_active_chunks,
        )

        self.context = context

        # A command queue is the OpenCL name for a CUDA stream. An abstract
        # command queue can either be implemented as an OpenCL command queue or
        # a CUDA stream depending on the context.
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        correlation_template = CorrelationTemplate(
            self.context,
            n_ants=outputs[0].antennas,
            n_channels=outputs[0].channels_per_substream,
            n_spectra_per_heap=n_spectra_per_heap,
        )
        self.correlation = correlation_template.instantiate(
            self._proc_command_queue, n_batches=heaps_per_fengine_per_chunk
        )

        # These queues are extended in the monitor class, allowing for the
        # monitor to track the number of items on each queue.
        # * The _rx_item_queue passes items from the _receiver_loop function to
        #   the _gpu_proc_loop function.
        # * The _tx_item_queue passes items from the _gpu_proc_loop to the
        #   _sender_loop function.
        # Once the destination function is finished with an item, it will pass
        # it back to the corresponding _(rx/tx)_free_item_queue to ensure that
        # all allocated buffers are in continuous circulation.
        self._rx_item_queue: asyncio.Queue[RxQueueItem | None] = self.monitor.make_queue("rx_item_queue", n_rx_items)
        self._rx_free_item_queue: asyncio.Queue[RxQueueItem] = self.monitor.make_queue("rx_free_item_queue", n_rx_items)
        self._tx_item_queue: asyncio.Queue[TxQueueItem | None] = self.monitor.make_queue("tx_item_queue", n_tx_items)
        self._tx_free_item_queue: asyncio.Queue[TxQueueItem] = self.monitor.make_queue("tx_free_item_queue", n_tx_items)

        allocator = accel.DeviceAllocator(self.context)
        for _ in range(n_rx_items):
            buffer_device = self.correlation.slots["in_samples"].allocate(allocator, bind=False)
            present = np.zeros(shape=(self.heaps_per_fengine_per_chunk, outputs[0].antennas), dtype=np.uint8)
            rx_item = RxQueueItem(buffer_device, present)
            self._rx_free_item_queue.put_nowait(rx_item)

        for _ in range(n_tx_items):
            buffer_device = self.correlation.slots["out_visibilities"].allocate(allocator, bind=False)
            saturated = self.correlation.slots["out_saturated"].allocate(allocator, bind=False)
            present_ants = np.zeros(shape=(outputs[0].antennas,), dtype=bool)
            tx_item = TxQueueItem(buffer_device, saturated, present_ants)
            self._tx_free_item_queue.put_nowait(tx_item)

        for _ in range(n_free_chunks):
            buf = self.correlation.slots["in_samples"].allocate_host(self.context)
            present = np.zeros(outputs[0].antennas * self.heaps_per_fengine_per_chunk, np.uint8)
            chunk = recv.Chunk(data=buf, present=present, stream=self.receiver_stream)
            chunk.recycle()  # Make available to the stream

        # NOTE: This value staggers the send so that packets within a heap are
        # transmitted onto the network across the entire time between dumps.
        # Care needs to be taken to ensure that this rate is not set too high.
        # If it is too high, the entire pipeline will stall needlessly waiting
        # for packets to be transmitted too slowly.
        self.dump_interval_s: float = self.timestamp_increment_per_accumulation / adc_sample_rate_hz

        self.send_stream = XSend(
            n_ants=outputs[0].antennas,
            n_channels=outputs[0].channels,
            n_channels_per_stream=outputs[0].channels_per_substream,
            dump_interval_s=self.dump_interval_s,
            send_rate_factor=outputs[0].send_rate_factor,
            channel_offset=self.channel_offset_value,  # Arbitrary for now - depends on F-Engine stream
            context=self.context,
            packet_payload=dst_packet_payload,
            stream_factory=lambda stream_config, buffers: make_stream(
                dest_ip=outputs[0].dst[0].host,
                dest_port=outputs[0].dst[0].port,
                interface_ip=dst_interface,
                ttl=dst_ttl,
                use_ibv=dst_ibv,
                affinity=dst_affinity,
                comp_vector=dst_comp_vector,
                stream_config=stream_config,
                buffers=buffers,
            ),
            tx_enabled=self._init_tx_enabled,
        )

    def populate_sensors(
        self,
        sensors: aiokatcp.SensorSet,
        rx_sensor_timeout: float,
        chan_range: tuple[int, int],
    ) -> None:
        """Define the sensors for an XBEngine.

        Parameters
        ----------
        rx_sensor_timeout
            See :meth:`.recv.make_sensors` for more information.
        chan_range
            Tuple of integers showing the two values of the chan-range sensor
            - (channel_offset, channel_offset + channels_per_substream)
        """
        # Static sensors
        sensors.add(
            aiokatcp.Sensor(
                str,
                "chan-range",
                "The range of channels processed by this XB-engine, inclusive",
                default=f"({chan_range[0]},{chan_range[1] - 1})",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )
        # Dynamic sensors
        sensors.add(
            aiokatcp.Sensor(
                bool,
                "rx.synchronised",
                "For the latest accumulation, was data present from all F-Engines.",
                default=False,
                initial_status=aiokatcp.Sensor.Status.ERROR,
                status_func=lambda value: aiokatcp.Sensor.Status.NOMINAL if value else aiokatcp.Sensor.Status.ERROR,
            )
        )
        sensors.add(
            aiokatcp.Sensor(
                int,
                "xeng-clip-cnt",
                "Number of visibilities that saturated",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )

        for sensor in recv.make_sensors(rx_sensor_timeout).values():
            sensors.add(sensor)

        sensors.add(DeviceStatusSensor(sensors))

        time_sync_task = add_time_sync_sensors(sensors)
        self.add_service_task(time_sync_task)
        self._cancel_tasks.append(time_sync_task)

    async def _receiver_loop(self) -> None:
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
        async for chunk in recv.recv_chunks(
            self.receiver_stream,
            self._src_layout,
            self.sensors,
            self.time_converter,
        ):
            # Get a free rx_item that will contain the GPU buffer to which the
            # received chunk will be transferred.
            item = await self._rx_free_item_queue.get()
            item.chunk = chunk
            # Need a separate attribute as the chunk gets reset
            item.timestamp = chunk.timestamp
            # Need to reshape chunk.present to get Heaps in one dimension
            item.present[:] = chunk.present.reshape(item.present.shape)
            # Initiate transfer from received chunk to rx_item buffer.
            # First wait for asynchronous GPU work on the buffer.
            item.enqueue_wait_for_events(self._upload_command_queue)
            item.buffer_device.set_async(self._upload_command_queue, chunk.data)
            item.add_marker(self._upload_command_queue)

            # Give the received item to the _gpu_proc_loop function.
            await self._rx_item_queue.put(item)

        # spead2 will (eventually) indicate that there are no chunks to async-for through
        logger.debug("_receiver_loop completed")
        self._rx_item_queue.put_nowait(None)

    async def _flush_accumulation(self, tx_item: TxQueueItem, next_accum: int) -> TxQueueItem:
        """Emit the current `tx_item` and prepare a new one."""
        if tx_item.batches == 0:
            # We never actually started this accumulation. We can just
            # update the timestamp and continue using it.
            tx_item.timestamp = next_accum * self.timestamp_increment_per_accumulation
            return tx_item

        # present_ants only takes into account batches that have
        # been seen. If some batches went missing entirely, the
        # whole accumulation is bad.
        if tx_item.batches != self.heap_accumulation_threshold:
            tx_item.present_ants.fill(False)

        # Update the sync sensor (converting np.bool_ to Python bool)
        self.sensors["rx.synchronised"].value = bool(tx_item.present_ants.all())

        self.correlation.reduce()
        tx_item.add_marker(self._proc_command_queue)
        await self._tx_item_queue.put(tx_item)

        # Prepare for the next accumulation (which might not be
        # contiguous with the previous one).
        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()
        tx_item.timestamp = next_accum * self.timestamp_increment_per_accumulation
        self.correlation.bind(out_visibilities=tx_item.buffer_device, out_saturated=tx_item.saturated)
        self.correlation.zero_visibilities()
        return tx_item

    async def _gpu_proc_loop(self) -> None:
        """
        Perform all GPU processing of received data in a continuous loop.

        This function performs the following steps:
        1. Retrieve an rx_item from the _rx_item_queue
        2.1 Apply the correlation kernel to small subsets of the data
            until all the data has been processed.
        2.2 If sufficient correlations have occured, transfer the correlated
            data to a tx_item, pass the tx_item to the _tx_item_queue and get a
            new item from the _tx_free_item_queue.

        The ratio of rx_items to tx_items is not one to one. There are expected
        to be many more rx_items in for every tx_item out.

        The above steps are performed in a loop until the running flag is set
        to false.

        .. todo::

            Add B-Engine processing in this function.
        """

        def do_correlation() -> None:
            """Apply correlation kernel to all pending batches."""
            first_batch = self.correlation.first_batch
            last_batch = self.correlation.last_batch
            if first_batch < last_batch:
                self.correlation()
                # Update the present ants tracker one last time
                assert rx_item is not None
                tx_item.present_ants[:] &= rx_item.present[first_batch:last_batch, :].all(axis=0)
                tx_item.batches += last_batch - first_batch
                self.correlation.first_batch = last_batch

        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()

        # Indicate that the timestamp still needs to be filled in.
        tx_item.timestamp = -1
        self.correlation.bind(out_visibilities=tx_item.buffer_device, out_saturated=tx_item.saturated)
        self.correlation.zero_visibilities()
        while True:
            # Get item from the receiver function.
            # - Wait for the HtoD transfers to complete, then
            # - Give the chunk back to the receiver for reuse.
            rx_item = await self._rx_item_queue.get()
            if rx_item is None:
                break
            await rx_item.async_wait_for_events()
            assert rx_item.chunk is not None  # mypy doesn't like the fact that the chunk is "optional".
            rx_item.chunk.recycle()

            current_timestamp = rx_item.timestamp
            if tx_item.timestamp < 0:
                # First heap seen. Round the timestamp down to the previous
                # accumulation boundary
                tx_item.timestamp = (
                    current_timestamp
                    // self.timestamp_increment_per_accumulation
                    * self.timestamp_increment_per_accumulation
                )

            self.correlation.bind(in_samples=rx_item.buffer_device)
            # Initially no work to do; as each batch is examined, last_batch
            # is extended.
            self.correlation.first_batch = 0
            self.correlation.last_batch = 0
            for i in range(self.heaps_per_fengine_per_chunk):
                # NOTE: The timestamp representing the end of an
                # accumulation does not necessarily line up with the chunk
                # timestamp. It will line up with a specific batch within a
                # chunk though, this is why this check has to happen for each
                # batch. This check is the equivalent of the MeerKAT SKARAB
                # X-Engine auto-resync logic.
                current_accum = current_timestamp // self.timestamp_increment_per_accumulation
                tx_accum = tx_item.timestamp // self.timestamp_increment_per_accumulation
                if current_accum != tx_accum:
                    do_correlation()
                    tx_item = await self._flush_accumulation(tx_item, current_accum)
                self.correlation.last_batch = i + 1
                current_timestamp += self.rx_heap_timestamp_step

            do_correlation()
            # If the last batch of the chunk was also the last batch of the
            # accumulation, we can flush it now without waiting for more data.
            # This is mostly a convenience for unit tests, since in practice
            # we'd expect to see more data soon.
            current_accum = current_timestamp // self.timestamp_increment_per_accumulation
            tx_accum = tx_item.timestamp // self.timestamp_increment_per_accumulation
            if current_accum != tx_accum:
                tx_item = await self._flush_accumulation(tx_item, current_accum)

            rx_item.reset()
            rx_item.add_marker(self._proc_command_queue)
            await self._rx_free_item_queue.put(rx_item)

        # When the stream is closed, if the sender loop is waiting for a tx item,
        # it will never exit. Upon receiving this NoneType, the sender_loop can
        # stop waiting and exit.
        logger.debug("_gpu_proc_loop completed")
        self._tx_item_queue.put_nowait(None)

    async def _sender_loop(self) -> None:
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

            heap = await self.send_stream.get_free_heap()

            # NOTE: We do not expect the time between dumps to be the same each
            # time as the time.time() function checks the wall time now, not
            # the actual time between timestamps. The difference between dump
            # timestamps is expected to be constant.
            new_time_s = time.time()
            time_difference_between_heaps_s = new_time_s - old_time_s

            logger.debug(
                "Current output heap timestamp: %#x, difference between timestamps: %#x, "
                "wall time between dumps %.2f s",
                item.timestamp,
                item.timestamp - old_timestamp,
                time_difference_between_heaps_s,
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

            item.buffer_device.get_async(self._download_command_queue, heap.buffer)
            item.saturated.get_async(self._download_command_queue, heap.saturated)
            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            if not np.any(item.present_ants):
                # All Antennas have missed data at some point, mark the entire dump missing
                logger.warning("All Antennas had a break in data during this accumulation")
                heap.buffer[...] = MISSING
                incomplete_accum_counter.inc(1)
            elif not item.present_ants.all():
                affected_baselines = Correlation.get_baselines_for_missing_ants(item.present_ants, self.n_ants)
                for affected_baseline in affected_baselines:
                    # Multiply by four as each baseline (antenna pair) has four
                    # associated correlation components (polarisation pairs).
                    affected_baseline_index = affected_baseline * 4
                    heap.buffer[:, affected_baseline_index : affected_baseline_index + 4, :] = MISSING

                incomplete_accum_counter.inc(1)
            # else: No F-Engines had a break in data for this accumulation

            heap.timestamp = item.timestamp
            if self.send_stream.tx_enabled:
                # Convert timestamp for the *end* of the heap (not the start)
                # to a UNIX time for the sensor update. NB: this should be done
                # *before* send_heap, because that gives away ownership of the
                # heap.
                end_adc_timestamp = item.timestamp + self.timestamp_increment_per_accumulation
                end_timestamp = self.time_converter.adc_to_unix(end_adc_timestamp)
                clip_cnt_sensor = self.sensors["xeng-clip-cnt"]
                clip_cnt_sensor.set_value(clip_cnt_sensor.value + int(heap.saturated), timestamp=end_timestamp)
            self.send_stream.send_heap(heap)

            item.reset()
            await self._tx_free_item_queue.put(item)

        await self.send_stream.send_stop_heap()
        logger.debug("_sender_loop completed")

    async def request_capture_start(self, ctx) -> None:
        """Start transmission of this baseline-correlation-products stream."""
        self.send_stream.tx_enabled = True

    async def request_capture_stop(self, ctx) -> None:
        """Stop transmission of this baseline-correlation-products stream."""
        self.send_stream.tx_enabled = False

    async def start(self, descriptor_interval_s: float = SPEAD_DESCRIPTOR_INTERVAL_S) -> None:
        """
        Start the engine.

        This function adds the receive, processing and transmit tasks onto the
        event loop. It also adds a task to continuously send the descriptor
        heap at an interval indicated by `descriptor_interval_s`.

        These functions will loop forever and only exit once the XBEngine
        receives a SIGINT or SIGTERM.

        Parameters
        ----------
        descriptor_interval_s
            The interval used to dictate the 'engine sleep interval' between
            sending the data descriptor.
        """
        # Create the descriptor task first to ensure descriptor will be sent
        # before any data makes its way through the pipeline.
        descriptor_sender = DescriptorSender(
            self.send_stream.source_stream,
            self.send_stream.descriptor_heap,
            descriptor_interval_s,
        )
        descriptor_task = asyncio.create_task(descriptor_sender.run(), name=DESCRIPTOR_TASK_NAME)
        self.add_service_task(descriptor_task)
        self._cancel_tasks.append(descriptor_task)

        base_recv.add_reader(
            self.receiver_stream,
            src=self._src,
            interface=self._src_interface,
            ibv=self._src_ibv,
            comp_vector=self._src_comp_vector,
            buffer=self._src_buffer,
        )

        self.add_service_task(asyncio.create_task(self._receiver_loop(), name=RECV_TASK_NAME))
        self.add_service_task(asyncio.create_task(self._gpu_proc_loop(), name=GPU_PROC_TASK_NAME))
        self.add_service_task(asyncio.create_task(self._sender_loop(), name=SEND_TASK_NAME))

        await super().start()

    async def on_stop(self) -> None:
        """
        Shut down processing when the device server is stopped.

        This is called by aiokatcp after closing the listening socket.
        Also handle any Exceptions thrown unexpectedly in any of the
        processing loops.
        """
        for task in self._cancel_tasks:
            task.cancel()
        self.receiver_stream.stop()
        # If any of the tasks are already done then we had an exception, and
        # waiting for the rest may hang as the shutdown path won't proceed
        # neatly.
        if not any(task.done() for task in self.service_tasks):
            for task in self.service_tasks:
                if task not in self._cancel_tasks:
                    await task
