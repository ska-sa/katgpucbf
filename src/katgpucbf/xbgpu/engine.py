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
This module defines the objects that implement an entire GPU XB-Engine pipeline.

The XBEngine comprises multiple Pipeline objects that facilitate output data
products. Additionally, the RxQueueItem and TxQueueItem objects, used in the
XBEngine for passing information between different async processing loops,
are defined here.

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
from abc import abstractmethod

import aiokatcp
import katsdpsigproc
import katsdpsigproc.resource
import numpy as np
import spead2.recv
from aiokatcp import DeviceServer
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext

from .. import (
    COMPLEX,
    DESCRIPTOR_TASK_NAME,
    GPU_PROC_TASK_NAME,
    N_POLS,
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
from . import DEFAULT_BPIPELINE_NAME, DEFAULT_N_RX_ITEMS, DEFAULT_N_TX_ITEMS, DEFAULT_XPIPELINE_NAME, bsend, recv
from .bsend import make_stream as make_bstream
from .correlation import Correlation, CorrelationTemplate
from .output import BOutput, Output, XOutput
from .xsend import XSend, incomplete_accum_counter
from .xsend import make_stream as make_xstream

logger = logging.getLogger(__name__)
MISSING = np.array([-(2**31), 1], dtype=np.int32)


class RxQueueItem(QueueItem):
    """
    Extension of the QueueItem for use in receive queues.

    The RxQueueItem holds a reference to the received Chunk because its data is
    copied into the GPU buffer asynchronously. Once the GPU processing loop is
    done with the Chunk's data, it hands it back to the receiving loop to reuse
    the resource. It is for this reason that the Chunk's heap presence is
    stored separately.

    The RxQueueItem also holds a reference count of the number of pipelines still
    using this item. This is to ensure the item isn't freed before each pipeline
    has had a chance to use the received data.
    """

    def __init__(self, buffer_device: accel.DeviceArray, present: np.ndarray, timestamp: int = 0) -> None:
        self.buffer_device = buffer_device
        self.present = present
        self.refcount = 0
        super().__init__(timestamp=timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp and chunk."""
        super().reset(timestamp)
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
        """Reset the timestamp and present antenna tracker."""
        super().reset(timestamp=timestamp)
        self.present_ants.fill(True)  # Assume they're fine until told otherwise
        self.batches = 0


class Pipeline:
    r"""Base Pipeline class to build on.

    SPEAD heaps utilised by this pipeline are first received at the
    :class:`.XBEngine`-level and uploaded to the GPU for processing.

    The pipeline then performs GPU-accelerated processing on the uploaded
    data before packetising and sending the results back out onto the network.

    Data processing occurs across three separate async methods. Data is passed
    between these methods using :class:`asyncio.Queue`\s. See each method for
    more details.

    - :meth:`.XBEngine._receiver_loop`,
    - :meth:`.gpu_proc_loop`,
    - :meth:`.sender_loop`.

    Items passed between queues may still have GPU operations in progress.
    :class:`.QueueItem` provides mechanisms to wait for the in-progress work to
    complete.

    .. todo::
        Update the `output` parameter to be a list of `Output`s once multiple
        beams are supported. This will require some updates to the plumbing
        that uses `output` to e.g. capture-{start, stop}.

    Parameters
    ----------
    output
        Output config for data product (BOutput or XOutput).
    name
        Name of Pipeline.
    engine
        The owning engine.
    context
        CUDA context for device work.
    """

    # NOTE: n_rx_items and n_tx_items dictate the number of GPU buffers.
    # Setting these values too high results in too much GPU memory being
    # consumed. The quantity of each needs to be sufficient so as not to starve
    # the different processing loops. The low single digits is suitable.
    # These values are not configurable as they have been acceptable for
    # most tests cases up until now. If the pipeline starts bottlenecking,
    # then maybe look at increasing these values.
    n_rx_items = DEFAULT_N_RX_ITEMS
    n_tx_items = DEFAULT_N_TX_ITEMS

    def __init__(self, output: Output, name: str, engine: "XBEngine", context: AbstractContext) -> None:
        self.output = output
        self.name = name
        self.engine = engine

        self._proc_command_queue = context.create_command_queue()
        self._download_command_queue = context.create_command_queue()

        # These queues are extended in the monitor class, allowing for the
        # monitor to track the number of items on each queue.
        # - The _rx_item_queue receives items from :meth:`.XBEngine._receiver_loop`
        #   to be used by :meth:`_gpu_proc_loop`.
        # - The _tx_item_queue receives items from :meth:`gpu_proc_loop` to be
        #   used by :meth:`sender_loop`.
        # Once the destination function is finished with an item, it will pass
        # it back to the corresponding free-item queue to ensure that all
        # allocated buffers are in continuous circulation.
        # NOTE: Pipelines must not place :class:`RxQueueItem`s directly back on
        # the `_rx_free_item_queue` as multiple pipelines will hold
        # references to a single :class:`RxQueueItem`. Instead, invoke
        # :meth:`.XBEngine.free_rx_item` to indicate this Pipeline no longer
        # holds a reference to the item.
        self._rx_item_queue: asyncio.Queue[RxQueueItem | None] = engine.monitor.make_queue(
            f"{name}.rx_item_queue", self.n_rx_items
        )
        self._tx_item_queue: asyncio.Queue[TxQueueItem | None] = engine.monitor.make_queue(
            f"{name}.tx_item_queue", self.n_tx_items
        )
        self._tx_free_item_queue: asyncio.Queue[TxQueueItem] = engine.monitor.make_queue(
            f"{name}.tx_free_item_queue", self.n_tx_items
        )

    def add_rx_item(self, item: RxQueueItem) -> None:
        """Append a newly-received :class:`RxQueueItem` to the :attr:`_rx_item_queue`."""
        self._rx_item_queue.put_nowait(item)

    def shutdown(self) -> None:
        """Start a graceful shutdown after the final call to :meth:`add_rx_item`."""
        self._rx_item_queue.put_nowait(None)

    @abstractmethod
    async def gpu_proc_loop(self) -> None:
        """Perform all GPU processing of received data in a continuous loop.

        This method does the following:

        - Get an RxQueueItem off the rx_item_queue
        - Ensure it is not a NoneType value (indicating shutdown sequence)
        - await any outstanding events associated with the RxQueueItem
        - Apply GPU processing to data in the RxQueueItem

            - Bind input buffer(s) accordingly

        - Obtain a free TxQueueItem from the tx_free_item_queue

            - Add event marker to wait for the proc_command_queue
            - Put the prepared TxQueueItem on the tx_item_queue

        NOTE: An initial TxQueueItem needs to be obtained from the tx_free_item_queue
        for the first round of processing:

        - The gpu_proc_loop requires logic to decipher the timestamp of the
          first heap output.
        - It also provides an opportunity to bind buffers before processing is queued.
        """
        raise NotImplementedError  # pragma: nocover

    @abstractmethod
    async def sender_loop(self) -> None:
        """Send heaps to the network in a continuous loop.

        This method does the following:

        - Get a TxQueueItem from the tx_item_queue
        - Ensure it is not a NoneType value (indicating shutdown sequence)
        - Wait for events on the item to complete (likely GPU processing)
        - Wait for an available heap buffer from the send_stream
        - Asynchronously Transfer/Download GPU buffer data into the heap buffer in system RAM

            - Wait for the transfer to complete, before

        - Transmit heap buffer onto the network
        - Place the TxQueueItem back on the tx_free_item_queue once complete
        """
        raise NotImplementedError  # pragma: nocover

    @abstractmethod
    def capture_enable(self, enable: bool = True) -> None:
        """Enable/Disable the transmission of this data product's stream."""
        raise NotImplementedError  # pragma: nocover


class BPipeline(Pipeline):
    """Processing pipeline for a collection of :class:`Beam`s."""

    output: BOutput

    def __init__(
        self,
        output: BOutput,
        engine: "XBEngine",
        context: AbstractContext,
        init_tx_enabled: bool,
        name: str = DEFAULT_BPIPELINE_NAME,
    ) -> None:
        super().__init__(output, name, engine, context)
        self.n_beams = 1

        # TODO: Obtain this in a neater way once the beamformer OpSequence is done
        buffer_shape = (
            engine.heaps_per_fengine_per_chunk,
            engine.n_channels_per_substream,
            engine.src_layout.n_spectra_per_heap,
            COMPLEX,
        )
        for _ in range(self.n_tx_items):
            # TODO: Declare shapes, dtypes, etc
            buffer_device = accel.DeviceArray(context, shape=buffer_shape, dtype=bsend.SEND_DTYPE)
            saturated = accel.DeviceArray(context, shape=(), dtype=np.uint32)
            present_ants = np.zeros(shape=(engine.n_ants,), dtype=bool)
            tx_item = TxQueueItem(buffer_device, saturated, present_ants)
            self._tx_free_item_queue.put_nowait(tx_item)

        # TODO: Collate all output.dst addresses into one list to give to BSend
        # The order doesn't *matter*, it's more for peace of mind
        # To keep the output.name and output.dst in lock-step, probably best to
        # pass `outputs` to BSend constructor?
        # TODO: The way this is imported will change once the OperationSequence is in place
        self.send_stream = bsend.BSend(
            output,
            engine.heaps_per_fengine_per_chunk,
            self.n_tx_items,
            n_channels_per_substream=engine.n_channels_per_substream,
            spectra_per_heap=engine.src_layout.n_spectra_per_heap,
            timestamp_step=engine.rx_heap_timestamp_step,
            send_rate_factor=engine.send_rate_factor,
            channel_offset=engine.channel_offset_value,
            context=context,
            packet_payload=engine.dst_packet_payload,
            stream_factory=lambda stream_config, buffers: make_bstream(
                endpoints=[output.dst],
                interface=engine.dst_interface,
                ttl=engine.dst_ttl,
                use_ibv=engine.dst_ibv,
                affinity=engine.dst_affinity,
                comp_vector=engine.dst_comp_vector,
                stream_config=stream_config,
                buffers=buffers,
            ),
            tx_enabled=init_tx_enabled,
        )

    def capture_enable(self, enable: bool = True) -> None:  # noqa: D102
        # TODO: Maybe pass this off to BSend?
        # If enable is True, ensure that this `beam_name` is present
        # else: Remove the entry

        pass

    async def gpu_proc_loop(self) -> None:  # noqa: D102
        while True:
            # Get item from the receiver loop.
            # - Wait for the HtoD transfers to complete, then
            # - Give the chunk back to the receiver for reuse.
            rx_item = await self._rx_item_queue.get()
            if rx_item is None:
                break
            await rx_item.async_wait_for_events()

            tx_item = await self._tx_free_item_queue.get()
            await tx_item.async_wait_for_events()
            # TODO: Is it fine to update the timestamp like this?
            tx_item.reset(rx_item.timestamp)

            # Bind input buffers

            # Queue GPU work
            # - For now, just fill it with zeros
            tx_item.buffer_device.zero(self._proc_command_queue)
            tx_item.saturated.zero(self._proc_command_queue)

            # Bind output buffers
            tx_item.add_marker(self._proc_command_queue)
            self._tx_item_queue.put_nowait(tx_item)

            # Finish with the rx_item
            rx_item.add_marker(self._proc_command_queue)
            self.engine.free_rx_item(rx_item)
        # When the stream is closed, if the sender loop is waiting for a tx item,
        # it will never exit. Upon receiving this NoneType, the sender_loop can
        # stop waiting and exit.
        logger.debug("gpu_proc_loop completed")
        self._tx_item_queue.put_nowait(None)

    async def sender_loop(self) -> None:  # noqa: D102
        # NOTE: This function passes the entire downloaded data to
        # chunk.send, which then takes care of directing data to each beam's
        # output destination.

        # Get a populated TxQueueItem from the tx_item_queue
        while True:
            item = await self._tx_item_queue.get()
            if item is None:
                break
            # The CPU doesn't need to wait, but the GPU does to ensure it
            # won't start the download before computation is done.
            item.enqueue_wait_for_events(self._download_command_queue)

            # Get a free Chunk from the send_stream
            chunk = await self.send_stream.get_free_chunk()

            # Kick off any data downloads from the GPU
            item.buffer_device.get_async(self._download_command_queue, chunk.data)
            item.saturated.get_async(self._download_command_queue, chunk.saturated)

            # Wait for the transfer
            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            # Set the Chunk timestamp
            chunk.timestamp = item.timestamp
            if self.send_stream.tx_enabled:
                # Update beng-clip-cnt sensor
                # But fgpu does it all in chunk.send by passing it
                # - Send stream(s),
                # - Frames (int, to send)
                # - time_converter, to get `end_time` for setting clip-cnt sensor
                # - engine.sensors, to obtain the clip-cnt sensor
                # - output_name, because the Chunk didn't have access to it
                #   (to find sensor name).
                pass
            self.send_stream.send_chunk(chunk, self.engine.time_converter, self.engine.sensors)
            self._tx_free_item_queue.put_nowait(item)

        await self.send_stream.send_stop_heap()
        logger.debug("sender_loop completed")


class XPipeline(Pipeline):
    """Processing pipeline for a single baseline-correlation-products stream."""

    output: XOutput

    def __init__(
        self,
        output: XOutput,
        engine: "XBEngine",
        context: AbstractContext,
        init_tx_enabled: bool,
        name: str = DEFAULT_XPIPELINE_NAME,
    ) -> None:
        super().__init__(output, name, engine, context)
        self.timestamp_increment_per_accumulation = output.heap_accumulation_threshold * engine.rx_heap_timestamp_step

        # NOTE: This value staggers the send so that packets within a heap are
        # transmitted onto the network across the entire time between dumps.
        # Care needs to be taken to ensure that this rate is not set too high.
        # If it is too high, the entire pipeline will stall needlessly waiting
        # for packets to be transmitted too slowly.
        self.dump_interval_s = self.timestamp_increment_per_accumulation / engine.adc_sample_rate_hz

        correlation_template = CorrelationTemplate(
            context=context,
            n_ants=engine.n_ants,
            n_channels=engine.n_channels_per_substream,
            n_spectra_per_heap=engine.src_layout.n_spectra_per_heap,
        )
        self.correlation = correlation_template.instantiate(
            self._proc_command_queue, n_batches=engine.src_layout.heaps_per_fengine_per_chunk
        )

        allocator = accel.DeviceAllocator(context=context)
        for _ in range(self.n_tx_items):
            buffer_device = self.correlation.slots["out_visibilities"].allocate(allocator, bind=False)
            saturated = self.correlation.slots["out_saturated"].allocate(allocator, bind=False)
            present_ants = np.zeros(shape=(engine.n_ants,), dtype=bool)
            tx_item = TxQueueItem(buffer_device, saturated, present_ants)
            self._tx_free_item_queue.put_nowait(tx_item)

        self.send_stream = XSend(
            output_name=output.name,
            n_ants=engine.n_ants,
            n_channels=engine.n_channels_total,
            n_channels_per_substream=engine.n_channels_per_substream,
            dump_interval_s=self.dump_interval_s,
            send_rate_factor=engine.send_rate_factor,
            channel_offset=engine.channel_offset_value,
            context=context,
            packet_payload=engine.dst_packet_payload,
            stream_factory=lambda stream_config, buffers: make_xstream(
                output_name=output.name,
                dest_ip=output.dst.host,
                dest_port=output.dst.port,
                interface_ip=engine.dst_interface,
                ttl=engine.dst_ttl,
                use_ibv=engine.dst_ibv,
                affinity=engine.dst_affinity,
                comp_vector=engine.dst_comp_vector,
                stream_config=stream_config,
                buffers=buffers,
            ),
            tx_enabled=init_tx_enabled,
        )

        self._populate_sensors()

    def _populate_sensors(self) -> None:
        sensors = self.engine.sensors
        # Static sensors
        sensors.add(
            aiokatcp.Sensor(
                str,
                f"{self.output.name}.chan-range",
                "The range of channels processed by this X-engine, inclusive",
                default=f"({self.engine.channel_offset_value},"
                f"{self.engine.channel_offset_value + self.engine.n_channels_per_substream - 1})",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )
        # Dynamic sensors
        sensors.add(
            aiokatcp.Sensor(
                bool,
                f"{self.output.name}.rx.synchronised",
                "For the latest accumulation, was data present from all F-Engines.",
                default=False,
                initial_status=aiokatcp.Sensor.Status.ERROR,
                status_func=lambda value: aiokatcp.Sensor.Status.NOMINAL if value else aiokatcp.Sensor.Status.ERROR,
            )
        )
        sensors.add(
            aiokatcp.Sensor(
                int,
                f"{self.output.name}.xeng-clip-cnt",
                "Number of visibilities that saturated",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )

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
        if tx_item.batches != self.output.heap_accumulation_threshold:
            tx_item.present_ants.fill(False)

        # Update the sync sensor (converting np.bool_ to Python bool)
        self.engine.sensors[f"{self.output.name}.rx.synchronised"].value = bool(tx_item.present_ants.all())

        self.correlation.reduce()
        tx_item.add_marker(self._proc_command_queue)
        self._tx_item_queue.put_nowait(tx_item)

        # Prepare for the next accumulation (which might not be
        # contiguous with the previous one).
        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()
        tx_item.reset(next_accum * self.timestamp_increment_per_accumulation)
        self.correlation.bind(out_visibilities=tx_item.buffer_device, out_saturated=tx_item.saturated)
        self.correlation.zero_visibilities()
        return tx_item

    async def gpu_proc_loop(self) -> None:  # noqa: D102
        # NOTE: The ratio of rx_items to tx_items is not one-to-one; there are expected
        # to be many more rx_items in for every tx_item out. For this reason, and in
        # addition to the steps outlined in :meth:`.Pipeline.gpu_proc_loop`, data is
        # only transferred to a `TxQueueItem` once sufficient correlations have occurred.
        rx_item: RxQueueItem | None

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
            for i in range(self.engine.heaps_per_fengine_per_chunk):
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
                current_timestamp += self.engine.rx_heap_timestamp_step

            do_correlation()
            # If the last batch of the chunk was also the last batch of the
            # accumulation, we can flush it now without waiting for more data.
            # This is mostly a convenience for unit tests, since in practice
            # we'd expect to see more data soon.
            current_accum = current_timestamp // self.timestamp_increment_per_accumulation
            tx_accum = tx_item.timestamp // self.timestamp_increment_per_accumulation
            if current_accum != tx_accum:
                tx_item = await self._flush_accumulation(tx_item, current_accum)

            rx_item.add_marker(self._proc_command_queue)
            self.engine.free_rx_item(rx_item)
        # When the stream is closed, if the sender loop is waiting for a tx item,
        # it will never exit. Upon receiving this NoneType, the sender_loop can
        # stop waiting and exit.
        logger.debug("gpu_proc_loop completed")
        self._tx_item_queue.put_nowait(None)

    async def sender_loop(self) -> None:  # noqa: D102
        # NOTE: The transfer from the GPU to the heap buffer and the sending onto
        # the network could be pipelined a bit better, but this is not really
        # required in this loop as this whole process occurs at a much slower
        # pace than the rest of the pipeline.
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
                incomplete_accum_counter.labels(self.output.name).inc(1)
            elif not item.present_ants.all():
                affected_baselines = Correlation.get_baselines_for_missing_ants(item.present_ants, self.engine.n_ants)
                for affected_baseline in affected_baselines:
                    # Multiply by four as each baseline (antenna pair) has four
                    # associated correlation components (polarisation pairs).
                    affected_baseline_index = affected_baseline * 4
                    heap.buffer[:, affected_baseline_index : affected_baseline_index + 4, :] = MISSING

                incomplete_accum_counter.labels(self.output.name).inc(1)
            # else: No F-Engines had a break in data for this accumulation

            heap.timestamp = item.timestamp
            if self.send_stream.tx_enabled:
                # Convert timestamp for the *end* of the heap (not the start)
                # to a UNIX time for the sensor update. NB: this should be done
                # *before* send_heap, because that gives away ownership of the
                # heap.
                end_adc_timestamp = item.timestamp + self.timestamp_increment_per_accumulation
                end_timestamp = self.engine.time_converter.adc_to_unix(end_adc_timestamp)
                clip_cnt_sensor = self.engine.sensors[f"{self.output.name}.xeng-clip-cnt"]
                clip_cnt_sensor.set_value(clip_cnt_sensor.value + int(heap.saturated), timestamp=end_timestamp)
            self.send_stream.send_heap(heap)

            await self._tx_free_item_queue.put(item)

        await self.send_stream.send_stop_heap()
        logger.debug("sender_loop completed")

    def capture_enable(self, enable: bool = True) -> None:  # noqa: D102
        self.send_stream.tx_enabled = enable


class XBEngine(DeviceServer):
    r"""GPU XB-Engine pipeline.

    Currently the B-Engine functionality has not been added. This class currently
    only creates an X-Engine pipeline.

    The XB-Engine conducts the reception of SPEAD heaps from F-engines and makes
    the data available to the constituent pipelines. In order to reduce the load
    on the main thread, received data is collected into chunks. A chunk consists
    of multiple batches of F-Engine heaps where a batch is a collection of heaps
    from all F-Engine with the same timestamp.

    There is a seperate function for sending descriptors onto the network.

    Class initialisers allocate all memory buffers to be used during the lifetime
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
    n_channels_per_substream
        The number of frequency channels contained per substream.
    n_samples_between_spectra
        The number of samples between frequency spectra received.
    n_spectra_per_heap
        The number of time samples received per frequency channel.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    sync_epoch
        UNIX time corresponding to timestamp zero
    channel_offset_value
        The index of the first channel in the subset of channels processed by
        this XB-Engine. Used to set the value in the XB-Engine output heaps for
        spectrum reassembly by the downstream receiver.
    outputs
        Output streams to generate. Currently XOutputs and a single BOutput is
        supported.
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
        send_rate_factor: float,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        sample_bits: int,
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
        context: AbstractContext,
    ):
        super().__init__(katcp_host, katcp_port)
        self._cancel_tasks: list[asyncio.Task] = []  # Tasks that need to be cancelled on shutdown

        if sample_bits != 8:
            raise ValueError("sample_bits must equal 8 - no other values supported at the moment.")

        for output in outputs:
            if channel_offset_value % n_channels_per_substream != 0:
                raise ValueError(f"{output.name}: channel_offset must be an integer multiple of channels_per_substream")

        # Array configuration parameters
        self.adc_sample_rate_hz = adc_sample_rate_hz
        self.time_converter = TimeConverter(sync_epoch, adc_sample_rate_hz)
        self.n_ants = n_ants
        self.n_channels_total = n_channels_total
        self.n_channels_per_substream = n_channels_per_substream
        self.sample_bits = sample_bits
        self.channel_offset_value = channel_offset_value

        self._src = src
        self._src_interface = src_interface
        self._src_ibv = src_ibv
        self._src_buffer = src_buffer
        self._src_comp_vector = src_comp_vector

        self.dst_interface = dst_interface
        self.dst_ttl = dst_ttl
        self.dst_ibv = dst_ibv
        self.dst_packet_payload = dst_packet_payload
        self.dst_affinity = dst_affinity
        self.dst_comp_vector = dst_comp_vector
        self.send_rate_factor = send_rate_factor

        self.monitor = monitor

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
        )

        # Sets the number of batches of heaps to store per chunk
        self.heaps_per_fengine_per_chunk = heaps_per_fengine_per_chunk

        # NOTE: n_free_chunks dictates the number of buffers in system RAM.
        # This can be set quite high as there is much more system RAM than GPU
        # RAM. It should be higher than max_active_chunks.
        # These values are not configurable as they have been acceptable for
        # most tests cases up until now. If the pipeline starts bottlenecking,
        # then maybe look at increasing these values.
        self.max_active_chunks: int = (
            math.ceil(rx_reorder_tol / self.rx_heap_timestamp_step / self.heaps_per_fengine_per_chunk) + 1
        )
        n_free_chunks: int = self.max_active_chunks + 8  # TODO: Abstract this 'naked' constant

        data_ringbuffer = ChunkRingbuffer(
            self.max_active_chunks, name="recv_data_ringbuffer", task_name=RECV_TASK_NAME, monitor=monitor
        )
        free_ringbuffer = spead2.recv.ChunkRingbuffer(n_free_chunks)
        self.src_layout = recv.Layout(
            n_ants=n_ants,
            n_channels_per_substream=n_channels_per_substream,
            n_spectra_per_heap=n_spectra_per_heap,
            sample_bits=self.sample_bits,
            timestamp_step=self.rx_heap_timestamp_step,
            heaps_per_fengine_per_chunk=self.heaps_per_fengine_per_chunk,
        )
        self.receiver_stream = recv.make_stream(
            layout=self.src_layout,
            data_ringbuffer=data_ringbuffer,
            free_ringbuffer=free_ringbuffer,
            src_affinity=src_affinity,
            max_active_chunks=self.max_active_chunks,
        )

        # Prevent multiple chunks from being in flight in pipelines at the same
        # time. This keeps the pipelines synchronised to avoid running out of
        # RxQueueItems.
        self._active_in_sem = asyncio.BoundedSemaphore(1)

        self._pipelines: list[Pipeline]
        x_outputs = [output for output in outputs if isinstance(output, XOutput)]
        b_outputs = [output for output in outputs if isinstance(output, BOutput)]
        self._pipelines = [XPipeline(output, self, context, tx_enabled) for output in x_outputs]
        if len(b_outputs) == 1:
            # TODO: Update once more BOutput's are supported
            self._pipelines.append(BPipeline(b_outputs[0], self, context, tx_enabled))

        self._upload_command_queue = context.create_command_queue()

        # This queue is extended in the monitor class, allowing for the
        # monitor to track the number of items on the queue.
        # - The XBEngine passes items from the :meth:`_receiver_loop` to each
        #   pipeline via :meth:`.Pipeline.add_rx_item`.
        # - Once the each pipeline is finished with an :class:`RxQueueItem`,
        #   it must pass it back to the _rx_free_item_queue via
        #   :meth:`free_rx_item` to ensure that all allocated buffers are in
        #   continuous circulation.
        # NOTE: Too high means too much GPU memory gets allocate
        self._rx_free_item_queue: asyncio.Queue[RxQueueItem] = monitor.make_queue(
            "rx_free_item_queue", DEFAULT_N_RX_ITEMS
        )

        rx_data_shape = (
            heaps_per_fengine_per_chunk,
            n_ants,
            n_channels_per_substream,
            n_spectra_per_heap,
            N_POLS,
            COMPLEX,
        )
        for _ in range(DEFAULT_N_RX_ITEMS):
            # TODO: Abstract dtype, as per NGC-1000
            buffer_device = accel.DeviceArray(context, rx_data_shape, dtype=np.int8)
            present = np.zeros(shape=(self.heaps_per_fengine_per_chunk, n_ants), dtype=np.uint8)
            rx_item = RxQueueItem(buffer_device, present)
            self._rx_free_item_queue.put_nowait(rx_item)

        for _ in range(n_free_chunks):
            buf = buffer_device.empty_like()
            present = np.zeros(n_ants * self.heaps_per_fengine_per_chunk, np.uint8)
            chunk = recv.Chunk(data=buf, present=present, sink=self.receiver_stream)
            chunk.recycle()  # Make available to the stream

    def populate_sensors(self, sensors: aiokatcp.SensorSet, rx_sensor_timeout: float) -> None:
        """Define the sensors for an XBEngine."""
        # Dynamic sensors
        for sensor in recv.make_sensors(rx_sensor_timeout).values():
            sensors.add(sensor)

        sensors.add(DeviceStatusSensor(sensors))

        time_sync_task = add_time_sync_sensors(sensors)
        self.add_service_task(time_sync_task)
        self._cancel_tasks.append(time_sync_task)

    def free_rx_item(self, item: RxQueueItem) -> None:
        """Return an RxQueueItem to the free queue if its refcount hits zero."""
        item.refcount -= 1
        if item.refcount == 0:
            # All Pipelines are done with this item
            self._active_in_sem.release()
            # NOTE: Recycle the chunk only as resetting of the item is done
            # when it is taken off the queue.
            assert item.chunk is not None
            item.chunk.recycle()
            self._rx_free_item_queue.put_nowait(item)

    async def _add_rx_item(self, item: RxQueueItem) -> None:
        """Push an :class:`RxQueueItem` to all the pipelines."""
        await self._active_in_sem.acquire()
        item.refcount = len(self._pipelines)
        for pipeline in self._pipelines:
            pipeline.add_rx_item(item)

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
            self.src_layout,
            self.sensors,
            self.time_converter,
        ):
            # Get a free rx_item that will contain the GPU buffer to which the
            # received chunk will be transferred.
            item = await self._rx_free_item_queue.get()
            # First wait for asynchronous GPU work on the buffer.
            item.enqueue_wait_for_events(self._upload_command_queue)
            item.reset()

            # Now populate the fresh item
            item.chunk = chunk
            # Need a separate attribute as the chunk gets reset
            item.timestamp = chunk.timestamp
            # Need to reshape chunk.present to get Heaps in one dimension
            item.present[:] = chunk.present.reshape(item.present.shape)
            # Initiate transfer from received chunk to rx_item buffer.
            item.buffer_device.set_async(self._upload_command_queue, chunk.data)
            item.add_marker(self._upload_command_queue)

            # Give the received item to the pipelines' gpu_proc_loop.
            await self._add_rx_item(item)

        # spead2 will (eventually) indicate that there are no chunks to async-for through
        logger.debug("_receiver_loop completed")
        for pipeline in self._pipelines:
            pipeline.shutdown()

    def _request_pipeline(self, stream_name: str) -> Pipeline:
        """Get the pipeline related to the katcp request.

        Return the first Pipeline that matches by name.

        Raises
        ------
        FailReply
            If the `stream_name` is not a known output.
        """
        # TODO: Need to update this logic when working with multiple beams
        # - e.g. Make `_pipelines` a property, and return according to
        #   the types of Pipelines it's parsed.
        #   - e.g. BPipeline.output could be a property containing all
        #     BOutputs.
        for pipeline in self._pipelines:
            if stream_name == pipeline.output.name:
                return pipeline
        raise aiokatcp.FailReply(f"No output stream called {stream_name!r}")

    async def request_capture_start(self, ctx, stream_name: str) -> None:
        """Start transmission of stream.

        Parameters
        ----------
        stream_name
            Output stream name.
        """
        pipeline = self._request_pipeline(stream_name)
        pipeline.capture_enable()

    async def request_capture_stop(self, ctx, stream_name: str) -> None:
        """Stop transmission of a stream.

        Parameters
        ----------
        stream_name
            Output stream name.
        """
        pipeline = self._request_pipeline(stream_name)
        pipeline.capture_enable(enable=False)

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
        for pipeline in self._pipelines:
            # TODO: Make this parametrisable somehow for XPipeline vs BPipeline
            # BPipeline needs to stagger the sending of descriptors
            descriptor_sender = DescriptorSender(
                pipeline.send_stream.stream,  # type: ignore
                pipeline.send_stream.descriptor_heap,  # type: ignore
                descriptor_interval_s,
            )
            descriptor_task = asyncio.create_task(
                descriptor_sender.run(), name=f"{pipeline.output.name}.{DESCRIPTOR_TASK_NAME}"
            )
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

        for pipeline in self._pipelines:
            proc_task = asyncio.create_task(
                pipeline.gpu_proc_loop(),
                name=f"{pipeline.output.name}.{GPU_PROC_TASK_NAME}",
            )
            self.add_service_task(proc_task)

            send_task = asyncio.create_task(
                pipeline.sender_loop(),
                name=f"{pipeline.output.name}.{SEND_TASK_NAME}",
            )
            self.add_service_task(send_task)

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
