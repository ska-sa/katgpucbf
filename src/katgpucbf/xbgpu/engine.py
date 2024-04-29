################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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
"""

import asyncio
import logging
import math
import time
from abc import abstractmethod
from typing import Generic, Sequence, TypeVar

import aiokatcp
import katsdpsigproc
import katsdpsigproc.resource
import numpy as np
import spead2.recv
import vkgdr.pycuda
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
from ..mapped_array import MappedArray
from ..monitor import Monitor
from ..queue_item import QueueItem
from ..recv import RX_SENSOR_TIMEOUT_CHUNKS, RX_SENSOR_TIMEOUT_MIN
from ..ringbuffer import ChunkRingbuffer
from ..send import DescriptorSender
from ..utils import DeviceStatusSensor, TimeConverter, add_time_sync_sensors, steady_state_timestamp_sensor
from . import DEFAULT_BPIPELINE_NAME, DEFAULT_N_RX_ITEMS, DEFAULT_N_TX_ITEMS, DEFAULT_XPIPELINE_NAME, recv
from .beamform import BeamformTemplate
from .bsend import BSend
from .bsend import make_stream as make_bstream
from .correlation import CorrelationTemplate
from .output import BOutput, Output, XOutput
from .xsend import XSend, incomplete_accum_counter
from .xsend import make_stream as make_xstream
from .xsend import skipped_accum_counter

logger = logging.getLogger(__name__)
_O = TypeVar("_O", bound=Output)
_T = TypeVar("_T", bound=QueueItem)


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


class XTxQueueItem(QueueItem):
    """
    Extension of the QueueItem to track antennas that have missed data.

    The TxQueueItem between the gpu-proc and sender loops needs to carry a
    record of which antennas have missed data at any point in the accumulation
    being processed. This is used to determine whether any output products were
    affected, and have their data zeroed accordingly.
    """

    def __init__(
        self,
        buffer_device: accel.DeviceArray,
        saturated: accel.DeviceArray,
        present_ants: np.ndarray,
        present_baselines: MappedArray,
        timestamp: int = 0,
    ) -> None:
        self.buffer_device = buffer_device
        self.saturated = saturated
        self.present_ants = present_ants
        self.present_baselines = present_baselines
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp and present antenna tracker."""
        super().reset(timestamp=timestamp)
        self.present_ants.fill(True)  # Assume they're fine until told otherwise
        self.batches = 0

    def update_present_baselines(self) -> None:
        """Recompute present_baselines from present_ants."""
        # See Correlation.get_baseline_index for the ordering.
        # We do a column of the triangle at a time for efficiency.
        n_ants = len(self.present_ants)
        offset = 0
        for i in range(n_ants):
            section = self.present_baselines.host[offset : offset + i + 1]
            if self.present_ants[i]:
                section[:] = self.present_ants[: i + 1]
            else:
                section[:] = 0
            offset += len(section)


class BTxQueueItem(QueueItem):
    """Transmit queue item for the beamformer pipeline.

    The `out`, `saturated`, `rand_states`, `weights` and `delays` must have the
    same shape and dtype as the corresponding slots in :class:`.Beamform`. This
    class needs to carry a record of which antennas have missed data in the
    current :class:`~katgpucbf.recv.Chunk` of data. This is used to determine
    whether any beam data has been affected, and have their data flagged
    accordingly.

    Parameters
    ----------
    out
        An int8 type :class:`~katsdpsigproc.accel.DeviceArray` with shape
        (frames, ants, channels, spectra_per_frame, N_POLS, COMPLEX).
    saturated
        An uint32 type :class:`~katsdpsigproc.accel.DeviceArray` with shape
        (n_beams,).
    present
        A boolean array with shape (heaps_per_fengine_per_chunk, n_ants).
    weights
        An np.complex64 :class:`~katgpucbf.mapped_array.MappedArray` with
        shape (n_ants, n_beams).
    delays
        An np.float32 :class:`~katgpucbf.mapped_array.MappedArray` with shape
        (n_ants, n_beams).
    timestamp
        ADC sample count of the received Chunk.

    .. todo::

        Potentially create (yet another) base class for B- and XTxQueueItems.
    """

    def __init__(
        self,
        out: accel.DeviceArray,
        saturated: accel.DeviceArray,
        present: np.ndarray,
        weights: MappedArray,
        delays: MappedArray,
        timestamp: int = 0,
    ) -> None:
        self.out = out
        self.saturated = saturated
        self.present = present
        self.weights = weights
        self.delays = delays
        #: Version of weights and delays (for comparison to BPipeline._weights_version)
        self.weights_version = -1
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the timestamp and the present antenna tracker."""
        super().reset(timestamp=timestamp)
        self.present.fill(True)


class Pipeline(Generic[_O, _T]):
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

    Parameters
    ----------
    outputs
        Sequence of Output config for data product (BOutput or XOutput).
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

    def __init__(self, outputs: Sequence[_O], name: str, engine: "XBEngine", context: AbstractContext) -> None:
        self.outputs = outputs
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
        self._tx_item_queue: asyncio.Queue[_T | None] = engine.monitor.make_queue(
            f"{name}.tx_item_queue", self.n_tx_items
        )
        self._tx_free_item_queue: asyncio.Queue[_T] = engine.monitor.make_queue(
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

        - Obtain a free QueueItem from the tx_free_item_queue

            - Add event marker to wait for the proc_command_queue
            - Put the prepared QueueItem on the tx_item_queue

        NOTE: An initial QueueItem needs to be obtained from the tx_free_item_queue
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

        - Get a QueueItem from the tx_item_queue
        - Ensure it is not a NoneType value (indicating shutdown sequence)
        - Wait for events on the item to complete (likely GPU processing)
        - Wait for an available heap buffer from the send_stream
        - Asynchronously Transfer/Download GPU buffer data into the heap buffer in system RAM

            - Wait for the transfer to complete, before

        - Transmit heap buffer onto the network
        - Place the QueueItem back on the tx_free_item_queue once complete
        """
        raise NotImplementedError  # pragma: nocover

    @abstractmethod
    def capture_enable(self, *, stream_id: int, enable: bool = True) -> None:
        """Enable/Disable the transmission of a data product's stream."""
        raise NotImplementedError  # pragma: nocover


class BPipeline(Pipeline[BOutput, BTxQueueItem]):
    """Processing pipeline for a collection of :class:`.output.BOutput`."""

    def __init__(
        self,
        outputs: Sequence[BOutput],
        engine: "XBEngine",
        context: AbstractContext,
        vkgdr_handle: vkgdr.Vkgdr,
        init_tx_enabled: bool,
        name: str = DEFAULT_BPIPELINE_NAME,
    ) -> None:
        super().__init__(outputs, name, engine, context)

        template = BeamformTemplate(
            context,
            [output.pol for output in outputs],
            n_spectra_per_frame=engine.src_layout.n_spectra_per_heap,
        )
        self._beamform = template.instantiate(
            self._proc_command_queue,
            n_frames=engine.heaps_per_fengine_per_chunk,
            n_ants=engine.n_ants,
            n_channels=engine.n_channels_per_substream,
            seed=int(engine.time_converter.sync_time),
            sequence_first=engine.channel_offset_value,
            sequence_step=engine.n_channels_total,
        )
        allocator = accel.DeviceAllocator(context=context)
        for _ in range(self.n_tx_items):
            out = self._beamform.slots["out"].allocate(allocator=allocator, bind=False)
            saturated = self._beamform.slots["saturated"].allocate(allocator=allocator, bind=False)
            present = np.zeros(shape=(engine.heaps_per_fengine_per_chunk, engine.n_ants), dtype=bool)
            weights = MappedArray.from_slot(vkgdr_handle, context, self._beamform.slots["weights"])
            delays = MappedArray.from_slot(vkgdr_handle, context, self._beamform.slots["delays"])
            tx_item = BTxQueueItem(out, saturated, present, weights, delays)
            self._tx_free_item_queue.put_nowait(tx_item)

        # These are the original weights, delays and gains as provided by the
        # user, rather than the processed values passed to the kernel.
        self._weights = np.ones((len(outputs), engine.n_ants), np.float64)
        self._delays = np.zeros((len(outputs), engine.n_ants, 2), np.float64)
        self._quant_gains = np.ones(len(outputs), np.float64)
        self._weights_version = 1
        # Timestamp which will include effect of a weights update made now
        self._weights_steady = 0

        self.send_stream = BSend(
            outputs=outputs,
            frames_per_chunk=engine.heaps_per_fengine_per_chunk,
            n_tx_items=self.n_tx_items,
            n_channels=engine.n_channels_total,
            n_channels_per_substream=engine.n_channels_per_substream,
            spectra_per_heap=engine.src_layout.n_spectra_per_heap,
            adc_sample_rate=engine.adc_sample_rate_hz,
            timestamp_step=engine.rx_heap_timestamp_step,
            send_rate_factor=engine.send_rate_factor,
            channel_offset=engine.channel_offset_value,
            context=context,
            packet_payload=engine.dst_packet_payload,
            stream_factory=lambda stream_config, buffers: make_bstream(
                output_names=[output.name for output in outputs],
                endpoints=[output.dst for output in outputs],
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

        self._populate_sensors()

    def _populate_sensors(self) -> None:
        sensors = self.engine.sensors
        for i, output in enumerate(self.outputs):
            # Static sensors
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"{output.name}.chan-range",
                    "The range of channels processed by this B-engine, inclusive",
                    default=f"({self.engine.channel_offset_value},"
                    f"{self.engine.channel_offset_value + self.engine.n_channels_per_substream - 1})",
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            # Dynamic sensors
            default_delays_str = ", ".join(str(value) for value in self._delays[i].flatten())
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"{output.name}.delay",
                    "The delay settings of the inputs for this beam. Each input has "
                    "a delay [s] and phase [rad]: (loadmcnt, delay0, phase0, delay1, "
                    "phase1, ...)",
                    default=f"(0, {default_delays_str})",
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    float,
                    f"{output.name}.quantiser-gain",
                    "Non-complex post-summation quantiser gain applied to this beam",
                    default=self._quant_gains[i],
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"{output.name}.weight",
                    "The summing weights applied to all the inputs of this beam",
                    # Cast to list first to add comma delimiter
                    default=str(list(self._weights[i])),
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    int,
                    f"{output.name}.beng-clip-cnt",
                    "Number of complex samples that saturated.",
                    default=0,
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )

    async def _get_rx_item(self) -> RxQueueItem | None:
        """Get the next :class:`RxQueueItem`.

        This is wrapped in a method so that it can be mocked.
        """
        return await self._rx_item_queue.get()

    async def gpu_proc_loop(self) -> None:  # noqa: D102
        while True:
            # Get item from the receiver loop.
            # - Wait for the HtoD transfers to complete, then
            # - Give the chunk back to the receiver for reuse.
            rx_item = await self._get_rx_item()
            if rx_item is None:
                break
            await rx_item.async_wait_for_events()

            tx_item = await self._tx_free_item_queue.get()
            await tx_item.async_wait_for_events()
            tx_item.reset(rx_item.timestamp)

            # After this point it's too late for set_weights etc to update
            # the weights for this timestamp.
            self._weights_steady = (
                rx_item.timestamp + self.engine.heaps_per_fengine_per_chunk * self.engine.rx_heap_timestamp_step
            )

            # Recompute the weights and delays if necessary
            if tx_item.weights_version != self._weights_version:
                channel_spacing = self.engine.bandwidth_hz / self.engine.n_channels_total
                # The user provides a fringe phase for the centre frequency. We
                # need to adjust that to the target fringe phase for the first
                # channel processed by this engine (channel_offset_value).
                centre_channel = self.engine.n_channels_total / 2
                fringe_scale = -2 * np.pi * channel_spacing * (self.engine.channel_offset_value - centre_channel)
                fringe_phase = self._delays.T[1] + fringe_scale * self._delays.T[0]
                fringe_rotator = np.exp(1j * fringe_phase)
                tx_item.weights.host[:] = self._weights.T * self._quant_gains * fringe_rotator
                # The factor of 2 combines with the factor of pi used by the
                # kernel to give a factor of 2pi, to convert cycles to radians.
                # The minus sign is because delaying the wave results in a
                # decrease in the phase at a fixed time.
                tx_item.delays.host[:] = -2 * channel_spacing * self._delays.T[0]
                tx_item.weights_version = self._weights_version

            # Queue GPU work
            tx_item.saturated.zero(self._proc_command_queue)
            self._beamform.bind(
                **{
                    "in": rx_item.buffer_device,
                    "out": tx_item.out,
                    "saturated": tx_item.saturated,
                    "weights": tx_item.weights.device,
                    "delays": tx_item.delays.device,
                }
            )
            self._beamform()

            tx_item.present[:] = rx_item.present

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
        while True:
            item = await self._tx_item_queue.get()
            if item is None:
                break
            # The CPU doesn't need to wait, but the GPU does to ensure it
            # won't start the download before computation is done.
            item.enqueue_wait_for_events(self._download_command_queue)

            chunk = await self.send_stream.get_free_chunk()

            item.out.get_async(self._download_command_queue, chunk.data)
            item.saturated.get_async(self._download_command_queue, chunk.saturated)

            event = self._download_command_queue.enqueue_marker()
            await katsdpsigproc.resource.async_wait_for_events([event])

            np.sum(item.present, axis=1, dtype=np.uint64, out=chunk.present_ants)
            chunk.timestamp = item.timestamp
            self.send_stream.send_chunk(chunk, self.engine.time_converter, self.engine.sensors)
            self._tx_free_item_queue.put_nowait(item)

        await self.send_stream.send_stop_heap()
        logger.debug("sender_loop completed")

    def capture_enable(self, *, stream_id: int, enable: bool = True) -> None:  # noqa: D102
        self.send_stream.enable_substream(stream_id=stream_id, enable=enable)

    def _weights_updated(self) -> None:
        """Update version tracking when weight-related parameters are updated."""
        self._weights_version += 1
        self.engine.update_steady_state_timestamp(self._weights_steady)

    def set_weights(self, stream_id: int, weights: np.ndarray) -> None:
        """Set the beam weights for one beam.

        Parameters
        ----------
        stream_id
            The index of the beam whose weights are being set.
        weights
            A 1D array containing real-valued weights (per input).
        """
        self._weights[stream_id] = weights
        self._weights_updated()

    def set_quant_gain(self, stream_id: int, gain: float) -> None:
        """Set the quantisation gain for one beam.

        Parameters
        ----------
        stream_id
            The index of the beam whose quantisation gain is being set.
        gain
            Real-valued quantisation gain.
        """
        self._quant_gains[stream_id] = gain
        self._weights_updated()

    def set_delays(self, stream_id: int, delays: np.ndarray) -> None:
        """Set the beam steering delays for one beam.

        Parameters
        ----------
        stream_id
            The index of the beam whose quantisation gain is being set.
        delays
            A 2D array of delay coefficients. The first axis corresponds to
            inputs. The second axis has length two, with the first element
            containing the delay in seconds, and the second containing a
            channel-independent phase rotation in radians.
        """
        self._delays[stream_id] = delays
        self._weights_updated()


class XPipeline(Pipeline[XOutput, XTxQueueItem]):
    """Processing pipeline for a single baseline-correlation-products stream."""

    def __init__(
        self,
        output: XOutput,
        engine: "XBEngine",
        context: AbstractContext,
        vkgdr_handle: vkgdr.Vkgdr,
        init_tx_enabled: bool,
        name: str = DEFAULT_XPIPELINE_NAME,
    ) -> None:
        super().__init__([output], name, engine, context)
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
            input_sample_bits=engine.sample_bits,
        )
        self.correlation = correlation_template.instantiate(
            self._proc_command_queue, n_batches=engine.src_layout.heaps_per_fengine_per_chunk
        )

        allocator = accel.DeviceAllocator(context=context)
        for _ in range(self.n_tx_items):
            buffer_device = self.correlation.slots["out_visibilities"].allocate(allocator, bind=False)
            saturated = self.correlation.slots["out_saturated"].allocate(allocator, bind=False)
            present_ants = np.zeros(shape=(engine.n_ants,), dtype=bool)
            present_baselines = MappedArray.from_slot(
                vkgdr_handle, context, self.correlation.slots["present_baselines"]
            )
            tx_item = XTxQueueItem(buffer_device, saturated, present_ants, present_baselines)
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

    @property
    def output(self) -> XOutput:
        """The single :class:`Output` produced by this pipeline."""
        return self.outputs[0]

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

    async def _flush_accumulation(self, tx_item: XTxQueueItem, next_accum: int) -> XTxQueueItem:
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

        tx_item.update_present_baselines()
        self.correlation.reduce()
        tx_item.add_marker(self._proc_command_queue)
        self._tx_item_queue.put_nowait(tx_item)

        # Prepare for the next accumulation (which might not be
        # contiguous with the previous one).
        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()
        tx_item.reset(next_accum * self.timestamp_increment_per_accumulation)
        self.correlation.bind(
            out_visibilities=tx_item.buffer_device,
            out_saturated=tx_item.saturated,
            present_baselines=tx_item.present_baselines.device,
        )
        self.correlation.zero_visibilities()
        return tx_item

    async def gpu_proc_loop(self) -> None:  # noqa: D102
        # NOTE: The ratio of rx_items to tx_items is not one-to-one; there are expected
        # to be many more rx_items in for every tx_item out. For this reason, and in
        # addition to the steps outlined in :meth:`.Pipeline.gpu_proc_loop`, data is
        # only transferred to a `XTxQueueItem` once sufficient correlations have occurred.
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
                # TODO: NGC-1308 Update the usage of tx_item.batches to check
                # against rx_item.present, i.e. whether it's actually received
                # any data for this batch.
                tx_item.batches += last_batch - first_batch
                self.correlation.first_batch = last_batch

        tx_item = await self._tx_free_item_queue.get()
        await tx_item.async_wait_for_events()

        # Indicate that the timestamp still needs to be filled in.
        tx_item.timestamp = -1
        self.correlation.bind(
            out_visibilities=tx_item.buffer_device,
            out_saturated=tx_item.saturated,
            present_baselines=tx_item.present_baselines.device,
        )
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

            if not np.any(item.present_ants):
                # All Antennas have missed data at some point, avoid sending altogether
                logger.warning("All Antennas had a break in data during this accumulation")
                skipped_accum_counter.labels(self.output.name).inc(1)
            else:
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

                if not np.all(item.present_ants):
                    incomplete_accum_counter.labels(self.output.name).inc(1)
                    if not np.any(item.present_ants):
                        logger.warning("All Antennas had a break in data during this accumulation")

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

    def capture_enable(self, *, stream_id: int, enable: bool = True) -> None:  # noqa: D102
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
    bandwidth_hz
        Total bandwidth across `n_channels_total` channels. This is used in
        delay calculations.
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
    sync_time
        UNIX time corresponding to timestamp zero
    channel_offset_value
        The index of the first channel in the subset of channels processed by
        this XB-Engine. Used to set the value in the XB-Engine output heaps for
        spectrum reassembly by the downstream receiver.
    outputs
        Output streams to generate.
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
        bandwidth_hz: float,
        send_rate_factor: float,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        sample_bits: int,
        sync_time: float,
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

        for output in outputs:
            if channel_offset_value % n_channels_per_substream != 0:
                raise ValueError(f"{output.name}: channel_offset must be an integer multiple of channels_per_substream")

        # Array configuration parameters
        self.adc_sample_rate_hz = adc_sample_rate_hz
        self.bandwidth_hz = bandwidth_hz
        self.time_converter = TimeConverter(sync_time, adc_sample_rate_hz)
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

        self.n_samples_between_spectra = n_samples_between_spectra
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

        with context:
            # We could quite easily make do with non-coherent mappings and
            # explicit flushing, but since NVIDIA currently only provides
            # host-coherent memory, this is a simpler option.
            vkgdr_handle = vkgdr.Vkgdr.open_current_context(vkgdr.OpenFlags.REQUIRE_COHERENT_BIT)

        self._pipelines: list[Pipeline] = []
        x_outputs = [output for output in outputs if isinstance(output, XOutput)]
        b_outputs = [output for output in outputs if isinstance(output, BOutput)]
        self._pipelines = [XPipeline(x_output, self, context, vkgdr_handle, tx_enabled) for x_output in x_outputs]
        if b_outputs:
            self._pipelines.append(BPipeline(b_outputs, self, context, vkgdr_handle, tx_enabled))
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
            # TODO: NGC-1106 update buffer_device dtype once 4-bit mode is supported
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
        sensors.add(steady_state_timestamp_sensor())
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

    def update_steady_state_timestamp(self, timestamp: int) -> None:
        """Update ``steady-state-timestamp`` sensor to at least `timestamp`."""
        sensor = self.sensors["steady-state-timestamp"]
        sensor.value = max(sensor.value, timestamp)

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
            # Zero data affected by missing antennas
            # TODO: NGC-1311 Update this once the region-based zeroing
            # feature is implemented in katsdpsigproc.
            for heap in range(self.heaps_per_fengine_per_chunk):
                for antenna in range(self.n_ants):
                    if not item.present[heap, antenna]:
                        chunk.data[heap, antenna, ...] = 0
            # Initiate transfer from received chunk to rx_item buffer.
            item.buffer_device.set_async(self._upload_command_queue, chunk.data)
            item.add_marker(self._upload_command_queue)

            # Give the received item to the pipelines' gpu_proc_loop.
            await self._add_rx_item(item)

        # spead2 will (eventually) indicate that there are no chunks to async-for through
        logger.debug("_receiver_loop completed")
        for pipeline in self._pipelines:
            pipeline.shutdown()

    def _request_pipeline(self, stream_name: str) -> tuple[Pipeline, int]:
        """Get the pipeline related to the katcp request.

        Return the first Pipeline that matches by name, as well as the index
        of the data-stream it transmits.

        Raises
        ------
        FailReply
            If the `stream_name` is not a known output.
        """
        for pipeline in self._pipelines:
            for i, output in enumerate(pipeline.outputs):
                if stream_name == output.name:
                    return pipeline, i
        raise aiokatcp.FailReply(f"No output stream called {stream_name!r}")

    def _request_bpipeline(self, stream_name: str) -> tuple[BPipeline, int]:
        """Get the :class:`BPipeline` related to the katcp request.

        This wraps :meth:`_request_bpipeline` to check that the requested
        stream is a tied-array-channelised-voltage stream.
        """
        pipeline, stream_id = self._request_pipeline(stream_name)
        if not isinstance(pipeline, BPipeline):
            raise aiokatcp.FailReply(f"Output {stream_name!r} is not a tied-array-channelised-voltage stream")
        return pipeline, stream_id

    async def request_capture_start(self, ctx, stream_name: str) -> None:
        """Start transmission of stream.

        Parameters
        ----------
        stream_name
            Output stream name.
        """
        pipeline, stream_id = self._request_pipeline(stream_name)
        pipeline.capture_enable(stream_id=stream_id)

    async def request_capture_stop(self, ctx, stream_name: str) -> None:
        """Stop transmission of a stream.

        Parameters
        ----------
        stream_name
            Output stream name.
        """
        pipeline, stream_id = self._request_pipeline(stream_name)
        pipeline.capture_enable(stream_id=stream_id, enable=False)

    async def request_beam_weights(self, ctx, stream_name: str, *weights: float) -> None:
        """Set the weights for all inputs of a given beam and update the sensor.

        Parameters
        ----------
        stream_name
            The beam to modify
        weights
            A sequence of real floating-point values (one per input).
        """
        pipeline, stream_id = self._request_bpipeline(stream_name)
        if len(weights) != self.n_ants:
            raise aiokatcp.FailReply(f"Incorrect number of weights (expected {self.n_ants}, received {len(weights)})")
        pipeline.set_weights(stream_id, np.array(weights))
        self.sensors[f"{stream_name}.weight"].set_value(str(list(weights)))

    async def request_beam_delays(self, ctx, stream_name: str, *delays: str) -> None:
        """Set the delays for all inputs of a given beam and update the sensor.

        Parameters
        ----------
        stream_name
            The beam to modify
        delays
            A sequence of strings (one per input). Each string has the form
            ``delay:fringe-offset``, where ``delay`` is a delay in seconds, and
            ``fringe-offset`` is the net phase adjustment at the centre
            frequency (of the whole stream, not of this engine).
        """
        pipeline, stream_id = self._request_bpipeline(stream_name)
        if len(delays) != self.n_ants:
            raise aiokatcp.FailReply(f"Incorrect number of delays (expected {self.n_ants}, received {len(delays)})")
        new_delays = np.empty((self.n_ants, 2), np.float64)
        for i, entry in enumerate(delays):
            delay_str, phase_str = entry.split(":")
            new_delays[i, 0] = float(delay_str)
            new_delays[i, 1] = float(phase_str)
        pipeline.set_delays(stream_id, new_delays)
        delays_formatted_str = ", ".join(str(value) for value in new_delays.flatten())
        self.sensors[f"{stream_name}.delay"].set_value(f"({pipeline._weights_steady}, {delays_formatted_str})")

    async def request_beam_quant_gains(self, ctx, stream_name: str, gain: float) -> None:
        """Set the quantisation gain for a beam and update the sensor.

        Parameters
        ----------
        stream_name
            The beam to modify.
        gain
            The new gain to apply.
        """
        pipeline, stream_id = self._request_bpipeline(stream_name)
        pipeline.set_quant_gain(stream_id, gain)
        self.sensors[f"{stream_name}.quantiser-gain"].set_value(gain)

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
                substreams=range(pipeline.send_stream.stream.num_substreams),  # type: ignore
            )
            descriptor_task = asyncio.create_task(
                descriptor_sender.run(), name=f"{pipeline.name}.{DESCRIPTOR_TASK_NAME}"
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
                name=f"{pipeline.name}.{GPU_PROC_TASK_NAME}",
            )
            self.add_service_task(proc_task)

            send_task = asyncio.create_task(
                pipeline.sender_loop(),
                name=f"{pipeline.name}.{SEND_TASK_NAME}",
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
