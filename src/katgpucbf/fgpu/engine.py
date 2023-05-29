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

"""Engine class, which combines all the processing steps for a single digitiser data stream."""

import asyncio
import logging
import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import aiokatcp
import katsdpsigproc.accel as accel
import numpy as np
import scipy.signal
import spead2.recv
import vkgdr.pycuda
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext, AbstractEvent
from katsdpsigproc.resource import async_wait_for_events

from .. import (
    BYTE_BITS,
    COMPLEX,
    DESCRIPTOR_TASK_NAME,
    GPU_PROC_TASK_NAME,
    MIN_SENSOR_UPDATE_PERIOD,
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
from . import DIG_RMS_DBFS_HIGH, DIG_RMS_DBFS_LOW, INPUT_CHUNK_PADDING, recv, send
from .compute import Compute, ComputeTemplate, NarrowbandConfig
from .delay import AbstractDelayModel, AlignedDelayModel, LinearDelayModel, MultiDelayModel, wrap_angle
from .output import NarrowbandOutput, Output, WidebandOutput

logger = logging.getLogger(__name__)


def _sample_models(
    delay_models: Iterable[AbstractDelayModel], start: int, stop: int, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call :meth:`.AbstractDelayModel.range` on multiple delay models and stack results."""
    # Each element of parts is a tuple of results from one delay model
    parts = [model.range(start, stop, step) for model in delay_models]
    # Transpose so that each element of groups is one result from all delay models
    return tuple(np.stack(group) for group in zip(*parts))  # type: ignore


def generate_pfb_weights(step: int, taps: int, w_cutoff: float) -> np.ndarray:
    """Generate Hann-window weights for the F-engine's PFB-FIR.

    The resulting weights are normalised such that the sum of
    squares is 1.

    Parameters
    ----------
    step
        Number of samples per spectrum.
    taps
        Number of taps in the PFB-FIR.
    w_cutoff
        Scaling factor for the width of the channel response.

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the weights for the PFB-FIR filters, as
        single-precision floats.
    """
    window_size = step * taps
    idx = np.arange(window_size)
    hann = np.square(np.sin(np.pi * idx / (window_size - 1)))
    sinc = np.sinc(w_cutoff * ((idx + 0.5) / step - taps / 2))
    weights = hann * sinc
    # Work around https://github.com/numpy/numpy/issues/21898
    weights /= np.sqrt(np.sum(np.square(weights)))  # type: ignore[misc]
    return weights.astype(np.float32)


def generate_ddc_weights(taps: int, subsampling: int, weight_pass: float) -> np.ndarray:
    """Generate equiripple filter weights for the narrowband low-pass filter.

    The filter is designed with the assumption that only the inner 50% of the
    band will be retained. The response in the outer 50% (and aliases
    thereof) are thus irrelevant.

    The resulting weights are normalised such that the sum of squares is 1.

    Parameters
    ----------
    taps
        Number of taps in the filter
    subsampling
        Subsampling factor for subsampling applied after filtering
    weight_pass
        Weight given to the passband in the filter design (relative to stopband
        weight of 1.0).
    """
    edges = [0, 0.25]
    desired = [1.0]
    weights = [weight_pass]
    for x in np.arange(0.75, 0.5 * subsampling):
        edges += [x, min(x + 0.5, 0.5 * subsampling)]
        desired.append(0.0)
        weights.append(1.0)
    coeff = scipy.signal.remez(taps, edges, desired, weights, fs=subsampling, maxiter=1000)
    coeff /= np.sqrt(np.sum(np.square(coeff)))
    return coeff.astype(np.float32)


@dataclass
class PolInItem:
    """Polarisation-specific elements of :class:`InItem`."""

    #: A device memory region for storing the raw samples.
    samples: accel.DeviceArray | None
    #: Bitmask indicating which packets were present in the chunk.
    present: np.ndarray
    #: Cumulative sum over :attr:`present`. It is up to the caller
    #: to compute it at the appropriate time.
    present_cumsum: np.ndarray
    #: Chunk to return to recv after processing (used with vkgdr only).
    chunk: recv.Chunk | None = None


class InItem(QueueItem):
    """Item for use in input queues.

    This Item references GPU memory regions for input samples from both
    polarisations, with metadata describing their dimensions (number of samples
    and bitwidth of samples) in addition to the features of :class:`QueueItem`.

    An example of usage is as follows:

    .. code-block:: python

        # In the receive function
        my_in_item.pol_data[pol].samples.set_region(...)  # start copying sample data to the GPU,
        my_in_item.add_marker(command_queue)
        self._in_queue.put_nowait(my_in_item)
        ...
        # in the processing function
        next_in_item = await self._in_queue.get() # get the item from the queue
        next_in_item.enqueue_wait_for_events(command_queue) # wait for its data to be completely copied
        ... # carry on executing kernels or whatever needs to be done with the data

    Parameters
    ----------
    context
        CUDA context in which to allocate memory.
    layout
        Layout of the source stream.
    n_samples
        Number of digitised samples to hold, per polarisation
    timestamp
        Timestamp of the oldest digitiser sample represented in the data.
    use_vkgdr
        Use vkgdr to write sample data directly to the GPU rather than staging in
        host memory.
    """

    #: Per-polarisation data
    pol_data: list[PolInItem]
    #: Number of samples in each :class:`~katsdpsigproc.accel.DeviceArray` in :attr:`PolInItem.samples`
    n_samples: int
    #: Bitwidth of the data in :attr:`PolInItem.samples`
    dig_sample_bits: int
    #: Number of pipelines still using this item
    refcount: int

    def __init__(
        self,
        context: AbstractContext,
        layout: recv.Layout,
        n_samples: int,
        timestamp: int = 0,
        *,
        use_vkgdr: bool = False,
    ) -> None:
        self.dig_sample_bits = layout.sample_bits
        self.pol_data = []
        present_size = accel.divup(n_samples, layout.heap_samples)
        data_size = n_samples * self.dig_sample_bits // BYTE_BITS
        for _pol in range(N_POLS):
            if use_vkgdr:
                # Memory belongs to the chunks, and we set samples when
                # initialising the item from the chunks.
                sample_data = None
            else:
                sample_data = accel.DeviceArray(
                    context, (data_size,), np.uint8, padded_shape=(data_size + INPUT_CHUNK_PADDING,)
                )
            self.pol_data.append(
                PolInItem(
                    samples=sample_data,
                    present=np.zeros(present_size, dtype=bool),
                    present_cumsum=np.zeros(present_size + 1, np.uint32),
                )
            )
        self.refcount = 0
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item.

        Zero the timestamp, empty the event list and set number of samples to
        zero.
        """
        super().reset(timestamp)
        self.n_samples = 0

    @property
    def capacity(self) -> int:  # noqa: D401
        """Memory capacity in samples.

        The amount of space allocated to each polarisation stored in
        :attr:`PolInData.samples`.
        """
        assert self.pol_data[0].samples is not None
        return self.pol_data[0].samples.shape[0] * BYTE_BITS // self.dig_sample_bits

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end (i.e. latest plus 1) timestamp of the item."""
        return self.timestamp + self.n_samples


@dataclass
class MappedArray:
    """An array visible in the address space of both the host and the device.

    The host view can be updated using normal numpy code, and the changes will
    be visible to subsequently enqueued kernels on the device.
    """

    host: np.ndarray
    device: accel.DeviceArray

    @classmethod
    def from_slot(cls, vkgdr_handle: vkgdr.Vkgdr, context: AbstractContext, slot: accel.IOSlotBase) -> "MappedArray":
        """Allocate a :class:`MappedArray` to match a slot.

        Parameters
        ----------
        vkgdr_handle
            Handle for allocating memory from vkgdr. It must be created from the same device
            as `context`.
        context
            CUDA context in which the device view will be used.
        slot
            Slot from which the dtype, shape and padded shape will be
            extracted. The parameter is annotated as
            :class:`~katsdpsigproc.accel.IOSlotBase` for convenience, but it
            must actually be an instance of :class:`~katsdpsigproc.accel.IOSlot`.
        """
        assert isinstance(slot, accel.IOSlot)
        padded_shape = slot.required_padded_shape()
        n_bytes = int(np.product(padded_shape)) * slot.dtype.itemsize
        with context:
            handle = vkgdr.pycuda.Memory(vkgdr_handle, n_bytes)
        # Slice out the shape from the padded shape
        index = tuple(slice(0, x) for x in slot.shape)
        host = np.asarray(handle).view(slot.dtype).reshape(padded_shape)[index]
        device = accel.DeviceArray(context, slot.shape, slot.dtype, padded_shape, raw=handle)
        return cls(host=host, device=device)


class OutItem(QueueItem):
    """Item for use in output queues.

    This Item references GPU memory regions for output spectra from both
    polarisations, with something about the fine delay, in addition to the
    features of :class:`QueueItem`.

    An example of usage is as follows:

    .. code-block:: python

        # In the processing function
        compute.run_some_dsp(my_out_item.spectra) # Run the DSP, whatever it is.
        my_out_item.add_marker(command_queue)
        self._out_queue.put_nowait(my_out_item)
        ...
        # in the transmit function
        next_out_item = await self._out_queue.get() # get the item from the queue
        next_out_item.enqueue_wait_for_events(download_queue) # wait for event indicating DSP is finished
        next_out_item.get_async(download_queue) # Start copying data back to the host
        ... # Normally you'd put a marker on the queue again so that you know when the
            # copy is finished, but this needn't be attached to the item unless
            # there's another queue afterwards.

    Parameters
    ----------
    compute
        F-engine Operation Sequence detailing the DSP happening on the data,
        including details for buffers, context, shapes, slots, etc.
    spectra_samples
        Number of ADC samples between spectra.
    timestamp
        Timestamp of the first spectrum in the `OutItem`.
    """

    #: Output data, a collection of spectra, arranged in memory by pol and by heap.
    spectra: accel.DeviceArray
    #: Output saturation count, per pol
    saturation: accel.DeviceArray
    #: Output sum of squared samples, per pol
    dig_total_power: list[accel.DeviceArray]
    #: Per-spectrum fine delays
    fine_delay: MappedArray
    #: Per-spectrum phase offsets
    phase: MappedArray
    #: Per-channel gains
    gains: MappedArray
    #: Gain version number matching `gains` (for comparison to Pipeline.gains_version)
    gains_version: int
    #: Bit-mask indicating which spectra contain valid data and should be transmitted.
    present: np.ndarray
    #: Number of spectra contained in :attr:`spectra`.
    n_spectra: int
    #: Number of ADC samples between spectra
    spectra_samples: int
    #: Corresponding chunk for transmission (only used in PeerDirect mode).
    chunk: send.Chunk | None = None

    def __init__(self, vkgdr_handle: vkgdr.Vkgdr, compute: Compute, spectra_samples: int, timestamp: int = 0) -> None:
        allocator = accel.DeviceAllocator(compute.template.context)
        self.spectra = compute.slots["out"].allocate(allocator, bind=False)
        self.saturated = compute.slots["saturated"].allocate(allocator, bind=False)
        if "dig_total_power0" in compute.slots:
            self.dig_total_power = [
                compute.slots[f"dig_total_power{pol}"].allocate(allocator, bind=False) for pol in range(N_POLS)
            ]
        else:
            self.dig_total_power = []
        context = compute.template.context
        self.fine_delay = MappedArray.from_slot(vkgdr_handle, context, compute.slots["fine_delay"])
        self.phase = MappedArray.from_slot(vkgdr_handle, context, compute.slots["phase"])
        self.gains = MappedArray.from_slot(vkgdr_handle, context, compute.slots["gains"])
        self.gains_version = -1
        self.present = np.zeros(self.fine_delay.host.shape[0], dtype=bool)
        self.spectra_samples = spectra_samples
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item.

        Zero the item's timestamp, empty the event list and set number of
        spectra to zero.

        This does *not* zero the dig_total_power counters. Use :meth:`reset_all`
        for that.
        """
        super().reset(timestamp)
        self.n_spectra = 0

    def reset_all(self, command_queue: AbstractCommandQueue, timestamp: int = 0) -> None:
        """Fully reset the item.

        In addition to the work done by :meth:`reset`, zero out GPU
        accumulators, using the given command queue. No events are added
        associated with this; it is assumed that the same command queue will
        be used to subsequently operate on the accumulators.
        """
        self.reset(timestamp)
        for buf in self.dig_total_power:
            buf.zero(command_queue)

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end timestamp of the item.

        Following Python's normal exclusive-end convention.
        """
        return self.timestamp + self.n_spectra * self.spectra_samples

    @property
    def channels(self) -> int:  # noqa: D401
        """Number of channels stored in the item."""
        return self.spectra.shape[1]

    @property
    def capacity(self) -> int:  # noqa: D401
        """Number of spectra stored in memory for each polarisation."""
        # PostProc's __init__ method gives this as (spectra // spectra_per_heap)*(spectra_per_heap), so
        # basically, the number of spectra.
        return self.spectra.shape[0] * self.spectra.shape[2]

    @property
    def pols(self) -> int:  # noqa: D401
        """Number of polarisations."""
        return self.spectra.shape[3]


def format_complex(value: numbers.Complex) -> str:
    """Format a complex number for a katcp request.

    The ICD specifies that complex numbers have the format real+imaginary j.
    Python's default formatting contains parentheses if real is non-zero, and
    omits the real part if it is zero. For a numpy value it also only includes
    enough significant figures for the dtype, which means that reading it back
    as a Python complex may not give exactly the same value.
    """
    return f"{value.real}{value.imag:+}j"


def dig_rms_dbfs_status(value: float) -> aiokatcp.Sensor.Status:
    """Compute status for dig-rms-dbfs sensor."""
    if DIG_RMS_DBFS_LOW <= value <= DIG_RMS_DBFS_HIGH:
        return aiokatcp.Sensor.Status.NOMINAL
    else:
        return aiokatcp.Sensor.Status.WARN


def _parse_gains(*values: str, channels: int, default_gain: complex | None) -> np.ndarray:
    """Parse the gains passed to :meth:`request-gain` or :meth:`request-gain-all`.

    If a single value is given it is expanded to a value per channel. If
    `default_gain` is not ``None``, the string "default" may be given to
    restore the default gains set via command line.

    The caller must ensure that `values` contains either 1 or `channels`
    items. :meth:`request_gain` handles the case where no values are given
    for querying existing gains.

    Failures are reported by raising an appropriate
    :exc:`aiokatcp.FailReply`.
    """
    if default_gain is not None and values == ("default",):
        gains = np.full(channels, default_gain, dtype=np.complex64)
    else:
        try:
            gains = np.array([complex(v) for v in values], dtype=np.complex64)
        except ValueError:
            raise aiokatcp.FailReply("invalid formatting of complex number")
    if not np.all(np.isfinite(gains)):
        raise aiokatcp.FailReply("non-finite gains are not permitted")
    if len(gains) == 1:
        gains = gains.repeat(channels)
    return gains


class Pipeline:
    """Processing pipeline for a single output stream.

    Parameters
    ----------
    output
        The output stream to produce
    engine
        The owning engine
    context
        CUDA context for device work
    dig_stats
        If true, this pipeline is responsible for producing the digitiser statistics
        such as dig-rms-dbfs.
    """

    def __init__(self, output: Output, engine: "Engine", context: AbstractContext, dig_stats: bool) -> None:
        assert isinstance(output, WidebandOutput) or not dig_stats, "wideband output required for digitiser stats"
        # Tuning knobs not exposed via arguments
        n_send = 4
        n_out = n_send if engine.use_peerdirect else 2

        self.engine = engine
        self.output = output
        self.dig_stats = dig_stats
        self._in_queue: asyncio.Queue[InItem | None] = engine.monitor.make_queue(f"{output.name}.in_queue", engine.n_in)
        self._out_queue: asyncio.Queue[OutItem | None] = engine.monitor.make_queue(f"{output.name}.out_queue", n_out)
        self._out_free_queue: asyncio.Queue[OutItem] = engine.monitor.make_queue(f"{output.name}.out_free_queue", n_out)
        self._send_free_queue: asyncio.Queue[send.Chunk] = engine.monitor.make_queue(
            f"{output.name}.send_free_queue", n_send
        )
        self._in_item: InItem | None = None

        # Initialise self._compute
        compute_queue = context.create_command_queue()
        self._download_queue = context.create_command_queue()
        if isinstance(output, NarrowbandOutput):
            narrowband_config = NarrowbandConfig(
                decimation=output.decimation,
                taps=output.ddc_taps,
                mix_frequency=-output.centre_frequency / engine.adc_sample_rate,
            )
        else:
            narrowband_config = None
        template = ComputeTemplate(
            context, output.taps, output.channels, engine.src_layout.sample_bits, narrowband=narrowband_config
        )
        self._compute = template.instantiate(compute_queue, engine.n_samples, self.spectra, engine.spectra_per_heap)
        device_pfb_weights = self._compute.slots["weights"].allocate(accel.DeviceAllocator(context))
        device_pfb_weights.set(
            compute_queue,
            generate_pfb_weights(output.spectra_samples // output.subsampling, output.taps, output.w_cutoff),
        )
        if isinstance(output, NarrowbandOutput):
            device_ddc_weights = self._compute.slots["ddc_weights"].allocate(accel.DeviceAllocator(context))
            device_ddc_weights.set(
                compute_queue,
                generate_ddc_weights(output.ddc_taps, output.subsampling, output.weight_pass),
            )

        # Initialize sending
        self._init_send(engine.use_peerdirect)
        self._out_item = self._out_free_queue.get_nowait()
        self._out_item.reset_all(compute_queue)

        # Initialize delays and gains
        self.delay_models: list[AlignedDelayModel[MultiDelayModel]] = []
        self.gains = np.zeros((output.channels, N_POLS), np.complex64)
        # A version number that is incremented every time the gains change
        self.gains_version = 0
        self._populate_sensors()
        self._init_delay_gain()

        self.descriptor_heap = send.make_descriptor_heap(
            channels_per_substream=output.channels // len(output.dst),
            spectra_per_heap=engine.spectra_per_heap,
        )

    def _populate_sensors(self) -> None:
        sensors = self.engine.sensors
        for pol in range(N_POLS):
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"{self.output.name}.input{pol}.eq",
                    "For this input, the complex, unitless, per-channel digital scaling factors "
                    "implemented prior to requantisation",
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"{self.output.name}.input{pol}.delay",
                    "The delay settings for this input: (loadmcnt <ADC sample "
                    "count when model was loaded>, delay <in seconds>, "
                    "delay-rate <unit-less or, seconds-per-second>, "
                    "phase <radians>, phase-rate <radians per second>).",
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    int,
                    f"{self.output.name}.input{pol}.feng-clip-cnt",
                    "Number of output samples that are saturated",
                    default=0,
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                    auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
                    auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
                )
            )

    def _init_delay_gain(self) -> None:
        """Initialise the delays and gains."""
        for pol in range(N_POLS):
            delay_sensor = self.engine.sensors[f"{self.output.name}.input{pol}.delay"]
            callback_func = partial(self.update_delay_sensor, delay_sensor=delay_sensor)
            delay_model = AlignedDelayModel(MultiDelayModel(callback_func), self.output.subsampling)
            self.delay_models.append(delay_model)

        for pol in range(N_POLS):
            self.set_gains(pol, np.full(self.output.channels, self.engine.default_gain, dtype=np.complex64))

    def _init_send(self, use_peerdirect: bool) -> None:
        """Initialise the send side of the pipeline."""
        send_chunks: list[send.Chunk] = []
        for _ in range(self._out_free_queue.maxsize):
            item = OutItem(self.engine.vkgdr_handle, self._compute, self.output.spectra_samples)
            if use_peerdirect:
                dev_buffer = item.spectra.buffer.gpudata.as_buffer(item.spectra.buffer.nbytes)
                # buf is structurally a numpy array, but the pointer in it is a CUDA
                # pointer and so actually trying to use it as such will cause a
                # segfault.
                buf = np.frombuffer(dev_buffer, dtype=item.spectra.dtype).reshape(item.spectra.shape)
                chunk = send.Chunk(
                    buf,
                    saturated=item.saturated.empty_like(),
                    n_substreams=len(self.output.dst),
                    feng_id=self.engine.feng_id,
                    spectra_samples=self.output.spectra_samples,
                )
                item.chunk = chunk
                send_chunks.append(chunk)
            self._out_free_queue.put_nowait(item)

        spectra = self._compute.spectra
        if not use_peerdirect:
            # When using PeerDirect, the chunks are created along with the items
            heaps = spectra // self.engine.spectra_per_heap
            send_shape = (heaps, self.output.channels, self.engine.spectra_per_heap, N_POLS, COMPLEX)
            for _ in range(self._send_free_queue.maxsize):
                send_chunks.append(
                    send.Chunk(
                        accel.HostArray(send_shape, send.SEND_DTYPE, context=self._compute.template.context),
                        accel.HostArray((heaps, N_POLS), np.uint32, context=self._compute.template.context),
                        n_substreams=len(self.output.dst),
                        feng_id=self.engine.feng_id,
                        spectra_samples=self.output.spectra_samples,
                    )
                )
                self._send_free_queue.put_nowait(send_chunks[-1])
        n_data_heaps = len(send_chunks) * self.spectra // self.engine.spectra_per_heap * len(self.output.dst)
        self._send_streams = self.engine.make_send_streams(self.output, n_data_heaps, send_chunks)

    @property
    def spectra(self) -> int:  # noqa: D401
        """Number of spectra per output chunk."""
        return self.engine.chunk_jones // self.output.channels

    def add_in_item(self, item: InItem) -> None:
        """Append a newly-received :class:`~.InItem`."""
        self._in_queue.put_nowait(item)

    def shutdown(self) -> None:
        """Start graceful shutdown after the final call to :meth:`add_in_item`."""
        self._in_queue.put_nowait(None)

    async def _fill_in(self) -> InItem | None:
        """Populate :attr:`_in_item` to continue processing.

        Retrieve the next :class:`InItem` from the queue if necessary. Returns the
        current :class:`InItem`, or ``None`` if there isn't one.
        """
        if self._in_item is None:
            with self.engine.monitor.with_state(f"{self.output.name}.run_processing", "wait in_queue"):
                self._in_item = await self._in_queue.get()
            if self._in_item is None:
                # shutdown was called, and there are no more items. Push the None
                # back into the queue so that if we call this method again we'll
                # see it again instead of blocking forever.
                self._in_queue.put_nowait(None)
            else:
                self._in_item.enqueue_wait_for_events(self._compute.command_queue)
                if isinstance(self.output, NarrowbandOutput):
                    samples = []
                    for pol_data in self._in_item.pol_data:
                        assert pol_data.samples is not None
                        samples.append(pol_data.samples)
                    self._compute.run_ddc(samples, self._in_item.timestamp)
                    self._in_item.add_marker(self._compute.command_queue)
        return self._in_item

    def _pop_in(self) -> None:
        """Remove the current InItem."""
        assert self._in_item is not None
        self.engine.free_in_item(self._in_item)
        self._in_item = None

    async def _next_out(self, new_timestamp: int) -> OutItem:
        """Grab the next free OutItem in the queue."""
        with self.engine.monitor.with_state(f"{self.output.name}.run_processing", "wait out_free_queue"):
            item = await self._out_free_queue.get()

        # Just make double-sure that all events associated with the item are past
        # and have already been waited for.
        assert not item.events
        item.reset_all(self._compute.command_queue, new_timestamp)
        return item

    async def _flush_out(self, new_timestamp: int) -> None:
        """Start the backend processing and prepare the data for transmission.

        Kick off the `run_backend()` processing, and put an event on the
        relevant command queue. This lets the next coroutine (run_transmit) know
        that the backend processing is finished, and the data can be transmitted
        out.

        Parameters
        ----------
        new_timestamp
            The timestamp that will immediately follow the current OutItem.
        """
        # Round down to a multiple of accs (don't send heap with partial
        # data).
        accs = self._out_item.n_spectra // self.engine.spectra_per_heap
        self._out_item.n_spectra = accs * self.engine.spectra_per_heap
        if self._out_item.n_spectra > 0:
            # Copy the gains to the device if they are out of date.
            if self._out_item.gains_version != self.gains_version:
                self._out_item.gains.host[:] = self.gains
                self._out_item.gains_version = self.gains_version
            # TODO: can limit postprocessing to the relevant range (the FFT
            # size is baked into the plan, so is harder to modify on the
            # fly). Without this, saturation counts can be wrong.
            self._compute.bind(
                fine_delay=self._out_item.fine_delay.device,
                phase=self._out_item.phase.device,
                gains=self._out_item.gains.device,
            )
            self._compute.run_backend(self._out_item.spectra, self._out_item.saturated)
            # Note: we also need to wait for any frontend calls because they
            # write directly to self._out_item.dig_total_power, but this
            # marker will take care of that too.
            self._out_item.add_marker(self._compute.command_queue)
            self._out_queue.put_nowait(self._out_item)
            # TODO: could set it to None, since we only need it when we're
            # ready to flush again?
            self._out_item = await self._next_out(new_timestamp)
        else:
            self._out_item.timestamp = new_timestamp

    async def run_processing(self) -> None:
        """Do the hard work of the F-engine.

        This function takes place entirely on the GPU. Coarse delay happens.
        Then a batch FFT operation is applied, and finally, fine-delay, phase
        correction, quantisation and corner-turn are performed.
        """
        # This is guaranteed by the way subsampling is defined; this assertion
        # is really as a reminder to the reader.
        assert self.output.spectra_samples % self.output.subsampling == 0
        while (in_item := await self._fill_in()) is not None:
            # TODO[nb]: add earlier checks to ensure this will be the case
            assert in_item.timestamp % self.output.subsampling == 0
            # If the input starts too late for the next expected timestamp,
            # we need to skip ahead to the next heap after the start, and
            # flush what we already have.
            start_timestamp = self._out_item.end_timestamp
            orig_start_timestamps = [model(start_timestamp)[0] for model in self.delay_models]
            if min(orig_start_timestamps) < in_item.timestamp:
                align = self.engine.spectra_per_heap * self.output.spectra_samples
                # This loop is needed because MultiDelayModel is not necessarily
                # monotonic, and so simply taking the larger of the two skip
                # results does not guarantee a suitable timestamp.
                while min(orig_start_timestamps) < in_item.timestamp:
                    start_timestamp = max(
                        model.skip(in_item.timestamp, start_timestamp + 1, align) for model in self.delay_models
                    )
                    orig_start_timestamps = [model(start_timestamp)[0] for model in self.delay_models]
                await self._flush_out(start_timestamp)
            # When we add new spectra they must follow contiguously from any
            # that we've already buffered.
            assert start_timestamp == self._out_item.end_timestamp

            # Compute the coarse delay for the first sample.
            # `orig_timestamp` is the timestamp of first sample from the input
            # to process in the PFB to produce the output spectrum with
            # `timestamp`. `offset` is the sample index corresponding to
            # `orig_timestamp` within the InItem.
            start_coarse_delays = [start_timestamp - orig_timestamp for orig_timestamp in orig_start_timestamps]
            offsets = [orig_timestamp - in_item.timestamp for orig_timestamp in orig_start_timestamps]
            # Convert from original samples to post-DDC samples
            pfb_offsets = [offset // self.output.subsampling for offset in offsets]

            # Identify a block of frontend work. We can grow it until
            # - we run out of the current input array;
            # - we fill up the output array; or
            # - the coarse delay changes.
            # We speculatively calculate delays until one of the first two is
            # met, then truncate if we observe a coarse delay change. Note:
            # max_end_in is computed assuming the coarse delay does not change.
            max_end_in = in_item.end_timestamp + min(start_coarse_delays) - self.output.window + 1
            max_end_out = self._out_item.timestamp + self._out_item.capacity * self.output.spectra_samples
            max_end = min(max_end_in, max_end_out)
            # Speculatively evaluate until one of the first two conditions is met
            timestamps = np.arange(start_timestamp, max_end, self.output.spectra_samples)
            orig_timestamps, fine_delays, phase = _sample_models(
                self.delay_models, start_timestamp, max_end, self.output.spectra_samples
            )
            # timestamps can be empty if we fast-forwarded the output right over the
            # end of the current input item, and it causes problems if we don't check
            # for it (argmax of an empty sequence).
            if timestamps.size:
                for pol in range(len(orig_timestamps)):
                    coarse_delays = timestamps - orig_timestamps[pol]
                    # Uses fact that argmax returns first maximum i.e. first true value
                    delay_change = int(np.argmax(coarse_delays != start_coarse_delays[pol]))
                    if coarse_delays[delay_change] != start_coarse_delays[pol]:
                        logger.debug(
                            "Coarse delay on pol %d changed from %d to %d at %d",
                            pol,
                            start_coarse_delays[pol],
                            coarse_delays[delay_change],
                            orig_timestamps[pol, delay_change],
                        )
                        timestamps = timestamps[:delay_change]
                        orig_timestamps = orig_timestamps[:, :delay_change]
                        fine_delays = fine_delays[:, :delay_change]
                        phase = phase[:, :delay_change]
                batch_spectra = orig_timestamps.shape[1]

                # Here we run the "frontend" which handles:
                # - 10-bit to float conversion (wideband only)
                # - Coarse delay
                # - The PFB-FIR.
                if batch_spectra > 0:
                    logging.debug("Processing %d spectra", batch_spectra)
                    out_slice = np.s_[self._out_item.n_spectra : self._out_item.n_spectra + batch_spectra]
                    # Convert units of fine delay from digitiser samples to phase slope
                    # across the band. Narrowband has `decimation` times less bandwidth,
                    # so the phase change across the band is that much less.
                    self._out_item.fine_delay.host[out_slice] = fine_delays.T / self.output.internal_decimation
                    # The phase is referenced to the centre frequency, but
                    # coarse delay in wideband affects the phase of the centre
                    # frequency. The centre frequency is 4 samples per cycle, so
                    # each sample shifts it by pi/2, and this needs to be
                    # corrected for. The 4-sample period also allows us to reduce
                    # the coarse delay mod 4, which bounds the magnitude of the
                    # correction and hence improves floating-point accuracy.
                    #
                    # In narrowband the centre frequency is shifted to DC by
                    # the mixer and no correction is needed.
                    phase = phase.T
                    if isinstance(self.output, WidebandOutput):
                        phase += 0.5 * np.pi * (np.array(start_coarse_delays) % 4)
                        phase = wrap_angle(phase)
                    # Divide by pi because the arguments of sincospif() used in the
                    # kernel are in radians/PI.
                    self._out_item.phase.host[out_slice] = phase / np.pi
                    samples = []
                    for pol_data in in_item.pol_data:
                        assert pol_data.samples is not None
                        samples.append(pol_data.samples)
                    if isinstance(self.output, NarrowbandOutput):
                        self._compute.run_narrowband_frontend(pfb_offsets, self._out_item.n_spectra, batch_spectra)
                    else:
                        self._compute.run_wideband_frontend(
                            samples,
                            self._out_item.dig_total_power,
                            pfb_offsets,
                            self._out_item.n_spectra,
                            batch_spectra,
                        )
                        # Only add the marker for wideband. In the narrowband case, only
                        # the DDC kernel depends on the digitiser samples.
                        in_item.add_marker(self._compute.command_queue)
                    self._out_item.n_spectra += batch_spectra
                    # Work out which output spectra contain missing data.
                    self._out_item.present[out_slice] = True
                    for pol, pol_data in enumerate(in_item.pol_data):
                        # Offset in the chunk of the first sample for each spectrum
                        first_offset = np.arange(
                            offsets[pol],
                            offsets[pol] + batch_spectra * self.output.spectra_samples,
                            self.output.spectra_samples,
                        )
                        # Offset of the last sample (inclusive, rather than past-the-end)
                        last_offset = first_offset + self.output.window - 1
                        first_packet = first_offset // self.engine.src_layout.heap_samples
                        # last_packet is exclusive
                        last_packet = last_offset // self.engine.src_layout.heap_samples + 1
                        present_packets = pol_data.present_cumsum[last_packet] - pol_data.present_cumsum[first_packet]
                        self._out_item.present[out_slice] &= present_packets == last_packet - first_packet

            # The _flush_out method calls the "backend" which triggers the FFT
            # and postproc operations.
            end_timestamp = self._out_item.end_timestamp
            if end_timestamp >= max_end_out:
                # We've filled up the output buffer.
                await self._flush_out(end_timestamp)

            if end_timestamp >= max_end_in:
                # We've exhausted the input buffer.
                self._pop_in()
        # Timestamp mostly doesn't matter because we're finished, but if a
        # katcp request arrives at this point we want to ensure the
        # steady-state-timestamp sensor is updated to a later timestamp than
        # anything we'll actually send.
        await self._flush_out(self._out_item.end_timestamp)
        logger.debug("run_processing completed")
        self._out_queue.put_nowait(None)

    async def _chunk_send_and_cleanup(
        self, streams: list["spead2.send.asyncio.AsyncStream"], n_frames: int, chunk: send.Chunk
    ) -> None:
        """Transmit a chunk's data and return it to the free queue.

        The returning of the chunk to the free queue happens whether the data
        transmission was successful or not.

        Parameters
        ----------
        streams
            The streams transmitting data.
        n_frames
            Number of frames of data to be transmitted.
        chunk
            :class:`~send.Chunk` used to facilitate data transmission.
        """
        try:
            await chunk.send(streams, n_frames, self.engine.time_converter, self.engine.sensors, self.output.name)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error sending chunk")
        finally:
            if chunk.cleanup is not None:
                chunk.cleanup()
                chunk.cleanup = None  # Potentially helps break reference cycles

    async def run_transmit(self) -> None:
        """Get the processed data from the GPU to the Network.

        This could be done either with or without PeerDirect. In the
        non-PeerDirect case, :class:`OutItem` objects are pulled from the
        `_out_queue`. We wait for the events that mark the end of the processing,
        then copy the data to host memory before turning it over to the
        :obj:`sender` for transmission on the network. The "empty" item is then
        returned to :meth:`run_processing` via the `_out_free_queue`, and once
        the chunk has been transmitted it is returned to `_send_free_queue`.

        In the PeerDirect case, the item and the chunk are bound together and
        share memory. In this case `_send_free_queue` is unused. The item is
        only returned to `_out_free_queue` once it has been fully transmitted.
        """
        task: asyncio.Future | None = None
        last_end_timestamp: int | None = None
        context = self._compute.template.context
        func_name = f"{self.output.name}.run_transmit"
        # Scratch space for transferring digitiser power
        if self.dig_stats:
            dig_total_power = [
                self._compute.slots[f"dig_total_power{pol}"].allocate_host(context) for pol in range(N_POLS)
            ]
        else:
            dig_total_power = []
        while True:
            with self.engine.monitor.with_state(func_name, "wait out_queue"):
                out_item = await self._out_queue.get()
            if not out_item:
                break
            events = []
            if out_item.chunk is not None:
                # We're using PeerDirect
                chunk = out_item.chunk
                chunk.cleanup = partial(self._out_free_queue.put_nowait, out_item)
                events.extend(out_item.events)
            else:
                with self.engine.monitor.with_state(func_name, "wait send_free_queue"):
                    chunk = await self._send_free_queue.get()
                chunk.cleanup = partial(self._send_free_queue.put_nowait, chunk)
                self._download_queue.enqueue_wait_for_events(out_item.events)
                assert isinstance(chunk.data, accel.HostArray)
                # TODO: use get_region since it might be partial
                out_item.spectra.get_async(self._download_queue, chunk.data)
            out_item.saturated.get_async(self._download_queue, chunk.saturated)
            for pol, trg in enumerate(dig_total_power):
                out_item.dig_total_power[pol].get_async(self._download_queue, trg)
            events.append(self._download_queue.enqueue_marker())

            chunk.timestamp = out_item.timestamp
            # Each frame is valid if all spectra in it are valid
            out_item.present.reshape(-1, self.engine.spectra_per_heap).all(axis=-1, out=chunk.present)
            with self.engine.monitor.with_state(func_name, "wait transfer"):
                await async_wait_for_events(events)

            for pol, trg in enumerate(dig_total_power):
                total_power = float(trg)
                avg_power = total_power / (out_item.n_spectra * self.output.spectra_samples)
                # Normalise relative to full scale. The factor of 2 is because we
                # want 1.0 to correspond to a sine wave rather than a square wave.
                avg_power /= ((1 << (self.engine.src_layout.sample_bits - 1)) - 1) ** 2 / 2
                avg_power_db = 10 * math.log10(avg_power) if avg_power else -math.inf
                self.engine.sensors[f"input{pol}.dig-rms-dbfs"].set_value(
                    avg_power_db, timestamp=self.engine.time_converter.adc_to_unix(out_item.end_timestamp)
                )

            n_frames = out_item.n_spectra // self.engine.spectra_per_heap
            if last_end_timestamp is not None and out_item.timestamp > last_end_timestamp:
                # Account for heaps skipped between the end of the previous out_item and the
                # start of the current one.
                skipped_samples = out_item.timestamp - last_end_timestamp
                skipped_frames = skipped_samples // (self.engine.spectra_per_heap * self.output.spectra_samples)
                send.skipped_heaps_counter.labels(self.output.name).inc(skipped_frames * len(self.output.dst))
            last_end_timestamp = out_item.end_timestamp
            out_item.reset()  # Safe to call in PeerDirect mode since it doesn't touch the raw data
            if out_item.chunk is None:
                # We're not in PeerDirect mode
                # (when we are the cleanup callback returns the item)
                self._out_free_queue.put_nowait(out_item)
            task = asyncio.create_task(
                self._chunk_send_and_cleanup(self._send_streams, n_frames, chunk),
                name="Chunk Send and Cleanup Task",
            )
            self.engine.add_service_task(task)

        if task:
            try:
                await task
            except Exception:
                pass  # It's already logged by _chunk_send_and_cleanup
        stop_heap = spead2.send.Heap(send.FLAVOUR)
        stop_heap.add_end()
        for substream_index in range(len(self.output.dst)):
            await self._send_streams[0].async_send_heap(stop_heap, substream_index=substream_index)
        logger.debug("run_transmit completed")

    def delay_update_timestamp(self) -> int:
        """Return a timestamp by which an update to the delay model will take effect."""
        # end_timestamp is updated whenever delays are written into the out_item
        return self._out_item.end_timestamp

    def update_delay_sensor(self, delay_models: Sequence[LinearDelayModel], *, delay_sensor: aiokatcp.Sensor) -> None:
        """Update the delay sensor upon loading of a new model.

        Accepting the delay_models as a read-only Sequence from the
        MultiDelayModel, even though we only need the first one to update
        the sensor.

        The delay and phase-rate values need to be scaled back to their
        original values (delay (s), phase-rate (rad/s)).
        """
        logger.debug("Updating delay sensor: %s", delay_sensor.name)

        orig_delay = delay_models[0].delay / self.engine.adc_sample_rate
        orig_phase = delay_models[0].phase
        orig_phase_rate = delay_models[0].phase_rate * self.engine.adc_sample_rate
        delay_sensor.value = (
            f"({delay_models[0].start}, "
            f"{orig_delay}, "
            f"{delay_models[0].delay_rate}, "
            f"{orig_phase}, "
            f"{orig_phase_rate})"
        )

    def set_gains(self, input: int, gains: np.ndarray) -> None:
        """Set the eq gains for one polarisation and update the sensor.

        The `gains` must contain one entry per channel; the shortcut of
        supplying a single value is handled by :meth:`request_gain`.
        """
        self.gains_version += 1
        self.gains[:, input] = gains
        # This timestamp is conservative: self._out_item.timestamp is almost
        # always valid, except while _flush_out is waiting to update
        # self._out_item. If a less conservative answer is needed, one would
        # need to track a separate timestamp in the class that is updated
        # as gains are copied to the OutItem.
        self.engine._update_steady_state_timestamp(self._out_item.end_timestamp)
        if np.all(gains == gains[0]):
            # All the values are the same, so it can be reported as a single value
            gains = gains[:1]
        sensor = self.engine.sensors[f"{self.output.name}.input{input}.eq"]
        sensor.value = "[" + ", ".join(format_complex(gain) for gain in gains) + "]"


class Engine(aiokatcp.DeviceServer):
    """Top-level class running the whole thing.

    .. todo::

      The :class:`Engine` needs to have more sensors and requests added to it,
      according to whatever the design is going to be. SKARAB didn't have katcp
      capability, that was all in corr2, so we need to figure out how to best
      control these engines. This docstring should also be updated to reflect
      its new nature as an inheritor of :class:`aiokatcp.DeviceServer`.

    Parameters
    ----------
    katcp_host
        Hostname or IP on which to listen for KATCP C&M connections.
    katcp_port
        Network port on which to listen for KATCP C&M connections.
    context
        The accelerator (OpenCL or CUDA) context to use for running the Engine.
    srcs
        A list of source endpoints for the incoming data.
    src_interface
        IP addresses of the network devices to use for input.
    src_ibv
        Use ibverbs for input.
    src_affinity
        List of CPU cores for input-handling threads. Must be one number per
        pol.
    src_comp_vector
        Completion vectors for source streams, or -1 for polling.
        See :class:`spead2.recv.UdpIbvConfig` for further information.
    src_packet_samples
        The number of samples per digitiser packet.
    src_buffer
        The size of the network receive buffer (per polarisation).
    dst_interface
        IP addresses of the network devices to use for output.
    dst_ttl
        TTL for outgoing packets.
    dst_ibv
        Use ibverbs for output.
    dst_packet_payload
        Size for output packets (voltage payload only, headers and padding are
        added to this).
    dst_affinity
        CPU core for output-handling thread.
    dst_comp_vector
        Completion vector for transmission, or -1 for polling.
        See :class:`spead2.send.UdpIbvConfig` for further information.
    outputs
        Output streams to generate. At present this must be a single
        WidebandOutput.
    adc_sample_rate
        Digitiser sampling rate (in Hz), used to determine transmission rate.
    send_rate_factor
        Configure the SPEAD2 sender with a rate proportional to this factor.
        This value is intended to dictate a data transmission rate slightly
        higher/faster than the ADC rate.
        NOTE:
        - A factor of zero (0) tells the sender to transmit as fast as possible.
    feng_id
        ID of the F-engine indicating which one in the array this is. Included
        in the output heaps so that the X-engine can determine where the data
        fits in.
    num_ants
        The number of antennas in the array. Used for numbering heaps so as
        not to collide with other antennas transmitting to the same X-engine.
    chunk_samples
        Number of samples in each input chunk, excluding padding samples.
    chunk_jones
        Number of Jones vectors in each output chunk.
    spectra_per_heap
        Number of spectra in each output heap.
    max_delay_diff
        Maximum supported difference between delays across polarisations (in samples).
    gain
        Initial eq gain for all channels.
    sync_epoch
        UNIX time at which the digitisers were synced.
    mask_timestamp
        Mask off bottom bits of timestamp (workaround for broken digitiser).
    use_vkgdr
        Assemble chunks directly in GPU memory (requires Vulkan).
    use_peerdirect
        Send chunks directly from GPU memory (requires supported GPU).
    monitor
        :class:`Monitor` to use for generating multiple :class:`~asyncio.Queue`
        objects needed to communicate between functions, and handling basic
        reporting for :class:`~asyncio.Queue` sizes and events.
    """

    # TODO: VERSION means interface version, rather than software version. It
    # will need to wait on a proper ICD for a release.
    VERSION = "katgpucbf-fgpu-icd-0.1"
    BUILD_STATE = __version__

    def __init__(
        self,
        *,
        katcp_host: str,
        katcp_port: int,
        context: AbstractContext,
        srcs: list[str | list[tuple[str, int]]],
        src_interface: list[str] | None,
        src_ibv: bool,
        src_affinity: list[int],
        src_comp_vector: list[int],
        src_packet_samples: int,
        src_buffer: int,
        dst_interface: list[str],
        dst_ttl: int,
        dst_ibv: bool,
        dst_packet_payload: int,
        dst_affinity: int,
        dst_comp_vector: int,
        outputs: list[Output],
        adc_sample_rate: float,
        send_rate_factor: float,
        feng_id: int,
        num_ants: int,
        chunk_samples: int,
        chunk_jones: int,
        spectra_per_heap: int,
        dig_sample_bits: int,
        max_delay_diff: int,
        gain: complex,
        sync_epoch: float,
        mask_timestamp: bool,
        use_vkgdr: bool,
        use_peerdirect: bool,
        monitor: Monitor,
    ) -> None:
        super().__init__(katcp_host, katcp_port)
        self._cancel_tasks: list[asyncio.Task] = []  # Tasks that need to be cancelled on shutdown
        self._populate_sensors(
            self.sensors, max(RX_SENSOR_TIMEOUT_MIN, RX_SENSOR_TIMEOUT_CHUNKS * chunk_samples / adc_sample_rate)
        )

        # Attributes copied or initialised from arguments
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_ibv = src_ibv
        self.src_layout = recv.Layout(dig_sample_bits, src_packet_samples, chunk_samples, mask_timestamp)
        self._dst_interface = dst_interface
        self._dst_ttl = dst_ttl
        self._dst_ibv = dst_ibv
        self._dst_packet_payload = dst_packet_payload
        self._dst_comp_vector = dst_comp_vector
        self._send_rate_factor = send_rate_factor
        self.adc_sample_rate = adc_sample_rate
        self.feng_id = feng_id
        self.n_ants = num_ants
        self.spectra_per_heap = spectra_per_heap
        self.chunk_jones = chunk_jones
        self.default_gain = gain
        self.time_converter = TimeConverter(sync_epoch, adc_sample_rate)
        self.monitor = monitor
        self.use_vkgdr = use_vkgdr
        self.use_peerdirect = use_peerdirect

        # Tuning knobs not exposed via arguments
        self.n_in = 3

        self._upload_queue = context.create_command_queue()
        # For copying head of each chunk to the tail of the previous chunk
        self._copy_queue = context.create_command_queue()

        extra_samples = max_delay_diff + max(output.window for output in outputs)
        if extra_samples > self.src_layout.chunk_samples:
            raise RuntimeError(f"chunk_samples is too small; it must be at least {extra_samples}")
        self.n_samples = self.src_layout.chunk_samples + extra_samples

        with context:
            # We could quite easily make do with non-coherent mappings and
            # explicit flushing, but since NVIDIA currently only provides
            # host-coherent memory, this is a simpler option.
            self.vkgdr_handle = vkgdr.Vkgdr.open_current_context(vkgdr.OpenFlags.REQUIRE_COHERENT_BIT)

        self._in_free_queue: asyncio.Queue[InItem] = monitor.make_queue("in_free_queue", self.n_in)
        self._init_recv(src_affinity, monitor)

        # Prevent multiple chunks from being in flight in pipelines at the same
        # time. This keeps the pipelines synchronised to avoid running out of
        # InItems.
        self._active_in_sem = asyncio.BoundedSemaphore(1)
        self._pipelines = []
        self._send_thread_pool = spead2.ThreadPool(1, [] if dst_affinity < 0 else [dst_affinity])
        for i, output in enumerate(outputs):
            # Wideband outputs are always placed first, but we need to consider
            # the case of there being no wideband output at all.
            dig_stats = i == 0 and isinstance(output, WidebandOutput)
            self._pipelines.append(Pipeline(output, self, context, dig_stats))

    def _init_recv(self, src_affinity: list[int], monitor: Monitor) -> None:
        """Initialise the receive side of the engine."""
        src_chunks_per_stream = 4
        ringbuffer_capacity = src_chunks_per_stream * N_POLS

        context = self._upload_queue.context
        for _ in range(self._in_free_queue.maxsize):
            self._in_free_queue.put_nowait(InItem(context, self.src_layout, self.n_samples, use_vkgdr=self.use_vkgdr))

        data_ringbuffer = ChunkRingbuffer(
            ringbuffer_capacity, name="recv_data_ringbuffer", task_name="run_receive", monitor=monitor
        )
        free_ringbuffers = [spead2.recv.ChunkRingbuffer(src_chunks_per_stream) for _ in range(N_POLS)]
        self._src_streams = recv.make_streams(self.src_layout, data_ringbuffer, free_ringbuffers, src_affinity)
        chunk_bytes = self.src_layout.chunk_samples * self.src_layout.sample_bits // BYTE_BITS
        for stream in self._src_streams:
            for _ in range(src_chunks_per_stream):
                if self.use_vkgdr:
                    array_bytes = self.n_samples * self.src_layout.sample_bits // BYTE_BITS
                    device_bytes = array_bytes + INPUT_CHUNK_PADDING
                    with context:
                        mem = vkgdr.pycuda.Memory(self.vkgdr_handle, device_bytes)
                    buf = np.array(mem, copy=False).view(np.uint8)
                    # The device buffer contains extra space for copying the head
                    # of the following chunk, but we don't need that in the host
                    # mapping.
                    buf = buf[:chunk_bytes]
                    device_array = accel.DeviceArray(
                        context, (array_bytes,), np.uint8, padded_shape=(device_bytes,), raw=mem
                    )
                else:
                    buf = accel.HostArray((chunk_bytes,), np.uint8, context=context)
                    device_array = None
                chunk = recv.Chunk(
                    data=buf,
                    device=device_array,
                    present=np.zeros(self.src_layout.chunk_samples // self.src_layout.heap_samples, np.uint8),
                    extra=np.zeros(self.src_layout.chunk_samples // self.src_layout.heap_samples, np.uint16),
                    stream=stream,
                )
                chunk.recycle()  # Make available to the stream

    def _populate_sensors(self, sensors: aiokatcp.SensorSet, rx_sensor_timeout: float) -> None:
        """Define the sensors for an engine (excluding pipeline-specific sensors)."""
        for pol in range(N_POLS):
            sensors.add(
                aiokatcp.Sensor(
                    int,
                    f"input{pol}.dig-clip-cnt",
                    "Number of digitiser samples that are saturated",
                    default=0,
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                    auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
                    auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    float,
                    f"input{pol}.dig-rms-dbfs",
                    "Digitiser ADC average power",
                    units="dBFS",
                    status_func=dig_rms_dbfs_status,
                    auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
                    auto_strategy_parameters=(MIN_SENSOR_UPDATE_PERIOD, math.inf),
                )
            )

        for sensor in recv.make_sensors(rx_sensor_timeout).values():
            sensors.add(sensor)
        sensors.add(
            aiokatcp.Sensor(
                int,
                "steady-state-timestamp",
                "Heaps with this timestamp or greater are guaranteed to "
                "reflect the effects of previous katcp requests.",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )

        sensors.add(DeviceStatusSensor(sensors))

        time_sync_task = add_time_sync_sensors(sensors)
        self.add_service_task(time_sync_task)
        self._cancel_tasks.append(time_sync_task)

    def make_send_streams(
        self, output: Output, n_data_heaps: int, chunks: Sequence[send.Chunk]
    ) -> list["spead2.send.asyncio.AsyncStream"]:
        """Create send streams for a pipeline.

        This method should only be called by :class:`Pipeline`.
        """
        return send.make_streams(
            output_name=output.name,
            thread_pool=self._send_thread_pool,
            endpoints=output.dst,
            interfaces=self._dst_interface,
            ttl=self._dst_ttl,
            ibv=self._dst_ibv,
            packet_payload=self._dst_packet_payload,
            comp_vector=self._dst_comp_vector,
            adc_sample_rate=self.adc_sample_rate,
            send_rate_factor=self._send_rate_factor * output.send_rate_factor,
            feng_id=self.feng_id,
            num_ants=self.n_ants,
            n_data_heaps=n_data_heaps,
            chunks=chunks,
        )

    def _update_steady_state_timestamp(self, timestamp: int) -> None:
        """Update the ``steady-state-timestamp`` sensor to at least a given value."""
        sensor = self.sensors["steady-state-timestamp"]
        sensor.value = max(sensor.value, timestamp)

    def free_in_item(self, item: InItem) -> None:
        """Return an InItem to the free queue if its refcount hits zero."""
        item.refcount -= 1
        if item.refcount == 0:
            self._active_in_sem.release()
            if self.use_vkgdr:
                chunks = []
                for pol_data in item.pol_data:
                    pol_data.samples = None
                    assert pol_data.chunk is not None
                    chunks.append(pol_data.chunk)
                    pol_data.chunk = None
                task = asyncio.create_task(
                    self._push_recv_chunks(chunks, item.events),
                    name="Receive Chunk Recycle Task",
                )
                self.add_service_task(task)
            self._in_free_queue.put_nowait(item)

    @staticmethod
    async def _push_recv_chunks(chunks: Iterable[recv.Chunk], events: Iterable[AbstractEvent]) -> None:
        """Return chunks to the streams once `events` have fired.

        This is only used when using vkgdr.
        """
        await async_wait_for_events(events)
        for chunk in chunks:
            chunk.recycle()

    async def _add_in_item(self, item: InItem) -> None:
        """Push an :class:`InItem` to all the pipelines.

        This also takes care of computing `present_cumsum` and initialising
        the refcount.
        """
        # np.cumsum doesn't provide an initial zero, so we output starting at
        # position 1.
        for pol_data in item.pol_data:
            np.cumsum(pol_data.present, dtype=pol_data.present_cumsum.dtype, out=pol_data.present_cumsum[1:])
        await self._active_in_sem.acquire()
        item.refcount = len(self._pipelines)
        for pipeline in self._pipelines:
            pipeline.add_in_item(item)

    def _copy_tail(self, prev_item: InItem, in_item: InItem | None) -> None:
        """Copy the head of `in_item` to the tail of `prev_item`.

        This allows for PFB windows to fit and for some protection against
        sharp changes in delay.

        If `in_item` is ``None`` or is not contiguous (in timestamp) with
        `prev_item` then the tail of `prev_item` is instead marked as absent.
        This can happen if we lose a whole input chunk from the digitiser.
        """
        chunk_heaps = prev_item.n_samples // self.src_layout.heap_samples
        copy_heaps = len(prev_item.pol_data[0].present) - chunk_heaps
        if in_item is not None and prev_item.end_timestamp == in_item.timestamp:
            sample_bits = self.src_layout.sample_bits
            copy_samples = prev_item.capacity - prev_item.n_samples
            copy_samples = min(copy_samples, in_item.n_samples)
            copy_bytes = copy_samples * sample_bits // BYTE_BITS
            # Must wait for the upload to complete before the copy starts
            self._copy_queue.enqueue_wait_for_events(in_item.events)
            for pol_data0, pol_data1 in zip(prev_item.pol_data, in_item.pol_data):
                assert pol_data0.samples is not None
                assert pol_data1.samples is not None
                pol_data1.samples.copy_region(
                    self._copy_queue,
                    pol_data0.samples,
                    np.s_[:copy_bytes],
                    np.s_[-copy_bytes:],
                )
                pol_data0.present[-copy_heaps:] = pol_data1.present[:copy_heaps]
            prev_item.n_samples += copy_samples
            prev_item.add_marker(self._copy_queue)
        else:
            for pol_data in prev_item.pol_data:
                pol_data.present[-copy_heaps:] = 0  # Mark tail as absent, for each pol

    async def _run_receive(self, streams: list[spead2.recv.ChunkRingStream], layout: recv.Layout) -> None:
        """Receive data from the network, queue it up for processing.

        This function receives chunk sets, which are chunks in groups of two -
        one per polarisation, from the spead2 receiver streams given. For each
        chunk set received, copies of the data to the GPU are initiated,
        awaited, and then the chunk containers are returned to the receiver
        stream so that the memory need not be expensively re-allocated every
        time.

        Additionally, the start of each chunk is copied to the tail of the
        previous chunk, and only then is the previous chunk pushed to the
        queues.

        In the GPU-direct case, <TODO clarify once I understand better>.

        Parameters
        ----------
        streams
            There should be only two of these because they each represent one of
            the digitiser's two polarisations.
        layout
            The structure of the streams.
        """
        prev_item = None
        async for chunks in recv.chunk_sets(streams, layout, self.sensors, self.time_converter):
            with self.monitor.with_state("run_receive", "wait in_free_queue"):
                in_item = await self._in_free_queue.get()
            # Make sure all the item's events are complete before overwriting
            # the data. This is not needed for vkgdr because in that case it's
            # handled by _push_recv_chunks.
            if not self.use_vkgdr:
                in_item.enqueue_wait_for_events(self._upload_queue)
            in_item.reset(chunks[0].timestamp)

            # In steady-state, chunks should be the same size, but during
            # shutdown, the last chunk may be short.
            in_item.n_samples = chunks[0].data.nbytes * BYTE_BITS // self.src_layout.sample_bits

            transfer_events = []
            for pol, (pol_data, chunk) in enumerate(zip(in_item.pol_data, chunks)):
                # Copy the present flags (synchronously).
                pol_data.present[: len(chunk.present)] = chunk.present
                # Update the digitiser saturation count (the "extra" field holds
                # per-heap values).
                assert chunk.extra is not None
                sensor = self.sensors[f"input{pol}.dig-clip-cnt"]
                sensor.set_value(
                    sensor.value + int(np.sum(chunk.extra, dtype=np.uint64)),
                    timestamp=self.time_converter.adc_to_unix(chunk.timestamp + layout.chunk_samples),
                )
            if self.use_vkgdr:
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is None
                    pol_data.samples = chunk.device  # type: ignore
                    pol_data.chunk = chunk
            else:
                # Copy each pol chunk to the right place on the GPU.
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is not None
                    pol_data.samples.set_region(
                        self._upload_queue, chunk.data, np.s_[: chunk.data.nbytes], np.s_[:], blocking=False
                    )
                    transfer_events.append(self._upload_queue.enqueue_marker())
                # Put events on the queue so that run_processing() knows when to
                # start.
                in_item.events.extend(transfer_events)

            if prev_item is not None:
                self._copy_tail(prev_item, in_item)
                await self._add_in_item(prev_item)
            prev_item = in_item

            if not self.use_vkgdr:
                # Wait until the copy is done, and then give the chunks of memory
                # back to the receiver streams for reuse.
                # NB: we don't use the Chunk context manager, because if
                # something goes wrong we won't have waited for the event, and
                # giving the chunk back to the stream while it's still in use
                # by the device could cause incorrect data to be transmitted.
                for pol in range(len(chunks)):
                    with self.monitor.with_state("run_receive", "wait transfer"):
                        await async_wait_for_events([transfer_events[pol]])
                    chunks[pol].recycle()

        if prev_item is not None:
            # Flush the final chunk to the pipelines
            self._copy_tail(prev_item, None)  # Mark tail as absent
            await self._add_in_item(prev_item)

        logger.debug("run_receive completed")
        for pipeline in self._pipelines:
            pipeline.shutdown()

    def _request_pipeline(self, stream_name: str) -> Pipeline:
        """Find the pipeline related to a katcp request.

        Raises
        ------
        FailReply
            If the `stream_name` is not a known output.
        """
        # Note: this takes O(n) time, but as there are expected to be less than 10
        # pipelines it is probably not worth keeping a dictionary.
        for pipeline in self._pipelines:
            if pipeline.output.name == stream_name:
                return pipeline
        raise aiokatcp.FailReply(f"no output stream called {stream_name!r}")

    async def request_gain(self, ctx, stream_name: str, input: int, *values: str) -> tuple[str, ...]:
        """Set or query the eq gains.

        If no values are provided, the gains are simply returned.

        Parameters
        ----------
        stream_name
            Output stream name
        input
            Input number (0 or 1)
        values
            Complex values. There must either be a single value (used for all
            channels), or a value per channel.
        """
        pipeline = self._request_pipeline(stream_name)
        output = pipeline.output
        if not 0 <= input < N_POLS:
            raise aiokatcp.FailReply("input is out of range")
        if len(values) not in {0, 1, output.channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 0, 1 or {output.channels})")
        if not values:
            gains = pipeline.gains[:, input]
        else:
            gains = _parse_gains(*values, channels=output.channels, default_gain=None)
            pipeline.set_gains(input, gains)

        # Return the current values.
        # If they're all the same, we can return just a single value.
        if np.all(gains == gains[0]):
            gains = gains[:1]
        return tuple(format_complex(gain) for gain in gains)

    async def request_gain_all(self, ctx, stream_name: str, *values: str) -> None:
        """Set the eq gains for all inputs.

        Parameters
        ----------
        stream_name
            Output stream name
        values
            Complex values. There must either be a single value (used for all
            channels), or a value per channel, or ``"default"`` to reset gains
            to the default.
        """
        pipeline = self._request_pipeline(stream_name)
        output = pipeline.output
        if len(values) not in {1, output.channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 1 or {output.channels})")
        gains = _parse_gains(*values, channels=output.channels, default_gain=self.default_gain)
        for i in range(N_POLS):
            pipeline.set_gains(i, gains)

    async def request_delays(self, ctx, stream_name: str, start_time: aiokatcp.Timestamp, *delays: str) -> None:
        """Add a new first-order polynomial to the delay and fringe correction model.

        .. todo::

          Make the request's fail replies more informative in the case of
          malformed requests.
        """

        def comma_string_to_float(comma_string: str) -> tuple[float, float]:
            a_str, b_str = comma_string.split(",")
            a = float(a_str)
            b = float(b_str)
            return a, b

        pipeline = self._request_pipeline(stream_name)
        if len(delays) != len(pipeline.delay_models):
            raise aiokatcp.FailReply(f"wrong number of delay coefficient sets (expected {len(pipeline.delay_models)})")

        # This will round the start time of the new delay model to the nearest
        # ADC sample. If the start time given doesn't coincide with an ADC sample,
        # then all subsequent delays for this model will be off by the product
        # of this delta and the delay_rate (same for phase).
        # This may be too small to be a concern, but if it is a concern,
        # then we'd need to compensate for that here.
        start_sample_count = round(self.time_converter.unix_to_adc(start_time))
        if start_sample_count < 0:
            raise aiokatcp.FailReply("Start time cannot be prior to the sync epoch")

        # Collect them in a temporary until they're all validated
        new_linear_models = []
        for coeffs in delays:
            delay_str, phase_str = coeffs.split(":")
            delay, delay_rate = comma_string_to_float(delay_str)
            phase, phase_rate = comma_string_to_float(phase_str)

            delay_samples = delay * self.adc_sample_rate
            new_linear_models.append(
                LinearDelayModel(
                    start_sample_count,
                    delay_samples,
                    delay_rate,
                    phase,
                    phase_rate / self.adc_sample_rate,
                )
            )

        for delay_model, new_linear_model in zip(pipeline.delay_models, new_linear_models):
            delay_model.base.add(new_linear_model)
        self._update_steady_state_timestamp(pipeline.delay_update_timestamp())

    async def start(self, descriptor_interval_s: float = SPEAD_DESCRIPTOR_INTERVAL_S) -> None:
        """Start the engine.

        This function adds the receive, processing and transmit tasks onto the
        event loop. It also adds a task to continuously send the descriptor
        heaps at an interval based on the `descriptor_interval_s`. See
        :meth:`_run_descriptors_loop` for more details.

        Parameters
        ----------
        descriptor_interval_s
            The base interval used as a multiplier on feng_id and n_ants to
            dictate the initial 'engine sleep interval' and 'send interval'
            respectively.
        """
        # Create the descriptor task first to ensure descriptors will be sent
        # before any data makes its way through the pipeline.
        for pipeline in self._pipelines:
            descriptor_sender = DescriptorSender(
                pipeline._send_streams[0],
                pipeline.descriptor_heap,
                self.n_ants * descriptor_interval_s,
                (self.feng_id + 1) * descriptor_interval_s,
                substreams=range(len(pipeline.output.dst)),
            )
            descriptor_task = asyncio.create_task(
                descriptor_sender.run(), name=f"{pipeline.output.name}.{DESCRIPTOR_TASK_NAME}"
            )
            self.add_service_task(descriptor_task)
            self._cancel_tasks.append(descriptor_task)

        for pol, stream in enumerate(self._src_streams):
            base_recv.add_reader(
                stream,
                src=self._srcs[pol],
                interface=self._src_interface[pol] if self._src_interface is not None else None,
                ibv=self._src_ibv,
                comp_vector=self._src_comp_vector[pol],
                buffer=self._src_buffer,
            )

        recv_task = asyncio.create_task(
            self._run_receive(self._src_streams, self.src_layout),
            name=RECV_TASK_NAME,
        )
        self.add_service_task(recv_task)

        for pipeline in self._pipelines:
            proc_task = asyncio.create_task(
                pipeline.run_processing(),
                name=f"{pipeline.output.name}.{GPU_PROC_TASK_NAME}",
            )
            self.add_service_task(proc_task)

            send_task = asyncio.create_task(
                pipeline.run_transmit(),
                name=f"{pipeline.output.name}.{SEND_TASK_NAME}",
            )
            self.add_service_task(send_task)

        await super().start()

    async def on_stop(self) -> None:
        """Shut down processing when the device server is stopped.

        This is called by aiokatcp after closing the listening socket.
        Also handle any Exceptions thrown unexpectedly in any of the
        processing loops.
        """
        for task in self._cancel_tasks:
            task.cancel()
        for stream in self._src_streams:
            stream.stop()
        # If any of the tasks are already done then we had an exception, and
        # waiting for the rest may hang as the shutdown path won't proceed
        # neatly.
        if not any(task.done() for task in self.service_tasks):
            for task in self.service_tasks:
                if task not in self._cancel_tasks:
                    await task
