################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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
import numbers
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Deque, Iterable, List, Optional, Sequence, Tuple, Union, cast

import aiokatcp
import katsdpsigproc.accel as accel
import numpy as np
import spead2.recv
from katsdpsigproc.abc import AbstractContext, AbstractEvent
from katsdpsigproc.resource import async_wait_for_events
from katsdptelstate.endpoint import Endpoint

from .. import (
    BYTE_BITS,
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
from ..ringbuffer import ChunkRingbuffer
from ..send import DescriptorSender
from . import SAMPLE_BITS, recv, send
from .compute import Compute, ComputeTemplate
from .delay import AbstractDelayModel, LinearDelayModel, MultiDelayModel, wrap_angle

logger = logging.getLogger(__name__)


def _device_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.DeviceArray:
    return accel.DeviceArray(context, slot.shape, slot.dtype, slot.required_padded_shape())


def _host_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.HostArray:
    return accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(), context=context)


def _sample_models(
    delay_models: Iterable[AbstractDelayModel], start: int, stop: int, step: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call :meth:`.AbstractDelayModel.range` on multiple delay models and stack results."""
    # Each element of parts is a tuple of results from one delay model
    parts = [model.range(start, stop, step) for model in delay_models]
    # Transpose so that each element of groups is one result from all delay models
    return tuple(np.stack(group) for group in zip(*parts))  # type: ignore


def generate_weights(channels: int, taps: int) -> np.ndarray:
    """Generate Hann-window weights for the F-engine's PFB-FIR.

    The resulting weights are normalised such that the sum of
    squares is 1.

    Parameters
    ----------
    channels
        Number of channels in the PFB.
    taps
        Number of taps in the PFB-FIR.

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the weights for the PFB-FIR filters, as
        single-precision floats.
    """
    step = 2 * channels
    window_size = step * taps
    idx = np.arange(window_size)
    hann = np.square(np.sin(np.pi * idx / (window_size - 1)))
    sinc = np.sinc((idx + 0.5) / step - taps / 2)
    weights = hann * sinc
    # Work around https://github.com/numpy/numpy/issues/21898
    weights /= np.sqrt(np.sum(np.square(weights)))  # type: ignore[misc]
    return weights.astype(np.float32)


@dataclass
class PolInItem:
    """Polarisation-specific elements of :class:`InItem`."""

    #: A device memory region for storing the raw samples.
    samples: Optional[accel.DeviceArray]
    #: Bitmask indicating which packets were present in the chunk.
    present: np.ndarray
    #: Cumulative sum over :attr:`present`. It is up to the caller
    #: to compute it at the appropriate time.
    present_cumsum: np.ndarray
    #: Chunk to return to recv after processing (used with vkgdr only).
    chunk: Optional[recv.Chunk] = None


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
    compute
        F-engine Operation Sequence detailing the computation operations which
        will take place on the data in :attr:`PolInItem.samples`.
    timestamp
        Timestamp of the oldest digitiser sample represented in the data.
    packet_samples
        Number of samples per digitiser packet (for sizing :attr:`PolInItem.present`).
    use_vkgdr
        Use vkgdr to write sample data directly to the GPU rather than staging in
        host memory.
    """

    #: Per-polarisation data
    pol_data: List[PolInItem]
    #: Number of samples in each :class:`~katsdpsigproc.accel.DeviceArray` in :attr:`PolInItem.samples`
    n_samples: int
    #: Bitwidth of the data in :attr:`PolInItem.samples`
    sample_bits: int

    def __init__(self, compute: Compute, timestamp: int = 0, *, packet_samples: int, use_vkgdr: bool = False) -> None:
        self.sample_bits = compute.sample_bits
        self.pol_data = []
        present_size = accel.divup(compute.samples, packet_samples)
        for pol in range(N_POLS):
            if use_vkgdr:
                # Memory belongs to the chunks, and we set samples when
                # initialising the item from the chunks.
                samples = None
            else:
                samples = _device_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots[f"in{pol}"]))
            self.pol_data.append(
                PolInItem(
                    samples=samples,
                    present=np.zeros(present_size, dtype=bool),
                    present_cumsum=np.zeros(present_size + 1, np.uint32),
                )
            )
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
        return self.pol_data[0].samples.shape[0] * BYTE_BITS // self.sample_bits

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end (i.e. latest plus 1) timestamp of the item."""
        return self.timestamp + self.n_samples


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
    timestamp
        Timestamp of the first spectrum in the `OutItem`.
    """

    #: Output data, a collection of spectra, arranged in memory by pol and by heap.
    spectra: accel.DeviceArray
    #: Provides a scratch space for collecting per-spectrum fine delays while
    #: the `OutItem` is being prepared. When the `OutItem` is placed onto the
    #: queue it is copied to the `Compute`.
    fine_delay: accel.HostArray
    #: A similar scratch space for collecting per-spectrum phase offsets while
    #: the :class:`OutItem` is being prepared.
    phase: accel.HostArray
    #: Per-channel gains
    gains: accel.HostArray
    #: Bit-mask indicating which spectra contain valid data and should be transmitted.
    present: np.ndarray
    #: Number of spectra contained in :attr:`spectra`.
    n_spectra: int
    #: Corresponding chunk for transmission (only used in PeerDirect mode).
    chunk: Optional[send.Chunk] = None

    def __init__(self, compute: Compute, timestamp: int = 0) -> None:
        self.spectra = _device_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["out"]))
        self.fine_delay = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["fine_delay"]))
        self.phase = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["phase"]))
        self.gains = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["gains"]))
        self.present = np.zeros(self.fine_delay.shape[0], dtype=bool)
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item.

        Zero the item's timestamp, empty the event list and set number of
        spectra to zero.
        """
        super().reset(timestamp)
        self.n_spectra = 0

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end timestamp of the item.

        Following Python's normal exclusive-end convention.
        """
        return self.timestamp + self.n_spectra * 2 * self.channels

    @property
    def channels(self) -> int:  # noqa: D401
        """Number of channels."""
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
        IP address of the network device to use for input.
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
    dst
        A list of destination endpoints for the outgoing data.
    dst_interface
        IP address of the network device to use for output.
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
    spectra
        Number of spectra that will be produced from a chunk of incoming
        digitiser data.
    spectra_per_heap
        Number of spectra in each output heap.
    channels
        Number of output channels to produce.
    taps
        Number of taps in each branch of the PFB-FIR.
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
        srcs: List[Union[str, List[Tuple[str, int]]]],
        src_interface: Optional[str],
        src_ibv: bool,
        src_affinity: List[int],
        src_comp_vector: List[int],
        src_packet_samples: int,
        src_buffer: int,
        dst: List[Endpoint],
        dst_interface: str,
        dst_ttl: int,
        dst_ibv: bool,
        dst_packet_payload: int,
        dst_affinity: int,
        dst_comp_vector: int,
        adc_sample_rate: float,
        send_rate_factor: float,
        feng_id: int,
        num_ants: int,
        chunk_samples: int,
        spectra: int,
        spectra_per_heap: int,
        channels: int,
        taps: int,
        max_delay_diff: int,
        gain: complex,
        sync_epoch: float,
        mask_timestamp: bool,
        use_vkgdr: bool,
        use_peerdirect: bool,
        monitor: Monitor,
    ) -> None:
        super(Engine, self).__init__(katcp_host, katcp_port)
        self._populate_sensors(self.sensors)

        # Attributes copied or initialised from arguments
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_ibv = src_ibv
        self._src_layout = recv.Layout(SAMPLE_BITS, src_packet_samples, chunk_samples, mask_timestamp)
        self._src_packet_samples = src_packet_samples
        self.adc_sample_rate = adc_sample_rate
        self.send_rate_factor = send_rate_factor
        self.feng_id = feng_id
        self.n_ants = num_ants
        self.default_gain = gain
        self.sync_epoch = sync_epoch
        self.monitor = monitor
        self.use_vkgdr = use_vkgdr

        # Tuning knobs not exposed via arguments
        n_in = 3
        n_send = 4
        n_out = n_send if use_peerdirect else 2

        # The type annotations have to be in comments because Python 3.8
        # doesn't support the syntax at runtime (Python 3.9 fixes that).
        self._in_queue = monitor.make_queue("in_queue", n_in)  # type: asyncio.Queue[Optional[InItem]]
        self._in_free_queue = monitor.make_queue("in_free_queue", n_in)  # type: asyncio.Queue[InItem]
        self._out_queue = monitor.make_queue("out_queue", n_out)  # type: asyncio.Queue[Optional[OutItem]]
        self._out_free_queue = monitor.make_queue("out_free_queue", n_out)  # type: asyncio.Queue[OutItem]
        self._send_free_queue = monitor.make_queue("send_free_queue", n_send)  # type: asyncio.Queue[send.Chunk]

        self._init_compute(
            context=context,
            spectra=spectra,
            spectra_per_heap=spectra_per_heap,
            channels=channels,
            taps=taps,
            max_delay_diff=max_delay_diff,
        )

        self._in_items: Deque[InItem] = deque()
        self._init_recv(src_affinity, monitor)

        send_chunks = self._init_send(len(dst), use_peerdirect)
        self._send_stream = send.make_stream(
            endpoints=dst,
            interface=dst_interface,
            ttl=dst_ttl,
            ibv=dst_ibv,
            packet_payload=dst_packet_payload,
            affinity=dst_affinity,
            comp_vector=dst_comp_vector,
            adc_sample_rate=adc_sample_rate,
            send_rate_factor=send_rate_factor,
            feng_id=feng_id,
            num_ants=num_ants,
            spectra=spectra,
            spectra_per_heap=spectra_per_heap,
            channels=channels,
            chunks=send_chunks,
        )
        self._out_item = self._out_free_queue.get_nowait()

        self.delay_models: List[MultiDelayModel] = []
        self.gains = np.zeros((self.channels, self.pols), np.complex64)
        self._init_delay_gain()

        self._descriptor_heap = send.make_descriptor_heap(
            channels_per_substream=channels // len(dst),
            spectra_per_heap=spectra_per_heap,
        )

    def _init_compute(
        self,
        context: AbstractContext,
        spectra: int,
        spectra_per_heap: int,
        channels: int,
        taps: int,
        max_delay_diff: int,
    ) -> None:
        """Initialise ``self._compute`` and related resources."""
        compute_queue = context.create_command_queue()
        self._upload_queue = context.create_command_queue()
        self._download_queue = context.create_command_queue()

        extra_samples = max_delay_diff + taps * channels * 2
        if extra_samples > self._src_layout.chunk_samples:
            raise RuntimeError(f"chunk_samples is too small; it must be at least {extra_samples}")
        samples = self._src_layout.chunk_samples + extra_samples

        template = ComputeTemplate(context, taps, channels)
        self._compute = template.instantiate(compute_queue, samples, spectra, spectra_per_heap)
        device_weights = self._compute.slots["weights"].allocate(accel.DeviceAllocator(context))
        device_weights.set(compute_queue, generate_weights(channels, taps))

    def _init_recv(self, src_affinity: List[int], monitor: Monitor) -> None:
        """Initialise the receive side of the engine."""
        src_chunks_per_stream = 4
        ringbuffer_capacity = src_chunks_per_stream * N_POLS

        for _ in range(self._in_free_queue.maxsize):
            self._in_free_queue.put_nowait(
                InItem(self._compute, packet_samples=self._src_packet_samples, use_vkgdr=self.use_vkgdr)
            )

        context = self._compute.template.context
        if self.use_vkgdr:
            import vkgdr.pycuda

            with context:
                # We could quite easily make do with non-coherent mappings and
                # explicit flushing, but since NVIDIA currently only provides
                # host-coherent memory, this is a simpler option.
                vkgdr_handle = vkgdr.Vkgdr.open_current_context(vkgdr.OpenFlags.REQUIRE_COHERENT_BIT)

        data_ringbuffer = ChunkRingbuffer(
            ringbuffer_capacity, name="recv_data_ringbuffer", task_name="run_receive", monitor=monitor
        )
        free_ringbuffers = [spead2.recv.ChunkRingbuffer(src_chunks_per_stream) for _ in range(N_POLS)]
        self._src_streams = recv.make_streams(self._src_layout, data_ringbuffer, free_ringbuffers, src_affinity)
        chunk_bytes = self._src_layout.chunk_samples * SAMPLE_BITS // BYTE_BITS
        for pol, stream in enumerate(self._src_streams):
            for _ in range(src_chunks_per_stream):
                if self.use_vkgdr:
                    device_bytes = self._compute.slots[f"in{pol}"].required_bytes()
                    with context:
                        mem = vkgdr.pycuda.Memory(vkgdr_handle, device_bytes)
                    buf = np.array(mem, copy=False).view(np.uint8)
                    # The device buffer contains extra space for copying the head
                    # of the following chunk, but we don't need that in the host
                    # mapping.
                    buf = buf[:chunk_bytes]
                    device_array = accel.DeviceArray(context, (device_bytes,), np.uint8, raw=mem)
                    chunk = recv.Chunk(data=buf, device=device_array, stream=stream)
                else:
                    buf = accel.HostArray((chunk_bytes,), np.uint8, context=context)
                    chunk = recv.Chunk(data=buf, stream=stream)
                chunk.present = np.zeros(self._src_layout.chunk_samples // self._src_packet_samples, np.uint8)
                chunk.recycle()  # Make available to the stream

    def _init_send(self, substreams: int, use_peerdirect: bool) -> List[send.Chunk]:
        """Initialise the send side of the engine, with the exception of ``_send_stream``."""
        send_chunks: List[send.Chunk] = []
        for _ in range(self._out_free_queue.maxsize):
            item = OutItem(self._compute)
            if use_peerdirect:
                dev_buffer = item.spectra.buffer.gpudata.as_buffer(item.spectra.buffer.nbytes)
                # buf is structurally a numpy array, but the pointer in it is a CUDA
                # pointer and so actually trying to use it as such will cause a
                # segfault.
                buf = np.frombuffer(dev_buffer, dtype=item.spectra.dtype).reshape(item.spectra.shape)
                chunk = send.Chunk(
                    buf,
                    substreams=substreams,
                    feng_id=self.feng_id,
                )
                item.chunk = chunk
                send_chunks.append(chunk)
            self._out_free_queue.put_nowait(item)

        spectra = self._compute.spectra
        if not use_peerdirect:
            # When using PeerDirect, the chunks are created along with the items
            send_shape = (spectra // self.spectra_per_heap, self.channels, self.spectra_per_heap, N_POLS, COMPLEX)
            for _ in range(self._send_free_queue.maxsize):
                send_chunks.append(
                    send.Chunk(
                        accel.HostArray(send_shape, send.SEND_DTYPE, context=self._compute.template.context),
                        substreams=substreams,
                        feng_id=self.feng_id,
                    )
                )
                self._send_free_queue.put_nowait(send_chunks[-1])
        return send_chunks

    def _init_delay_gain(self) -> None:
        """Initialise the delays and gains."""
        for pol in range(N_POLS):
            delay_model = MultiDelayModel(
                callback_func=partial(self.update_delay_sensor, delay_sensor=self.sensors[f"input{pol}-delay"])
            )
            self.delay_models.append(delay_model)

        for pol in range(N_POLS):
            self.set_gains(pol, np.full(self.channels, self.default_gain, dtype=np.complex64))

    @property
    def channels(self) -> int:  # noqa: D401
        """Number of channels into which the incoming signal is decomposed."""
        return self._compute.template.channels

    @property
    def taps(self) -> int:  # noqa: D401
        """Number of taps in the PFB-FIR filter."""
        return self._compute.template.taps

    @property
    def spectra_per_heap(self) -> int:  # noqa: D401
        """The number of spectra which will be transmitted per output heap."""
        return self._compute.spectra_per_heap

    @property
    def sample_bits(self) -> int:  # noqa: D401
        """Bitwidth of the incoming digitiser samples."""
        return self._compute.sample_bits

    @property
    def spectra_samples(self) -> int:  # noqa: D401
        """Number of incoming digitiser samples needed per spectrum.

        Note that this is the spacing between spectra. Each spectrum uses
        an overlapping window with more samples than this.
        """
        return 2 * self.channels

    @property
    def pols(self) -> int:  # noqa: D401
        """Number of polarisations."""
        return N_POLS

    @staticmethod
    def _populate_sensors(sensors: aiokatcp.SensorSet) -> None:
        """Define the sensors for an engine."""
        for pol in range(N_POLS):
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"input{pol}-eq",
                    "For this input, the complex, unitless, per-channel digital scaling factors "
                    "implemented prior to requantisation",
                    initial_status=aiokatcp.Sensor.Status.NOMINAL,
                )
            )
            sensors.add(
                aiokatcp.Sensor(
                    str,
                    f"input{pol}-delay",
                    "The delay settings for this input: (loadmcnt <ADC sample "
                    "count when model was loaded>, delay <in seconds>, "
                    "delay-rate <unit-less or, seconds-per-second>, "
                    "phase <radians>, phase-rate <radians per second>).",
                )
            )
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

    async def _next_in(self) -> Optional[InItem]:
        """Load next InItem for processing.

        Move the next :class:`InItem` from the `_in_queue` to `_in_items`, where
        it will be picked up by the processing.
        """
        with self.monitor.with_state("run_processing", "wait in_queue"):
            item = await self._in_queue.get()

        if item is not None:
            self._in_items.append(item)
            # print(f'Received input with timestamp {self._in_items[-1].timestamp}, '
            #       f'{self._in_items[-1].n_samples} samples')

            # Make sure that all events associated with the item are past.
            self._in_items[-1].enqueue_wait_for_events(self._compute.command_queue)
        else:
            # To keep _run_processing simple, it may make further calls to
            # _next_in after receiving a None. To keep things simple, put
            # a None back into the queue so that the next call also gets
            # None rather than hanging.
            self._in_queue.put_nowait(None)
        return item

    async def _fill_in(self) -> bool:
        """Load sufficient InItems to continue processing.

        Tries to get at least two items into ``self._in_items``, and if
        loading a second item that is adjacent to the first, copies the overlap
        region.

        Returns true if processing can proceed, false if the stream is
        exhausted.
        """
        if len(self._in_items) == 0:
            if not (await self._next_in()):
                return False
        if len(self._in_items) == 1:
            # Copy the head of the new chunk to the tail of the older chunk
            # to allow for PFB windows to fit and for some protection against
            # sharp changes in delay.
            #
            # This could only fail if we'd lost a whole input chunk of
            # data from the digitiser. In that case the data we'd like
            # to copy is missing so we can't do this step.
            chunk_packets = self._in_items[0].n_samples // self._src_packet_samples
            copy_packets = len(self._in_items[0].pol_data[0].present) - chunk_packets
            if (await self._next_in()) and self._in_items[0].end_timestamp == self._in_items[1].timestamp:
                sample_bits = self._in_items[0].sample_bits
                copy_samples = self._in_items[0].capacity - self._in_items[0].n_samples
                copy_samples = min(copy_samples, self._in_items[1].n_samples)
                copy_bytes = copy_samples * sample_bits // BYTE_BITS
                for pol_data0, pol_data1 in zip(self._in_items[0].pol_data, self._in_items[1].pol_data):
                    assert pol_data0.samples is not None
                    assert pol_data1.samples is not None
                    pol_data1.samples.copy_region(
                        self._compute.command_queue,
                        pol_data0.samples,
                        np.s_[:copy_bytes],
                        np.s_[-copy_bytes:],
                    )
                    pol_data0.present[-copy_packets:] = pol_data1.present[:copy_packets]
                self._in_items[0].n_samples += copy_samples
            else:
                for pol_data in self._in_items[0].pol_data:
                    pol_data.present[-copy_packets:] = 0  # Mark tail as absent, for each pol
            # Update the cumulative sums. Note that during shutdown this may be
            # done more than once, but since it is shutdown the performance
            # implications aren't too important.
            # np.cumsum doesn't provide an initial zero, so we output starting at
            # position 1.
            for pol_data in self._in_items[0].pol_data:
                np.cumsum(pol_data.present, dtype=pol_data.present_cumsum.dtype, out=pol_data.present_cumsum[1:])
        return True

    def _pop_in(self) -> None:
        """Remove the oldest InItem."""
        item = self._in_items.popleft()
        event = self._compute.command_queue.enqueue_marker()
        if self.use_vkgdr:
            chunks = []
            for pol_data in item.pol_data:
                pol_data.samples = None
                assert pol_data.chunk is not None
                chunks.append(pol_data.chunk)
                pol_data.chunk = None
            asyncio.create_task(self._push_recv_chunks(chunks, event))
        else:
            item.events.append(event)
        self._in_free_queue.put_nowait(item)

    async def _next_out(self, new_timestamp: int) -> OutItem:
        """Grab the next free OutItem in the queue."""
        with self.monitor.with_state("run_processing", "wait out_free_queue"):
            item = await self._out_free_queue.get()

        # Just make double-sure that all events associated with the item are past.
        item.enqueue_wait_for_events(self._compute.command_queue)
        item.reset(new_timestamp)
        return item

    async def _flush_out(self, new_timestamp: int) -> None:
        """Start the backend processing and prepare the data for transmission.

        Kick off the `run_backend()` processing, and put an event on the
        relevant command queue. This lets the next coroutine (_run_transmit) know
        that the backend processing is finished, and the data can be transmitted
        out.

        Parameters
        ----------
        new_timestamp
            The timestamp that will immediately follow the current OutItem.
        """
        # Round down to a multiple of accs (don't send heap with partial
        # data).
        accs = self._out_item.n_spectra // self.spectra_per_heap
        self._out_item.n_spectra = accs * self.spectra_per_heap
        if self._out_item.n_spectra > 0:
            # Take a copy of the gains synchronously. This avoids race conditions
            # with gains being updated at the same time as they're in the
            # middle of being transferred.
            self._out_item.gains[:] = self.gains
            # TODO: only need to copy the relevant region, and can limit
            # postprocessing to the relevant range (the FFT size is baked into
            # the plan, so is harder to modify on the fly).
            self._compute.buffer("fine_delay").set_async(self._compute.command_queue, self._out_item.fine_delay)
            self._compute.buffer("phase").set_async(self._compute.command_queue, self._out_item.phase)
            self._compute.buffer("gains").set_async(self._compute.command_queue, self._out_item.gains)
            self._compute.run_backend(self._out_item.spectra)
            self._out_item.add_marker(self._compute.command_queue)
            self._out_queue.put_nowait(self._out_item)
            # TODO: could set it to None, since we only need it when we're
            # ready to flush again?
            self._out_item = await self._next_out(new_timestamp)
        else:
            self._out_item.timestamp = new_timestamp

    @staticmethod
    async def _push_recv_chunks(chunks: Iterable[recv.Chunk], event: AbstractEvent) -> None:
        """Return chunks to the streams once `event` has fired.

        This is only used when using vkgdr.
        """
        await async_wait_for_events([event])
        for chunk in chunks:
            chunk.recycle()

    async def _run_processing(self) -> None:
        """Do the hard work of the F-engine.

        This function takes place entirely on the GPU. First, a little bit of
        the next chunk is copied to the end of the previous one, to allow for
        the overlap required by the PFB. Coarse delay happens. Then a batch FFT
        operation is applied, and finally, fine-delay, phase correction,
        quantisation and corner-turn are performed.
        """
        while await self._fill_in():
            # If the input starts too late for the next expected timestamp,
            # we need to skip ahead to the next heap after the start, and
            # flush what we already have.
            start_timestamp = self._out_item.end_timestamp
            orig_start_timestamps = [model(start_timestamp)[0] for model in self.delay_models]
            if min(orig_start_timestamps) < self._in_items[0].timestamp:
                align = self.spectra_per_heap * self.spectra_samples
                # This loop is needed because MultiDelayModel is not necessarily
                # monotonic, and so simply taking the larger of the two skip
                # results does not guarantee a suitable timestamp.
                while min(orig_start_timestamps) < self._in_items[0].timestamp:
                    start_timestamp = max(
                        model.skip(self._in_items[0].timestamp, start_timestamp + 1, align)
                        for model in self.delay_models
                    )
                    orig_start_timestamps = [model(start_timestamp)[0] for model in self.delay_models]
                await self._flush_out(start_timestamp)
            # When we add new spectra they must follow contiguously for any
            # that we've already buffered.
            assert start_timestamp == self._out_item.end_timestamp

            # Compute the coarse delay for the first sample.
            # `orig_timestamp` is the timestamp of first sample from the input
            # to process in the PFB to produce the output spectrum with
            # `timestamp`. `offset` is the sample index corresponding to
            # `orig_timestamp` within the InItem.
            start_coarse_delays = [start_timestamp - orig_timestamp for orig_timestamp in orig_start_timestamps]
            offsets = [orig_timestamp - self._in_items[0].timestamp for orig_timestamp in orig_start_timestamps]

            # Identify a block of frontend work. We can grow it until
            # - we run out of the current input array;
            # - we fill up the output array; or
            # - the coarse delay changes.
            # We speculatively calculate delays until one of the first two is
            # met, then truncate if we observe a coarse delay change. Note:
            # max_end_in is computed assuming the coarse delay does not change.
            max_end_in = (
                self._in_items[0].end_timestamp + min(start_coarse_delays) - self.taps * self.spectra_samples + 1
            )
            max_end_out = self._out_item.timestamp + self._out_item.capacity * self.spectra_samples
            max_end = min(max_end_in, max_end_out)
            # Speculatively evaluate until one of the first two conditions is met
            timestamps = np.arange(start_timestamp, max_end, self.spectra_samples)
            orig_timestamps, fine_delays, phase = _sample_models(
                self.delay_models, start_timestamp, max_end, self.spectra_samples
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
                # - 10-bit to float conversion
                # - Coarse delay
                # - The PFB-FIR.
                if batch_spectra > 0:
                    logging.debug("Processing %d spectra", batch_spectra)
                    out_slice = np.s_[self._out_item.n_spectra : self._out_item.n_spectra + batch_spectra]
                    self._out_item.fine_delay[out_slice] = fine_delays.T
                    # Divide by pi because the arguments of sincospif() used in the
                    # kernel are in radians/PI.
                    self._out_item.phase[out_slice] = phase.T / np.pi
                    samples = []
                    for pol_data in self._in_items[0].pol_data:
                        assert pol_data.samples is not None
                        samples.append(pol_data.samples)
                    self._compute.run_frontend(samples, offsets, self._out_item.n_spectra, batch_spectra)
                    self._out_item.n_spectra += batch_spectra
                    # Work out which output spectra contain missing data.
                    self._out_item.present[out_slice] = True
                    for pol, pol_data in enumerate(self._in_items[0].pol_data):
                        # Offset in the chunk of the first sample for each spectrum
                        first_offset = np.arange(
                            offsets[pol],
                            offsets[pol] + batch_spectra * self.spectra_samples,
                            self.spectra_samples,
                        )
                        # Offset of the last sample (inclusive, rather than past-the-end)
                        last_offset = first_offset + self.taps * self.spectra_samples - 1
                        first_packet = first_offset // self._src_packet_samples
                        # last_packet is exclusive
                        last_packet = last_offset // self._src_packet_samples + 1
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
                # TODO: should maybe also do this if _in_items[1] would work
                # just as well and we've filled the output buffer.
                self._pop_in()
        # Timestamp mostly doesn't matter because we're finished, but if a
        # katcp request arrives at this point we want to ensure the
        # steady-state-timestamp sensor is updated to a later timestamp than
        # anything we'll actually send.
        await self._flush_out(self._out_item.end_timestamp)
        logger.debug("_run_processing completed")
        self._out_queue.put_nowait(None)

    async def _run_receive(self, streams: List[spead2.recv.ChunkRingStream], layout: recv.Layout) -> None:
        """Receive data from the network, queue it up for processing.

        This function receives chunk sets, which are chunks in groups of two -
        one per polarisation, from the spead2 receiver streams given. For each
        chunk set received, copies of the data to the GPU are initiated,
        awaited, and then the chunk containers are returned to the receiver
        stream so that the memory need not be expensively re-allocated every
        time.

        In the GPU-direct case, <TODO clarify once I understand better>.

        Parameters
        ----------
        streams
            There should be only two of these because they each represent one of
            the digitiser's two polarisations.
        layout
            The structure of the streams.
        """
        async for chunks in recv.chunk_sets(streams, layout):
            with self.monitor.with_state("run_receive", "wait in_free_queue"):
                in_item = await self._in_free_queue.get()
            with self.monitor.with_state("run_receive", "wait events"):
                # Make sure all the item's events are past.
                await in_item.async_wait_for_events()
            in_item.reset(chunks[0].timestamp)

            # In steady-state, chunks should be the same size, but during
            # shutdown, the last chunk may be short.
            in_item.n_samples = chunks[0].data.nbytes * BYTE_BITS // self.sample_bits

            transfer_events = []
            for pol_data, chunk in zip(in_item.pol_data, chunks):
                # Copy the present flags (synchronously).
                pol_data.present[: len(chunk.present)] = chunk.present
            if self.use_vkgdr:
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is None
                    pol_data.samples = chunk.device  # type: ignore
                    pol_data.chunk = chunk
                self._in_queue.put_nowait(in_item)
            else:
                # Copy each pol chunk to the right place on the GPU.
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is not None
                    pol_data.samples.set_region(
                        self._upload_queue, chunk.data, np.s_[: chunk.data.nbytes], np.s_[:], blocking=False
                    )
                    transfer_events.append(self._upload_queue.enqueue_marker())

                # Put events on the queue so that _run_processing() knows when to
                # start.
                in_item.events.extend(transfer_events)
                self._in_queue.put_nowait(in_item)

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
        logger.debug("run_receive completed")
        self._in_queue.put_nowait(None)

    def _chunk_finished(self, chunk: send.Chunk, future: asyncio.Future) -> None:
        """Return a chunk to the free queue after it has completed transmission.

        This is intended to be used as a callback on an :class:`asyncio.Future`.
        """
        if chunk.cleanup is not None:
            chunk.cleanup()
            chunk.cleanup = None  # Potentially helps break reference cycles
        try:
            future.result()  # No result, but want the exception
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error sending chunk")

    async def _run_transmit(self, stream: "spead2.send.asyncio.AsyncStream") -> None:
        """Get the processed data from the GPU to the Network.

        This could be done either with or without PeerDirect. In the
        non-PeerDirect case, :class:`OutItem` objects are pulled from the
        `_out_queue`. We wait for the events that mark the end of the processing,
        then copy the data to host memory before turning it over to the
        :obj:`sender` for transmission on the network. The "empty" item is then
        returned to :meth:`_run_processing` via the `_out_free_queue`, and once
        the chunk has been transmitted it is returned to `_send_free_queue`.

        In the PeerDirect case, the item and the chunk are bound together and
        share memory. In this case `_send_free_queue` is unused. The item is
        only returned to `_out_free_queue` once it has been fully transmitted.

        Parameters
        ----------
        stream
            The stream transmitting data.
        """
        task: Optional[asyncio.Future] = None
        last_end_timestamp: Optional[int] = None
        while True:
            with self.monitor.with_state("run_transmit", "wait out_queue"):
                out_item = await self._out_queue.get()
            if not out_item:
                break
            if out_item.chunk is not None:
                # We're using PeerDirect
                chunk = out_item.chunk
                chunk.cleanup = partial(self._out_free_queue.put_nowait, out_item)
                events = out_item.events
            else:
                with self.monitor.with_state("run_transmit", "wait send_free_queue"):
                    chunk = await self._send_free_queue.get()
                chunk.cleanup = partial(self._send_free_queue.put_nowait, chunk)
                self._download_queue.enqueue_wait_for_events(out_item.events)
                assert isinstance(chunk.data, accel.HostArray)
                # TODO: use get_region since it might be partial
                out_item.spectra.get_async(self._download_queue, chunk.data)
                events = [self._download_queue.enqueue_marker()]

            chunk.timestamp = out_item.timestamp
            # Each frame is valid if all spectra in it are valid
            out_item.present.reshape(-1, self.spectra_per_heap).all(axis=-1, out=chunk.present)
            with self.monitor.with_state("run_transmit", "wait transfer"):
                await async_wait_for_events(events)
            n_frames = out_item.n_spectra // self.spectra_per_heap
            if last_end_timestamp is not None and out_item.timestamp > last_end_timestamp:
                # Account for heaps skipped between the end of the previous out_item and the
                # start of the current one.
                skipped_samples = out_item.timestamp - last_end_timestamp
                skipped_frames = skipped_samples // (self.spectra_per_heap * self.spectra_samples)
                send.skipped_heaps_counter.inc(skipped_frames * stream.num_substreams)
            last_end_timestamp = out_item.end_timestamp
            out_item.reset()  # Safe to call in PeerDirect mode since it doesn't touch the raw data
            if out_item.chunk is None:
                # We're not in PeerDirect mode
                # (when we are the cleanup callback returns the item)
                self._out_free_queue.put_nowait(out_item)
            task = asyncio.create_task(chunk.send(stream, n_frames))
            task.add_done_callback(partial(self._chunk_finished, chunk))

        if task:
            try:
                await task
            except Exception:
                pass  # It's already logged by the chunk_finished callback
        stop_heap = spead2.send.Heap(send.FLAVOUR)
        stop_heap.add_end()
        for substream_index in range(stream.num_substreams):
            await stream.async_send_heap(stop_heap, substream_index=substream_index)
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

        orig_delay = delay_models[0].delay / self.adc_sample_rate
        phase_rate_correction = 0.5 * np.pi * delay_models[0].delay_rate
        orig_phase = wrap_angle(delay_models[0].phase - 0.5 * np.pi * delay_models[0].delay)
        orig_phase_rate = (delay_models[0].phase_rate - phase_rate_correction) * self.adc_sample_rate
        delay_sensor.value = (
            f"({delay_models[0].start}, "
            f"{orig_delay}, "
            f"{delay_models[0].delay_rate}, "
            f"{orig_phase}, "
            f"{orig_phase_rate})"
        )

    def _update_steady_state_timestamp(self, timestamp: int) -> None:
        """Update the ``steady-state-timestamp`` sensor to at least a given value."""
        sensor = self.sensors["steady-state-timestamp"]
        sensor.value = max(sensor.value, timestamp)

    def set_gains(self, input: int, gains: np.ndarray) -> None:
        """Set the eq gains for one polarisation and update the sensor.

        The `gains` must contain one entry per channel; the shortcut of
        supplying a single value is handled by :meth:`request_gain`.
        """
        self.gains[:, input] = gains
        # This timestamp is conservative: self._out_item.timestamp is almost
        # always valid, except while _flush_out is waiting to update
        # self._out_item. If a less conservative answer is needed, one would
        # need to track a separate timestamp in the class that is updated
        # as gains are copied to the OutItem.
        self._update_steady_state_timestamp(self._out_item.end_timestamp)
        if np.all(gains == gains[0]):
            # All the values are the same, so it can be reported as a single value
            gains = gains[:1]
        self.sensors[f"input{input}-eq"].value = "[" + ", ".join(format_complex(gain) for gain in gains) + "]"

    def _parse_gains(self, *values: str, allow_default: bool) -> np.ndarray:
        """Parse the gains passed to :meth:`request-gain` or :meth:`request-gain-all`.

        If a single value is given it is expanded to a value per channel. If
        `allow_default` is true, the string "default" may be given to restore
        the default gains set via command line.

        The caller must ensure that `values` contains either 1 or `channels`
        items. :meth:`request_gain` handles the case where no values are given
        for querying existing gains.

        Failures are reported by raising an appropriate
        :exc:`aiokatcp.FailReply`.
        """
        if allow_default and values == ("default",):
            gains = np.full(self.channels, self.default_gain, dtype=np.complex64)
        else:
            try:
                gains = np.array([complex(v) for v in values], dtype=np.complex64)
            except ValueError:
                raise aiokatcp.FailReply("invalid formatting of complex number")
        if not np.all(np.isfinite(gains)):
            raise aiokatcp.FailReply("non-finite gains are not permitted")
        if len(gains) == 1:
            gains = gains.repeat(self.channels)
        return gains

    async def request_gain(self, ctx, input: int, *values: str) -> Tuple[str, ...]:
        """Set or query the eq gains.

        If no values are provided, the gains are simply returned.

        Parameters
        ----------
        input
            Input number (0 or 1)
        values
            Complex values. There must either be a single value (used for all
            channels), or a value per channel.
        """
        if not 0 <= input < N_POLS:
            raise aiokatcp.FailReply("input is out of range")
        if len(values) not in {0, 1, self.channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 0, 1 or {self.channels})")
        if not values:
            gains = self.gains[:, input]
        else:
            gains = self._parse_gains(*values, allow_default=False)
            self.set_gains(input, gains)

        # Return the current values.
        # If they're all the same, we can return just a single value.
        if np.all(gains == gains[0]):
            gains = gains[:1]
        return tuple(format_complex(gain) for gain in gains)

    async def request_gain_all(self, ctx, *values: str) -> None:
        """Set the eq gains for all inputs.

        Parameters
        ----------
        values
            Complex values. There must either be a single value (used for all
            channels), or a value per channel, or ``"default"`` to reset gains
            to the default.
        """
        if len(values) not in {1, self.channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 1 or {self.channels})")
        gains = self._parse_gains(*values, allow_default=True)
        for i in range(N_POLS):
            self.set_gains(i, gains)

    async def request_delays(self, ctx, start_time: aiokatcp.Timestamp, *delays: str) -> None:
        """Add a new first-order polynomial to the delay and fringe correction model.

        .. todo::

          Make the request's fail replies more informative in the case of
          malformed requests.
        """

        def comma_string_to_float(comma_string: str) -> Tuple[float, float]:
            a_str, b_str = comma_string.split(",")
            a = float(a_str)
            b = float(b_str)
            return a, b

        if len(delays) != len(self.delay_models):
            raise aiokatcp.FailReply(f"wrong number of delay coefficient sets (expected {len(self.delay_models)})")

        # This will round the start time of the new delay model to the nearest
        # ADC sample. If the start time given doesn't coincide with an ADC sample,
        # then all subsequent delays for this model will be off by the product
        # of this delta and the delay_rate (same for phase).
        # This may be too small to be a concern, but if it is a concern,
        # then we'd need to compensate for that here.
        start_sample_count = round((start_time - self.sync_epoch) * self.adc_sample_rate)
        if start_sample_count < 0:
            raise aiokatcp.FailReply("Start time cannot be prior to the sync epoch")

        # Collect them in a temporary until they're all validated
        new_linear_models = []
        for coeffs in delays:
            delay_str, phase_str = coeffs.split(":")
            delay, delay_rate = comma_string_to_float(delay_str)
            phase, phase_rate = comma_string_to_float(phase_str)
            if delay < 0:
                raise aiokatcp.FailReply("delay cannot be negative")
            if delay + 5 * delay_rate < 0:
                logger.warning("delay will become negative within 5s")

            delay_samples = delay * self.adc_sample_rate
            # For compatibility with MeerKAT, the phase given is the net change in
            # phase for the centre frequency, including delay, and we need to
            # compensate for the effect of the delay at that frequency. The centre
            # frequency is 4 samples per cycle, so each sample of delay reduces
            # phase by pi/2 radians.
            delay_phase_correction = 0.5 * np.pi * delay_samples
            phase += delay_phase_correction
            phase_rate_correction = 0.5 * np.pi * delay_rate
            new_linear_models.append(
                LinearDelayModel(
                    start_sample_count,
                    delay_samples,
                    delay_rate,
                    phase,
                    phase_rate / self.adc_sample_rate + phase_rate_correction,
                )
            )

        for delay_model, new_linear_model in zip(self.delay_models, new_linear_models):
            delay_model.add(new_linear_model)
        self._update_steady_state_timestamp(self.delay_update_timestamp())

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
        descriptor_sender = DescriptorSender(
            self._send_stream,
            self._descriptor_heap,
            self.n_ants * descriptor_interval_s,
            (self.feng_id + 1) * descriptor_interval_s,
        )
        self._descriptor_task = asyncio.create_task(descriptor_sender.run(), name=DESCRIPTOR_TASK_NAME)
        self.add_service_task(self._descriptor_task)

        for pol, stream in enumerate(self._src_streams):
            base_recv.add_reader(
                stream,
                src=self._srcs[pol],
                interface=self._src_interface,
                ibv=self._src_ibv,
                comp_vector=self._src_comp_vector[pol],
                buffer=self._src_buffer,
            )

        proc_task = asyncio.create_task(
            self._run_processing(),
            name=GPU_PROC_TASK_NAME,
        )
        self.add_service_task(proc_task)

        recv_task = asyncio.create_task(
            self._run_receive(self._src_streams, self._src_layout),
            name=RECV_TASK_NAME,
        )
        self.add_service_task(recv_task)

        send_task = asyncio.create_task(
            self._run_transmit(self._send_stream),
            name=SEND_TASK_NAME,
        )
        self.add_service_task(send_task)

        await super().start()

    async def on_stop(self) -> None:
        """Shut down processing when the device server is stopped.

        This is called by aiokatcp after closing the listening socket.
        Also handle any Exceptions thrown unexpectedly in any of the
        processing loops.
        """
        self._descriptor_task.cancel()
        for stream in self._src_streams:
            stream.stop()
        # If any of the tasks are already done then we had an exception, and
        # waiting for the rest may hang as the shutdown path won't proceed
        # neatly.
        if not any(task.done() for task in self.service_tasks):
            for task in self.service_tasks:
                if task is not self._descriptor_task:
                    await task
