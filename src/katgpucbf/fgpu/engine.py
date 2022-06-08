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
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import aiokatcp
import katsdpsigproc.accel as accel
import numpy as np
import spead2.recv
from katsdpsigproc.abc import AbstractContext
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
from ..ringbuffer import ChunkRingbuffer
from . import SAMPLE_BITS, recv, send
from .delay import LinearDelayModel, MultiDelayModel
from .process import Processor

logger = logging.getLogger(__name__)


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
    """A logical grouping to combine a `Processor` with other things it needs.

    The Engine class is essentially a wrapper around a
    :class:`~katgpucbf.fgpu.process.Processor` object, but adds a delay model,
    and source and sender functionality.

    .. todo::

      Perhaps in a future iteration, :class:`~katgpucbf.fgpu.process.Processor`
      could be folded into :class:`Engine`.

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
        self.populate_sensors(self.sensors)
        self.feng_id = feng_id
        self.n_ants = num_ants

        if use_vkgdr:
            import vkgdr.pycuda

            with context:
                # We could quite easily make do with non-coherent mappings and
                # explicit flushing, but since NVIDIA currently only provides
                # host-coherent memory, this is a simpler option.
                vkgdr_handle = vkgdr.Vkgdr.open_current_context(vkgdr.OpenFlags.REQUIRE_COHERENT_BIT)

        self.sync_epoch = sync_epoch
        self.adc_sample_rate = adc_sample_rate
        self.send_rate_factor = send_rate_factor
        self.default_gain = gain
        self.delay_models = []

        for pol in range(N_POLS):
            delay_model = MultiDelayModel(
                callback_func=partial(self.update_delay_sensor, delay_sensor=self.sensors[f"input{pol}-delay"])
            )
            self.delay_models.append(delay_model)

        extra_samples = max_delay_diff + taps * channels * 2
        if extra_samples > chunk_samples:
            raise RuntimeError(f"chunk_samples is too small; it must be at least {extra_samples}")
        self._processor = Processor(
            context,
            taps,
            chunk_samples + extra_samples,
            src_packet_samples,
            spectra,
            spectra_per_heap,
            channels,
            self.delay_models,
            use_vkgdr,
            monitor,
        )
        chunk_bytes = chunk_samples * SAMPLE_BITS // BYTE_BITS
        for pol in range(N_POLS):
            self.set_gains(pol, np.full(channels, gain, dtype=np.complex64))

        ringbuffer_capacity = 2
        data_ringbuffer = ChunkRingbuffer(
            ringbuffer_capacity, name="recv_data_ringbuffer", task_name="run_receive", monitor=monitor
        )
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_ibv = src_ibv
        self._src_layout = recv.Layout(SAMPLE_BITS, src_packet_samples, chunk_samples, mask_timestamp)
        src_chunks_per_stream = 4
        free_ringbuffers = [spead2.recv.ChunkRingbuffer(src_chunks_per_stream) for _ in range(N_POLS)]
        self._src_streams = recv.make_streams(self._src_layout, data_ringbuffer, free_ringbuffers, src_affinity)
        for pol, stream in enumerate(self._src_streams):
            for _ in range(src_chunks_per_stream):
                if use_vkgdr:
                    device_bytes = self._processor.compute.slots[f"in{pol}"].required_bytes()
                    with context:
                        mem = vkgdr.pycuda.Memory(vkgdr_handle, device_bytes)
                    buf = np.array(mem, copy=False).view(np.uint8)
                    # The device buffer contains extra space for copying the head
                    # of the following chunk, but we don't need that in the host
                    # mapping.
                    buf = buf[:chunk_bytes]
                    device_array = accel.DeviceArray(context, (device_bytes,), np.uint8, raw=mem)
                    chunk = recv.Chunk(data=buf, device=device_array)
                else:
                    buf = accel.HostArray((chunk_bytes,), np.uint8, context=context)
                    chunk = recv.Chunk(data=buf)
                chunk.present = np.zeros(chunk_samples // src_packet_samples, np.uint8)
                stream.add_free_chunk(chunk)

        send_chunks = []
        send_shape = (spectra // spectra_per_heap, channels, spectra_per_heap, N_POLS, COMPLEX)
        send_dtype = np.dtype(np.int8)
        for _ in range(self._processor.send_free_queue.maxsize):
            dev: Optional[accel.DeviceArray]
            if use_peerdirect:
                dev = accel.DeviceArray(context, send_shape, send_dtype)
                dev_buffer = dev.buffer.gpudata.as_buffer(int(np.product(send_shape) * send_dtype.itemsize))
                buf = np.frombuffer(dev_buffer, dtype=send_dtype).reshape(send_shape)
            else:
                dev = None
                buf = accel.HostArray(send_shape, send_dtype, context=context)
            send_chunks.append(
                send.Chunk(
                    buf,
                    device=dev,
                    substreams=len(dst),
                    feng_id=feng_id,
                )
            )

        extra_memory_regions = self._processor.peerdirect_memory_regions if use_peerdirect else []
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
            extra_memory_regions=extra_memory_regions,
        )
        for schunk in send_chunks:
            self._processor.send_free_queue.put_nowait(schunk)

        self._descriptor_heap_reflist = send.make_descriptor_heaps(
            data_type=send_dtype,
            channels=channels,
            substreams=len(dst),
            spectra_per_heap=spectra_per_heap,
        )

    @staticmethod
    def populate_sensors(sensors: aiokatcp.SensorSet) -> None:
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
        orig_phase_rate = (delay_models[0].phase_rate - phase_rate_correction) * self.adc_sample_rate
        delay_sensor.value = (
            f"({delay_models[0].start}, "
            f"{orig_delay}, "
            f"{delay_models[0].delay_rate}, "
            f"{delay_models[0].phase}, "
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
        self._update_steady_state_timestamp(self._processor.set_gains(input, gains))
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
        channels = self._processor.channels
        if allow_default and values == ("default",):
            gains = np.full(channels, self.default_gain, dtype=np.complex64)
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
        channels = self._processor.channels
        if len(values) not in {0, 1, channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 0, 1 or {channels})")
        if not values:
            gains = self._processor.gains[:, input]
        else:
            gains = self._parse_gains(*values, allow_default=False)
            self.set_gains(input, gains)

        # Return the current values.
        # If they're all the same, we can return just a single value.
        if np.all(gains == gains[0]):
            gains = gains[:1]
        return tuple(format_complex(gain) for gain in gains)

    async def request_gain_all(self, ctx, *values: str) -> None:
        """Set or query the eq gains for all inputs.

        Parameters
        ----------
        values
            Complex values. There must either be a single value (used for all
            channels), or a value per channel, or ``"default"`` to reset gains
            to the default.
        """
        channels = self._processor.channels
        if len(values) not in {1, channels}:
            raise aiokatcp.FailReply(f"invalid number of values provided (must be 1 or {channels})")
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
        start_sample_count = int((start_time - self.sync_epoch) * self.adc_sample_rate)

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
        self._update_steady_state_timestamp(self._processor.delay_update_timestamp())

    async def start(self, descriptor_interval_s: float = SPEAD_DESCRIPTOR_INTERVAL_S) -> None:
        """Start the engine.

        This function adds the receive, processing and transmit tasks onto the
        event loop. It also adds a task to continuously send the descriptor
        heaps at an interval based on the `descriptor_interval_s`. See
        :meth:`~Processor.run_descriptors_loop` for more details.

        Parameters
        ----------
        descriptor_interval_s
            The base interval used as a multiplier on feng_id and n_ants to
            dictate the initial 'engine sleep interval' and 'send interval'
            respectively.
        """
        # Create the descriptor task first to ensure descriptors will be sent
        # before any data makes its way through the pipeline.
        self._descriptor_task = asyncio.create_task(
            self._processor.run_descriptors_loop(
                self._send_stream,
                self._descriptor_heap_reflist,
                self.n_ants,
                self.feng_id,
                descriptor_interval_s,
            ),
            name=DESCRIPTOR_TASK_NAME,
        )
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
            self._processor.run_processing(self._src_streams),
            name=GPU_PROC_TASK_NAME,
        )
        self.add_service_task(proc_task)

        recv_task = asyncio.create_task(
            self._processor.run_receive(self._src_streams, self._src_layout),
            name=RECV_TASK_NAME,
        )
        self.add_service_task(recv_task)

        send_task = asyncio.create_task(
            self._processor.run_transmit(self._send_stream),
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
