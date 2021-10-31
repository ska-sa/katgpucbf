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

"""Engine class, which combines all the processing steps for a single digitiser data stream."""

import asyncio
import logging
from typing import List, Optional, Tuple, TypedDict, Union

import aiokatcp
import katsdpsigproc.accel as accel
import numpy as np
import spead2.recv
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from .. import COMPLEX, N_POLS, __version__
from ..monitor import Monitor
from . import BYTE_BITS, recv, send
from .compute import ComputeTemplate
from .delay import LinearDelayModel, MultiDelayModel
from .process import Processor

logger = logging.getLogger(__name__)


def generate_weights(channels: int, taps: int) -> np.ndarray:
    """Generate Hann-window weights for the F-engine's PFB-FIR.

    .. todo::

      Check for off-by-one/off-by-half issues. Seems to produce a filter which
      is not perfectly symmetrical, which could produce some phase response in
      the frequency domain.

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
    sinc = np.sinc(idx / step - taps / 2)
    weights = hann * sinc
    return weights.astype(np.float32)


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
    spectra
        Number of spectra that will be produced from a chunk of incoming
        digitiser data.
    spectra_per_heap
        Number of spectra in each output heap.
    channels
        Number of output channels to produce.
    taps
        Number of taps in each branch of the PFB-FIR
    quant_gain
        Rescaling factor to apply before 8-bit requantisation.
    sync_epoch
        UNIX time at which the digitisers were synced.
    mask_timestamp
        Mask off bottom bits of timestamp (workaround for broken digitiser).
    use_gdrcopy
        Assemble chunks directly in GPU memory (requires supported GPU).
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
        spectra: int,
        spectra_per_heap: int,
        channels: int,
        taps: int,
        quant_gain: float,
        sync_epoch: float,
        mask_timestamp: bool,
        use_gdrcopy: bool,
        use_peerdirect: bool,
        monitor: Monitor,
    ) -> None:
        super(Engine, self).__init__(katcp_host, katcp_port)
        self.populate_sensors(self.sensors)
        self.sensors["quant-gain"].value = quant_gain

        if use_gdrcopy:
            import gdrcopy.pycuda

            gdr = gdrcopy.Gdr()

        self.sync_epoch = sync_epoch
        self.adc_sample_rate = adc_sample_rate
        self.send_rate_factor = send_rate_factor
        self.delay_models = [MultiDelayModel() for _ in range(N_POLS)]
        queue = context.create_command_queue()
        template = ComputeTemplate(context, taps)
        chunk_samples = spectra * channels * 2
        extra_samples = taps * channels * 2
        compute = template.instantiate(queue, chunk_samples + extra_samples, spectra, spectra_per_heap, channels)
        chunk_bytes = chunk_samples * compute.sample_bits // BYTE_BITS
        device_weights = compute.slots["weights"].allocate(accel.DeviceAllocator(context))
        device_weights.set(queue, generate_weights(channels, taps))
        compute.quant_gain = quant_gain
        self._processor = Processor(compute, self.delay_models, use_gdrcopy, monitor, self.sensors)

        ringbuffer_capacity = 2
        ring = spead2.recv.asyncio.ChunkRingbuffer(ringbuffer_capacity)
        monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_ibv = src_ibv
        self._src_layout = recv.Layout(compute.sample_bits, src_packet_samples, chunk_samples, mask_timestamp)
        self._src_streams = [
            recv.make_stream(
                pol,
                self._src_layout,
                ring,
                src_affinity[pol],
                monitor=monitor,
            )
            for pol in range(N_POLS)
        ]
        src_chunks_per_stream = 4
        monitor.event_qsize("free_chunks", 0, src_chunks_per_stream * len(self._src_streams))
        for pol, stream in enumerate(self._src_streams):
            for _ in range(src_chunks_per_stream):
                if use_gdrcopy:
                    device_bytes = compute.slots[f"in{pol}"].required_bytes()
                    with context:
                        device_raw, buf_raw, _ = gdrcopy.pycuda.allocate_raw(gdr, device_bytes)
                    buf = np.frombuffer(buf_raw, np.uint8)
                    # The device buffer contains extra space for copying the head
                    # of the following chunk, but we don't need that in the host
                    # mapping.
                    buf = buf[:chunk_bytes]
                    # Hack to work around limitations in katsdpsigproc and pycuda
                    device_array = accel.DeviceArray(context, (device_bytes,), np.uint8, raw=int(device_raw))
                    device_array.buffer.base = device_raw
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

    @staticmethod
    def populate_sensors(sensors: aiokatcp.SensorSet) -> None:
        """Define the sensors for an engine."""
        # No more than once per second, at least once every 10 seconds.
        # The TypedDict is necessary for mypy to believe the use as
        # kwargs is valid.
        AutoStrategy = TypedDict(  # noqa: N806
            "AutoStrategy",
            {
                "auto_strategy": aiokatcp.SensorSampler.Strategy,
                "auto_strategy_parameters": Tuple[float, float],
            },
        )
        auto_strategy = AutoStrategy(
            auto_strategy=aiokatcp.SensorSampler.Strategy.EVENT_RATE,
            auto_strategy_parameters=(1.0, 10.0),
        )
        sensor_list: List[aiokatcp.Sensor] = [
            aiokatcp.Sensor(
                int,
                "input-heaps-total",
                "number of heaps received (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                int,
                "input-chunks-total",
                "number of chunks received (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                int,
                "input-bytes-total",
                "number of bytes of digitiser samples received (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                int,
                "input-missing-heaps-total",
                "number of heaps dropped on the input (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                # TODO: Think about what status_func should do for the status of the
                # sensor. If it goes into "warning" as soon as a single packet is
                # dropped, then it may not be too useful. Having the information
                # necessary to implement this may involve shifting things between
                # classes.
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                int,
                "output-heaps-total",
                "number of heaps transmitted (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                int,
                "output-bytes-total",
                "number of payload bytes transmitted (prometheus: counter)",
                default=0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                **auto_strategy,
            ),
            aiokatcp.Sensor(
                float,
                "quant-gain",
                "rescaling factor to apply before 8-bit requantisation (prometheus: gauge)",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            ),
        ]
        for sensor in sensor_list:
            sensors.add(sensor)

    async def request_quant_gain(self, ctx, quant_gain: float) -> None:
        """Set the quant gain."""
        self._processor.compute.quant_gain = quant_gain
        # We'll use the actual value of the property instead of the argument
        # passed here, in case there's some kind of setter function which may
        # modify it in any way.
        self.sensors["quant-gain"].set_value(self._processor.compute.quant_gain)

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

        for coeffs, delay_model in zip(delays, self.delay_models):
            delay_str, phase_str = coeffs.split(":")
            delay, delay_rate = comma_string_to_float(delay_str)
            phase, phase_rate = comma_string_to_float(phase_str)

            delay_samples = delay * self.adc_sample_rate
            # For compatibility with MeerKAT, the phase given is the net change in
            # phase for the centre frequency, including delay, and we need to
            # compensate for the effect of the delay at that frequency. The centre
            # frequency is 4 samples per cycle, so each sample of delay reduces
            # phase by pi/2 radians.
            delay_phase_correction = 0.5 * np.pi * delay_samples
            phase += delay_phase_correction
            phase_rate_correction = 0.5 * np.pi * delay_rate
            new_linear_model = LinearDelayModel(
                start_sample_count,
                delay_samples,
                delay_rate,
                phase,
                phase_rate / self.adc_sample_rate + phase_rate_correction,
            )

            delay_model.add(new_linear_model)

    async def start(self) -> None:
        """Start the engine.

        This function adds the receive, processing and transmit tasks onto the
        event loop and does the `gather` so that they can do their thing
        concurrently.
        """
        for pol, stream in enumerate(self._src_streams):
            recv.add_reader(
                stream,
                src=self._srcs[pol],
                interface=self._src_interface,
                ibv=self._src_ibv,
                comp_vector=self._src_comp_vector[pol],
                buffer=self._src_buffer,
            )
        self._processing_task = asyncio.gather(
            asyncio.create_task(self._processor.run_processing(self._src_streams)),
            asyncio.create_task(self._processor.run_receive(self._src_streams, self._src_layout)),
            asyncio.create_task(self._processor.run_transmit(self._send_stream)),
        )

        def done_callback(future: asyncio.Future) -> None:
            try:
                future.result()  # Evaluate just for exceptions
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Processing failed with exception")

        self._processing_task.add_done_callback(done_callback)
        await super().start()

    async def on_stop(self):
        """Shut down processing when the device server is stopped.

        This is called by aiokatcp after closing the listening socket.
        """
        for stream in self._src_streams:
            stream.stop()
        try:
            await self._processing_task
        except Exception:
            pass  # Errors get logged by the done_callback above
