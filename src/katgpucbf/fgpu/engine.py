"""Engine class, which combines all the processing steps for a single digitiser data stream."""

import asyncio
from typing import List, Optional, Tuple, TypedDict, Union

import aiokatcp
import katsdpsigproc.accel as accel
import numpy as np
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from .. import __version__
from . import recv, send
from .compute import ComputeTemplate
from .delay import LinearDelayModel, MultiDelayModel
from .monitor import Monitor
from .process import Processor


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
    adc_rate
        Digitiser sampling rate (in Hz), used to determine transmission rate.
    feng_id
        ID of the F-engine indicating which one in the array this is. Included
        in the output heaps so that the X-engine can determine where the data
        fits in.
    spectra
        Number of spectra that will be produced from a chunk of incoming
        digitiser data.
    acc_len
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
        adc_rate: float,
        feng_id: int,
        spectra: int,
        acc_len: int,
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
        # The type ignore are because mypy doesn't
        sensors: List[aiokatcp.Sensor] = [
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
                float,
                "quant-gain",
                "rescaling factor to apply before 8-bit requantisation (prometheus: gauge)",
                default=quant_gain,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            ),
        ]
        for sensor in sensors:
            self.sensors.add(sensor)

        if use_gdrcopy:
            import gdrcopy.pycuda

            gdr = gdrcopy.Gdr()
        self.sync_epoch = sync_epoch
        self.adc_rate = adc_rate
        self.delay_model = MultiDelayModel()
        queue = context.create_command_queue()
        template = ComputeTemplate(context, taps)
        chunk_samples = spectra * channels * 2
        extra_samples = taps * channels * 2
        compute = template.instantiate(queue, chunk_samples + extra_samples, spectra, acc_len, channels)
        device_weights = compute.slots["weights"].allocate(accel.DeviceAllocator(context))
        device_weights.set(queue, generate_weights(channels, taps))
        compute.quant_gain = quant_gain
        pols = compute.pols
        self._processor = Processor(compute, self.delay_model, use_gdrcopy, monitor, self.sensors)

        ringbuffer_capacity = 2
        ring = recv.Ringbuffer(ringbuffer_capacity)
        monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_streams = [
            recv.Stream(
                pol,
                compute.sample_bits,
                src_packet_samples,
                chunk_samples,
                ring,
                src_affinity[pol],
                mask_timestamp=mask_timestamp,
                use_gdrcopy=use_gdrcopy,
                monitor=monitor,
            )
            for pol in range(pols)
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
                    buf = buf[: stream.chunk_bytes]
                    # Hack to work around limitations in katsdpsigproc and pycuda
                    device_array = accel.DeviceArray(context, (device_bytes,), np.uint8, raw=int(device_raw))
                    device_array.buffer.base = device_raw
                    chunk = recv.Chunk(buf, device_array)
                else:
                    buf = accel.HostArray((stream.chunk_bytes,), np.uint8, context=context)
                    chunk = recv.Chunk(buf)
                stream.add_chunk(chunk)
        send_chunks = []
        send_shape = (spectra // acc_len, channels, acc_len, pols, 2)
        send_dtype = np.dtype(np.int8)
        for _ in range(4):
            if use_peerdirect:
                dev = accel.DeviceArray(context, send_shape, send_dtype)
                buf = dev.buffer.gpudata.as_buffer(int(np.product(send_shape) * send_dtype.itemsize))
                send_chunks.append(send.Chunk(buf, dev))
            else:
                buf = accel.HostArray(send_shape, send_dtype, context=context)
                send_chunks.append(send.Chunk(buf))
        memory_regions = [chunk.base for chunk in send_chunks]
        if use_peerdirect:
            memory_regions.extend(self._processor.peerdirect_memory_regions)
        # Send a bit faster than nominal rate to account for header overheads
        rate = pols * adc_rate * send_dtype.itemsize * 1.1
        # There is a SPEAD header, 8 item pointers, and 3 padding pointers for
        # a 96 byte header, matching the MeerKAT packet format.
        self._sender = send.Sender(
            len(send_chunks),
            memory_regions,
            dst_affinity,
            dst_comp_vector,
            feng_id,
            [(d.host, d.port) for d in dst],
            dst_ttl,
            dst_interface,
            dst_ibv,
            dst_packet_payload + 96,  # TODO make this into some kind of parameter. A naked 96 makes me nervous.
            rate,
            len(send_chunks) * spectra // acc_len * len(dst),
            monitor,
        )
        monitor.event_qsize("send_free_ringbuffer", 0, len(send_chunks))
        for schunk in send_chunks:
            self._sender.push_free_ring(schunk)

    async def request_quant_gain(self, ctx, quant_gain: float) -> None:
        """Set the quant gain."""
        self._processor.compute.quant_gain = quant_gain
        # We'll use the actual value of the property instead of the argument
        # passed here, in case there's some kind of setter function which may
        # modify it in any way.
        self.sensors["quant-gain"].set_value(self._processor.compute.quant_gain)

    async def request_delays(self, ctx, start_time: aiokatcp.Timestamp, delays: str) -> None:
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

        delay_str, phase_str = delays.split(":")
        delay, delay_rate = comma_string_to_float(delay_str)
        phase, phase_rate = comma_string_to_float(phase_str)

        # This will round the start time of the new delay model to the nearest
        # ADC sample. If the start time given doesn't coincide with an ADC sample,
        # then all subsequent delays for this model will be off by the product
        # of this delta and the delay_rate (same for phase).
        # This may be too small to be a concern, but if it is a concern,
        # then we'd need to compensate for that here.
        start_sample_count = int((start_time - self.sync_epoch) * self.adc_rate)

        new_linear_model = LinearDelayModel(start_sample_count, delay, delay_rate, phase, phase_rate)

        self.delay_model.add(new_linear_model)

    async def run(self) -> None:
        """Run the engine.

        This function adds the receive, processing and transmit tasks onto the
        event loop and does the `gather` so that they can do their thing
        concurrently.

        .. todo::

          The shutdown process is something of a wild west at the moment. It
          needs fairly serious cleaning up. One aspect of this will be the
          `await self.stop()` in the `finally:` statement. If the task is
          cancelled, it will raise an exception, and if the finally was already
          doing cleanup from an exception then I think you lose one of the
          exceptions. Which is a problem.
        """
        loop = asyncio.get_event_loop()
        await self.start()
        try:
            for pol, stream in enumerate(self._src_streams):
                src = self._srcs[pol]
                if isinstance(src, str):
                    stream.add_udp_pcap_file_reader(src)
                else:
                    if self._src_interface is None:
                        raise ValueError("src_interface is required for UDP sources")
                    # TODO: use src_ibv                     ...?
                    stream.add_udp_ibv_reader(src, self._src_interface, self._src_buffer, self._src_comp_vector[pol])
            tasks = [
                loop.create_task(self._processor.run_processing(self._src_streams)),
                loop.create_task(self._processor.run_receive(self._src_streams)),
                loop.create_task(self._processor.run_transmit(self._sender)),
            ]
            await asyncio.gather(*tasks)
        finally:
            for stream in self._src_streams:
                stream.stop()
            self._sender.stop()
            await self.stop()
