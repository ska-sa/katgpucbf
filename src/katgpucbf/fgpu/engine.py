import asyncio
from typing import List, Tuple, Union, Optional

import numpy as np
import katsdpsigproc.accel as accel

from . import recv, send
from .compute import ComputeTemplate
from .process import Processor
from .delay import MultiDelayModel
from .monitor import Monitor
from .types import AbstractContext


def generate_weights(channels: int, taps: int) -> np.ndarray:
    step = 2 * channels
    window_size = step * taps
    idx = np.arange(window_size)
    hann = np.square(np.sin(np.pi * idx / (window_size - 1)))
    sinc = np.sinc(idx / step - taps / 2)
    weights = hann * sinc
    return weights.astype(np.float32)


class Engine:
    """Combines all the processing steps for a single digitiser."""

    def __init__(self, *,
                 context: AbstractContext,
                 srcs: List[Union[str, List[Tuple[str, int]]]],
                 src_interface: Optional[str],
                 src_ibv: bool,
                 src_affinity: List[int],
                 src_comp_vector: List[int],
                 src_packet_samples: int,
                 src_buffer: int,
                 dst: List[Tuple[str, int]],
                 dst_interface: str,
                 dst_ttl: int,
                 dst_ibv: bool,
                 dst_packet_payload: int,
                 dst_affinity: int,
                 dst_comp_vector: int,
                 adc_rate: float,
                 spectra: int, acc_len: int,
                 channels: int, taps: int,
                 quant_scale: float,
                 mask_timestamp: bool,
                 use_gdrcopy: bool,
                 use_peerdirect: bool,
                 monitor: Monitor) -> None:
        if use_gdrcopy:
            import gdrcopy.pycuda
            gdr = gdrcopy.Gdr()

        self.delay_model = MultiDelayModel()
        queue = context.create_command_queue()
        template = ComputeTemplate(context, taps)
        chunk_samples = spectra * channels * 2
        extra_samples = taps * channels * 2 - 8
        compute = template.instantiate(
            queue, chunk_samples + extra_samples, spectra, acc_len, channels)
        device_weights = compute.slots['weights'].allocate(accel.DeviceAllocator(context))
        device_weights.set(queue, generate_weights(channels, taps))
        compute.quant_scale = quant_scale
        pols = compute.pols
        self._processor = Processor(compute, self.delay_model, use_gdrcopy, monitor)

        ringbuffer_capacity = 2
        ring = recv.Ringbuffer(ringbuffer_capacity)
        monitor.event_qsize('recv_ringbuffer', 0, ringbuffer_capacity)
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_streams = [recv.Stream(pol, compute.sample_bits, src_packet_samples,
                                         chunk_samples, ring, src_affinity[pol],
                                         mask_timestamp=mask_timestamp,
                                         use_gdrcopy=use_gdrcopy,
                                         monitor=monitor)
                             for pol in range(pols)]
        src_chunks_per_stream = 4
        monitor.event_qsize('free_chunks', 0, src_chunks_per_stream * len(self._src_streams))
        for pol, stream in enumerate(self._src_streams):
            for i in range(src_chunks_per_stream):
                if use_gdrcopy:
                    device_bytes = compute.slots[f'in{pol}'].required_bytes()
                    with context:
                        device_raw, buf_raw, _ = gdrcopy.pycuda.allocate_raw(
                            gdr, device_bytes)
                    buf = np.frombuffer(buf_raw, np.uint8)
                    # The device buffer contains extra space for copying the head
                    # of the following chunk, but we don't need that in the host
                    # mapping.
                    buf = buf[:stream.chunk_bytes]
                    # Hack to work around limitations in katsdpsigproc and pycuda
                    device_array = accel.DeviceArray(
                        context, (device_bytes,), np.uint8, raw=int(device_raw))
                    device_array.buffer.base = device_raw
                    chunk = recv.Chunk(buf, device_array)
                else:
                    buf = accel.HostArray((stream.chunk_bytes,), np.uint8, context=context)
                    chunk = recv.Chunk(buf)
                stream.add_chunk(chunk)
        send_chunks = []
        send_shape = (spectra // acc_len, channels, acc_len, pols, 2)
        send_dtype = np.dtype(np.int8)
        for i in range(4):
            if use_peerdirect:
                dev = accel.DeviceArray(context, send_shape, send_dtype)
                buf = dev.buffer.gpudata.as_buffer(
                    int(np.product(send_shape) * send_dtype.itemsize))
                send_chunks.append(send.Chunk(buf, dev))
            else:
                buf = accel.HostArray(send_shape, send_dtype, context=context)
                send_chunks.append(send.Chunk(buf))
        memory_regions = [chunk.base for chunk in send_chunks]
        if use_peerdirect:
            memory_regions.extend(self._processor.peerdirect_memory_regions)
        # Send a bit faster than nominal rate to account for header overheads
        rate = pols * adc_rate * send_dtype.itemsize * 1.1
        # There is a SPEAD header, 8 item pointers,
        # and 3 padding pointers, for a 96 byte header.
        self._sender = send.Sender(
            len(send_chunks), memory_regions, dst_affinity, dst_comp_vector,
            [(d.host, d.port) for d in dst], dst_ttl, dst_interface, dst_ibv,
            dst_packet_payload + 96, rate, len(send_chunks) * spectra // acc_len * len(dst),
            monitor)
        monitor.event_qsize('send_free_ringbuffer', 0, len(send_chunks))
        for schunk in send_chunks:
            self._sender.push_free_ring(schunk)

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            for pol, stream in enumerate(self._src_streams):
                src = self._srcs[pol]
                if isinstance(src, str):
                    stream.add_udp_pcap_file_reader(src)
                else:
                    if self._src_interface is None:
                        raise ValueError('src_interface is required for UDP sources')
                    # TODO: use src_ibv
                    stream.add_udp_ibv_reader(src, self._src_interface, self._src_buffer,
                                              self._src_comp_vector[pol])
            tasks = [
                loop.create_task(self._processor.run_processing(self._src_streams)),
                loop.create_task(self._processor.run_receive(self._src_streams)),
                loop.create_task(self._processor.run_transmit(self._sender))
            ]
            await asyncio.gather(*tasks)
        finally:
            for stream in self._src_streams:
                stream.stop()
            self._sender.stop()
