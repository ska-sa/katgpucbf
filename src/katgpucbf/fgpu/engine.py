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
                 dst_affinity: List[int],
                 dst_comp_vector: List[int],
                 adc_rate: float,
                 spectra: int, acc_len: int,
                 channels: int, taps: int,
                 quant_scale: float,
                 mask_timestamp: bool,
                 monitor: Monitor) -> None:
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
        self._processor = Processor(compute, self.delay_model, monitor)

        ring = recv.Ringbuffer(2)
        self._srcs = list(srcs)
        self._src_comp_vector = list(src_comp_vector)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_streams = [recv.Stream(pol, compute.sample_bits, src_packet_samples,
                                         chunk_samples, ring, src_affinity[pol],
                                         mask_timestamp=mask_timestamp)
                             for pol in range(pols)]
        for stream in self._src_streams:
            for i in range(4):
                buf = accel.HostArray((stream.chunk_bytes,), np.uint8, context=context)
                stream.add_chunk(recv.Chunk(buf))
        send_bufs = []
        for i in range(2):
            buf = accel.HostArray((spectra // acc_len, channels, acc_len, pols, 2), np.int8,
                                  context=context)
            send_bufs.append(buf)
        # Send a bit faster than nominal rate to account for header overheads
        rate = pols * adc_rate * buf.dtype.itemsize * 1.1
        # There is a SPEAD header, 8 item pointers,
        # and 3 padding pointers, for a 96 byte header.
        self._sender = send.Sender(
            send_bufs, dst_affinity, dst_comp_vector,
            [(d.host, d.port) for d in dst], dst_ttl, dst_interface, dst_ibv,
            dst_packet_payload + 96, rate, len(send_bufs) * spectra // acc_len * len(dst))

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
                loop.create_task(self._processor.run_processing()),
                loop.create_task(self._processor.run_receive(self._src_streams)),
                loop.create_task(self._processor.run_transmit(self._sender))
            ]
            await asyncio.gather(*tasks)
        finally:
            for stream in self._src_streams:
                stream.stop()
            self._sender.stop()
