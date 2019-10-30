import asyncio
from typing import List, Tuple, Union, Optional

import numpy as np
import katsdpsigproc.accel as accel

from . import recv, send
from .compute import ComputeTemplate
from .process import Processor
from .delay import MultiDelayModel
from .types import AbstractContext


class Engine:
    """Combines all the processing steps for a single digitiser."""

    def __init__(self, *,
                 context: AbstractContext,
                 srcs: List[Union[str, List[Tuple[str, int]]]],
                 src_interface: Optional[str],
                 src_ibv: bool,
                 src_affinity: List[int],
                 src_packet_samples: int,
                 src_buffer: int,
                 dst: List[Tuple[str, int]],
                 dst_interface: str,
                 dst_ttl: int,
                 dst_ibv: bool,
                 dst_max_packet_size: int,
                 dst_affinity: List[int],
                 bandwidth: float,
                 spectra: int, acc_len: int,
                 channels: int, taps: int) -> None:
        self.delay_model = MultiDelayModel()
        queue = context.create_command_queue()
        template = ComputeTemplate(context, taps)
        chunk_samples = spectra * channels * 2
        extra_samples = taps * channels * 2 - 8
        compute = template.instantiate(
            queue, chunk_samples + extra_samples, spectra, acc_len, channels)
        pols = compute.pols
        self._processor = Processor(compute, self.delay_model)

        ring = recv.Ringbuffer(2)
        self._srcs = list(srcs)
        self._src_interface = src_interface
        self._src_buffer = src_buffer
        self._src_streams = [recv.Stream(pol, compute.sample_bits, src_packet_samples,
                                         chunk_samples, ring, src_affinity[pol])
                             for pol in range(pols)]
        self._sender = send.Sender(2, dst_affinity)
        for stream in self._src_streams:
            for i in range(4):
                buf = accel.HostArray((stream.chunk_bytes,), np.uint8, context=context)
                stream.add_chunk(recv.Chunk(buf))
        for i in range(2):
            buf = accel.HostArray((spectra // acc_len, channels, acc_len, pols, 2), np.int8,
                                  context=context)
            self._sender.free_ring.try_push(send.Chunk(buf))
        # Send a bit faster than nominal bandwidth to account for header overheads
        rate = pols * bandwidth * buf.dtype.itemsize * 1.1
        for i, (host, port) in enumerate(dst):
            self._sender.add_udp_stream(host, port, dst_ttl, dst_interface, dst_ibv,
                                        dst_max_packet_size, rate, 2 * spectra // acc_len)

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            for pol, stream in enumerate(self._src_streams):
                if isinstance(self._srcs[pol], str):
                    stream.add_udp_pcap_file_reader(self._srcs[pol])
                else:
                    # TODO: use src_ibv
                    stream.add_udp_ibv_reader(self._srcs[pol], self._src_interface,
                                              self._src_buffer, pol)
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
