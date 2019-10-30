#!/usr/bin/env python3

import argparse
import asyncio
import ipaddress
from typing import List, Tuple, Union

from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser
import numpy as np
import katsdpsigproc.accel as accel

import katfgpu.recv
from katfgpu.compute import ComputeTemplate
from katfgpu.process import Processor
from katfgpu.delay import MultiDelayModel


SAMPLE_BITS = 10
N_POL = 2
PACKET_SAMPLES = 4096

CHANNELS = 4096
CHUNK_SAMPLES = 2**26
SPECTRA = CHUNK_SAMPLES // (2 * CHANNELS)
OVERLAP_SAMPLES = 2**23
CHUNK_BYTES = CHUNK_SAMPLES * 10 // 8
ACC_LEN = 256
TAPS = 4


def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    try:
        endpoints = endpoint_list_parser(None)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
            if endpoint.port is None:
                raise ValueError
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', '-i', type=get_interface_address,
                        help='Interface for live capture')
    parser.add_argument('sources', type=parse_source, nargs=N_POL)
    args = parser.parse_args()
    for source in args.sources:
        if not isinstance(source, str) and args.interface is None:
            parser.error('Live source requires --interface')
    return args


async def main() -> None:
    loop = asyncio.get_event_loop()
    args = parse_args()
    ctx = accel.create_some_context()
    queue = ctx.create_command_queue()
    template = ComputeTemplate(ctx, TAPS)
    compute = template.instantiate(
        queue, CHUNK_SAMPLES + OVERLAP_SAMPLES, SPECTRA, ACC_LEN, CHANNELS)
    processor = Processor(compute, MultiDelayModel())

    ring = katfgpu.recv.Ringbuffer(2)
    streams = [katfgpu.recv.Stream(pol, SAMPLE_BITS, PACKET_SAMPLES, CHUNK_SAMPLES, ring)
               for pol in range(N_POL)]
    sender = katfgpu.send.Sender(2)
    try:
        for pol in range(N_POL):
            for i in range(4):
                buf = accel.HostArray((streams[pol].chunk_bytes,), np.uint8, context=ctx)
                streams[pol].add_chunk(katfgpu.recv.Chunk(buf))
            if isinstance(args.sources[pol], str):
                streams[pol].add_udp_pcap_file_reader(args.sources[pol])
            else:
                streams[pol].add_udp_ibv_reader(args.sources[pol], args.interface, 32 * 1024 * 1024, pol)

        for i in range(2):
            buf = accel.HostArray((SPECTRA // ACC_LEN, CHANNELS, ACC_LEN, N_POL, 2), np.int8, context=ctx)
            sender.free_ring.try_push(katfgpu.send.Chunk(buf))
        sender.add_udp_stream('239.102.123.0', 7149, 1, '127.0.0.1', False,
                              8872, 0, 2 * SPECTRA // ACC_LEN)

        tasks = [
            loop.create_task(processor.run_processing()),
            loop.create_task(processor.run_receive(streams)),
            loop.create_task(processor.run_transmit(sender))
        ]
        await asyncio.gather(*tasks)
        print('Done!')
    finally:
        for stream in streams:
            stream.stop()
        sender.stop()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
