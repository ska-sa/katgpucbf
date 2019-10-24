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


SAMPLE_BITS = 10
CHUNK_SAMPLES = 2**28
PACKET_SAMPLES = 4096
CHUNK_BYTES = CHUNK_SAMPLES * 10 // 8
N_POL = 2


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
    args = parse_args()
    ctx = accel.create_some_context()
    queue = ctx.create_command_queue()

    ring = katfgpu.recv.Ringbuffer(2)
    recv = [katfgpu.recv.Receiver(pol, SAMPLE_BITS, PACKET_SAMPLES, CHUNK_SAMPLES, ring)
            for pol in range(N_POL)]
    try:
        dev_samples = [accel.DeviceArray(ctx, (recv[pol].chunk_bytes,), np.uint8)
                       for pol in range(N_POL)]
        dev_timestamp = [None] * N_POL
        for pol in range(N_POL):
            for i in range(4):
                buf = accel.HostArray((recv[pol].chunk_bytes,), np.uint8, context=ctx)
                recv[pol].add_chunk(katfgpu.recv.Chunk(buf))
            if isinstance(args.sources[pol], str):
                recv[pol].add_udp_pcap_file_reader(args.sources[pol])
            else:
                recv[pol].add_udp_ibv_reader(args.sources[pol], args.interface, 32 * 1024 * 1024, pol)

        lost = 0
        async for chunk in ring:
            total = len(chunk.present)
            good = sum(chunk.present)
            lost += total - good
            print('Received chunk: timestamp={chunk.timestamp} pol={chunk.pol} ({good}/{total}, lost {lost})'.format(
                chunk=chunk, good=good, total=total, lost=lost))
            try:
                buf = chunk.base
                dev_samples[chunk.pol].set(queue, buf)
                dev_timestamp[chunk.pol] = chunk.timestamp
                if all(t == chunk.timestamp for t in dev_timestamp):
                    print(f'Ready to process timestamp {chunk.timestamp}')
            finally:
                recv[chunk.pol].add_chunk(chunk)
        print('Done!')
    finally:
        for r in recv:
            r.stop()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
