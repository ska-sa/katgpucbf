#!/usr/bin/env python3

"""Compare output of two F-engines.

This can be used to ensure that a change to F-engine code produces
bit-identical results. It subscribes to only one multicast group per F-engine,
so does not necessarily require high performance.

The two F-engines to compare must use F-engine IDs 0 and 1. Note that they
must be given bit-identical inputs!

See xbgpu for help on the command-line arguments.
"""

import argparse
import asyncio

import numpy as np
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import endpoint_parser

import katgpucbf.recv
import katgpucbf.xbgpu.recv
from katgpucbf import COMPLEX, N_POLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs=2, type=endpoint_parser(7148), help="Source groups for the two engines to compare")
    parser.add_argument("--interface", type=get_interface_address, help="Interface on which to listen")
    parser.add_argument("--ibv", action="store_true", help="Use ibverbs")
    parser.add_argument("--array-size", type=int, required=True)
    parser.add_argument("--channels-per-stream", type=int, required=True)
    parser.add_argument("--spectra-per-heap", type=int, default=256)
    parser.add_argument("--samples-between-spectra", type=int, required=True)
    parser.add_argument("--heaps-per-fengine-per-chunk", type=int, default=16)
    args = parser.parse_args()
    return args


async def main() -> None:
    args = parse_args()
    layout = katgpucbf.xbgpu.recv.Layout(
        n_ants=args.array_size,
        n_channels_per_stream=args.channels_per_stream,
        n_spectra_per_heap=args.spectra_per_heap,
        timestamp_step=args.samples_between_spectra * args.spectra_per_heap,
        sample_bits=8,
        heaps_per_fengine_per_chunk=args.heaps_per_fengine_per_chunk,
    )
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(2)
    free_ringbuffer = spead2.recv.ChunkRingbuffer(4)
    stream = katgpucbf.xbgpu.recv.make_stream(layout, data_ringbuffer, free_ringbuffer, -1, 2)
    for _ in range(free_ringbuffer.maxsize):
        shape = (
            layout.heaps_per_fengine_per_chunk,
            layout.n_ants,
            layout.n_channels_per_stream,
            layout.n_spectra_per_heap,
            N_POLS,
            COMPLEX,
        )
        data = np.ones(shape, np.int8)
        present = np.zeros(shape[:2], np.uint8)
        chunk = katgpucbf.recv.Chunk(data=data, present=present, stream=stream)
        chunk.recycle()

    srcs = [(ep.host, ep.port) for ep in args.src]
    katgpucbf.recv.add_reader(
        stream, src=srcs, interface=args.interface, ibv=args.ibv, comp_vector=12, buffer=32 * 1024 * 1024
    )
    async for chunk in data_ringbuffer:  # type: ignore
        with chunk:
            timestamp = chunk.chunk_id * layout.timestamp_step
            if not np.all(chunk.present[:, :2]):
                print(f"Received a chunk with timestamp {timestamp} but not all data present")
                print(chunk.present)
                continue
            if not np.all(chunk.data[:, 0] == chunk.data[:, 1]):
                print(f"Mismatch in chunk with timestamp {timestamp}")
                np.testing.assert_equal(chunk.data[:, 0], chunk.data[:, 1])
                break
            print(f"Chunk with timestamp {timestamp} is good")


if __name__ == "__main__":
    asyncio.run(main())
