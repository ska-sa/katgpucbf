################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Digitiser simulator.

Simulates the packet structure of MeerKAT digitisers.
"""

import argparse
import asyncio
import logging
import time
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import katsdpservices
import numpy as np
import spead2.send
import toolz
from katsdptelstate.endpoint import endpoint_list_parser

from .. import BYTE_BITS, DEFAULT_TTL
from . import send, signal

logger = logging.getLogger(__name__)


def parse_args(arglist: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="dsim")
    parser.add_argument("--adc-sample-rate", type=float, default=1712e6, help="Digitiser sampling rate [%(default)s]")
    parser.add_argument(
        "--signal-freq", type=float, default=232101234.0, help="Frequency of simulated tone [%(default)s]"
    )
    parser.add_argument("--sync-time", type=float, help="Sync time in UNIX epoch seconds (must be in the past)")
    parser.add_argument("--interface", default="lo", help="Network interface on which to send packets [%(default)s]")
    parser.add_argument("--max-heaps", type=int, default=128, help="Depth of send queue (per interface) [%(default)s]")
    parser.add_argument("--heap-samples", type=int, default=4096, help="Number of samples per heap [%(default)s]")
    parser.add_argument("--sample-bits", type=int, default=10, help="Number of bits per sample [%(default)s]")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="IP TTL for multicast [%(default)s]")
    parser.add_argument("--ibv", action="store_true", help="Use ibverbs for acceleration")
    parser.add_argument(
        "--signal-heaps", type=int, default=32768, help="Length of pre-computed signal in heaps [%(default)s]"
    )
    parser.add_argument(
        "dest",
        nargs="+",
        type=endpoint_list_parser(None),
        metavar="X.X.X.X[+N]:PORT",
        help="Destination addresses (one per polarisation)",
    )

    args = parser.parse_args()
    if args.max_heaps <= 0:
        parser.error("--max-heaps must be positive")
    if args.signal_heaps <= 0:
        parser.error("--signal-heaps must be positive")
    for dest in args.dest:
        for ep in dest:
            if ep.port is None:
                parser.error("port must be specified on destinations")
    # Round target frequency to fit an integer number of waves into signal_heaps
    waves = max(1, round(args.signal_heaps * args.heap_samples * args.signal_freq / args.adc_sample_rate))
    args.signal_freq = waves * args.adc_sample_rate / args.signal_heaps / args.heap_samples
    logger.info(f"Rounded tone frequency to {args.signal_freq} Hz")
    return args


def first_timestamp(sync_time: float, now: float, adc_sample_rate: float, heap_samples: int) -> int:
    """Determine ADC timestamp for first sample."""
    # Convert to heap count (rounding)
    first_heap = round((now - sync_time) * adc_sample_rate / heap_samples)
    if first_heap < 0:
        raise ValueError("sync time is in the future")
    # Convert to a sample count
    return first_heap * heap_samples


async def async_main() -> None:
    """Asynchronous main entry point."""
    args = parse_args()

    timestamp = 0
    if args.sync_time is not None:
        timestamp = first_timestamp(args.sync_time, time.time(), args.adc_sample_rate, args.heap_samples)
    sig = signal.CW(amplitude=1.0, frequency=args.signal_freq)
    sig_data = sig.sample(timestamp, args.heap_samples * args.signal_heaps, args.adc_sample_rate)
    sig_data = signal.quantise(sig_data, args.sample_bits)
    sig_data = signal.packbits(sig_data, args.sample_bits)

    substream_offset = 0
    heap_size = args.heap_samples * args.sample_bits // BYTE_BITS
    heap_sets: List[send.HeapSet] = []
    endpoints: List[Tuple[str, int]] = []
    for i, pol_dest in enumerate(args.dest):
        heap_set = send.HeapSet(args.signal_heaps, len(pol_dest), substream_offset, heap_size, i)
        heap_set.payload[:] = sig_data
        heap_sets.append(heap_set)
        substream_offset += len(pol_dest)
        for ep in pol_dest:
            endpoints.append((ep.host, ep.port))

    stream = send.make_stream(
        endpoints=endpoints,
        heap_sets=heap_sets,
        n_pols=len(args.dest),
        adc_sample_rate=args.adc_sample_rate,
        heap_samples=args.heap_samples,
        sample_bits=args.sample_bits,
        max_heaps=args.max_heaps,
        ttl=args.ttl,
        interface_address=katsdpservices.get_interface_address(args.interface),
        ibv=args.ibv,
    )
    # Interleave the heaps from the different polarisations
    heaps = toolz.interleave(heap_set.heaps for heap_set in heap_sets)
    # Group them into chunks for bulk transmission
    # partition_all produces tuples, but spead2 wants lists
    chunks = [list(chunk) for chunk in toolz.partition_all(args.max_heaps // 2, heaps)]
    futures: Deque[asyncio.Future] = deque()
    logger.info("Starting transmission")
    heap_set_samples = args.signal_heaps * args.heap_samples
    # TODO: might be more efficient to share timestamps between the heap sets
    for heap_set in heap_sets:
        heap_set.timestamps[:] = np.arange(0, args.signal_heaps, dtype=">u8") * args.heap_samples + timestamp
    while True:
        for chunk in chunks:
            # Each heap is a single packet, so despite ROUND_ROBIN they will be
            # sent sequentially.
            futures.append(stream.async_send_heaps(chunk, spead2.send.GroupMode.ROUND_ROBIN))
            while len(futures) > 1:
                await futures.popleft()
        for heap_set in heap_sets:
            heap_set.timestamps += heap_set_samples


def main() -> None:
    """Run main program."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
