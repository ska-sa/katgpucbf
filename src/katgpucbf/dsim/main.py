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
from signal import SIGINT, SIGTERM
from typing import List, Optional, Sequence, Tuple

import katsdpservices
import numpy as np
import prometheus_async
import pyparsing as pp
from katsdptelstate.endpoint import endpoint_list_parser

from .. import BYTE_BITS, DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_TTL
from . import send, signal
from .server import DeviceServer

logger = logging.getLogger(__name__)


def parse_args(arglist: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="dsim")
    parser.add_argument(
        "--adc-sample-rate", type=float, default=1712e6, help="Digitiser sampling rate (Hz) [%(default)s]"
    )
    parser.add_argument(
        "--signals",
        default="cw(1.0, 232101234.0);",
        help="Specification for the signals to generate (see the docstring for parse_signals). "
        "The specification must produce either a single signal, or one per output stream. [%(default)s]",
    )
    parser.add_argument(
        "--signal-freq", type=float, default=232101234.0, help="Frequency of simulated tone (Hz) [%(default)s]"
    )
    parser.add_argument("--sync-time", type=float, help="Sync time in UNIX epoch seconds (must be in the past)")
    parser.add_argument("--interface", default="lo", help="Network interface on which to send packets [%(default)s]")
    parser.add_argument("--heap-samples", type=int, default=4096, help="Number of samples per heap [%(default)s]")
    parser.add_argument("--sample-bits", type=int, default=10, help="Number of bits per sample [%(default)s]")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="IP TTL for multicast [%(default)s]")
    parser.add_argument("--ibv", action="store_true", help="Use ibverbs for acceleration")
    parser.add_argument("--affinity", type=int, default=-1, help="Core affinity for the sending thread [not bound]")
    parser.add_argument(
        "--signal-heaps", type=int, default=32768, help="Length of pre-computed signal in heaps [%(default)s]"
    )
    parser.add_argument(
        "--katcp-host",
        type=str,
        default=DEFAULT_KATCP_HOST,
        help="Hostname or IP on which to listen for KATCP C&M connections [all interfaces]",
    )
    parser.add_argument(
        "--katcp-port",
        type=int,
        default=DEFAULT_KATCP_PORT,
        help="Network port on which to listen for KATCP C&M connections [%(default)s]",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        help="Network port on which to serve Prometheus metrics [none]",
    )
    parser.add_argument(
        "dest",
        nargs="+",
        type=endpoint_list_parser(None),
        metavar="X.X.X.X[+N]:PORT",
        help="Destination addresses (one per polarisation)",
    )

    args = parser.parse_args()
    if args.signal_heaps < 2:
        parser.error("--signal-heaps must be at least 2")
    for dest in args.dest:
        for ep in dest:
            if ep.port is None:
                parser.error("port must be specified on destinations")
    try:
        signals = signal.parse_signals(args.signals)
    except pp.ParseBaseException as exc:
        parser.error(f"invalid --signals: {exc}")
    if len(signals) == 1:
        signals *= len(args.dest)
    if len(signals) != len(args.dest):
        parser.error(f"expected 1 or {len(args.dest)} signals, found {len(signals)}")
    args.signals = signals
    return args


def first_timestamp(sync_time: float, now: float, adc_sample_rate: float, heap_samples: int) -> int:
    """Determine ADC timestamp for first sample."""
    # Convert to heap count (rounding)
    first_heap = round((now - sync_time) * adc_sample_rate / heap_samples)
    if first_heap < 0:
        raise ValueError("sync time is in the future")
    # Convert to a sample count
    return first_heap * heap_samples


def add_signal_handlers(server: DeviceServer) -> None:
    """Arrange for clean shutdown on SIGINT (Ctrl-C) or SIGTERM."""
    signums = [SIGINT, SIGTERM]

    def handler():
        # Remove the handlers so that if it fails to shut down, the next
        # attempt will try harder.
        logger.info("Received signal, shutting down")
        for signum in signums:
            loop.remove_signal_handler(signum)
        server.halt()

    loop = asyncio.get_event_loop()
    for signum in signums:
        loop.add_signal_handler(signum, handler)


async def async_main() -> None:
    """Asynchronous main entry point."""
    args = parse_args()
    heap_size = args.heap_samples * args.sample_bits // BYTE_BITS

    if args.prometheus_port is not None:
        await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)

    timestamp = 0
    if args.sync_time is not None:
        timestamp = first_timestamp(args.sync_time, time.time(), args.adc_sample_rate, args.heap_samples)
    sig_data = []
    for sig in args.signals:
        # TODO: cache shared signals to reduce computation time
        data = sig.sample(timestamp, args.heap_samples * args.signal_heaps, args.adc_sample_rate)
        data = signal.quantise(data, args.sample_bits)
        data = signal.packbits(data, args.sample_bits)
        sig_data.append(data)

    timestamps = np.zeros(args.signal_heaps, dtype=">u8")
    heap_set = send.HeapSet.create(
        timestamps, [len(pol_dest) for pol_dest in args.dest], heap_size, range(len(args.dest))
    )

    endpoints: List[Tuple[str, int]] = []
    for i, pol_dest in enumerate(args.dest):
        heap_set.data.payload.isel(pol=i).data.ravel()[:] = sig_data[i]
        for ep in pol_dest:
            endpoints.append((ep.host, ep.port))
    stream = send.make_stream(
        endpoints=endpoints,
        heap_sets=[heap_set],
        n_pols=len(args.dest),
        adc_sample_rate=args.adc_sample_rate,
        heap_samples=args.heap_samples,
        sample_bits=args.sample_bits,
        max_heaps=heap_set.data["heaps"].size,
        ttl=args.ttl,
        interface_address=katsdpservices.get_interface_address(args.interface),
        ibv=args.ibv,
        affinity=args.affinity,
    )
    sender = send.Sender(stream, heap_set, timestamp, args.heap_samples)
    server = DeviceServer(sender, args.katcp_host, args.katcp_port)
    await server.start()
    add_signal_handlers(server)

    logger.info("Starting transmission")
    await sender.run()
    await server.join()


def main() -> None:
    """Run main program."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
