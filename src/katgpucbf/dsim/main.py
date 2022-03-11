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
import atexit
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from signal import SIGINT, SIGTERM
from typing import List, Optional, Sequence, Tuple

import dask.config
import dask.system
import katsdpservices
import numpy as np
import prometheus_async
import pyparsing as pp
from katsdptelstate.endpoint import endpoint_list_parser

from katgpucbf import BYTE_BITS, DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_TTL
from katgpucbf.dsim import descriptors, send, signal
from katgpucbf.dsim.server import DeviceServer

# from .. import BYTE_BITS, DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_TTL
# from . import descriptors, send, signal
# from .server import DeviceServer

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
    args.signals_orig = args.signals
    args.signals = signals
    return args


def first_timestamp(sync_time: float, now: float, adc_sample_rate: float, align: int) -> Tuple[int, float]:
    """Determine ADC timestamp for first sample and the time at which to start sending.

    The resulting value will be a multiple of `align`.
    """
    # Convert to repeat count (rounding)
    first_block = math.ceil((now - sync_time) * adc_sample_rate / align)
    if first_block < 0:
        raise ValueError("sync time is in the future")
    # Convert to a sample count
    samples = first_block * align
    return samples, sync_time + samples / adc_sample_rate


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

    # Override dask's default thread pool with one that runs with SCHED_IDLE
    # priority. This ensures that when the networking code needs a CPU core,
    # it gets priority over dask.
    # The type: ignore is because typeshed is currently missing os.sched_param
    # (it is fixed in https://github.com/python/typeshed/pull/6442).
    pool = ThreadPoolExecutor(
        dask.config.get("num_workers", dask.system.CPU_COUNT),
        initializer=os.sched_setscheduler,
        initargs=(0, os.SCHED_IDLE, os.sched_param(0)),  # type: ignore
    )
    atexit.register(pool.shutdown)
    dask.config.set(pool=pool)

    if args.prometheus_port is not None:
        await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)

    timestamps = np.zeros(args.signal_heaps, dtype=">u8")
    heap_sets = [
        send.HeapSet.create(timestamps, [len(pol_dest) for pol_dest in args.dest], heap_size, range(len(args.dest)))
        for _ in range(2)
    ]

    timestamp = 0
    if args.sync_time is not None:
        timestamp, start_time = first_timestamp(
            args.sync_time, time.time(), args.adc_sample_rate, args.signal_heaps * args.heap_samples
        )
        # Sleep until start_time. Python doesn't seem to have an interface
        # for sleeping until an absolute time, so this will be wrong by the
        # time that elapsed from calling time.time until calling time.sleep,
        # but that's small change.
        time.sleep(max(0, start_time - time.time()))
    else:
        args.sync_time = time.time()
    logger.info("First timestamp will be %#x", timestamp)

    endpoints: List[Tuple[str, int]] = []
    for pol_dest in args.dest:
        for ep in pol_dest:
            endpoints.append((ep.host, ep.port))

    # Start descriptor sender first so descriptors are sent before dsim data.
    descriptor_sender = descriptors.DescriptorSender(args.interface, args.heap_samples, args.ttl, timestamp, endpoints)
    descriptor_task = asyncio.create_task(descriptor_sender.run())

    await signal.sample_async(args.signals, 0, args.adc_sample_rate, args.sample_bits, heap_sets[0].data["payload"])

    stream = send.make_stream(
        endpoints=endpoints,
        heap_sets=heap_sets,
        n_pols=len(args.dest),
        adc_sample_rate=args.adc_sample_rate,
        heap_samples=args.heap_samples,
        sample_bits=args.sample_bits,
        max_heaps=heap_sets[0].data["heaps"].size,
        ttl=args.ttl,
        interface_address=katsdpservices.get_interface_address(args.interface),
        ibv=args.ibv,
        affinity=args.affinity,
    )

    sender = send.Sender(stream, heap_sets[0], timestamp, args.heap_samples, args.sync_time, args.adc_sample_rate)

    server = DeviceServer(
        sender=sender,
        descriptor_sender=descriptor_sender,
        spare=heap_sets[1],
        adc_sample_rate=args.adc_sample_rate,
        first_timestamp=timestamp,
        sample_bits=args.sample_bits,
        signals_str=args.signals_orig,
        signals=args.signals,
        host=args.katcp_host,
        port=args.katcp_port,
    )
    await server.start()

    logger.debug("Setting up descriptors")
    add_signal_handlers(server)

    logger.info("Starting transmission")
    await asyncio.gather(sender.run(), descriptor_task, server.join())


def main() -> None:
    """Run main program."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
