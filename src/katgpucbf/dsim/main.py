################################################################################
# Copyright (c) 2021-2024, National Research Foundation (SARAO)
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
import math
import os
import time
from collections.abc import Sequence

import katsdpservices
import numpy as np
import prometheus_async
import pyparsing as pp
from katsdptelstate.endpoint import endpoint_list_parser

from .. import BYTE_BITS, DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_TTL, SPEAD_DESCRIPTOR_INTERVAL_S
from ..send import DescriptorSender
from ..utils import TimeConverter, add_gc_stats, add_signal_handlers
from . import descriptors, send, signal
from .server import DeviceServer

logger = logging.getLogger(__name__)


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="dsim")
    parser.add_argument(
        "--adc-sample-rate", type=float, default=1712e6, help="Digitiser sampling rate (Hz) [%(default)s]"
    )
    parser.add_argument(
        "--signals",
        default="cw(1.0, 232101234.0);",
        help="Specification for the signals to generate (see the usage guide). "
        "The specification must produce either a single signal, or one per output stream. [%(default)s]",
    )
    parser.add_argument("--sync-time", type=float, help="Sync time in UNIX epoch seconds (must be in the past)")
    parser.add_argument("--interface", default="lo", help="Network interface on which to send packets [%(default)s]")
    parser.add_argument("--heap-samples", type=int, default=4096, help="Number of samples per heap [%(default)s]")
    parser.add_argument("--sample-bits", type=int, default=10, help="Number of bits per sample [%(default)s]")
    parser.add_argument("--first-id", type=int, default=0, help="Digitiser ID for first stream [%(default)s]")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="IP TTL for multicast [%(default)s]")
    parser.add_argument("--ibv", action="store_true", help="Use ibverbs for acceleration")
    parser.add_argument("--affinity", type=int, default=-1, help="Core affinity for the sending thread [not bound]")
    parser.add_argument(
        "--main-affinity", type=int, default=-1, help="Core affinity for the main Python thread [not bound]"
    )
    parser.add_argument(
        "--period", type=int, default=2**27, help="Period of pre-computed signal in samples [%(default)s]"
    )
    parser.add_argument(
        "--max-period", type=int, help="Maximum supported period for pre-computed signals [same as --period]"
    )
    parser.add_argument("--dither-seed", type=int, help="Fixed seed for reproducible dithering [random]")
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

    args = parser.parse_args(arglist)
    if args.period <= 0:
        parser.error("--period must be positive")

    if args.max_period is None:
        args.max_period = args.period
    if args.max_period <= 0:
        parser.error("--max-period must be positive")
    if args.max_period % args.period:
        parser.error("--max-period must be a multiple of --period")
    if args.max_period % args.heap_samples:
        parser.error("--max-period must be a multiple of --heap-samples")
    if args.max_period < 2 * args.heap_samples:
        parser.error("--max-period must be at least 2 heaps")

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


def first_timestamp(time_converter: TimeConverter, now: float, align: int) -> int:
    """Determine ADC timestamp for first sample and the time at which to start sending.

    The resulting value will be a multiple of `align`.

    Parameters
    ----------
    time_converter
        Time converter between UNIX timestamps and ADC samples
    now
        Lower bound on first timestamp, expressed as UNIX timestamp
    align
        Alignment requirement on the returned ADC sample count
    """
    # Convert to repeat count (rounding)
    first_block = math.ceil(time_converter.unix_to_adc(now) / align)
    if first_block < 0:
        raise ValueError("sync time is in the future")
    # Convert to a sample count
    samples = first_block * align
    return samples


async def _async_main(tg: asyncio.TaskGroup) -> None:
    """Real implementation of :func:`async_main`.

    This is split into a separate function to avoid having to indent the whole
    thing inside the `tg` context manager.
    """
    args = parse_args()
    heap_size = args.heap_samples * args.sample_bits // BYTE_BITS

    if args.prometheus_port is not None:
        await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)

    timestamps = np.zeros(args.max_period // args.heap_samples, dtype=">u8")
    heap_sets = [
        send.HeapSet.create(
            timestamps,
            [len(pol_dest) for pol_dest in args.dest],
            heap_size,
            range(args.first_id, args.first_id + len(args.dest)),
        )
        for _ in range(2)
    ]

    endpoints: list[tuple[str, int]] = []
    for pol_dest in args.dest:
        for ep in pol_dest:
            endpoints.append((ep.host, ep.port))

    config = descriptors.create_config()
    interface_address = katsdpservices.get_interface_address(args.interface)
    descriptor_stream = send.make_stream_base(
        endpoints=endpoints,
        config=config,
        ttl=args.ttl,
        interface_address=interface_address,
        ibv=args.ibv,
    )
    descriptor_stream.set_cnt_sequence(1, 2)

    # Start descriptor sender first so descriptors are sent before dsim data.
    descriptor_heap = descriptors.create_descriptors_heap()
    descriptor_sender = DescriptorSender(descriptor_stream, descriptor_heap, SPEAD_DESCRIPTOR_INTERVAL_S)
    tg.create_task(descriptor_sender.run())

    if args.dither_seed is None:
        args.dither_seed = np.random.SeedSequence().entropy  # Generate a random seed
        assert isinstance(args.dither_seed, int)  # Keeps mypy happy

    # Enable real-time scheduling after creating signal_service (which spawns a
    # bunch of processes we don't want to have it) but before creating the send
    # stream (which we do want to have it).
    try:
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(1))
    except PermissionError:
        logger.info("Real-time scheduling could not be enabled (permission denied)")
    else:
        logger.info("Real-time scheduling enabled")

    stream = send.make_stream(
        endpoints=endpoints,
        heap_sets=heap_sets,
        n_pols=len(args.dest),
        adc_sample_rate=args.adc_sample_rate,
        heap_samples=args.heap_samples,
        sample_bits=args.sample_bits,
        max_heaps=heap_sets[0].data["heaps"].size,
        ttl=args.ttl,
        interface_address=interface_address,
        ibv=args.ibv,
        affinity=args.affinity,
    )
    # Set spead stream to have heap id in even numbers for dsim data.
    stream.set_cnt_sequence(2, 2)
    sender = send.Sender(stream, heap_sets[0], args.heap_samples)

    server = DeviceServer(
        sender=sender,
        descriptor_sender=descriptor_sender,
        spare=heap_sets[1],
        adc_sample_rate=args.adc_sample_rate,
        dither_seed=args.dither_seed,
        sample_bits=args.sample_bits,
        host=args.katcp_host,
        port=args.katcp_port,
    )
    await server.set_signals(args.signals, args.signals_orig, args.period)

    # Only set this affinity after constructing DeviceServer, which creates
    # a separate process for the signal service that shouldn't inherit this.
    if args.main_affinity >= 0:
        os.sched_setaffinity(0, [args.main_affinity])

    if args.sync_time is None:
        args.sync_time = time.time()
    server.sensors["sync-time"].value = args.sync_time
    await server.start()

    add_signal_handlers(server)
    add_gc_stats()

    time_converter = TimeConverter(args.sync_time, args.adc_sample_rate)
    timestamp = first_timestamp(time_converter, time.time(), args.max_period)
    start_time = time_converter.adc_to_unix(timestamp)

    logger.info("First timestamp will be %#x", timestamp)
    # Sleep until start_time. Python doesn't seem to have an interface
    # for sleeping until an absolute time, so this will be wrong by the
    # time that elapsed from calling time.time until calling
    # asyncio.sleep, but that's small change.
    await asyncio.sleep(max(0, start_time - time.time()))
    logger.info("Starting transmission")
    tg.create_task(sender.run(timestamp, time_converter))
    tg.create_task(server.join())
    # The caller will exit the scope of tg, thus waiting for everything to finish


async def async_main() -> None:
    """Asynchronous main entry point."""
    async with asyncio.TaskGroup() as tg:
        await _async_main(tg)


def main() -> None:
    """Run main program."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
