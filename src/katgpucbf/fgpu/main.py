#!/usr/bin/env python3
"""katfgpu main script.

This is what kicks everything off. Command-line arguments are parsed, and used
to create an :class:`~katfgpu.engine.Engine` object, which then takes over the
actual running of the processing.
"""

import argparse
import asyncio
import ipaddress
import logging
import signal
from typing import List, Tuple, Union, Optional, TypeVar, Callable

import katsdpservices
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import endpoint_list_parser
import katsdpsigproc.accel as accel

from .engine import Engine
from .monitor import Monitor, FileMonitor, NullMonitor


_T = TypeVar("_T")
N_POL = 2  # TODO trace this. I'm fairly certain that number of pols comes up elsewhere. Does this change everything?
logger = logging.getLogger(__name__)


def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return value


def comma_split(base_type: Callable[[str], _T], count: Optional[int] = None) -> Callable[[str], List[_T]]:
    """Return a function to split a comma-delimited str into a list of type _T.

    This function is used to parse lists of CPU core numbers, which come from
    the command-line as comma-separated strings, but are obviously more useful
    as a list of ints. It's generic enough that it could process lists of other
    types as well though if necessary.

    Parameters
    ----------
    base_type
        The base type of thing you expect in the list, e.g. `int`, `float`.
    count
        How many of them you expect to be in the list. `None` means the list
        could be any length.
    """

    def func(value: str) -> List[_T]:  # noqa: D102
        parts = value.split(",")
        n = len(parts)
        if count is not None and n != count:
            raise ValueError(f"Expected {count} comma-separated fields, received {n}")
        return [base_type(part) for part in parts]

    return func


def parse_args() -> argparse.Namespace:
    """Declare and parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-interface", type=get_interface_address, help="Name of input network device")
    parser.add_argument("--src-ibv", action="store_true", help="Use ibverbs for input [no]")
    parser.add_argument(
        "--src-affinity",
        type=comma_split(int, N_POL),
        metavar="CORE,CORE",
        default=[-1] * N_POL,
        help="Cores for input-handling threads (comma-separated) [not bound]",
    )
    parser.add_argument(
        "--src-comp-vector",
        type=comma_split(int, N_POL),
        metavar="VECTOR,VECTOR",
        default=[0] * N_POL,
        help="Completion vectors for source streams, or -1 for polling [0]",
    )
    parser.add_argument(
        "--src-packet-samples", type=int, default=4096, help="Number of samples per digitiser packet [%(default)s]"
    )
    parser.add_argument(
        "--src-buffer",
        type=int,
        default=32 * 1024 * 1024,
        metavar="BYTES",
        help="Size of network receive buffer (per pol) [32MiB]",
    )
    parser.add_argument(
        "--dst-interface", type=get_interface_address, required=True, help="Name of output network device"
    )
    parser.add_argument("--dst-ttl", type=int, default=4, help="TTL for outgoing packets [%(default)s]")
    parser.add_argument("--dst-ibv", action="store_true", help="Use ibverbs for output [no]")
    parser.add_argument(
        "--dst-packet-payload",
        type=int,
        default=1024,
        metavar="BYTES",
        help="Size for output packets (voltage payload only) [%(default)s]",
    )
    parser.add_argument(
        "--dst-affinity", type=int, default=-1, metavar="CORE,...", help="Cores for output-handling threads [not bound]"
    )
    parser.add_argument(
        "--dst-comp-vector",
        type=int,
        default=0,
        metavar="VECTOR,...",
        help="Completion vector for transmission, or -1 for polling [0]",
    )
    parser.add_argument(
        "--adc-rate",
        type=float,
        default=0.0,
        metavar="HZ",
        help="Digitiser sampling rate, used to determine transmission rate [fast as possible]",
    )
    parser.add_argument("--channels", type=int, required=True, help="Number of output channels to produce")
    parser.add_argument(
        "--acc-len", type=int, default=256, metavar="SPECTRA", help="Spectra in each output heap [%(default)s]"
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=2 ** 26,
        metavar="SAMPLES",
        help="Number of digitiser samples to process at a time (per pol). If not a multiple of 2*channels*acc-len,"
        "it will be rounded up to the next multiple. [%(default)s]",
    )
    parser.add_argument("--taps", type=int, default=16, help="Number of taps in polyphase filter bank [%(default)s]")
    parser.add_argument(
        "--quant-scale", type=float, default=0.001, help="Rescaling factor before 8-bit requantisation [%(default)s]"
    )
    parser.add_argument(
        "--mask-timestamp",
        action="store_true",
        help="Mask off bottom bits of timestamp (workaround for broken digitiser)",
    )
    parser.add_argument(
        "--use-gdrcopy", action="store_true", help="Assemble chunks directly in GPU memory (requires supported GPU)"
    )
    parser.add_argument(
        "--use-peerdirect", action="store_true", help="Send chunks directly from GPU memory (requires supported GPU)"
    )
    parser.add_argument("--monitor-log", help="File to write performance-monitoring data to")

    parser.add_argument("src", type=parse_source, nargs=N_POL, help="Source endpoints (or pcap file)")
    parser.add_argument("dst", type=endpoint_list_parser(7148), help="Destination endpoints")
    args = parser.parse_args()

    if args.use_peerdirect and not args.dst_ibv:
        parser.error("--use-peerdirect requires --dst-ibv")
    for src in args.src:
        if not isinstance(src, str) and args.src_interface is None:
            parser.error("Live source requires --src-interface")
    return args


def add_signal_handlers():
    """Arrange for clean shutdown on SIGINT (Ctrl-C) or SIGTERM.

    .. todo::

       This is still not particularly clean, sometimes hangs if digitiser data
       is still flowing, and possibly exits in the middle of sending a heap.
       However, it's still useful as it ensures CUDA is shut down properly,
       which is necessary for NSight to operate correctly.
    """
    signums = [signal.SIGINT, signal.SIGTERM]

    def handler():
        # Remove the handlers so that if it fails to shut down, the next
        # attempt will try harder.
        logger.info("Received signal, shutting down")
        for signum in signums:
            loop.remove_signal_handler(signum)
        task.cancel()

    loop = asyncio.get_event_loop()
    task = asyncio.current_task()
    for signum in signums:
        loop.add_signal_handler(signum, handler)


async def async_main() -> None:
    """Start the F-Engine asynchronously."""
    add_signal_handlers()
    args = parse_args()
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)

    chunk_samples = accel.roundup(args.chunk_samples, 2 * args.channels * args.acc_len)

    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(args.monitor_log)
    else:
        monitor = NullMonitor()
    with monitor:
        engine = Engine(
            context=ctx,
            srcs=args.src,
            src_interface=args.src_interface,
            src_ibv=args.src_ibv,
            src_affinity=args.src_affinity,
            src_comp_vector=args.src_comp_vector,
            src_packet_samples=args.src_packet_samples,
            src_buffer=args.src_buffer,
            dst=args.dst,
            dst_interface=args.dst_interface,
            dst_ttl=args.dst_ttl,
            dst_ibv=args.dst_ibv,
            dst_packet_payload=args.dst_packet_payload,
            dst_affinity=args.dst_affinity,
            dst_comp_vector=args.dst_comp_vector,
            adc_rate=args.adc_rate,
            spectra=chunk_samples // (2 * args.channels),
            acc_len=args.acc_len,
            channels=args.channels,
            taps=args.taps,
            quant_scale=args.quant_scale,
            mask_timestamp=args.mask_timestamp,
            use_gdrcopy=args.use_gdrcopy,
            use_peerdirect=args.use_peerdirect,
            monitor=monitor,
        )
        await engine.run()


def main() -> None:
    """Start the F-engine."""
    katsdpservices.setup_logging()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(async_main())
    except asyncio.CancelledError:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
