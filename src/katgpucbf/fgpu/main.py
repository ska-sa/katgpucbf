################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

"""fgpu main script.

This is what kicks everything off. Command-line arguments are parsed, and used
to create an :class:`~katgpucbf.fgpu.engine.Engine` object, which then takes over the
actual running of the processing.
"""

import argparse
import asyncio
import logging
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import katsdpservices
import katsdpsigproc.accel as accel
import prometheus_async
from katsdpservices import get_interface_address
from katsdpservices.aiomonitor import add_aiomonitor_arguments, start_aiomonitor
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import endpoint_list_parser

from .. import DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_PACKET_PAYLOAD_BYTES, DEFAULT_TTL, N_POLS, __version__
from ..monitor import FileMonitor, Monitor, NullMonitor
from ..utils import add_signal_handlers, parse_source
from .engine import Engine

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


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


def parse_args(arglist: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Declare and parse command-line arguments.

    Parameters
    ----------
    arglist
        You can pass a list of argument strings in this parameter, for example
        in test situations, to make use of the configured defaults. If None,
        arguments from ``sys.argv`` will be used.
    """
    parser = argparse.ArgumentParser(prog="fgpu")
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
    add_aiomonitor_arguments(parser)
    parser.add_argument("--src-interface", type=get_interface_address, help="Name of input network device")
    parser.add_argument("--src-ibv", action="store_true", help="Use ibverbs for input [no]")
    parser.add_argument(
        "--src-affinity",
        type=comma_split(int, N_POLS),
        metavar="CORE,CORE",
        default=[-1] * N_POLS,
        help="Cores for input-handling threads (comma-separated) [not bound]",
    )
    parser.add_argument(
        "--src-comp-vector",
        type=comma_split(int, N_POLS),
        metavar="VECTOR,VECTOR",
        default=[0] * N_POLS,
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
    parser.add_argument("--dst-ttl", type=int, default=DEFAULT_TTL, help="TTL for outgoing packets [%(default)s]")
    parser.add_argument("--dst-ibv", action="store_true", help="Use ibverbs for output [no]")
    parser.add_argument(
        "--dst-packet-payload",
        type=int,
        default=DEFAULT_PACKET_PAYLOAD_BYTES,
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
        "--adc-sample-rate",
        type=float,
        required=True,
        metavar="HZ",
        help="Digitiser sampling rate, used to determine transmission rate and calculate delays",
    )
    parser.add_argument(
        "--send-rate-factor",
        type=float,
        default=1.1,
        metavar="FACTOR",
        help="Target transmission rate faster than ADC sample rate by this factor. \
            Set to zero to send as fast as possible. [%(default)s]",
    )
    parser.add_argument(
        "--feng-id",
        type=int,
        default=0,
        help="ID of the F-engine indicating which one in the array it is. [%(default)s]",
    )
    parser.add_argument(
        "--array-size",
        type=int,
        default=65536,
        help="The number of antennas in the array. [%(default)s]",
    )
    parser.add_argument("--channels", type=int, required=True, help="Number of output channels to produce")
    parser.add_argument(
        "--spectra-per-heap",
        type=int,
        default=256,
        metavar="SPECTRA",
        help="Spectra in each output heap [%(default)s]",
    )
    parser.add_argument(
        "--src-chunk-samples",
        type=int,
        default=2**24,
        metavar="SAMPLES",
        help="Number of digitiser samples to process at a time (per pol). [%(default)s]",
    )
    parser.add_argument(
        "--dst-chunk-jones",
        type=int,
        default=2**23,
        metavar="VECTORS",
        help="Number of Jones vectors in output chunks. If not a multiple of "
        "2*channels*spectra-per-heap, it will be rounded up to the next multiple. [%(default)s]",
    )
    parser.add_argument("--taps", type=int, default=16, help="Number of taps in polyphase filter bank [%(default)s]")
    parser.add_argument(
        "--max-delay-diff",
        type=int,
        default=1048576,
        help="Maximum supported difference between delays across polarisations (in samples) [%(default)s]",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Initial eq gains [%(default)s]")
    parser.add_argument(
        "--sync-epoch",
        type=int,  # AFAIK, the digitisers sync on PPS signals, so it makes sense for this to be an int.
        required=True,
        help="UNIX time at which digitisers were synced.",
    )
    parser.add_argument(
        "--mask-timestamp",
        action="store_true",
        help="Mask off bottom bits of timestamp (workaround for broken digitiser)",
    )
    parser.add_argument(
        "--use-vkgdr", action="store_true", help="Assemble chunks directly in GPU memory (requires Vulkan)"
    )
    parser.add_argument(
        "--use-peerdirect", action="store_true", help="Send chunks directly from GPU memory (requires supported GPU)"
    )
    parser.add_argument("--monitor-log", help="File to write performance-monitoring data to")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("src", type=parse_source, nargs=N_POLS, help="Source endpoints (or pcap file)")
    parser.add_argument("dst", type=endpoint_list_parser(7148), help="Destination endpoints")
    args = parser.parse_args(arglist)

    if args.use_peerdirect:
        # TODO: make it work again
        parser.error("--use-peerdirect is currently broken")
    if args.use_peerdirect and not args.dst_ibv:
        parser.error("--use-peerdirect requires --dst-ibv")
    for src in args.src:
        if not isinstance(src, str) and args.src_interface is None:
            parser.error("Live source requires --src-interface")
    return args


def make_engine(ctx: AbstractContext, args: argparse.Namespace) -> Tuple[Engine, Monitor]:
    """Make an :class:`Engine` object, given a GPU context.

    Parameters
    ----------
    ctx
        The GPU context in which the :class:`.Engine` will operate.
    args
        Parsed arguments returned from :func:`parse_args`.
    """
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(args.monitor_log)
    else:
        monitor = NullMonitor()

    chunk_jones = accel.roundup(args.dst_chunk_jones, args.channels * args.spectra_per_heap)
    engine = Engine(
        katcp_host=args.katcp_host,
        katcp_port=args.katcp_port,
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
        adc_sample_rate=args.adc_sample_rate,
        send_rate_factor=args.send_rate_factor,
        feng_id=args.feng_id,
        num_ants=args.array_size,
        chunk_samples=args.src_chunk_samples,
        spectra=chunk_jones // args.channels,
        spectra_per_heap=args.spectra_per_heap,
        channels=args.channels,
        taps=args.taps,
        max_delay_diff=args.max_delay_diff,
        gain=args.gain,
        sync_epoch=float(args.sync_epoch),  # CLI arg is an int, but SDP can handle a float downstream.
        mask_timestamp=args.mask_timestamp,
        use_vkgdr=args.use_vkgdr,
        use_peerdirect=args.use_peerdirect,
        monitor=monitor,
    )

    return engine, monitor


async def async_main() -> None:
    """Start the F-Engine asynchronously."""
    args = parse_args()
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    logger.info("Initialising F-engine on %s", ctx.device.name)
    engine, monitor = make_engine(ctx, args)
    add_signal_handlers(engine)
    prometheus_server: Optional[prometheus_async.aio.web.MetricsHTTPServer] = None
    if args.prometheus_port is not None:
        prometheus_server = await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)
    with monitor, start_aiomonitor(asyncio.get_running_loop(), args, locals()):
        await engine.start()
        await engine.join()
        if prometheus_server:
            await prometheus_server.close()


def main() -> None:
    """Start the F-engine."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
