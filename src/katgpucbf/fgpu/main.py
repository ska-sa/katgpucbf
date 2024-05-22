################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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
import gc
import logging
import math
from collections.abc import Callable, Sequence
from typing import TypedDict, TypeVar

import katsdpservices
import katsdpsigproc.accel as accel
import prometheus_async
import vkgdr
from katsdpservices import get_interface_address
from katsdpservices.aiomonitor import add_aiomonitor_arguments, start_aiomonitor
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser

from .. import (
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_KATCP_HOST,
    DEFAULT_KATCP_PORT,
    DEFAULT_PACKET_PAYLOAD_BYTES,
    DEFAULT_TTL,
    DIG_SAMPLE_BITS,
    __version__,
)
from ..mapped_array import make_vkgdr
from ..monitor import FileMonitor, Monitor, NullMonitor
from ..spead import DEFAULT_PORT
from ..utils import add_gc_stats, add_signal_handlers, comma_split, parse_source
from . import DIG_SAMPLE_BITS_VALID
from .engine import Engine
from .output import NarrowbandOutput, WidebandOutput

_T = TypeVar("_T")
_OD = TypeVar("_OD", bound="_OutputDict")
logger = logging.getLogger(__name__)
DEFAULT_TAPS = 16
DEFAULT_W_CUTOFF = 1.0
#: Ratio of decimation factor to tap count
DEFAULT_DDC_TAPS_RATIO = 12
DEFAULT_WEIGHT_PASS = 0.005


class _OutputDict(TypedDict, total=False):
    """Configuration options for an output stream.

    Unlike :class:`WidebandOutput` or :class:`NarrowbandOutput`, all the fields
    are optional, so that it can be built up incrementally. They must all be
    filled in before using it to construct an :class:`Output`.
    """

    name: str
    channels: int
    jones_per_batch: int
    dst: list[Endpoint]
    taps: int
    w_cutoff: float


class _WidebandOutputDict(_OutputDict, total=False):
    """Configuration options for a wideband output.

    See :class:`_OutputDict` for further information.
    """

    pass


class _NarrowbandOutputDict(_OutputDict, total=False):
    """Configuration options for a narrowband output.

    See :class:`_OutputDict` for further information.
    """

    centre_frequency: float
    decimation: int
    ddc_taps: int
    weight_pass: float


def _parse_stream(value: str, kws: _OD, field_callback: Callable[[_OD, str, str], None]) -> None:
    """Parse a wideband or narrowband stream description.

    This populates `kws` (which should initially be empty) from key=value pairs
    in `value`. It handles the common fields directly, and type-specific fields
    are handled by a provided field callback. The callback is invoked with
    `kws`, the key and the value. If it does not recognise the key, it should
    raise ValueError.
    """
    for part in value.split(","):
        match part.split("=", 1):
            case [key, data]:
                match key:
                    case _ if key in kws:
                        raise ValueError(f"{key} specified twice")
                    case "name":
                        kws[key] = data
                    case "channels" | "taps" | "jones_per_batch":
                        kws[key] = int(data)
                    case "w_cutoff":
                        kws[key] = float(data)
                    case "dst":
                        kws[key] = endpoint_list_parser(DEFAULT_PORT)(data)
                    case _:
                        field_callback(kws, key, data)
            case _:
                raise ValueError(f"missing '=' in {part}")
    for key in ["name", "channels", "dst"]:
        if key not in kws:
            raise ValueError(f"{key} is missing")


def parse_wideband(value: str) -> WidebandOutput:
    """Parse a string with a wideband configuration.

    The string has a comma-separated list of key=value pairs. See
    :class:`WidebandOutput` for the valid keys and types. The following
    keys are required:

    - name
    - channels
    - dst
    """

    def field_callback(kws: _WidebandOutputDict, key: str, data: str) -> None:
        raise ValueError(f"unknown key {key}")

    try:
        kws: _WidebandOutputDict = {}
        _parse_stream(value, kws, field_callback)
        kws = {
            "taps": DEFAULT_TAPS,
            "w_cutoff": DEFAULT_W_CUTOFF,
            "jones_per_batch": DEFAULT_JONES_PER_BATCH,
            **kws,
        }
        return WidebandOutput(**kws)
    except ValueError as exc:
        raise ValueError(f"--wideband: {exc}") from exc


def parse_narrowband(value: str) -> NarrowbandOutput:
    """Parse a string with a narrowband configuration.

    The string has a comma-separated list of key=value pairs. See
    :class:`NarrowbandOutput` for the valid keys and types. The following
    keys are required:

    - name
    - channels
    - centre_frequency
    - decimation
    - dst
    """

    def field_callback(kws: _NarrowbandOutputDict, key: str, data: str) -> None:
        match key:
            case "centre_frequency" | "weight_pass":
                kws[key] = float(data)
            case "decimation" | "ddc_taps":
                kws[key] = int(data)
            case _:
                raise ValueError(f"unknown key {key}")

    try:
        kws: _NarrowbandOutputDict = {}
        _parse_stream(value, kws, field_callback)
        for key in ["centre_frequency", "decimation"]:
            if key not in kws:
                raise ValueError(f"{key} is missing")
        # Note that using **kws at the end means these are only defaults which
        # can be overridden by the user.
        # The ignore is to work around https://github.com/python/mypy/issues/9408
        kws = {
            "taps": DEFAULT_TAPS,
            "w_cutoff": DEFAULT_W_CUTOFF,
            "jones_per_batch": DEFAULT_JONES_PER_BATCH,
            "weight_pass": DEFAULT_WEIGHT_PASS,
            "ddc_taps": DEFAULT_DDC_TAPS_RATIO * kws["decimation"],
            **kws,
        }
        return NarrowbandOutput(**kws)
    except ValueError as exc:
        raise ValueError(f"--narrowband: {exc}") from exc


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--narrowband",
        type=parse_narrowband,
        default=[],
        action="append",
        metavar="KEY=VALUE[,KEY=VALUE...]",
        help=(
            "Add a narrowband output (may be repeated). "
            "The required keys are: name, centre_frequency, decimation, channels, dst. "
            f"Optional keys: taps [{DEFAULT_TAPS}], ddc_taps [{DEFAULT_DDC_TAPS_RATIO}*decimation], "
            f"w_cutoff [{DEFAULT_W_CUTOFF}], weight_pass."
        ),
    )
    parser.add_argument(
        "--wideband",
        type=parse_wideband,
        default=[],
        action="append",
        metavar="KEY=VALUE[,KEY=VALUE...]",
        help=(
            "Add a wideband output (may be repeated). The required keys are: name, channels, dst. "
            f"Optional keys: taps [{DEFAULT_TAPS}], w_cutoff [{DEFAULT_W_CUTOFF}]"
        ),
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
    add_aiomonitor_arguments(parser)
    parser.add_argument(
        "--src-interface",
        type=comma_split(get_interface_address),
        help="Name(s) of input network device(s)",
    )
    parser.add_argument("--src-ibv", action="store_true", help="Use ibverbs for input [no]")
    parser.add_argument(
        "--src-affinity",
        type=comma_split(int),
        metavar="CORE,...",
        default=[-1],
        help="Cores for input-handling threads (comma-separated) [not bound]",
    )
    parser.add_argument(
        "--src-comp-vector",
        type=comma_split(int),
        metavar="VECTOR,...",
        default=[0],
        help="Completion vectors for source streams, or -1 for polling [0]",
    )
    parser.add_argument(
        "--src-packet-samples", type=int, default=4096, help="Number of samples per digitiser packet [%(default)s]"
    )
    parser.add_argument(
        "--src-buffer",
        type=int,
        default=128 * 1024 * 1024,
        metavar="BYTES",
        help="Size of network receive buffer [128MiB]",
    )
    parser.add_argument(
        "--dst-interface", type=comma_split(get_interface_address), required=True, help="Name of output network device"
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
        metavar="VECTOR",
        help="Completion vector for transmission, or -1 for polling [0]",
    )
    parser.add_argument(
        "--dst-buffer",
        type=int,
        default=1024 * 1024,
        metavar="BYTES",
        help="Size of network send buffer [1MiB]",
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
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        metavar="SAMPLES",
        help="Jones vectors in each output batch [%(default)s]",
    )
    parser.add_argument(
        "--src-chunk-samples",
        type=int,
        default=2**25,
        metavar="SAMPLES",
        help="Number of digitiser samples to process at a time (per pol). [%(default)s]",
    )
    parser.add_argument(
        "--dst-chunk-jones",
        type=int,
        default=2**23,
        metavar="VECTORS",
        help="Number of Jones vectors in output chunks. If not a multiple of "
        "jones-per-batch, it will be rounded up to the next multiple. [%(default)s]",
    )
    parser.add_argument(
        "--max-delay-diff",
        type=int,
        default=1048576,
        help="Maximum supported difference between delays across polarisations (in samples) [%(default)s]",
    )
    parser.add_argument(
        "--dig-sample-bits",
        type=int,
        default=DIG_SAMPLE_BITS,
        choices=DIG_SAMPLE_BITS_VALID,
        metavar="BITS",
        help="Number of bits per digitised sample [%(default)s]",
    )
    parser.add_argument(
        "--dst-sample-bits",
        type=int,
        default=8,
        choices=[4, 8],
        metavar="BITS",
        help="Number of bits per output sample real component [%(default)s]",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Initial eq gains [%(default)s]")
    parser.add_argument(
        "--sync-time",
        type=float,
        required=True,
        help="UNIX time at which digitisers were synced.",
    )
    parser.add_argument(
        "--mask-timestamp",
        action="store_true",
        help="Mask off bottom bits of timestamp (workaround for broken digitiser)",
    )
    parser.add_argument(
        "--use-vkgdr",
        action="store_true",
        help="Assemble chunks directly in GPU memory (requires sufficient GPU BAR space)",
    )
    parser.add_argument(
        "--use-peerdirect", action="store_true", help="Send chunks directly from GPU memory (requires supported GPU)"
    )
    parser.add_argument("--monitor-log", help="File to write performance-monitoring data to")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("src", type=parse_source, help="Source endpoints (or pcap file)")
    args = parser.parse_args(arglist)

    if args.use_peerdirect and not args.dst_ibv:
        parser.error("--use-peerdirect requires --dst-ibv")
    if not isinstance(args.src, str) and args.src_interface is None:
        parser.error("Live source requires --src-interface")
    if args.src_ibv and len(args.src_affinity) != len(args.src_comp_vector):
        parser.error("--src-comp-vector must have same length as --src-affinity")

    # Convert from _*OutputDict to *Output
    used_names = set()
    args.outputs = []
    for output_group in [args.wideband, args.narrowband]:
        for output in output_group:
            name = output.name
            if output.name in used_names:
                parser.error(f"output name {name} used twice")
            if len(output.dst) % len(args.dst_interface) != 0:
                parser.error(f"{name}: number of destinations must be divisible by number of destination interfaces")
            used_names.add(name)
            args.outputs.append(output)
    if not args.outputs:
        parser.error("At least one --wideband or --narrowband argument is required")

    return args


def make_engine(ctx: AbstractContext, vkgdr_handle: vkgdr.Vkgdr, args: argparse.Namespace) -> tuple[Engine, Monitor]:
    """Make an :class:`Engine` object, given a GPU context.

    Parameters
    ----------
    ctx
        The GPU context in which the :class:`.Engine` will operate.
    vkgdr_handle
        The Vkgdr handle in which the :class:`.Engine` will operate. It
        must use the same device as `ctx`.
    args
        Parsed arguments returned from :func:`parse_args`.
    """
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(args.monitor_log)
    else:
        monitor = NullMonitor()

    batch_jones_lcm = math.lcm(*(output.jones_per_batch for output in args.outputs))
    chunk_jones = accel.roundup(args.dst_chunk_jones, batch_jones_lcm)
    engine = Engine(
        katcp_host=args.katcp_host,
        katcp_port=args.katcp_port,
        context=ctx,
        vkgdr_handle=vkgdr_handle,
        srcs=args.src,
        src_interface=args.src_interface,
        src_ibv=args.src_ibv,
        src_affinity=args.src_affinity,
        src_comp_vector=args.src_comp_vector,
        src_packet_samples=args.src_packet_samples,
        src_buffer=args.src_buffer,
        dst_interface=args.dst_interface,
        dst_ttl=args.dst_ttl,
        dst_ibv=args.dst_ibv,
        dst_packet_payload=args.dst_packet_payload,
        dst_affinity=args.dst_affinity,
        dst_comp_vector=args.dst_comp_vector,
        dst_buffer=args.dst_buffer,
        outputs=args.outputs,
        adc_sample_rate=args.adc_sample_rate,
        send_rate_factor=args.send_rate_factor,
        feng_id=args.feng_id,
        n_ants=args.array_size,
        chunk_samples=args.src_chunk_samples,
        chunk_jones=chunk_jones,
        dig_sample_bits=args.dig_sample_bits,
        dst_sample_bits=args.dst_sample_bits,
        max_delay_diff=args.max_delay_diff,
        gain=args.gain,
        sync_time=args.sync_time,
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
    vkgdr_handle = make_vkgdr(ctx.device)
    logger.info("Initialising F-engine on %s", ctx.device.name)
    engine, monitor = make_engine(ctx, vkgdr_handle, args)
    add_signal_handlers(engine)
    add_gc_stats()
    prometheus_server: prometheus_async.aio.web.MetricsHTTPServer | None = None
    if args.prometheus_port is not None:
        prometheus_server = await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)
    with monitor, start_aiomonitor(asyncio.get_running_loop(), args, locals()):
        await engine.start()
        # Avoid garbage collections needing to iterate over all the objects
        # allocated so far. That makes garbage collection much faster, and we
        # don't expect to free up much of what's currently allocated.
        gc.freeze()
        await engine.join()
        gc.unfreeze()  # Allow objects to be tidied away during shutdown
        if prometheus_server:
            await prometheus_server.close()


def main() -> None:
    """Start the F-engine."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
