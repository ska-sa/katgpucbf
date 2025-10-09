################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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
to create an :class:`.FEngine` object, which then takes over the actual running
of the processing.
"""

import argparse
import asyncio
import contextlib
import logging
import math
from collections.abc import MutableMapping, Sequence

import katsdpsigproc.accel as accel
import vkgdr
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import endpoint_list_parser

from .. import (
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_PACKET_PAYLOAD_BYTES,
    DIG_SAMPLE_BITS,
)
from ..main import (
    SubParser,
    add_common_arguments,
    add_recv_arguments,
    add_send_arguments,
    add_time_converter_arguments,
    engine_main,
    parse_dither,
    parse_enum,
    parse_source,
)
from ..mapped_array import make_vkgdr
from ..monitor import FileMonitor, Monitor, NullMonitor
from ..spead import DEFAULT_PORT
from ..utils import DitherType
from . import DIG_SAMPLE_BITS_VALID
from .engine import FEngine
from .output import NarrowbandOutput, NarrowbandOutputDiscard, NarrowbandOutputNoDiscard, WidebandOutput, WindowFunction

logger = logging.getLogger(__name__)
DEFAULT_TAPS = 16
DEFAULT_W_CUTOFF = 1.0
#: Ratio of decimation factor to tap count
DEFAULT_DDC_TAPS_RATIO = 16
DEFAULT_WEIGHT_PASS = 0.005


def _parse_window_function(value: str) -> WindowFunction:
    """Parse a :class:`.WindowFunction` from a command-line argument."""
    return parse_enum("window_function", value, WindowFunction)


def _add_stream_args(parser: SubParser) -> None:
    parser.add_argument("name", type=str, required=True)
    parser.add_argument("channels", type=int, required=True)
    parser.add_argument("jones_per_batch", type=int, default=DEFAULT_JONES_PER_BATCH)
    parser.add_argument("dst", type=endpoint_list_parser(DEFAULT_PORT), required=True)
    parser.add_argument("taps", type=int, default=DEFAULT_TAPS)
    parser.add_argument("w_cutoff", type=float, default=DEFAULT_W_CUTOFF)
    parser.add_argument("window_function", type=_parse_window_function, default=WindowFunction.DEFAULT)
    parser.add_argument("dither", type=parse_dither, default=DitherType.DEFAULT)


def parse_wideband(value: str) -> WidebandOutput:
    """Parse a string with a wideband configuration.

    The string has a comma-separated list of key=value pairs. See
    :class:`WidebandOutput` for the valid keys and types. The following
    keys are required:

    - name
    - channels
    - dst
    """
    parser = SubParser()
    _add_stream_args(parser)
    args = parser(value)
    return WidebandOutput(**vars(args))


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
    parser = SubParser()
    _add_stream_args(parser)
    parser.add_argument("centre_frequency", type=float, required=True)
    parser.add_argument("decimation", type=int, required=True)
    parser.add_argument("ddc_taps", type=int)  # Default is computed later
    parser.add_argument("weight_pass", type=float, default=DEFAULT_WEIGHT_PASS)
    parser.add_argument("pass_bandwidth", type=float)
    args = parser(value)
    if args.ddc_taps is None:
        args.ddc_taps = DEFAULT_DDC_TAPS_RATIO * args.decimation
        if args.pass_bandwidth is not None:
            args.ddc_taps *= 2  # sampling = 2 * decimation in this case
    if args.pass_bandwidth is not None:
        return NarrowbandOutputNoDiscard(**vars(args))
    else:
        del args.pass_bandwidth
        return NarrowbandOutputDiscard(**vars(args))


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
            f"Optional keys: taps [{DEFAULT_TAPS}], ddc_taps [{DEFAULT_DDC_TAPS_RATIO}*subsampling], "
            f"w_cutoff [{DEFAULT_W_CUTOFF}], window_function [hann], weight_pass [{DEFAULT_WEIGHT_PASS}], "
            "dither [uniform]."
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
            f"Optional keys: taps [{DEFAULT_TAPS}], w_cutoff [{DEFAULT_W_CUTOFF}], "
            "window_function [hann], dither [uniform]"
        ),
    )
    add_common_arguments(parser)
    add_recv_arguments(parser, multi=True)
    parser.add_argument(
        "--recv-packet-samples", type=int, default=4096, help="Number of samples per digitiser packet [%(default)s]"
    )
    add_send_arguments(parser, multi=True)
    parser.add_argument(
        "--send-packet-payload",
        type=int,
        default=DEFAULT_PACKET_PAYLOAD_BYTES,
        metavar="BYTES",
        help="Size for output packets (voltage payload only) [%(default)s]",
    )
    # TODO (NGC-1758): add this argument to xbgpu/dsim so it can be
    # incorporated into add_send_arguments.
    parser.add_argument(
        "--send-buffer",
        type=int,
        default=1024 * 1024,
        metavar="BYTES",
        help="Size of network send buffer [1MiB]",
    )
    add_time_converter_arguments(parser)
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
        "--recv-chunk-samples",
        type=int,
        default=2**25,
        metavar="SAMPLES",
        help="Number of digitiser samples to process at a time (per pol). [%(default)s]",
    )
    parser.add_argument(
        "--send-chunk-jones",
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
        "--send-sample-bits",
        type=int,
        default=8,
        choices=[4, 8],
        metavar="BITS",
        help="Number of bits per output sample real component [%(default)s]",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Initial eq gains [%(default)s]")
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
    parser.add_argument("src", type=parse_source, help="Source endpoints (or pcap file)")
    args = parser.parse_args(arglist)

    if args.use_peerdirect and not args.send_ibv:
        parser.error("--use-peerdirect requires --send-ibv")
    if not isinstance(args.src, str) and args.recv_interface is None:
        parser.error("Live source requires --recv-interface")
    if args.recv_ibv and len(args.recv_affinity) != len(args.recv_comp_vector):
        parser.error("--recv-comp-vector must have same length as --recv-affinity")

    # Convert from _*OutputDict to *Output
    used_names = set()
    args.outputs = []
    for output_group in [args.wideband, args.narrowband]:
        for output in output_group:
            name = output.name
            if output.name in used_names:
                parser.error(f"output name {name} used twice")
            if len(output.dst) % len(args.send_interface) != 0:
                parser.error(f"{name}: number of destinations must be divisible by number of destination interfaces")
            used_names.add(name)
            args.outputs.append(output)
    if not args.outputs:
        parser.error("At least one --wideband or --narrowband argument is required")

    return args


def make_engine(ctx: AbstractContext, vkgdr_handle: vkgdr.Vkgdr, args: argparse.Namespace) -> tuple[FEngine, Monitor]:
    """Make an :class:`.FEngine` object, given a GPU context.

    Parameters
    ----------
    ctx
        The GPU context in which the :class:`.FEngine`
        will operate.
    vkgdr_handle
        The Vkgdr handle in which the :class:`.FEngine`
        will operate. It must use the same device as `ctx`.
    args
        Parsed arguments returned from :func:`parse_args`.
    """
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(args.monitor_log)
    else:
        monitor = NullMonitor()

    batch_jones_lcm = math.lcm(*(output.jones_per_batch for output in args.outputs))
    chunk_jones = accel.roundup(args.send_chunk_jones, batch_jones_lcm)
    engine = FEngine(
        katcp_host=args.katcp_host,
        katcp_port=args.katcp_port,
        context=ctx,
        vkgdr_handle=vkgdr_handle,
        srcs=args.src,
        recv_interface=args.recv_interface,
        recv_ibv=args.recv_ibv,
        recv_affinity=args.recv_affinity,
        recv_comp_vector=args.recv_comp_vector,
        recv_packet_samples=args.recv_packet_samples,
        recv_buffer=args.recv_buffer,
        send_interface=args.send_interface,
        send_ttl=args.send_ttl,
        send_ibv=args.send_ibv,
        send_packet_payload=args.send_packet_payload,
        send_affinity=args.send_affinity,
        send_comp_vector=args.send_comp_vector,
        send_buffer=args.send_buffer,
        outputs=args.outputs,
        adc_sample_rate=args.adc_sample_rate,
        send_rate_factor=args.send_rate_factor,
        feng_id=args.feng_id,
        n_ants=args.array_size,
        chunk_samples=args.recv_chunk_samples,
        chunk_jones=chunk_jones,
        dig_sample_bits=args.dig_sample_bits,
        send_sample_bits=args.send_sample_bits,
        max_delay_diff=args.max_delay_diff,
        gain=args.gain,
        sync_time=args.sync_time,
        mask_timestamp=args.mask_timestamp,
        use_vkgdr=args.use_vkgdr,
        use_peerdirect=args.use_peerdirect,
        monitor=monitor,
    )

    return engine, monitor


async def start_engine(
    args: argparse.Namespace,
    tg: asyncio.TaskGroup,
    exit_stack: contextlib.AsyncExitStack,
    locals_: MutableMapping[str, object],
) -> FEngine:
    """Start the F-Engine asynchronously.

    See Also
    --------
    katgpucbf.main.engine_main
    """
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    vkgdr_handle = make_vkgdr(ctx.device)
    logger.info("Initialising F-engine on %s", ctx.device.name)
    engine, monitor = make_engine(ctx, vkgdr_handle, args)
    exit_stack.enter_context(monitor)
    locals_.update(locals())
    await engine.start()
    return engine


def main() -> None:
    """Run the F-engine."""
    engine_main(parse_args(), start_engine)


if __name__ == "__main__":
    main()
