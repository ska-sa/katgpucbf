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
from collections.abc import Callable, MutableMapping, Sequence
from typing import TypedDict

import katsdpsigproc.accel as accel
import vkgdr
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser

from .. import (
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_PACKET_PAYLOAD_BYTES,
    DIG_SAMPLE_BITS,
)
from ..main import (
    add_common_arguments,
    add_recv_arguments,
    add_send_arguments,
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
    window_function: WindowFunction
    dither: DitherType


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
    pass_bandwidth: float


def _parse_stream[OD: _OutputDict](value: str, kws: OD, field_callback: Callable[[OD, str, str], None]) -> None:
    """Parse a wideband or narrowband stream description.

    This populates `kws` (which should initially be empty) from key=value pairs
    in `value`. It handles the common fields directly, and type-specific fields
    are handled by a provided field callback. The callback is invoked with
    `kws`, the key and the value. If it does not recognise the key, it should
    raise :exc:`ValueError`.
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
                    case "window_function":
                        kws[key] = parse_enum("window_function", data, WindowFunction)
                    case "dst":
                        kws[key] = endpoint_list_parser(DEFAULT_PORT)(data)
                    case "dither":
                        kws[key] = parse_dither(data)
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
        raise ValueError(f"unknown key {key!r}")

    try:
        kws: _WidebandOutputDict = {}
        _parse_stream(value, kws, field_callback)
        kws = {
            "taps": DEFAULT_TAPS,
            "w_cutoff": DEFAULT_W_CUTOFF,
            "window_function": WindowFunction.DEFAULT,
            "jones_per_batch": DEFAULT_JONES_PER_BATCH,
            "dither": DitherType.DEFAULT,
            **kws,
        }
        return WidebandOutput(**kws)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
            case "centre_frequency" | "weight_pass" | "pass_bandwidth":
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
        default_taps = DEFAULT_DDC_TAPS_RATIO * kws["decimation"]
        if "pass_bandwidth" in kws:
            default_taps *= 2  # sampling = 2 * decimation in this case
        kws = {
            "taps": DEFAULT_TAPS,
            "w_cutoff": DEFAULT_W_CUTOFF,
            "window_function": WindowFunction.DEFAULT,
            "jones_per_batch": DEFAULT_JONES_PER_BATCH,
            "weight_pass": DEFAULT_WEIGHT_PASS,
            "ddc_taps": default_taps,
            "dither": DitherType.DEFAULT,
            **kws,
        }
        if "pass_bandwidth" in kws:
            return NarrowbandOutputNoDiscard(**kws)
        else:
            # mypy isn't smart enough to realise that "pass_bandwidth"
            # isn't going to be in **kws.
            return NarrowbandOutputDiscard(**kws)  # type: ignore[misc]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
