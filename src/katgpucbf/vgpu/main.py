################################################################################
# Copyright (c) 2025 National Research Foundation (SARAO)
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

"""vgpu main script."""

import argparse
import asyncio
import contextlib
import re
from collections.abc import MutableMapping, Sequence

import aiokatcp
from katsdptelstate.endpoint import endpoint_list_parser

from .. import DEFAULT_JONES_PER_BATCH
from ..main import (
    add_common_arguments,
    add_recv_arguments,
    add_send_arguments,
    add_time_converter_arguments,
    comma_split,
    engine_main,
    parse_source_ipv4,
)
from ..monitor import FileMonitor, Monitor, NullMonitor
from .engine import VEngine

VTP_DEFAULT_PORT = 52030
_ARGUMENT_PARSER = argparse.ArgumentParser  # Modified by unit tests


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = _ARGUMENT_PARSER(prog="vgpu")
    add_common_arguments(parser)
    add_recv_arguments(parser)
    add_send_arguments(parser, ibverbs=False)
    add_time_converter_arguments(parser)
    parser.add_argument("--recv-channels", type=int, metavar="CHANNELS", required=True, help="Number of input channels")
    parser.add_argument(
        "--recv-channels-per-substream",
        type=int,
        metavar="CHANNELS",
        required=True,
        help="Number of input channels in heap",
    )
    parser.add_argument(
        "--recv-jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        help="Number of tied-array-channelised-voltage Jones vectors in each batch. [%(default)s]",
    )
    parser.add_argument(
        "--recv-samples-between-spectra",
        type=int,
        metavar="SAMPLES",
        required=True,
        help="Timestamp increment between spectra",
    )
    parser.add_argument(
        "--recv-batches-per-chunk", type=int, metavar="BATCHES", default=1, help="Number of batches per input chunk"
    )
    parser.add_argument(
        "--recv-sample-bits",
        type=int,
        metavar="BITS",
        default=8,
        choices=[8],
        help="Number of bits in each real sample [%(default)s]",
    )
    parser.add_argument("--recv-bandwidth", type=float, metavar="HZ", required=True, help="Input bandwidth")
    parser.add_argument(
        "--recv-pols",
        type=comma_split(str, 2),
        metavar="[+-]P,[+-]P",
        required=True,
        help="Input polarisations (±x, ±y, ±L or ±R)",
    )
    parser.add_argument("--send-bandwidth", type=float, metavar="HZ", required=True, help="Output bandwidth")
    parser.add_argument(
        "--send-pols",
        type=comma_split(str, 2),
        metavar="P,P",
        required=True,
        help="Output polarisations (x, y, L or R)",
    )
    parser.add_argument(
        "--send-samples-per-frame",
        type=int,
        metavar="SAMPLES",
        default=20000,
        help="Samples per VDIF frame [%(default)s]",
    )
    parser.add_argument("--send-station", type=str, metavar="ID", required=True, help="VDIF Station ID")
    parser.add_argument("--fir-taps", type=int, required=True, metavar="TAPS", help="Number of taps in rational filter")
    parser.add_argument(
        "--hilbert-taps", type=int, default=201, metavar="TAPS", help="Number of taps in Hilbert filter [%(default)s]"
    )
    parser.add_argument(
        "--passband",
        type=float,
        default=0.9,
        metavar="FRACTION",
        help="Fraction of band to retain in passband filter [%(default)s]",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.969,
        metavar="SIGMA",
        help="Threshold (in σ) between quantisation levels [%(default)s]",
    )
    parser.add_argument("--monitor-log", help="File to write performance-monitoring data to")
    parser.add_argument("src", type=parse_source_ipv4, nargs=2, help="Source endpoints")
    parser.add_argument("dst", type=endpoint_list_parser(VTP_DEFAULT_PORT), help="Destination endpoints")

    args = parser.parse_args(arglist)
    if args.recv_channels % args.recv_channels_per_substream != 0:
        parser.error(
            f"--recv-channels ({args.recv_channels}) "
            f"must be a multiple of --recv-channels-per-substream ({args.recv_channels_per_substream})"
        )
    if args.recv_jones_per_batch % args.recv_channels != 0:
        parser.error(
            f"--recv-jones-per-batch ({args.recv_jones_per_batch}) "
            f"must be a multiple of --recv-channels ({args.recv_channels})"
        )
    for pol in args.recv_pols:
        if not re.fullmatch(r"^[-+]?[xyLR]", pol):
            parser.error(f"{pol!r} is not a valid --recv-pol value")
    if set(pol[-1] for pol in args.recv_pols) not in [{"x", "y"}, {"L", "R"}]:
        parser.error("--recv-pol is not an orthogonal polarisation basis")
    for pol in args.send_pols:
        if pol not in ["x", "y", "L", "R"]:
            parser.error(f"{pol!r} is not a valid --send-pol value")
    return args


def make_engine(args: argparse.Namespace) -> VEngine:
    """Create the :class:`.VEngine`."""
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(args.monitor_log)
    else:
        monitor = NullMonitor()

    return VEngine(
        katcp_host=args.katcp_host,
        katcp_port=args.katcp_port,
        sync_time=args.sync_time,
        adc_sample_rate=args.adc_sample_rate,
        n_channels=args.recv_channels,
        n_channels_per_substream=args.recv_channels_per_substream,
        n_spectra_per_heap=args.recv_jones_per_batch // args.recv_channels,
        n_samples_between_spectra=args.recv_samples_between_spectra,
        n_batches_per_chunk=args.recv_batches_per_chunk,
        sample_bits=args.recv_sample_bits,
        srcs=args.src,
        recv_interface=args.recv_interface,
        recv_ibv=args.recv_ibv,
        recv_affinity=args.recv_affinity,
        recv_comp_vector=args.recv_comp_vector,
        recv_buffer=args.recv_buffer,
        recv_pols=tuple(args.recv_pols),
        send_pols=tuple(args.send_pols),
        monitor=monitor,
    )


async def start_engine(
    args: argparse.Namespace,
    tg: asyncio.TaskGroup,
    exit_stack: contextlib.AsyncExitStack,
    locals_: MutableMapping[str, object],
) -> aiokatcp.DeviceServer:
    """Start the V-engine asynchronously.

    See Also
    --------
    katgpucbf.main.engine_main
    """
    engine = make_engine(args)
    locals_.update(locals())
    await engine.start()
    return engine


def main():
    """Run the V-engine."""
    engine_main(parse_args(), start_engine)


if __name__ == "__main__":
    main()
