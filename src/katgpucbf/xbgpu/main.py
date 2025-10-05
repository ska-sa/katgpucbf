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

"""
Module to launch the XB-Engine.

This module parses all command line arguments required to configure the
XB-Engine and creates an XBEngine object. The XBEngine object then manages
everything required to run the XB-Engine.

.. todo::
    - Checks need to be put in place to ensure the command line parameters are
      correct:

        - Is the port number valid, is the IP address a multicast address, is
          the array size >0, etc.
"""

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import MutableMapping, Sequence

import katsdpsigproc.accel
import vkgdr
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import endpoint_parser

from katgpucbf.xbgpu.engine import XBEngine

from .. import (
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_PACKET_PAYLOAD_BYTES,
)
from ..main import (
    SubParser,
    add_common_arguments,
    add_recv_arguments,
    add_send_arguments,
    engine_main,
    parse_dither,
    parse_source_ipv4,
)
from ..mapped_array import make_vkgdr
from ..monitor import FileMonitor, Monitor, NullMonitor
from ..spead import DEFAULT_PORT
from ..utils import DitherType
from .correlation import device_filter
from .output import BOutput, XOutput

logger = logging.getLogger(__name__)


def _add_stream_args(parser: SubParser) -> None:
    parser.add_argument("name", type=str, required=True)
    parser.add_argument("dst", type=endpoint_parser(DEFAULT_PORT), required=True)


def _parse_pol(value: str) -> int:
    pol = int(value)
    if pol not in {0, 1}:
        raise argparse.ArgumentTypeError("pol must be either 0 or 1")
    return pol


def parse_beam(value: str) -> BOutput:
    """Parse a string with beam configuration data.

    The string has a comma-separated list of key=value pairs. See
    :class:`BOutput` for the valid keys and types. The following
    keys are required:

    - name
    - dst
    - pol
    """
    parser = SubParser()
    _add_stream_args(parser)
    parser.add_argument("pol", type=_parse_pol, required=True)
    parser.add_argument("dither", type=parse_dither, default=DitherType.DEFAULT)
    args = parser(value)
    return BOutput(**vars(args))


def parse_corrprod(value: str) -> XOutput:
    """Parse a string with correlation product configuration data.

    The string has comma-separated list of key=value pairs. See
    :class:`XOutput` for the valid keys and types. The following keys
    are required:

    - name
    - dst
    - heap_accumulation_threshold
    """
    parser = SubParser()
    _add_stream_args(parser)
    parser.add_argument("heap_accumulation_threshold", type=int, required=True)
    args = parser(value)
    return XOutput(**vars(args))


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse all command line parameters for the XB-Engine and ensure that they are valid."""
    parser = argparse.ArgumentParser(description="Launch an XB-Engine for a single multicast stream.")
    parser.add_argument(
        "--beam",
        type=parse_beam,
        default=[],
        action="append",
        metavar="KEY=VALUE[,KEY=VALUE...]",
        help="Add a half-beam output (may be repeated). The required keys are: name, dst, pol. Optional keys: dither.",
    )
    parser.add_argument(
        "--corrprod",
        type=parse_corrprod,
        default=[],
        action="append",
        metavar="KEY=VALUE[,KEY=VALUE...]",
        help="Add a baseline-correlation-products output (may be repeated). The required keys are: "
        "name, dst, heap_accumulation_threshold.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--adc-sample-rate",
        type=float,
        required=True,
        help="Digitiser sample rate (Hz). If this value is set lower than the actual rate, the pipeline will stall.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        help="Total bandwidth (Hz) of all channels (--channels) [computed assuming critically-sampled PFB].",
    )
    parser.add_argument(
        "--send-rate-factor",
        type=float,
        default=1.1,
        help="Target transmission rate faster than ADC sample rate by this factor. "
        "Set to zero to send as fast as possible. [%(default)s]",
    )
    parser.add_argument("--array-size", type=int, required=True, help="Number of antennas in the array.")
    parser.add_argument(
        "--channels",
        type=int,
        required=True,
        help="Total number of channels in the stream.",
    )
    parser.add_argument(
        "--channels-per-substream",
        type=int,
        required=True,
        help="Number of channels in the multicast stream that this engine receives data from.",
    )
    parser.add_argument(
        "--samples-between-spectra",
        type=int,
        required=True,
        help="Number of samples between spectra.",
    )
    parser.add_argument(
        "--channel-offset-value",
        type=int,
        default=0,
        help="Index of the first channel in the subset of channels processed "
        "by this XB-Engine. Used to set the value in the XB-Engine "
        "output heaps for spectrum reassembly by the downstream receiver. "
        "[%(default)s]",
    )
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        help="Number of antenna-channelised-voltage Jones vectors in each F-engine batch. [%(default)s]",
    )
    parser.add_argument(
        "--sample-bits",
        type=int,
        default=8,
        choices=[8],
        help="Number of bits for each real and imaginary value in a sample. [%(default)s]",
    )
    parser.add_argument(
        "--heaps-per-fengine-per-chunk",
        type=int,
        default=32,
        help="A batch is a collection of heaps from different F-Engines with "
        "the same timestamp. This parameter specifies the number of "
        "consecutive batches to store in the same chunk. The higher this "
        "value is, the more GPU and system RAM is allocated, the lower "
        "this value is, the more work the python processing thread "
        "is required to do. [%(default)s]",
    )
    parser.add_argument(
        "--recv-reorder-tol",
        type=int,
        default=2**29,
        help="Maximum time (in ADC ticks) that packets can be delayed relative to others "
        "and still be accepted. [%(default)s]",
    )
    parser.add_argument(
        "--sync-time",
        type=float,
        required=True,
        help="UNIX time at which digitisers were synced.",
    )
    add_recv_arguments(parser, multi=False)
    add_send_arguments(parser, multi=False)
    parser.add_argument(
        "--send-packet-payload",
        type=int,
        default=DEFAULT_PACKET_PAYLOAD_BYTES,
        help="Size in bytes for output packets (payload only) [%(default)s]",
    )
    parser.add_argument(
        "--send-enabled",
        action="store_true",
        help="Start with correlator output transmission enabled, without having to issue a katcp command.",
    )
    parser.add_argument("--monitor-log", type=str, help="File to write performance-monitoring data to")
    parser.add_argument("src", type=parse_source_ipv4, help="Multicast address data is received from.")

    args = parser.parse_args(arglist)
    if args.jones_per_batch % args.channels != 0:
        parser.error(f"--jones-per-batch ({args.jones_per_batch}) must be a multiple of --channels ({args.channels})")

    if args.bandwidth is None:
        args.bandwidth = args.adc_sample_rate / args.samples_between_spectra * args.channels

    used_names = set()
    args.outputs = []
    for output_group in [args.corrprod, args.beam]:
        for output in output_group:
            name = output.name
            if name in used_names:
                parser.error(f"output name {name} already used.")
            used_names.add(name)
            args.outputs.append(output)

    return args


def make_engine(
    context: AbstractContext, vkgdr_handle: vkgdr.Vkgdr, args: argparse.Namespace
) -> tuple[XBEngine, Monitor]:
    """Make an :class:`XBEngine` object, given a GPU context.

    Parameters
    ----------
    context
        The GPU context in which the :class:`.XBEngine` will operate.
    vkgdr_handle
        Handle to vkgdr for the same device as `context`.
    args
        Parsed arguments returned from :func:`parse_args`.
    """
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(filename=args.monitor_log)
    else:
        monitor = NullMonitor()

    logger.info("Initialising XB-Engine on %s", context.device.name)
    xbengine = XBEngine(
        katcp_host=args.katcp_host,
        katcp_port=args.katcp_port,
        adc_sample_rate=args.adc_sample_rate,
        bandwidth=args.bandwidth,
        send_rate_factor=args.send_rate_factor,
        n_ants=args.array_size,
        n_channels=args.channels,
        n_channels_per_substream=args.channels_per_substream,
        n_samples_between_spectra=args.samples_between_spectra,
        n_spectra_per_heap=args.jones_per_batch // args.channels,
        sample_bits=args.sample_bits,
        sync_time=args.sync_time,
        channel_offset_value=args.channel_offset_value,
        outputs=args.outputs,
        src=args.src,
        recv_interface=args.recv_interface,
        recv_ibv=args.recv_ibv,
        recv_affinity=args.recv_affinity,
        recv_comp_vector=args.recv_comp_vector,
        recv_buffer=args.recv_buffer,
        send_interface=args.send_interface,
        send_ttl=args.send_ttl,
        send_ibv=args.send_ibv,
        send_packet_payload=args.send_packet_payload,
        send_affinity=args.send_affinity,
        send_comp_vector=args.send_comp_vector,
        heaps_per_fengine_per_chunk=args.heaps_per_fengine_per_chunk,
        recv_reorder_tol=args.recv_reorder_tol,
        send_enabled=args.send_enabled,
        monitor=monitor,
        context=context,
        vkgdr_handle=vkgdr_handle,
    )
    return xbengine, monitor


async def start_engine(
    args: argparse.Namespace,
    tg: asyncio.TaskGroup,
    exit_stack: contextlib.AsyncExitStack,
    locals_: MutableMapping[str, object],
) -> XBEngine:
    """Create and launch the XB-Engine.

    Attach the ibverbs sender transport to the XBEngine object and then tell
    the object to launch all its internal asyncio functions.

    See Also
    --------
    katgpucbf.main.engine_main
    """
    context = katsdpsigproc.accel.create_some_context(device_filter=device_filter)
    vkgdr_handle = make_vkgdr(context.device)
    xbengine, monitor = make_engine(context, vkgdr_handle, args)
    exit_stack.enter_context(monitor)

    # katsdpcontroller launches us with real-time scheduling, but we don't
    # want that for the main Python thread since it can starve the
    # latency-sensitive network threads.
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))

    logger.info("Starting main processing loop")
    locals_.update(locals())
    await xbengine.start()
    return xbengine


def main() -> None:
    """Run the XB-engine."""
    engine_main(parse_args(), start_engine)


if __name__ == "__main__":
    main()
