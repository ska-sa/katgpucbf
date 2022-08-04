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
import logging
from typing import Optional, Sequence, Tuple

import katsdpsigproc.accel
import prometheus_async
from katsdpservices import get_interface_address, setup_logging
from katsdpservices.aiomonitor import add_aiomonitor_arguments, start_aiomonitor
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import endpoint_parser

from katgpucbf.xbgpu.engine import XBEngine

from .. import (
    DEFAULT_KATCP_HOST,
    DEFAULT_KATCP_PORT,
    DEFAULT_PACKET_PAYLOAD_BYTES,
    DEFAULT_TTL,
    DESCRIPTOR_TASK_NAME,
    SPEAD_DESCRIPTOR_INTERVAL_S,
    __version__,
)
from ..monitor import FileMonitor, Monitor, NullMonitor
from ..utils import add_signal_handlers, parse_source
from .correlation import device_filter

logger = logging.getLogger(__name__)


def parse_args(arglist: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse all command line parameters for the XB-Engine and ensure that they are valid."""
    parser = argparse.ArgumentParser(description="Launch an XB-Engine for a single multicast stream.")
    parser.add_argument(
        "--katcp-host",
        type=str,
        default=DEFAULT_KATCP_HOST,
        help="Hostname or IP address on which to listen for KATCP C&M connections [all interfaces]",
    )
    parser.add_argument(
        "--katcp-port",
        type=int,
        default=DEFAULT_KATCP_PORT,
        help="TCP port on which to listen for KATCP C&M connections [%(default)s]",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        help="Network port on which to serve Prometheus metrics [none]",
    )
    add_aiomonitor_arguments(parser)
    parser.add_argument(
        "--adc-sample-rate",
        type=float,
        required=True,
        help="Digitiser sample rate (Hz). If this value is set lower than the actual rate, the pipeline will stall.",
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
        help="Total number of channels out of the F-Engine PFB.",
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
        "--spectra-per-heap",
        type=int,
        default=256,
        help="Number of packed spectra in every received channel. [%(default)s]",
    )
    parser.add_argument(
        "--sample-bits",
        type=int,
        default=8,
        help="Number of bits for each real and imaginary value in a sample. [%(default)s]",
    )
    parser.add_argument(
        "--heaps-per-fengine-per-chunk",
        type=int,
        default=5,
        help="A batch is a collection of heaps from different F-Engines with "
        "the same timestamp. This parameter specifies the number of "
        "consecutive batches to store in the same chunk. The higher this "
        "value is, the more GPU and system RAM is allocated, the lower "
        "this value is, the more work the python processing thread "
        "is required to do. [%(default)s]",
    )
    parser.add_argument(
        "--rx-reorder-tol",
        type=int,
        default=2**29,
        help="Maximum time (in ADC ticks) that packets can be delayed relative to others "
        "and still be accepted. [%(default)s]",
    )
    parser.add_argument(
        "--heap-accumulation-threshold",
        type=int,
        default=52,
        help="Number of batches of heaps to accumulate in a single dump. [%(default)s]",
    )
    parser.add_argument(
        "--src-affinity", type=int, default=-1, help="Core to which the receiver thread will be bound [not bound]."
    )
    parser.add_argument(
        "--src-comp-vector",
        type=int,
        default=0,
        help="Completion vector for source streams, or -1 for polling [%(default)s].",
    )
    parser.add_argument(
        "--src-interface",
        type=get_interface_address,
        required=True,
        help="Name of the interface receiving data from the F-Engines, e.g. eth0.",
    )
    parser.add_argument("--src-ibv", action="store_true", help="Use ibverbs for input [no].")
    parser.add_argument(
        "--src-buffer",
        type=int,
        default=32 * 1024 * 1024,
        metavar="BYTES",
        help="Size of network receive buffer [32MiB]",
    )
    parser.add_argument(
        "--dst-affinity", type=int, default=-1, help="Core to which the sender thread will be bound [not bound]."
    )
    parser.add_argument(
        "--dst-comp-vector",
        type=int,
        default=1,
        help="Completion vector for transmission, or -1 for polling [%(default)s].",
    )
    parser.add_argument(
        "--dst-interface",
        type=get_interface_address,
        required=True,
        help="Name of the interface that this engine will transmit data on, e.g. eth1.",
    )
    parser.add_argument(
        "--dst-packet-payload",
        type=int,
        default=DEFAULT_PACKET_PAYLOAD_BYTES,
        help="Size in bytes for output packets (baseline correlation products payload only) [%(default)s]",
    )
    parser.add_argument("--dst-ttl", type=int, default=DEFAULT_TTL, help="TTL for outgoing packets [%(default)s]")
    parser.add_argument("--dst-ibv", action="store_true", help="Use ibverbs for output [no].")
    parser.add_argument(
        "--tx-enabled",
        action="store_true",
        help="Start with correlator output transmission enabled, without having to issue a katcp command.",
    )
    parser.add_argument("--monitor-log", type=str, help="File to write performance-monitoring data to")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("src", type=parse_source, help="Multicast address data is received from.")
    parser.add_argument("dst", type=endpoint_parser(7148), help="Multicast address data is sent on.")

    args = parser.parse_args(arglist)

    if args.sample_bits != 8:
        parser.error("Only 8-bit values are currently supported.")

    return args


def make_engine(context: AbstractContext, args: argparse.Namespace) -> Tuple[XBEngine, Monitor]:
    """Make an :class:`XBEngine` object, given a GPU context.

    Parameters
    ----------
    context
        The GPU context in which the :class:`.XBEngine` will operate.
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
        adc_sample_rate_hz=args.adc_sample_rate,
        send_rate_factor=args.send_rate_factor,
        n_ants=args.array_size,
        n_channels_total=args.channels,
        n_channels_per_stream=args.channels_per_substream,
        n_spectra_per_heap=args.spectra_per_heap,
        n_samples_between_spectra=args.samples_between_spectra,
        sample_bits=args.sample_bits,
        heap_accumulation_threshold=args.heap_accumulation_threshold,
        channel_offset_value=args.channel_offset_value,
        src=args.src,
        src_interface=args.src_interface,
        src_ibv=args.src_ibv,
        src_affinity=args.src_affinity,
        src_comp_vector=args.src_comp_vector,
        src_buffer=args.src_buffer,
        heaps_per_fengine_per_chunk=args.heaps_per_fengine_per_chunk,
        rx_reorder_tol=args.rx_reorder_tol,
        tx_enabled=args.tx_enabled,
        monitor=monitor,
        context=context,
    )
    return xbengine, monitor


async def async_main(args: argparse.Namespace) -> None:
    """Create and launch the XB-Engine.

    Attach the ibverbs sender transport to the XBEngine object and then tell
    the object to launch all its internal asyncio functions.

    Parameters
    ----------
    args
        Parsed arguments returned from :func:`parse_args`.
    """
    context = katsdpsigproc.accel.create_some_context(device_filter=device_filter)
    xbengine, monitor = make_engine(context, args)

    # Attach this transport to send the baseline correlation products to the
    # network.
    xbengine.add_udp_sender_transport(
        dest_ip=args.dst.host,
        dest_port=args.dst.port,
        interface_ip=args.dst_interface,
        ttl=args.dst_ttl,
        thread_affinity=args.dst_affinity,
        comp_vector=args.dst_comp_vector,
        packet_payload=args.dst_packet_payload,
        use_ibv=args.dst_ibv,
    )

    prometheus_server: Optional[prometheus_async.aio.web.MetricsHTTPServer] = None
    if args.prometheus_port is not None:
        prometheus_server = await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)

    with monitor, start_aiomonitor(asyncio.get_running_loop(), args, locals()):
        logger.info("Starting main processing loop")

        # TODO: Work the descriptor task into being handled by xbengine.start
        # see NGC-589.
        descriptor_task = asyncio.create_task(
            xbengine.run_descriptors_loop(SPEAD_DESCRIPTOR_INTERVAL_S), name=DESCRIPTOR_TASK_NAME
        )
        xbengine.add_service_task(descriptor_task)
        add_signal_handlers(xbengine)

        await xbengine.start()
        await xbengine.join()

        if prometheus_server:
            await prometheus_server.close()


def main() -> None:
    """
    Launch the XB-Engine pipeline.

    This method only sets up the asyncio loop and calls the async_main() method
    which is where the real work is done.
    """
    args = parse_args()
    setup_logging()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
