################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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
import signal
from typing import List, Optional

import katsdpsigproc.accel
import prometheus_async
from katsdpservices import get_interface_address, setup_logging
from katsdptelstate.endpoint import endpoint_parser

from katgpucbf.xbgpu.engine import XBEngine

from .. import DEFAULT_PACKET_PAYLOAD_BYTES, SPEAD_DESCRIPTOR_INTERVAL_S, __version__
from ..monitor import FileMonitor, Monitor, NullMonitor
from .correlation import device_filter
from .engine import done_callback

DEFAULT_KATCP_PORT = 7147
DEFAULT_KATCP_HOST = ""  # Default to all interfaces, but user can override with a specific one.

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--array-size", type=int, help="Number of antennas in the array.")
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
        "--chunk-spectra",
        type=int,
        default=5,
        help="A batch is a collection of heaps from different F-Engines with "
        "the same timestamp. This parameter specifies the number of "
        "consecutive spectra to store in the same chunk. The higher this "
        "value is, the more GPU and system RAM is allocated, the lower "
        "this value is, the more work the python processing thread "
        "is required to do. [%(default)s]",
    )
    parser.add_argument(
        "--rx-reorder-tol",
        type=int,
        default=2 ** 29,
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
    parser.add_argument("--dst-ttl", type=int, default=4, help="TTL for outgoing packets [%(default)s]")
    parser.add_argument("--dst-ibv", action="store_true", help="Use ibverbs for output [no].")
    parser.add_argument("--monitor-log", type=str, help="File to write performance-monitoring data to")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("src", type=endpoint_parser(7148), help="Multicast address data is received from.")
    parser.add_argument("dst", type=endpoint_parser(7148), help="Multicast address data is sent on.")

    args = parser.parse_args()

    if args.sample_bits != 8:
        parser.error("Only 8-bit values are currently supported.")

    return args


def add_signal_handlers(engine: XBEngine, standalone_tasks: List[asyncio.Task]) -> None:
    """Arrange for clean shutdown on SIGINT (Ctrl-C) or SIGTERM.

    Parameters
    ----------
    engine
        The XBEngine instance launched to facilitate XBEngine operations.
    standalone_tasks
        A list of Tasks that are created/run outside the XBEngine's realm of operation,
        e.g. Sending of Heap Descriptors.
    """
    signums = [signal.SIGINT, signal.SIGTERM]

    def handler():
        # Remove the handlers so that if it fails to shut down, the next
        # attempt will try harder.
        logger.info("Received signal, shutting down")
        for signum in signums:
            loop.remove_signal_handler(signum)
        for task in standalone_tasks:
            task.cancel()
        engine.halt()

    loop = asyncio.get_event_loop()
    for signum in signums:
        loop.add_signal_handler(signum, handler)


async def async_main(args: argparse.Namespace) -> None:
    """
    Create and launch the XB-Engine.

    This function creates the XBEngine object. It attaches the ibverbs sender
    and receiver transports to the XBEngine object and then tells the object to
    launch all its internal asyncio functions.

    Parameters
    ----------
    args: argparse.Namespace
        Command line parameter arguments generated by argparse.
    """
    monitor: Monitor
    if args.monitor_log is not None:
        monitor = FileMonitor(filename=args.monitor_log)
    else:
        monitor = NullMonitor()

    context = katsdpsigproc.accel.create_some_context(device_filter=device_filter)
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
        sample_bits=args.sample_bits,
        heap_accumulation_threshold=args.heap_accumulation_threshold,
        channel_offset_value=args.channel_offset_value,
        src_affinity=args.src_affinity,
        chunk_spectra=args.chunk_spectra,
        rx_reorder_tol=args.rx_reorder_tol,
        monitor=monitor,
        context=context,
    )

    if args.src_ibv:
        # Attach this transport to receive channelisation products from the network
        # at high rates.
        xbengine.add_udp_ibv_receiver_transport(
            src_ip=args.src.host,
            src_port=args.src.port,
            interface_ip=args.src_interface,
            comp_vector=args.src_comp_vector,
            buffer_size=args.src_buffer,
        )
    else:
        # Attach a plain-old UDP Receiver
        xbengine.add_udp_receiver_transport(
            src_ip=args.src.host,
            src_port=args.src.port,
            interface_ip=args.src_interface,
            buffer_size=args.src_buffer,
        )

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

    logger.info("Starting main processing loop")

    descriptor_task = asyncio.create_task(xbengine.run_descriptors_loop(SPEAD_DESCRIPTOR_INTERVAL_S))
    descriptor_task.add_done_callback(done_callback)

    add_signal_handlers(engine=xbengine, standalone_tasks=[descriptor_task])

    await xbengine.start()

    await asyncio.gather(
        asyncio.create_task(xbengine.join()),
        descriptor_task,
        return_exceptions=True,
    )

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
