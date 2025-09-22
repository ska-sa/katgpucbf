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

"""Utilities for writing the main programs of engines."""

import argparse
import asyncio
import contextlib
import enum
import gc
import ipaddress
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable, MutableMapping

import aiokatcp
import katsdpservices
import prometheus_async
import prometheus_client
from katsdpservices import get_interface_address
from katsdpservices.aiomonitor import add_aiomonitor_arguments, start_aiomonitor
from katsdptelstate.endpoint import endpoint_list_parser

from . import DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, DEFAULT_TTL, __version__, spead, utils

logger = logging.getLogger(__name__)


def parse_enum[E: enum.Enum](name: str, value: str, cls: type[E]) -> E:
    """Parse a command-line argument into an enum type."""
    table = {member.name.lower(): member for member in cls}
    try:
        return table[value]
    except KeyError:
        raise ValueError(f"Invalid {name} value {value} (valid values are {list(table.keys())})") from None


def parse_dither(value: str) -> utils.DitherType:
    """Parse a string into a dither type."""
    # Note: this allows only the non-aliases, so excludes DEFAULT
    return parse_enum("dither", value, utils.DitherType)


def parse_source_ipv4(value: str) -> list[tuple[str, int]]:
    """Parse a string into a list of IPv4 endpoints."""
    endpoints = endpoint_list_parser(spead.DEFAULT_PORT)(value)
    for endpoint in endpoints:
        ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
    return [(ep.host, ep.port) for ep in endpoints]


def parse_source(value: str) -> list[tuple[str, int]] | str:
    """Parse a string into a list of IP endpoints or a filename."""
    try:
        return parse_source_ipv4(value)
    except ValueError:
        if os.path.exists(value):
            return value
        raise ValueError(f"{value} is not an endpoint list or a filename") from None


def comma_split[T](
    base_type: Callable[[str], T], count: int | None = None, allow_single=False
) -> Callable[[str], list[T]]:
    """Return a function to split a comma-delimited str into a list of type T.

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
    allow_single
        If true (defaults to false), allow a single value to be used when
        `count` is greater than 1. In this case, it will be repeated `count`
        times.
    """

    def func(value: str) -> list[T]:
        parts = value.split(",")
        if parts == [""]:
            parts = []
        n = len(parts)
        if count is not None and n == 1 and allow_single:
            parts = parts * count
        elif count is not None and n != count:
            raise ValueError(f"Expected {count} comma-separated fields, received {n}")
        return [base_type(part) for part in parts]

    return func


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    katcp: bool = True,
    prometheus: bool = True,
    aiomonitor: bool = True,
    version: bool = True,
) -> None:
    """Add command-line arguments to the parser."""
    if katcp:
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
    if prometheus:
        parser.add_argument(
            "--prometheus-port",
            type=int,
            help="Network port on which to serve Prometheus metrics [none]",
        )
    if aiomonitor:
        add_aiomonitor_arguments(parser)
    if version:
        parser.add_argument("--version", action="version", version=__version__)


def _multi_add_argument(multi: bool, parser: argparse.ArgumentParser, *args, **kwargs) -> None:
    """Add an argument to a parser in either singular to multi form.

    In the multi form, the user passes a comma-separated list, and the argument
    becomes an array. To make the help text appropriate to either case, the
    `metavar` and `help` arguments are processed with :meth:`str.format`,
    with the following substitutions:

    ====== ============= ==============
    Key    multi=False   multi=True
    ====== ============= ==============
    s      ""            "(s)"
    dots   ""            ",..."
    ====== ============= ==============

    If a default is specified, it will become a singleton array in the multi
    case. However, if no default is specified, the default remains ``None``
    regardless of `multi`.
    """
    if multi:
        subst = {"s": "(s)", "dots": ",..."}
        kwargs["type"] = comma_split(kwargs.get("type", str))
        if "default" in kwargs:
            # Avoid showing the array in the help text. This is somewhat
            # hacky because it only handles one specific way of writing
            # the default. Unfortunately there isn't an easy way to do
            # %-formatting with only one token substituted.
            default = kwargs["default"]
            if "help" in kwargs:
                kwargs["help"] = kwargs["help"].replace("%(default)s", str(default))
            kwargs["default"] = [default]
    else:
        subst = {"s": "", "dots": ""}
    for key in ["metavar", "help"]:
        if key in kwargs:
            kwargs[key] = kwargs[key].format(**subst)
    parser.add_argument(*args, **kwargs)


def add_recv_arguments(parser: argparse.ArgumentParser, *, multi: bool = False) -> None:
    """Add arguments for receiving interface (supporting ibverbs).

    If `multi` is true, the arguments take comma-separated lists.
    """
    _multi_add_argument(
        multi,
        parser,
        "--recv-interface",
        type=get_interface_address,
        metavar="IFACE{dots}",
        help="Name{s} of input network device{s}",
    )
    parser.add_argument("--recv-ibv", action="store_true", help="Use ibverbs for receiving [no]")
    _multi_add_argument(
        multi,
        parser,
        "--recv-affinity",
        type=int,
        metavar="CORE{dots}",
        default=-1,
        help="Core{s} for input-handling thread{s} [not bound]",
    )
    _multi_add_argument(
        multi,
        parser,
        "--recv-comp-vector",
        type=int,
        metavar="VECTOR{dots}",
        default=0,
        help="Completion vector{s} for source streams, or -1 for polling [0]",
    )
    parser.add_argument(
        "--recv-buffer",
        type=int,
        default=128 * 1024 * 1024,
        metavar="BYTES",
        help="Size of network receive buffer [128MiB]",
    )


def add_send_arguments(parser: argparse.ArgumentParser, *, prefix: str = "send-", multi: bool = False) -> None:
    """Add arguments for sending interface (supporting ibverbs).

    Parameters
    ----------
    parser
        Parser to which arguments are added.
    prefix
        Prefix to use on argument names.
    multi
        If true, multiple interfaces are supported.
    """
    parser.add_argument(
        f"--{prefix}affinity",
        type=int,
        default=-1,
        metavar="CORE",
        help="Core for output-handling thread [not bound]",
    )
    parser.add_argument(
        f"--{prefix}comp-vector",
        type=int,
        default=0,
        metavar="VECTOR",
        help="Completion vector for transmission, or -1 for polling [%(default)s]",
    )
    _multi_add_argument(
        multi,
        parser,
        f"--{prefix}interface",
        type=get_interface_address,
        required=True,
        metavar="IFACE{dots}",
        help="Name{s} of output network device{s}",
    )
    parser.add_argument(
        f"--{prefix}ttl", type=int, default=DEFAULT_TTL, metavar="TTL", help="TTL for outgoing packets [%(default)s]"
    )
    parser.add_argument(f"--{prefix}ibv", action="store_true", help="Use ibverbs for output [no]")


def add_signal_handlers(server: aiokatcp.DeviceServer) -> None:
    """Arrange for clean shutdown on SIGINT (Ctrl-C) or SIGTERM."""
    signums = [signal.SIGINT, signal.SIGTERM]

    def handler():
        # Remove the handlers so that if it fails to shut down, the next
        # attempt will try harder.
        logger.info("Received signal, shutting down")
        for signum in signums:
            loop.remove_signal_handler(signum)
        server.halt()

    loop = asyncio.get_running_loop()
    for signum in signums:
        loop.add_signal_handler(signum, handler)


def add_gc_stats() -> None:
    """Add Prometheus metrics for garbage collection timing.

    It is only safe to call this once.
    """
    gc_time = prometheus_client.Histogram(
        "python_gc_time_seconds",
        "Time spent in garbage collection",
        buckets=[0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100],
        labelnames=["generation"],
    )
    # Make all the metrics exist, before any GC calls happen
    for generation in range(3):
        gc_time.labels(str(generation))
    start_time = 0.0

    def callback(phase: str, info: dict) -> None:
        nonlocal start_time
        if phase == "start":
            start_time = time.monotonic()
        else:
            started = start_time  # Copy as early as possible, before any more GC can happen
            elapsed = time.monotonic() - started
            gc_time.labels(str(info["generation"])).observe(elapsed)

    gc.callbacks.append(callback)


async def _engine_main_async(
    args: argparse.Namespace,
    start_engine: Callable[
        [argparse.Namespace, asyncio.TaskGroup, contextlib.AsyncExitStack, MutableMapping[str, object]],
        Awaitable[aiokatcp.DeviceServer],
    ],
) -> None:
    katsdpservices.setup_logging()
    add_gc_stats()
    locals_: dict[str, object] = {}
    async with contextlib.AsyncExitStack() as exit_stack:
        if getattr(args, "prometheus_port", None) is not None:
            prometheus_server = await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)
            exit_stack.push_async_callback(prometheus_server.close)

        if getattr(args, "aiomonitor", False):
            exit_stack.enter_context(start_aiomonitor(asyncio.get_running_loop(), args, locals_))

        tg = asyncio.TaskGroup()
        await exit_stack.enter_async_context(tg)

        engine = await start_engine(args, tg, exit_stack, locals_)
        add_signal_handlers(engine)
        # Avoid garbage collections needing to iterate over all the objects
        # allocated so far. That makes garbage collection much faster, and we
        # don't expect to free up much of what's currently allocated.
        gc.freeze()
        exit_stack.callback(gc.unfreeze)
        tg.create_task(engine.join())


def engine_main(
    args: argparse.Namespace,
    start_engine: Callable[
        [argparse.Namespace, asyncio.TaskGroup, contextlib.AsyncExitStack, MutableMapping],
        Awaitable[aiokatcp.DeviceServer],
    ],
) -> None:
    """Run an engine.

    This takes care of:

    - running an event loop;
    - setting up logging;
    - running a web server for Prometheus scraping if requested on the command line;
    - running aiomonitor
    - adding Prometheus statistics for the garbage collector (GC);
    - freezing the GC after starting the engine and unfreezing it on shutdown;

    Parameters
    ----------
    args
        The command-line arguments.
    start_engine
        The function that sets up and starts the engine. It takes the following parameters:

        - The command-line arguments (`args`)
        - A task group that can be used to schedule on-going work that will be
          waited on before shutting down.
        - An asynchronous exit stack that can be used to enter contexts or
          schedule cleanup work.
        - A dictionary of variables to expose to aiomonitor.
    """
    asyncio.run(_engine_main_async(args, start_engine))
