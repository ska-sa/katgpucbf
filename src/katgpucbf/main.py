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
import gc
import logging
import signal
import time
from collections.abc import Awaitable, Callable, MutableMapping

import aiokatcp
import katsdpservices
import prometheus_async
import prometheus_client
from katsdpservices.aiomonitor import add_aiomonitor_arguments, start_aiomonitor

from . import DEFAULT_KATCP_HOST, DEFAULT_KATCP_PORT, __version__

logger = logging.getLogger(__name__)


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
