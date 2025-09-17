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
from collections.abc import MutableMapping, Sequence

import aiokatcp
from katsdpservices.aiomonitor import add_aiomonitor_arguments

from .. import (
    DEFAULT_KATCP_HOST,
    DEFAULT_KATCP_PORT,
    __version__,
)
from ..main import engine_main
from .engine import Engine


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="vgpu")
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
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args(arglist)
    return args


def make_engine(args: argparse.Namespace) -> Engine:
    """Create the :class:`~katgpucbf.vgpu.Engine`."""
    return Engine(args.katcp_host, args.katcp_port)


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
