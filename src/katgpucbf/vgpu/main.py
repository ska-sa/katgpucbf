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

from ..main import add_common_arguments, engine_main
from .engine import VEngine


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="vgpu")
    add_common_arguments(parser)

    args = parser.parse_args(arglist)
    return args


def make_engine(args: argparse.Namespace) -> VEngine:
    """Create the :class:`.VEngine`."""
    return VEngine(args.katcp_host, args.katcp_port)


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
