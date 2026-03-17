################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Fixtures for use in vgpu unit tests."""

from collections.abc import AsyncGenerator

import pytest
import spead2

from katgpucbf.vgpu.engine import VEngine
from katgpucbf.vgpu.main import make_engine, parse_args


@pytest.fixture
async def engine(
    request: pytest.FixtureRequest,
    mock_recv_streams: list[spead2.InprocQueue],
    engine_arglist: list[str],
) -> AsyncGenerator[VEngine, None]:
    """Create a dummy :class:`.VEngine` for unit testing.

    The command-line arguments are provided by a class-level `engine_arglist`
    fixture.
    """
    args = parse_args(engine_arglist)
    server = make_engine(args)
    await server.start()
    yield server
    await server.stop()
