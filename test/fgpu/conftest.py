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

"""Fixtures for use in fgpu unit tests."""

import aiokatcp
import pytest

from katgpucbf.fgpu.main import make_engine


@pytest.fixture
async def engine_server(request, context):
    """Create a dummy :class:`.fgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.fgpu.main`, and are a set of simple parameters just to
    get the :class:`~.fgpu.Engine` running so that the KATCP interface can be
    tested.
    """
    server, _monitor = make_engine(context, arglist=request.cls.engine_arglist)

    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def engine_client(engine_server):
    """Create a KATCP client for communicating with the dummy server."""
    host, port = engine_server.server.sockets[0].getsockname()[:2]
    client = await aiokatcp.Client.connect(host, port)
    yield client
    client.close()
    await client.wait_closed()
