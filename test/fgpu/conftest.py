################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

from collections.abc import AsyncGenerator

import aiokatcp
import async_timeout
import pytest
from katsdpsigproc.abc import AbstractContext

import katgpucbf.fgpu.engine
import katgpucbf.fgpu.recv
from katgpucbf import N_POLS
from katgpucbf.fgpu.engine import Engine
from katgpucbf.fgpu.main import make_engine, parse_args


@pytest.fixture
def n_src_streams() -> int:  # noqa: D401
    """Number of source streams for an fgpu instance."""
    return N_POLS


@pytest.fixture
def recv_max_chunks_one(monkeypatch) -> None:
    """Change :data:`.recv.MAX_CHUNKS` to 1 for the test.

    This simplifies the process of reliably injecting data.
    """
    monkeypatch.setattr(katgpucbf.fgpu.recv, "MAX_CHUNKS", 1)


def check_vkgdr(context: AbstractContext) -> None:
    """Check whether vkgdr works on `context`, and skip the test if not."""
    vkgdr = pytest.importorskip("vkgdr")
    pytest.importorskip("vkgdr.pycuda")
    try:
        with context:
            handle = vkgdr.Vkgdr.open_current_context()
            vkgdr.pycuda.Memory(handle, 16 * 1024 * 1024)
    except Exception as exc:
        pytest.skip(f"vkgdr not functional on this GPU: {exc}")


@pytest.fixture
async def engine_server(
    request,
    engine_arglist: list[str],
    mock_recv_streams,
    mock_send_stream,
    recv_max_chunks_one,
    context: AbstractContext,
) -> AsyncGenerator[Engine, None]:
    """Create a dummy :class:`.fgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.fgpu.main`, and are a set of simple parameters just to
    get the :class:`~.fgpu.Engine` running so that the KATCP interface can be
    tested.

    Extra command-line arguments can be added using a ``cmdline_args`` marker.
    """
    arglist = list(engine_arglist)  # Copy, to ensure we don't alter original
    if request.node.get_closest_marker("use_vkgdr"):
        check_vkgdr(context)
        arglist.append("--use-vkgdr")
    # iter_markers works closest-to-furthest, but we want the opposite so
    # that more specific markers append options to the end, overriding those
    # added by less-specific markers.
    for marker in reversed(list(request.node.iter_markers("cmdline_args"))):
        arglist.extend(marker.args)

    args = parse_args(arglist)
    server, _monitor = make_engine(context, args)

    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def engine_client(engine_server: Engine) -> AsyncGenerator[aiokatcp.Client, None]:
    """Create a KATCP client for communicating with the dummy server."""
    host, port = engine_server.sockets[0].getsockname()[:2]
    async with async_timeout.timeout(5):  # To fail the test quickly if unable to connect
        client = await aiokatcp.Client.connect(host, port)
    yield client
    client.close()
    await client.wait_closed()
