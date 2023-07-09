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
from ipaddress import IPv4Address, IPv4Network

import aiokatcp
import async_timeout
import pytest
import spead2.recv
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext

import katgpucbf.fgpu.engine
import katgpucbf.fgpu.recv
from katgpucbf.fgpu.engine import Engine
from katgpucbf.fgpu.main import make_engine, parse_args


@pytest.fixture
def mock_send_stream_network() -> IPv4Network:
    """Network mask to filter the queues returned by :func:`mock_send_stream`.

    Test classes can override this to select only a subset.
    """
    return IPv4Network("0.0.0.0/0")


@pytest.fixture
def mock_send_stream(mocker, mock_send_stream_network: IPv4Network) -> list[spead2.InprocQueue]:
    """Mock out creation of the send stream.

    Each time a :class:`spead2.send.asyncio.UdpStream` is created, it instead
    creates an in-process stream and appends an equivalent number of inproc
    queues to the list returned by the fixture.

    The queues returned can be filtered by IP address by overriding the
    :func:`mock_send_stream_network` fixture.
    """
    queues: list[spead2.InprocQueue] = []

    def constructor(thread_pool, endpoints, config, *args, **kwargs):
        stream_queues = [spead2.InprocQueue() for _ in endpoints]
        queues.extend(
            queue
            for queue, endpoint in zip(stream_queues, endpoints)
            if IPv4Address(endpoint[0]) in mock_send_stream_network
        )
        return spead2.send.asyncio.InprocStream(thread_pool, stream_queues, config)

    mocker.patch("spead2.send.asyncio.UdpStream", autospec=True, side_effect=constructor)
    return queues


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
    mock_recv_stream,
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
