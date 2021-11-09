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

from typing import AsyncGenerator, List, Optional, Tuple, Union

import aiokatcp
import async_timeout
import numpy as np
import pytest
import spead2.recv
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext

import katgpucbf.fgpu.recv
from katgpucbf import N_POLS
from katgpucbf.fgpu.engine import Engine
from katgpucbf.fgpu.main import make_engine, parse_args


@pytest.fixture
def mock_recv_streams(mocker) -> List[spead2.InprocQueue]:
    """Mock out :func:`katgpucbf.fgpu.recv.add_reader` to use in-process queues.

    Returns
    -------
    queues
        An in-process queue to use for sending to each polarisation.
    """
    queues = [spead2.InprocQueue() for _ in range(N_POLS)]
    queue_iter = iter(queues)  # Each call to add_reader gets the next queue

    def add_reader(
        stream: spead2.recv.ChunkRingStream,
        *,
        src: Union[str, List[Tuple[str, int]]],
        interface: Optional[str],
        ibv: bool,
        comp_vector: int,
        buffer: int,
    ) -> None:
        """Mock implementation of :func:`katgpucbf.fgpu.recv.add_reader`."""
        queue = next(queue_iter)
        stream.add_inproc_reader(queue)

    mocker.patch("katgpucbf.fgpu.recv.add_reader", autospec=True, side_effect=add_reader)
    return queues


@pytest.fixture
def mock_send_stream(mocker) -> List[spead2.InprocQueue]:
    """Mock out creation of the send stream.

    Each time a :class:`spead2.send.asyncio.UdpStream` is created, it instead
    creates an in-process stream and appends an equivalent number of inproc
    queues to the list returned by the fixture.
    """
    queues: List[spead2.InprocQueue] = []

    def constructor(thread_pool, endpoints, config, *args, **kwargs):
        stream_queues = [spead2.InprocQueue() for _ in endpoints]
        queues.extend(stream_queues)
        return spead2.send.asyncio.InprocStream(thread_pool, queues, config)

    mocker.patch("spead2.send.asyncio.UdpStream", autospec=True, side_effect=constructor)
    return queues


@pytest.fixture
def recv_max_chunks_one(monkeypatch) -> None:
    """Change :data:`.recv.MAX_CHUNKS` to 1 for the test.

    This simplifies the process of reliably injecting data.
    """
    monkeypatch.setattr(katgpucbf.fgpu.recv, "MAX_CHUNKS", 1)


def check_gdrcopy(context: AbstractContext) -> None:
    """Check whether gdrcopy works on `context`, and skip the test if not."""
    gdrcopy = pytest.importorskip("gdrcopy")
    pytest.importorskip("gdrcopy.pycuda")
    try:
        gdr = gdrcopy.Gdr()
        gdrcopy.pycuda.allocate(gdr, (32 * 1024 * 1024,), np.uint8)
    except Exception as exc:
        pytest.skip(f"gdrcopy not functional on this GPU: {exc}")


@pytest.fixture
async def engine_server(
    request, mock_recv_streams, mock_send_stream, recv_max_chunks_one, context: AbstractContext
) -> AsyncGenerator[Engine, None]:
    """Create a dummy :class:`.fgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.fgpu.main`, and are a set of simple parameters just to
    get the :class:`~.fgpu.Engine` running so that the KATCP interface can be
    tested.
    """
    arglist = list(request.cls.engine_arglist)
    if request.node.get_closest_marker("use_gdrcopy"):
        check_gdrcopy(context)
        arglist.append("--use-gdrcopy")
    args = parse_args(arglist)
    server, _monitor = make_engine(context, args)

    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def engine_client(engine_server: Engine) -> AsyncGenerator[aiokatcp.Client, None]:
    """Create a KATCP client for communicating with the dummy server."""
    assert engine_server.server is not None
    assert engine_server.server.sockets is not None
    host, port = engine_server.server.sockets[0].getsockname()[:2]
    with async_timeout.timeout(5):  # To fail the test quickly if unable to connect
        client = await aiokatcp.Client.connect(host, port)
    yield client
    client.close()
    await client.wait_closed()
