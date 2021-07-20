"""Fixtures for use in fgpu unit tests."""

import aiokatcp
import katsdpsigproc.accel as accel
import pytest

from katgpucbf.fgpu.main import make_engine


@pytest.fixture
async def gpu_context():
    """Generate a GPU context."""
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    return ctx


@pytest.fixture
async def engine_server(request, gpu_context):
    """Create a dummy :class:`.fgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.fgpu.main`, and are a set of simple parameters just to
    get the :class:`~.fgpu.Engine` running so that the KATCP interface can be
    tested.
    """
    server, _monitor = make_engine(gpu_context, arglist=request.cls.engine_arglist)

    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def engine_client(engine_server):
    """Create a KATCP client for communicating with the dummy server."""
    address = engine_server.server.sockets[0].getsockname()
    client = await aiokatcp.Client.connect(address[0], address[1])
    yield client
    client.close()
    await client.wait_closed()
