"""Collection of tests for the KATCP interface of katgpucbf.fgpu."""

import asyncio
import time

import aiokatcp
import katsdpsigproc.accel as accel
import pytest

from katgpucbf.fgpu.main import make_engine

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def event_loop():
    """Re-define the ``event_loop`` fixture, to give it a module scope."""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def gpu_context():
    """Generate a GPU context."""
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda, interactive=False)
    return ctx


@pytest.fixture(scope="module")
async def engine_server(gpu_context):
    """Create a dummy :class:`.fgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.fgpu.main`, and are a set of simple parameters just to
    get the :class:`~.fgpu.Engine` running so that the KATCP interface can be
    tested.
    """
    arglist = [
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        "--channels=4096",
        "239.10.10.0+7:7149",  # src1
        "239.10.10.8+7:7149",  # src2
        "239.10.11.0+15:7149",  # dst
    ]
    server = make_engine(gpu_context, arglist=arglist)

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


class TestKatcpRequests:
    """Unit tests for the Engine's KATCP requests."""

    async def test_quant_gain_set(self, engine_client, engine_server):
        """Test that the quant gain is correctly set."""
        _reply, _informs = await engine_client.request("quant-scale", 0.2)
        assert engine_server._processor.compute.quant_scale == 0.2

    @pytest.mark.xfail
    async def test_delay_model_update(self, engine_client, engine_server):
        """Test that the delay model is correctly updated.

        The new element is a piecewise-linear section, and it should go onto the
        end of the :class:`.MultiDelayModel` list.
        """
        start_time = int(time.time()) + 10
        _reply, _informs = await engine_client.request("delays", str(start_time), "3.76,0.12:7.322,1.91")
        assert engine_server._processor.delay_model._models[-1].start == start_time
        assert engine_server._processor.delay_model._models[-1].delay == 3.76
        assert engine_server._processor.delay_model._models[-1].delay_rate == 0.12
        assert engine_server._processor.delay_model._models[-1].phase == 7.322
        assert engine_server._processor.delay_model._models[-1].phase_rate == 1.91


class TestKatcpSensors:
    """Unit tests for the Engine's KATCP sensors.

    .. todo:: Write some tests!
    """

    pass
