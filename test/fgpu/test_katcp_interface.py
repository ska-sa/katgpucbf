"""Collection of tests for the KATCP interface of katgpucbf.fgpu."""

import asyncio
import time

import aiokatcp
import katsdpsigproc.accel as accel
import pytest
from katsdptelstate.endpoint import endpoint_list_parser

from katgpucbf.fgpu.engine import Engine
from katgpucbf.fgpu.main import DEFAULT_KATCP_HOST
from katgpucbf.fgpu.monitor import NullMonitor

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
    endpoint_func = endpoint_list_parser(default_port=7148)
    src_endpoints = [("239.0.0.0", 7148), ("239.0.0.1", 7148)]
    dst_endpoints = endpoint_func("239.1.0.0+15")
    LOCALHOST = "127.0.0.1"
    monitor = NullMonitor()
    server = Engine(
        katcp_host=DEFAULT_KATCP_HOST,
        katcp_port=0,  # This lets the OS assign an unused port, avoiding any conflicts.
        context=gpu_context,
        srcs=src_endpoints,
        src_interface=LOCALHOST,
        src_ibv=False,
        src_affinity=[-1, -1],
        src_comp_vector=[0],
        src_packet_samples=4096,
        src_buffer=32 * 1024 * 1024,
        dst=dst_endpoints,
        dst_interface=LOCALHOST,
        dst_ttl=4,
        dst_ibv=False,
        dst_packet_payload=1024,
        dst_affinity=-1,
        dst_comp_vector=0,
        adc_rate=0,
        feng_id=0,
        spectra=2 ** 26 // (2 * 4096),
        acc_len=256,
        channels=4096,
        taps=4,
        quant_scale=0.001,
        mask_timestamp=True,
        use_gdrcopy=False,
        use_peerdirect=False,
        monitor=monitor,
    )
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
