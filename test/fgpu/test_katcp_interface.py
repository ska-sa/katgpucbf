"""Collection of tests for the KATCP interface of katgpucbf.fgpu."""
import time

import aiokatcp
import pytest

pytestmark = pytest.mark.asyncio


class TestKatcpRequests:
    """Unit tests for the Engine's KATCP requests."""

    engine_arglist = [
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        "--channels=4096",
        "--sync-epoch=1632561921",
        "239.10.10.0+7:7149",  # src1
        "239.10.10.8+7:7149",  # src2
        "239.10.11.0+15:7149",  # dst
    ]

    async def test_quant_gain_set(self, engine_client, engine_server):
        """Test that the quant gain is correctly set."""
        _reply, _informs = await engine_client.request("quant-gain", 0.2)
        assert engine_server._processor.compute.quant_gain == 0.2

    async def test_delay_model_update(self, engine_client, engine_server):
        """Test that the delay model is correctly updated.

        The new element is a piecewise-linear section, and it should go onto the
        end of the :class:`.MultiDelayModel` list.

        .. todo::

          Don't just compare the start_time value, compute the actual timestamp
          number in sync_epoch terms.
        """
        start_time = int(time.time()) + 10
        _reply, _informs = await engine_client.request("delays", start_time, "3.76,0.12:7.322,1.91")
        assert engine_server._processor.delay_model._models[-1].start == start_time
        assert engine_server._processor.delay_model._models[-1].delay == 3.76
        assert engine_server._processor.delay_model._models[-1].delay_rate == 0.12
        assert engine_server._processor.delay_model._models[-1].phase == 7.322
        assert engine_server._processor.delay_model._models[-1].phase_rate == 1.91

    async def test_delay_model_update_malformed(self, engine_client, engine_server):
        """Test that a malformed delay model is rejected."""
        start_time = int(time.time()) + 10
        with pytest.raises(aiokatcp.FailReply):
            # Bad delay-string
            _reply, _informs = await engine_client.request("delays", start_time, "3.76-0.12<>7.322-1.91")
        with pytest.raises(aiokatcp.FailReply):
            # Missing start time argument
            _reply, _informs = await engine_client.request("delays", "3.76,0.12:7.322,1.91")


class TestKatcpSensors:
    """Unit tests for the Engine's KATCP sensors.

    .. todo:: Write some tests!
    """

    pass
