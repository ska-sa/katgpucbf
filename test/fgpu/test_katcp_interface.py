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

"""Collection of tests for the KATCP interface of katgpucbf.fgpu."""
import aiokatcp
import numpy as np
import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.cuda_only]

SYNC_EPOCH = 1632561921


class TestKatcpRequests:
    """Unit tests for the Engine's KATCP requests."""

    engine_arglist = [
        "--katcp-host=127.0.0.1",
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        "--channels=4096",
        f"--sync-epoch={SYNC_EPOCH}",
        "--adc-sample-rate=1.712e9",
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
        """
        assert engine_server.adc_sample_rate == 1.712e9

        start_time = SYNC_EPOCH + 10
        _reply, _informs = await engine_client.request("delays", start_time, "3.76,0.12:7.322,1.91")

        # We expect the start time to be (10 seconds * 1.712e9 samples / second) i.e. 1.712e10.
        assert engine_server._processor.delay_model._models[-1].start == int(1.712e10)
        assert engine_server._processor.delay_model._models[-1].delay == 3.76 * 1.712e9
        assert engine_server._processor.delay_model._models[-1].delay_rate == 0.12
        assert engine_server._processor.delay_model._models[-1].phase == pytest.approx(7.322 - 2 * np.pi)
        assert engine_server._processor.delay_model._models[-1].phase_rate == 1.91

    @pytest.mark.parametrize(
        "malformed_delay_string",
        [
            "3.76,0.12;7.322,1.91",  # Missing colon
            "3.76-0.12:7.322,1.91",  # Missing comma, delay half
            "3.76,0.12:7.322-1.91",  # Missing comma, phase half
            "3.76,0.12:apple,1.91",  # Non-float value for phase
            "3.76,pear:7.322,1.91",  # Non-float value for delay rate
        ],
    )
    async def test_delay_model_update_malformed(self, engine_client, malformed_delay_string):
        """Test that a malformed delay model is rejected.

        We test for various combinations of malformations of the delay string.
        """
        start_time = SYNC_EPOCH + 10
        with pytest.raises(aiokatcp.FailReply):
            _reply, _informs = await engine_client.request("delays", start_time, malformed_delay_string)

    async def test_delay_model_update_missing_argument(self, engine_client):
        """Test that a delay request with a missing argument is rejected."""
        with pytest.raises(aiokatcp.FailReply):
            # Missing start time argument
            _reply, _informs = await engine_client.request("delays", "3.76,0.12:7.322,1.91")


class TestKatcpSensors:
    """Unit tests for the Engine's KATCP sensors.

    .. todo:: Write some tests!
    """

    pass
