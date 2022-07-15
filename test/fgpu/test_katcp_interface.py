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

import re
from typing import Tuple

import aiokatcp
import numpy as np
import pytest
from numpy import safe_eval

from katgpucbf import N_POLS
from katgpucbf.fgpu.delay import wrap_angle
from katgpucbf.fgpu.engine import Engine

from .. import get_sensor

pytestmark = [pytest.mark.cuda_only]

CHANNELS = 4096
SYNC_EPOCH = 1632561921
GAIN = 0.125  # Exactly representable to avoid some rounding issues


def assert_valid_complex(value: str) -> None:
    """Check that `value` is a valid encoding of a complex number according to the ICD."""
    # This check is actually somewhat stricter e.g. it doesn't allow scientific
    # notation and requires the decimal point to the present.
    assert re.fullmatch(r"-?[0-9]+\.[0-9]+[+-][0-9]+\.[0-9]+j", value)
    complex(value)  # The regex should be sufficient, but this provides extra validation


def assert_valid_complex_list(value: str) -> None:
    """Check that `value` is a valid encoding of a list of complex numbers."""
    assert value[0] == "["
    assert value[-1] == "]"
    for term in value[1:-1].split(","):
        assert_valid_complex(term.strip(" "))


class TestKatcpRequests:
    """Unit tests for the Engine's KATCP requests."""

    engine_arglist = [
        "--katcp-host=127.0.0.1",
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        f"--channels={CHANNELS}",
        f"--sync-epoch={SYNC_EPOCH}",
        f"--gain={GAIN}",
        "--adc-sample-rate=1.712e9",
        "239.10.10.0+7:7149",  # src1
        "239.10.10.8+7:7149",  # src2
        "239.10.11.0+15:7149",  # dst
    ]

    @pytest.mark.parametrize("pol", range(N_POLS))
    async def test_initial_gain(self, engine_client: aiokatcp.Client, pol: int) -> None:
        """Test that the command-line gain is set correctly."""
        reply, _informs = await engine_client.request("gain", pol)
        assert reply == [b"0.125+0.0j"]
        sensor_value = await get_sensor(engine_client, f"input{pol}-eq")
        assert sensor_value == "[0.125+0.0j]"

    @pytest.mark.parametrize("pol", range(N_POLS))
    async def test_gain_set_scalar(self, engine_client: aiokatcp.Client, engine_server: Engine, pol: int) -> None:
        """Test that the eq gain is correctly set with a scalar value."""
        reply, _informs = await engine_client.request("gain", pol, "0.2-3j")
        assert len(reply) == 1
        value = aiokatcp.decode(str, reply[0])
        assert_valid_complex(value)
        assert complex(value) == pytest.approx(0.2 - 3j)

        sensor_value = await get_sensor(engine_client, f"input{pol}-eq")
        assert_valid_complex_list(sensor_value)
        assert safe_eval(sensor_value) == pytest.approx([0.2 - 3j])
        np.testing.assert_equal(engine_server._processor.gains[:, pol], np.full(CHANNELS, 0.2 - 3j, np.complex64))
        # Other pol must not have been affected
        np.testing.assert_equal(engine_server._processor.gains[:, 1 - pol], np.full(CHANNELS, GAIN, np.complex64))

    async def test_gain_set_vector(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that the eq gain is correctly set with a vector of values."""
        # This test doesn't parametrize over pols. It's assumed that anything
        # causing the wrong pol to be set would be picked up by the scalar
        # test.
        gains = np.arange(CHANNELS, dtype=np.float32) * (2 + 3j)
        reply, _informs = await engine_client.request("gain", 0, *(str(gain) for gain in gains))
        np.testing.assert_equal(engine_server._processor.gains[:, 0], gains)
        assert len(reply) == CHANNELS
        for value in reply:
            assert_valid_complex(aiokatcp.decode(str, value))
        reply_array = np.array([complex(aiokatcp.decode(str, value)) for value in reply])
        np.testing.assert_equal(reply_array, gains)

        sensor_value = await get_sensor(engine_client, "input0-eq")
        assert_valid_complex_list(sensor_value)
        np.testing.assert_equal(np.array(safe_eval(sensor_value)), gains)

    async def test_gain_not_complex(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if a value passed to ``?gain`` is not a finite complex number."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", 0, "i am not a complex number")
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", 0, "nan")
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", 0, "inf+infj")

    async def test_gain_bad_input(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if the input number passed to ``?gain`` is not a finite complex number."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", 2)

    async def test_gain_wrong_length(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain`` is used with the wrong number of arguments."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", 0, "1", "2")

    async def test_gain_all_set_scalar(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that ``?gain-all`` works correctly with a vector of values."""
        reply, _informs = await engine_client.request("gain-all", "0.2-3j")
        assert reply == []
        for pol in range(N_POLS):
            sensor_value = await get_sensor(engine_client, f"input{pol}-eq")
            assert_valid_complex_list(sensor_value)
            assert safe_eval(sensor_value) == pytest.approx([0.2 - 3j])
            np.testing.assert_equal(engine_server._processor.gains[:, pol], np.full(CHANNELS, 0.2 - 3j, np.complex64))

    async def test_gain_all_set_vector(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that ``?gain-all`` works correctly with a scalar value."""
        gains = np.arange(CHANNELS, dtype=np.float32) * (2 + 3j)
        reply, _informs = await engine_client.request("gain-all", *(str(gain) for gain in gains))
        assert reply == []
        for pol in range(N_POLS):
            np.testing.assert_equal(engine_server._processor.gains[:, pol], gains)
            sensor_value = await get_sensor(engine_client, f"input{pol}-eq")
            assert_valid_complex_list(sensor_value)
            np.testing.assert_equal(np.array(safe_eval(sensor_value)), gains)

    async def test_gain_all_set_default(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test ``?gain-all default``."""
        await engine_client.request("gain-all", "2+3j")
        await engine_client.request("gain-all", "default")
        for pol in range(N_POLS):
            sensor_value = await get_sensor(engine_client, f"input{pol}-eq")
            assert sensor_value == "[0.125+0.0j]"

    async def test_gain_all_empty(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain-all`` is used with no values."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain-all")

    @pytest.mark.parametrize("correct_delay_strings", [("3.76e-9,0.12e-9:7.322,1.91", "2.67e-9,0.02e-9:5.678,1.81")])
    async def test_delay_model_update_correct(self, engine_client, correct_delay_strings):
        """Test correctly-formed delay strings and validate the updates.

        The validation is done by comparing it against the corresponding delay sensor readings.
        """

        def parse_delay_string(delay_str: str) -> Tuple[float, float, float, float]:
            delay_str, phase_str = delay_str.split(":")
            delay, delay_rate = [float(value) for value in delay_str.split(",")]
            phase, phase_rate = [float(value) for value in phase_str.split(",")]
            return delay, delay_rate, wrap_angle(phase), phase_rate

        start_time = 0
        await engine_client.request("delays", start_time, correct_delay_strings[0], correct_delay_strings[1])

        for pol in range(N_POLS):
            sensor_reading = await get_sensor(engine_client, f"input{pol}-delay")
            sensor_values = sensor_reading[1:-1].split(",")[1:]  # Drop the timestamp
            sensor_values = (float(field.strip()) for field in sensor_values)

            for actual_value, expected_value in zip(sensor_values, parse_delay_string(correct_delay_strings[pol])):
                assert actual_value == pytest.approx(expected=expected_value)

    @pytest.mark.parametrize(
        "malformed_delay_string",
        [
            "3.76,0.12;7.322,1.91",  # Missing colon
            "3.76-0.12:7.322,1.91",  # Missing comma, delay half
            "3.76,0.12:7.322-1.91",  # Missing comma, phase half
            "3.76,0.12:apple,1.91",  # Non-float value for phase
            "3.76,pear:7.322,1.91",  # Non-float value for delay rate
            "-1.0,0.0:0.0,0.0",  # Negative delay
        ],
    )
    async def test_delay_model_update_malformed(self, engine_client, malformed_delay_string):
        """Test that a malformed delay model is rejected.

        We test for various combinations of malformations of the delay string.
        """
        start_time = SYNC_EPOCH + 10
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("delays", start_time, malformed_delay_string, malformed_delay_string)

    async def test_delay_model_update_missing_argument(self, engine_client):
        """Test that a delay request with a missing argument is rejected."""
        with pytest.raises(aiokatcp.FailReply):
            # Missing start time argument
            await engine_client.request("delays", "3.76,0.12:7.322,1.91")
        with pytest.raises(aiokatcp.FailReply):
            # Only one of the two models
            await engine_client.request("delays", "123456789.0", "3.76,0.12:7.322,1.91")

    async def test_delay_model_update_too_many_arguments(self, engine_client):
        """Test that a delay request with too many arguments is rejected."""
        coeffs = "3.76,0.12:7.322,1.91"
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("delays", "123456789.0", coeffs, coeffs, coeffs)
