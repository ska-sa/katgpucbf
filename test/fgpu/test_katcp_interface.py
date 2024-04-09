################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

import aiokatcp
import numpy as np
import pytest
from numpy import safe_eval

from katgpucbf import N_POLS
from katgpucbf.fgpu.delay import wrap_angle
from katgpucbf.fgpu.engine import Engine

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

    @pytest.fixture
    def engine_arglist(self) -> list[str]:
        return [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            "--src-interface=lo",
            "--dst-interface=lo",
            f"--sync-epoch={SYNC_EPOCH}",
            f"--gain={GAIN}",
            "--adc-sample-rate=1.712e9",
            f"--wideband=name=wideband,dst=239.10.11.0+15:7149,channels={CHANNELS}",
            "239.10.10.0+15:7149",  # src
        ]

    @pytest.mark.parametrize("pol", range(N_POLS))
    async def test_initial_gain(self, engine_client: aiokatcp.Client, pol: int) -> None:
        """Test that the command-line gain is set correctly."""
        reply, _informs = await engine_client.request("gain", "wideband", pol)
        assert reply == [b"0.125+0.0j"]
        sensor_value = await engine_client.sensor_value(f"wideband.input{pol}.eq", str)
        assert sensor_value == "[0.125+0.0j]"

    @pytest.mark.parametrize("pol", range(N_POLS))
    async def test_gain_set_scalar(self, engine_client: aiokatcp.Client, engine_server: Engine, pol: int) -> None:
        """Test that the eq gain is correctly set with a scalar value."""
        # TODO[nb]: need to update for multiple pipelines
        reply, _informs = await engine_client.request("gain", "wideband", pol, "0.2-3j")
        assert reply == []

        # Read back the value
        reply, _informs = await engine_client.request("gain", "wideband", pol)
        assert len(reply) == 1
        value = aiokatcp.decode(str, reply[0])
        assert_valid_complex(value)
        assert complex(value) == pytest.approx(0.2 - 3j)

        sensor_value = await engine_client.sensor_value(f"wideband.input{pol}.eq", str)
        assert_valid_complex_list(sensor_value)
        assert safe_eval(sensor_value) == pytest.approx([0.2 - 3j])
        np.testing.assert_equal(engine_server._pipelines[0].gains[:, pol], np.full(CHANNELS, 0.2 - 3j, np.complex64))
        # Other pol must not have been affected
        np.testing.assert_equal(engine_server._pipelines[0].gains[:, 1 - pol], np.full(CHANNELS, GAIN, np.complex64))

    async def test_gain_set_vector(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that the eq gain is correctly set with a vector of values."""
        # This test doesn't parametrize over pols. It's assumed that anything
        # causing the wrong pol to be set would be picked up by the scalar
        # test.
        # TODO[nb]: need to update for multiple pipelines
        gains = np.arange(CHANNELS, dtype=np.float32) * (2 + 3j)
        reply, _informs = await engine_client.request("gain", "wideband", 0, *(str(gain) for gain in gains))
        np.testing.assert_equal(engine_server._pipelines[0].gains[:, 0], gains)
        assert reply == []

        # Read back the values
        reply, _informs = await engine_client.request("gain", "wideband", 0)
        assert len(reply) == CHANNELS
        for value in reply:
            assert_valid_complex(aiokatcp.decode(str, value))
        reply_array = np.array([complex(aiokatcp.decode(str, value)) for value in reply])
        np.testing.assert_equal(reply_array, gains)

        sensor_value = await engine_client.sensor_value("wideband.input0.eq", str)
        assert_valid_complex_list(sensor_value)
        np.testing.assert_equal(np.array(safe_eval(sensor_value)), gains)

    async def test_gain_not_complex(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if a value passed to ``?gain`` is not a finite complex number."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", "wideband", 0, "i am not a complex number")
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", "wideband", 0, "nan")
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", "wideband", 0, "inf+infj")

    async def test_gain_bad_input(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if the input number passed to ``?gain`` is not a finite complex number."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", "wideband", 2)

    async def test_gain_bad_output(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain`` is used with a bad stream name."""
        with pytest.raises(aiokatcp.FailReply, match="badstream"):
            await engine_client.request("gain", "badstream", 0)

    async def test_gain_wrong_length(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain`` is used with the wrong number of arguments."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain", "wideband", 0, "1", "2")

    async def test_gain_all_set_scalar(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that ``?gain-all`` works correctly with a vector of values."""
        # TODO[nb]: need to update for multiple pipelines
        reply, _informs = await engine_client.request("gain-all", "wideband", "0.2-3j")
        assert reply == []
        for pol in range(N_POLS):
            sensor_value = await engine_client.sensor_value(f"wideband.input{pol}.eq", str)
            assert_valid_complex_list(sensor_value)
            assert safe_eval(sensor_value) == pytest.approx([0.2 - 3j])
            np.testing.assert_equal(
                engine_server._pipelines[0].gains[:, pol], np.full(CHANNELS, 0.2 - 3j, np.complex64)
            )

    async def test_gain_all_set_vector(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test that ``?gain-all`` works correctly with a scalar value."""
        # TODO[nb]: need to update for multiple pipelines
        gains = np.arange(CHANNELS, dtype=np.float32) * (2 + 3j)
        reply, _informs = await engine_client.request("gain-all", "wideband", *(str(gain) for gain in gains))
        assert reply == []
        for pol in range(N_POLS):
            np.testing.assert_equal(engine_server._pipelines[0].gains[:, pol], gains)
            sensor_value = await engine_client.sensor_value(f"wideband.input{pol}.eq", str)
            assert_valid_complex_list(sensor_value)
            np.testing.assert_equal(np.array(safe_eval(sensor_value)), gains)

    async def test_gain_all_set_default(self, engine_client: aiokatcp.Client, engine_server: Engine) -> None:
        """Test ``?gain-all default``."""
        await engine_client.request("gain-all", "wideband", "2+3j")
        await engine_client.request("gain-all", "wideband", "default")
        for pol in range(N_POLS):
            sensor_value = await engine_client.sensor_value(f"wideband.input{pol}.eq", str)
            assert sensor_value == "[0.125+0.0j]"

    async def test_gain_all_empty(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain-all`` is used with no values."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("gain-all", "wideband")

    async def test_gain_all_bad_output(self, engine_client: aiokatcp.Client) -> None:
        """Test that an error is raised if ``?gain-all`` is used with an unknown output."""
        with pytest.raises(aiokatcp.FailReply, match="badstream"):
            await engine_client.request("gain-all", "badstream", "1")

    @pytest.mark.parametrize("correct_delay_strings", [("3.76e-9,0.12e-9:7.322,1.91", "2.67e-9,0.02e-9:5.678,1.81")])
    async def test_delay_model_update_correct(
        self, engine_server: Engine, engine_client: aiokatcp.Client, correct_delay_strings: tuple[str, str]
    ) -> None:
        """Test correctly-formed delay strings and validate the updates.

        The validation is done by comparing it against the corresponding delay sensor readings.
        """

        def parse_delay_string(delay_str: str) -> tuple[float, float, float, float]:
            delay_str, phase_str = delay_str.split(":")
            delay, delay_rate = (float(value) for value in delay_str.split(","))
            phase, phase_rate = (float(value) for value in phase_str.split(","))
            return delay, delay_rate, wrap_angle(phase), phase_rate

        start_time = SYNC_EPOCH
        await engine_client.request(
            "delays", "wideband", start_time, correct_delay_strings[0], correct_delay_strings[1]
        )
        # The delay model won't become current until some data is received, but
        # we're not simulating any. Poke the delay model manually to make it
        # update the sensor.
        # TODO[nb]: need to update for multiple pipelines
        for model in engine_server._pipelines[0].delay_models:
            model(1)

        for pol in range(N_POLS):
            sensor_reading = await engine_client.sensor_value(f"wideband.input{pol}.delay", str)
            sensor_values_str = sensor_reading[1:-1].split(",")[1:]  # Drop the timestamp
            sensor_values = (float(field.strip()) for field in sensor_values_str)

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
        ],
    )
    async def test_delay_model_update_malformed(
        self, engine_client: aiokatcp.Client, malformed_delay_string: str
    ) -> None:
        """Test that a malformed delay model is rejected.

        We test for various combinations of malformations of the delay string.
        """
        start_time = SYNC_EPOCH + 10
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request(
                "delays", "wideband", start_time, malformed_delay_string, malformed_delay_string
            )

    async def test_delay_model_update_missing_argument(self, engine_client: aiokatcp.Client) -> None:
        """Test that a delay request with a missing argument is rejected."""
        with pytest.raises(aiokatcp.FailReply):
            # Missing start time argument
            await engine_client.request("delays", "wideband", "3.76,0.12:7.322,1.91")
        with pytest.raises(aiokatcp.FailReply):
            # Only one of the two models
            await engine_client.request("delays", "wideband", SYNC_EPOCH, "3.76,0.12:7.322,1.91")

    async def test_delay_model_update_too_many_arguments(self, engine_client: aiokatcp.Client) -> None:
        """Test that a delay request with too many arguments is rejected."""
        coeffs = "3.76,0.12:7.322,1.91"
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("delays", "wideband", SYNC_EPOCH, coeffs, coeffs, coeffs)

    async def test_delay_model_update_before_sync_epoch(self, engine_client: aiokatcp.Client) -> None:
        """Test that a delay model loaded prior to the sync epoch is rejected."""
        with pytest.raises(aiokatcp.FailReply):
            await engine_client.request("delays", "wideband", SYNC_EPOCH - 1.0, "0,0:0,0", "0,0:0,0")
