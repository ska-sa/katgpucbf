################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Gain test."""

import numpy as np
import pytest

from katgpucbf import DIG_SAMPLE_BITS

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0117")
async def test_gain(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test that the ``?beam-quant-gains`` command functions.

    Verification method
    -------------------
    Verification by means of test. Set gain factors of :math:`\frac{1}{4}`, 1
    and 4 on three different beams. Check that the outputs values are in these
    ratios.
    """
    receiver = receive_tied_array_channelised_voltage
    # TODO: the test can't be implemented until katsdpcontroller supports
    # setting the gain (NGC-446). For now just check that the beams have
    # approximately the expected power. That also tends to fail, because
    # without control of the gains, we have to set the input level very low
    # to avoid saturation, and that leads to quantisation effects.
    pdf_report.step("Configure the D-sim with Gaussian noise.")
    dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
    amplitude = 32 / dig_max / receiver.n_ants
    await cbf.dsim_clients[0].request("signals", f"common=wgn({amplitude});common;common;")
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}")

    _, data = await receiver.next_complete_chunk()
    power = np.sum(np.square(data, dtype=np.float32), axis=(1, 2, 3)) / (data.shape[1] * data.shape[2])
    pdf_report.detail(f"Power measured as {list(power)}")
    expected_power = np.square(amplitude * dig_max * receiver.n_ants)
    pdf_report.detail(f"Expected power is {expected_power}")
    # The variation from expected power seems to be dominated by quantisation
    # effects rather than noise. I've picked 5% tolerance since that's plenty
    # of wiggle room while still being tight enough to catch major errors.
    np.testing.assert_equal(power, pytest.approx(expected_power, rel=0.05))
