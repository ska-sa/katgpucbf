################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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
from pytest_check import check

from katgpucbf import DIG_SAMPLE_BITS

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0117")
async def test_gain(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    r"""Test that the ``?beam-quant-gains`` command functions.

    Verification method
    -------------------
    Verification by means of test. Set gain factors of :math:`\frac{1}{2}`, 1
    and 2 on three different beams. Check that the output values are in these
    ratios.
    """
    receiver = receive_tied_array_channelised_voltage
    pcc = cbf.product_controller_client

    pdf_report.step("Configure the D-sim with Gaussian noise.")
    dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
    amplitude = 32 / dig_max
    await pcc.request("dsim-signals", cbf.dsim_names[0], f"common=nodither(wgn({amplitude}));common;common;")
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}.")

    gains = [0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    stream_names = receiver.stream_names[: len(gains)]  # Only need 3 beams for the test
    pdf_report.step("Set weights to 1/n_ants")
    weights = (1.0 / receiver.n_ants,) * receiver.n_ants
    for stream_name in stream_names:
        await pcc.request("beam-weights", stream_name, *weights)
        pdf_report.detail(f"Set weights for {stream_name} to {weights}.")

    pdf_report.step("Set gains to 1/2, 1 and 2.")
    for gain, stream_name in zip(gains, stream_names, strict=True):
        await pcc.request("beam-quant-gains", stream_name, gain)
        pdf_report.detail(f"Set gain for {stream_name} to {gain}.")

    pdf_report.step("Collect data.")
    timestamp, data = await receiver.next_complete_chunk()
    pdf_report.detail(f"Received chunk with timestamp {timestamp}.")

    pdf_report.step("Check power level of each beam.")
    data = data[: len(gains), pass_channels]
    for gain, stream_name, stream_data in zip(gains, stream_names, data, strict=True):
        power = np.sum(np.square(stream_data, dtype=np.float32)) / (stream_data.shape[0] * stream_data.shape[1])
        expected_power = np.square(amplitude * dig_max * gain)
        pdf_report.detail(f"{stream_name}: power measured as {power:.2f}, expected {expected_power:.2f}.")
        # The variation from expected power seems to be dominated by quantisation
        # effects rather than noise. I've picked 5% tolerance since that's plenty
        # of wiggle room while still being tight enough to catch major errors.
        assert power == pytest.approx(expected_power, rel=0.05)

    pdf_report.step("Compare beams to each other.")
    # Compare like pols because the dithering is independent between pols and may rarely cause
    # differences greater than expected.
    for i in range(2, len(gains)):
        scale = gains[i] / gains[i % 2]
        expected = data[i % 2] * scale
        # The actual data gets clamped, so we should clamp expected values too
        max_value = 2 ** (receiver.n_bits_per_sample - 1) - 1
        expected = np.clip(expected, -max_value, max_value)
        with check:
            # Differences are typically limited to 0.5 * scale, but when
            # n_ants is not a power of 2, the weight (1/n_ants) is inexact
            # and so the pre-quantisation value is not exactly an integer.
            # In that case dithering can have an effect.
            np.testing.assert_allclose(data[i], expected, atol=scale, rtol=0.0)
            pdf_report.detail(f"Beams {stream_names[0]} and {stream_names[i]} agree.")
