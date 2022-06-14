################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""CBF lineraity test."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl, antenna_channelised_voltage
from ..reporter import Reporter


async def test_cbf_linearity(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that baseline Correlation Products are linear when input CW is scaled.

    Requirements verified:
        CBF.V.A.IF: CBF Linearity

    Verification method:

    Verify lineraity by checking the channelised output scales linearly with a
    linear change to the CW input amplitude.
    """
    pdf_report.step("Capture channelised data for various input CW scales and check linearity.")

    pdf_report.step("CW scales selection.")
    cw_scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]
    pdf_report.detail(f"CW scales: {cw_scales}")

    pdf_report.step("Channel selection. Compute desired channel frequency for Dsim.")
    rng = np.random.default_rng(seed=2021)
    random_channel_center = int(rng.uniform(0, correlator.n_chans))
    pdf_report.detail(f"Random channel selected: {random_channel_center}")
    pdf_report.detail(
        f"Channel frequency for selected channel: {round((random_channel_center*(856e6/correlator.n_chans))/1e6,2)}MHz"
    )

    pdf_report.step("Set EQ gain.")
    gain = 0.003
    pdf_report.detail(f"Setting gain to: {gain}")
    await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

    base_corr_prod = []
    for scale in cw_scales:
        base_corr_prod.append(
            await antenna_channelised_voltage.sample_tone_response(
                rel_freqs=random_channel_center,
                amplitude=scale,
                correlator=correlator,
                receiver=receive_baseline_correlation_products,
            )
        )

    linear_scale_result = []
    for product in base_corr_prod:
        linear_scale_result.append(product[random_channel_center])

    linear_test_result = np.sqrt(linear_scale_result / np.max(linear_scale_result))

    pdf_report.step("Compute Mean Square Error (MSE).")
    mse = np.sum(np.square(cw_scales - linear_test_result)) / len(cw_scales)
    pdf_report.detail(f"MSE is: {mse}")
    assert mse == pytest.approx(mse, rel=0.01)

    # Generate plot with reference
    # TODO: This plot should be automagically added into the report. Not yet sure how to do that.
    plt.figure()
    plt.plot(10 * np.log10(np.square(cw_scales)), label="Ref")
    plt.plot(10 * np.log10(np.square(linear_test_result)), label="CBF Lin Test")
    plt.legend()
    plt.xlabel("CW Scale")
    plt.ylabel("dB")
    plt.title("CBF Linearity Test")
    plt.text(0, -50, f"Channel Under Test: {random_channel_center}", color="green", style="italic")
    plt.text(
        0,
        -53,
        f"Channel Frequency: {round((random_channel_center*(856e6/correlator.n_chans))/1e6,2)}MHz",
        color="green",
        style="italic",
    )
    labels = np.round(cw_scales, 5)
    labels = [
        "$2^{0}$",
        "$2^{-1}$",
        "$2^{-2}$",
        "$2^{-3}$",
        "$2^{-4}$",
        "$2^{-5}$",
        "$2^{-6}$",
        "$2^{-7}$",
        "$2^{-8}$",
        "$2^{-9}$",
    ]
    plt.xticks(np.arange(0, len(linear_test_result), step=1), labels=labels)
    plt.savefig("linearity.png")
