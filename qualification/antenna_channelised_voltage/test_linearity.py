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

"""CBF linearity test."""
import numpy as np
from matplotlib.figure import Figure

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter
from . import compute_tone_gain, sample_tone_response


async def test_linearity(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that baseline Correlation Products are linear when input CW is scaled.

    Requirements verified:
        CBF.V.A.IF: CBF Linearity

    Verification method:

    Verify linearity by checking the channelised output scales linearly with a
    linear change to the CW input amplitude.
    """
    receiver = receive_baseline_correlation_products
    pdf_report.step("Capture channelised data for various input CW scales and check linearity.")

    pdf_report.step("Select a range of CW scales for testing.")
    cw_scales = [0.5**i for i in range(10)]
    pdf_report.detail(f"CW scales: {cw_scales}")

    pdf_report.step("Select a channel and compute the channel center frequency for the D-sim.")
    sel_chan_center = receiver.n_chans // 3
    channel_frequency = sel_chan_center * (receiver.bandwidth / receiver.n_chans)
    pdf_report.detail(
        f"Channel {sel_chan_center} selected, with center frequency " + f"{channel_frequency/1e6:.2f} MHz."
    )

    pdf_report.step("Set EQ gain.")
    gain = compute_tone_gain(receiver=receiver, amplitude=max(cw_scales), target_voltage=110)

    pdf_report.detail(f"Setting gain to: {gain}")
    await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

    base_corr_prod = await sample_tone_response(
        rel_freqs=sel_chan_center,
        amplitude=cw_scales,
        receiver=receiver,
    )

    linear_scale_result = base_corr_prod[:, sel_chan_center]

    # Normalise and compute the effective received voltage value (from power) for comparison to the requested value.
    linear_test_result = np.sqrt(linear_scale_result / np.max(linear_scale_result))

    pdf_report.step("Compute RMS Voltage.")
    rms_voltage = np.sqrt(np.max(linear_scale_result) / receiver.n_spectra_per_acc)
    pdf_report.detail(f"RMS voltage: {rms_voltage:.3f}.")

    pdf_report.step("Compute Mean Square Error (MSE).")
    mse = np.square(cw_scales - linear_test_result).mean()
    pdf_report.detail(f"MSE is: {mse}")

    # Check for any zeros at the end of the sequence prior to plotting. Remove if so.
    zero_locations = np.where(linear_test_result == 0)[0]
    if len(zero_locations):
        linear_test_result = linear_test_result[: zero_locations[0]]

    # Generate plot with reference
    labels = [f"$2^{{-{i}}}$" for i in range(len(cw_scales))]
    title = "Power relative to input CW level"
    xticks = np.arange(len(cw_scales))
    fig = Figure()
    ax = fig.subplots()
    ax.plot(20 * np.log10(cw_scales), label="Reference")
    ax.plot(20 * np.log10(linear_test_result), label="Measured")
    ax.set_title(title)
    ax.set_xlabel("CW Scale")
    ax.set_ylabel("dB")
    ax.legend()
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    pdf_report.figure(fig)
