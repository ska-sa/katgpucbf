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
from ..antenna_channelised_voltage import compute_tone_gain, sample_tone_response
from ..reporter import Reporter


async def test_linearity(
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

    pdf_report.step("Select a range of CW scales for testing.")
    cw_scales = [0.5**i for i in range(10)]
    pdf_report.detail(f"CW scales: {cw_scales}")

    pdf_report.step("Select a channel and compute the channel center frequency for the D-sim.")
    sel_chan_center = receive_baseline_correlation_products.n_chans // 3
    channel_frequency = sel_chan_center * (
        receive_baseline_correlation_products.bandwidth / receive_baseline_correlation_products.n_chans
    )
    pdf_report.detail(
        f"Channel {sel_chan_center} selected, with center frequency " + f"{channel_frequency/1e6:.2f} MHz."
    )

    pdf_report.step("Set EQ gain.")
    gain = compute_tone_gain(receiver=receive_baseline_correlation_products)

    pdf_report.detail(f"Setting gain to: {gain}")
    await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

    base_corr_prod = await sample_tone_response(
        rel_freqs=sel_chan_center,
        amplitude=cw_scales,
        receiver=receive_baseline_correlation_products,
    )

    linear_scale_result = base_corr_prod[:, sel_chan_center]
    linear_test_result = np.sqrt(linear_scale_result / np.max(linear_scale_result))

    rms_voltage = np.sqrt(np.max(linear_scale_result) / receive_baseline_correlation_products.n_spectra_per_acc)
    pdf_report.step("Compute RMS Voltage.")
    pdf_report.detail(f"RMS voltage: {rms_voltage:.3f}.")

    pdf_report.step("Compute Mean Square Error (MSE).")
    mse = np.sum(np.square(cw_scales - linear_test_result)) / len(cw_scales)
    pdf_report.detail(f"MSE is: {mse}")

    # Generate plot with reference
    labels = [f"$2^{{-{i}}}$" for i in range(len(cw_scales))]
    for xticks, ymin, title in [
        (np.arange(len(cw_scales)), -100, "CBF Linearity Test"),
    ]:
        fig = Figure()
        ax = fig.subplots()
        # pgfplots seems to struggle if data is too far outside ylim
        ax.plot(10 * np.log10(np.square(cw_scales)), label="Reference")
        ax.plot(10 * np.log10(np.square(linear_test_result)), label="Measured")
        ax.set_title(title)
        ax.set_xlabel("CW Scale")
        ax.set_ylabel("dB")
        ax.legend()
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(ymin, 0)

        for y in [-3, -53]:
            if ymin < y:
                ax.axhline(y, dashes=(1, 1), color="black")
                ax.annotate(f"{y} dB", (xticks[-1], y), horizontalalignment="right", verticalalignment="top")
        # tikzplotlib.clean_figure doesn't like data outside the ylim at all
        pdf_report.figure(
            fig, clean_figure=False, tikzplotlib_kwargs=dict(axis_width=r"0.8\textwidth", axis_height=r"0.5\textwidth")
        )
