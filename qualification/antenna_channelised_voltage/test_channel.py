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

"""CBF Channel test."""

from typing import List

import numpy as np
from matplotlib.figure import Figure

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter
from . import compute_tone_gain, sample_tone_response


def measure_sfdr(data: np.ndarray, base_channel: np.ndarray) -> List:
    """Measure Spurious Free Dynamic Range (SFDR) of the response at a given power level.

    Estimate the SFDR by measuring the power (dB) of the next strongest
    tone in the spectrum (ignoring the fundamental tone). The SFDR is the
    difference in these two values.

    Returns
    -------
    A list of tuples with channel, difference, height of next peak, and the index.

    """
    sfdr_measurements = []

    for idx, channel in enumerate(base_channel):
        peak = np.max(data[idx, channel])
        below_peak_idxs = np.nonzero(data[idx, :] < peak)
        next_peak = np.max(data[idx, below_peak_idxs])
        next_peak_idx = np.where(data[idx, :] == next_peak)[0][0]
        peak_diff = peak - next_peak
        sfdr_measurements.append([channel, peak_diff, next_peak_idx, next_peak])

    return sfdr_measurements


async def test_channel(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    r"""Test channel position and measure SFDR per channel under test.

    Requirements verified:

    CBF-REQ-0126
        The CBF shall perform channelisation such that the 53 dB attenuation bandwidth
        is :math:`\le 2\times` (twice) the pass band width.
    """
    receiver = receive_baseline_correlation_products

    channel_skip = 32

    # Arbitrary channels, not too near the edges, skipping every 'channel_skip' channels
    selected_channels = np.arange(8, receiver.n_chans, channel_skip)

    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)
    # Determine the ideal F-engine output level at the peak. Maximum target_voltage is 127, but some headroom is good.
    gain = compute_tone_gain(receiver=receiver, amplitude=amplitude, target_voltage=110)

    pdf_report.step("Set gain and measure channel position.")
    gain_step = 100.0
    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.

    # Pre-allocate to hold prior hdr_data for each next iteration for each spectrum
    hdr_data = np.zeros((len(selected_channels), receiver.n_chans))

    for i in range(3):

        pdf_report.detail(f"Set gain to {gain}.")
        await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

        pdf_report.detail(f"Collect power measurements for {len(selected_channels)} channels.")
        data = await sample_tone_response(selected_channels, amplitude, receiver)
        data = data.astype(np.float64)

        # Iterate through all selected channels (per gain setting) and check if the position is correct.
        # Store gain adjusted data for SFDR measurement.
        for idx, spectrum in enumerate(data):
            # Use current gain to capture spikes, then adjust gain
            if i == 0:
                peak_data = np.max(spectrum)
                peak_channel = np.where(spectrum == peak_data)[0][0]
                hdr_data[idx] = spectrum

                # Extract which channel we are interested in checking
                expected_channel = selected_channels[idx]

                # Check if tone captured is in the correct channel
                pdf_report.detail(f"Expected channel: {expected_channel}. Actual channel: {peak_channel}")
                assert peak_channel == expected_channel
            else:
                power_scale = gain_step ** (i * 2)
                hdr_data[idx] = np.where(
                    hdr_data[idx] >= peak_data / power_scale, hdr_data[idx], spectrum / power_scale
                )
        gain *= gain_step

    # The maximum is to avoid errors when data is 0
    db_sfdr = 10 * np.log10(np.maximum(hdr_data, 1e-100) / peak_data)
    db_sfdr = np.round(
        db_sfdr, 3
    )  # rounded to 3 places to present serialisation of long numbers causing issues for LaTeX

    # Measure SFDR per captured spectrum
    pdf_report.step("Check SFDR attenuation.")
    sfdr_measurements = measure_sfdr(db_sfdr, selected_channels)

    for entry in sfdr_measurements:
        pdf_report.detail(
            f"SFDR: {entry[1]:.3f}dB for base channel {entry[0]}. Next peak channel {entry[2]} value {entry[3]:.3f}dB."
        )
        expect(entry[1] >= 53)

    # Select an arbitrary channel, not too near the edges for plotting
    selected_plot_idx = len(selected_channels) // 2
    plot_channel = selected_channels[selected_plot_idx]
    pdf_report.step(f"SFDR plot for base channel {plot_channel}.")

    xticks = np.arange(0, (8192 + 1024), 1024)
    ymin = -100
    title = f"SFDR for channel {plot_channel}"
    x = np.linspace(0, 8191, len(db_sfdr[selected_plot_idx, :]))
    db_plot = db_sfdr[selected_plot_idx, :]

    fig = Figure()
    ax = fig.subplots()
    # pgfplots seems to struggle if data is too far outside ylim
    ax.plot(x, np.maximum(db_plot, ymin - 10))
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("dB")
    ax.set_xticks(xticks)
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
