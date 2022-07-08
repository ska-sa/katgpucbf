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


def measure_sfdr(db: np.ndarray, base_channel: np.ndarray) -> List[float]:
    """Measure Spurious Free Dynamic Range (SFDR) of the response at a given power level.

    Estimate the SFDR by measuring the power (dB) of the next strongest
    tone in the spectrum (ignoring the fundamental tone). The SFDR is the
    difference in these two values.

    Returns
    -------
    A list of differences computed from peak_value - next_peak_value.
    """
    sfdr_measurements = []

    for idx, channel in enumerate(base_channel):
        peak_value = np.max(db[idx, channel])
        below_peak_idxs = np.nonzero(db[idx, :] < peak_value)
        next_peak_value = np.max(db[idx, below_peak_idxs])
        peak_diff = peak_value - next_peak_value
        sfdr_measurements.append(peak_diff)

    return sfdr_measurements


async def test_channelisation_and_sfdr(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    r"""Test channel position and measure SFDR per channel under test.

    Requirements verified:

    CBF-REQ-0046
        The channel spacing shall not be less than half of that which is specified for that configuration.

    CBF-REQ-TDB
        The CBF, when configured for Wideband intermediate resolution channelisation, shall channelise
        the L-band pass band into equispaced frequency channels with a channel spacing of delta frequency <= 105kHz.
    """
    receiver = receive_baseline_correlation_products

    required_sfdr_db = 53.0
    channel_range_start = 8
    channel_skip = 32

    # Arbitrary channels, not too near the edges, skipping every 'channel_skip' channels
    selected_channels = np.arange(channel_range_start, receiver.n_chans, channel_skip)

    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)
    # Determine the ideal F-engine output level at the peak. Maximum target_voltage is 127, but some headroom is good.
    gain = compute_tone_gain(receiver=receiver, amplitude=amplitude, target_voltage=110)

    pdf_report.step("Set gain and measure channel position.")
    gain_step = 100.0
    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.

    # Pre-allocate to hold prior hdr_data for next iteration for each spectrum
    hdr_data = np.zeros((len(selected_channels), receiver.n_chans))
    peak_data_hist = np.zeros(len(selected_channels))
    for i in range(3):

        pdf_report.detail(f"Set gain to {gain}.")
        await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

        pdf_report.detail(f"Collect power measurements for {len(selected_channels)} channels.")
        data = await sample_tone_response(selected_channels, amplitude, receiver)
        data = data.astype(np.float64)

        # Seed next_expected_channel if test just starting.
        if i == 0:
            next_expected_channel = channel_range_start

        # Iterate through all selected channels (per gain setting) and check if the position is correct.
        # Store gain adjusted data for SFDR measurement.
        for idx, spectrum in enumerate(data):
            # Use current gain to capture spikes, then adjust gain.
            if i == 0:
                peak_data = np.max(spectrum)
                peak_channel = np.where(spectrum == peak_data)[0][0]
                peak_data_hist[idx] = peak_data
                hdr_data[idx] = spectrum

                # Test1: Test if channel with tone received is correct based on channel skip and detected channel.
                expect(peak_channel == next_expected_channel)
                next_expected_channel = peak_channel + channel_skip

                # Test2: Channel test based on known requested channels.
                expected_channel = selected_channels[idx]

                # Check if tone captured is in the correct channel.
                expect(peak_channel == expected_channel)
            else:
                # Compute HDR data
                power_scale = gain_step ** (i * 2)
                hdr_data[idx] = np.where(
                    hdr_data[idx] >= peak_data_hist[idx] / power_scale, hdr_data[idx], spectrum / power_scale
                )
        gain *= gain_step

    # The maximum is to avoid errors when data is 0
    hdr_data_db = 10 * np.log10(np.maximum(hdr_data, 1e-100) / peak_data)

    # Measure SFDR per captured spectrum
    pdf_report.step("Check SFDR attenuation.")
    sfdr_measurements = measure_sfdr(hdr_data_db, selected_channels)

    sfdr_mean = 0.0
    for sfdr in sfdr_measurements:
        sfdr_mean += sfdr
        expect(sfdr >= required_sfdr_db)
    sfdr_mean /= len(selected_channels)

    pdf_report.detail(f"SFDR (mean): {sfdr_mean:.3f}dB for {len(selected_channels)} channels.")

    # Figure out worst SFDR measurement and plot that one.
    pdf_report.step("Worst SFDR measurement.")
    worst_sfdr_measurement_value = np.min(sfdr_measurements)
    selected_plot_idx = np.where(sfdr_measurements == worst_sfdr_measurement_value)[0][0]
    pdf_report.detail(f"{worst_sfdr_measurement_value:.3f}dB for channel {selected_channels[selected_plot_idx]}.")

    # Round to 3 places to present serialisation of long numbers causing issues for LaTeX
    hdr_data_db = np.round(hdr_data_db, 3)

    plot_channel = selected_channels[selected_plot_idx]
    pdf_report.step(f"SFDR plot for base channel {plot_channel}.")

    xticks = np.arange(0, (receiver.n_chans + 1024), 1024)
    ymin = -100
    title = f"SFDR for channel {plot_channel}"
    x = np.linspace(0, receiver.n_chans - 1, len(hdr_data_db[selected_plot_idx, :]))
    db_plot = hdr_data_db[selected_plot_idx, :]

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
    if ymin < -required_sfdr_db:
        ax.axhline(-required_sfdr_db, dashes=(1, 1), color="black")
        ax.annotate(
            f"{-required_sfdr_db} dB",
            (xticks[-1], -required_sfdr_db),
            horizontalalignment="right",
            verticalalignment="top",
        )
    # tikzplotlib.clean_figure doesn't like data outside the ylim at all
    pdf_report.figure(
        fig, clean_figure=False, tikzplotlib_kwargs=dict(axis_width=r"0.8\textwidth", axis_height=r"0.5\textwidth")
    )
