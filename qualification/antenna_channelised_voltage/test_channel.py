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

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import POTLocator, Reporter
from . import sample_tone_response_hdr


def measure_sfdr(hdr_data_db: np.ndarray, base_channel: np.ndarray) -> list[float]:
    """Measure Spurious Free Dynamic Range (SFDR) of the response at a given power level.

    Estimate the SFDR by measuring the power (dB) of the next strongest
    tone in the spectrum (ignoring the fundamental tone). The SFDR is the
    difference in these two values.

    Parameters
    ----------
    hdr_data_db:
        A 2-dimensional array where axis 0 represents the different captured spectra,
        and axis 1 represents the samples in a spectrum (in dB).
    base_channel:
        Channel numbers corresponding to the first axis of `hdr_data_db`.

    Returns
    -------
    A list of differences computed from peak_value - next_peak_value.
    """
    sfdr_measurements = []

    for spectrum, channel in zip(hdr_data_db, base_channel):
        peak_value = spectrum[channel]
        next_peak_value = max(np.max(spectrum[:channel]), np.max(spectrum[channel + 1 :]))
        peak_diff = peak_value - next_peak_value
        sfdr_measurements.append(peak_diff)
    return sfdr_measurements


@pytest.mark.name("Channelisation and SFDR")
@pytest.mark.requirements("CBF-REQ-0126")
async def test_channelisation_and_sfdr(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test channel position and measure SFDR per channel under test.

    Verification method
    -------------------
    Channel Position:
    Verification by means of test. The expected channel position is verified by pre-selecting a range
    of channels (frequencies) and producing independent spectra per frequency. The position of the
    tone in each spectrum is determined and compared to the position expected.

    Spurious Free Dynamic Range (SFDR):
    Verification by means of test. During the execution of the channel position test the minimum
    SFDR is computed as the difference between the peak detected in the channel and the next strongest
    peak detected in the spectrum where no intentional tone is positioned. This is reported in dB.
    """
    receiver = receive_baseline_correlation_products

    required_sfdr_db = 53.0
    channel_range_start = 8
    # Odd number to ensure that the channel indices tested cover a range of least significant bits as
    # this may influence the performance of the FFT.
    channel_skip = 31

    # Arbitrary channels, not too near the edges, skipping every 'channel_skip' channels
    rel_freqs = np.arange(channel_range_start, receiver.n_chans, channel_skip)
    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)

    pdf_report.step("Set gain and measure channel position.")
    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.

    pdf_report.detail(f"Collect power measurements for {len(rel_freqs)} channels.")
    hdr_data = await sample_tone_response_hdr(
        correlator=correlator,
        receiver=receiver,
        pdf_report=pdf_report,
        amplitude=amplitude,
        rel_freqs=rel_freqs,
    )

    # Check tone positions w.r.t. requested channels
    pdf_report.step("Check tone positions.")
    for idx, sel_chan in enumerate(rel_freqs):
        peak_chan = np.argmax(hdr_data[idx])
        with check:
            assert sel_chan == peak_chan

    # The maximum is to avoid errors when data is 0
    hdr_data_db = 10 * np.log10(np.maximum(hdr_data, 1e-100) / np.max(hdr_data))

    # Measure SFDR per captured spectrum
    pdf_report.step("Check SFDR attenuation.")
    sfdr_measurements = measure_sfdr(hdr_data_db, rel_freqs)
    # Figure out worst SFDR measurement
    sfdr_min = np.min(sfdr_measurements)

    # Check that minimum SFDR measurement meets the requirement.
    with check:
        assert sfdr_min >= required_sfdr_db
    sfdr_mean = np.mean(sfdr_measurements)

    pdf_report.detail(f"SFDR (mean): {sfdr_mean:.3f}dB for {len(rel_freqs)} channels.")

    # Report and plot worst SFDR measurement.
    pdf_report.step("Minimum SFDR measurement.")

    selected_plot_idx = np.argmin(sfdr_measurements)
    pdf_report.detail(f"{sfdr_min:.3f}dB for channel {rel_freqs[selected_plot_idx]}.")

    plot_channel = rel_freqs[selected_plot_idx]
    pdf_report.step(f"SFDR plot for base channel {plot_channel}.")

    ymin = -100
    title = f"SFDR for channel {plot_channel}"
    x = np.linspace(0, receiver.n_chans - 1, len(hdr_data_db[selected_plot_idx, :]))
    db_plot = hdr_data_db[selected_plot_idx, :]

    fig = Figure()
    ax = fig.subplots()
    ax.plot(x, db_plot)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    ax.set_ylabel("dB")
    ax.xaxis.set_major_locator(POTLocator())
    ax.set_ylim(ymin, -0.05 * ymin)
    if ymin < -required_sfdr_db:
        ax.axhline(-required_sfdr_db, dashes=(1, 1), color="black")
        ax.annotate(
            f"{-required_sfdr_db} dB",
            (0, -required_sfdr_db),
            xytext=(-3, -3),
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="top",
        )
    pdf_report.figure(fig)
