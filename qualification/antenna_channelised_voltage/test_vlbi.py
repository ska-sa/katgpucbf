################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Test filter response for VLBI narrowband mode."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import POTLocator, Reporter
from . import sample_tone_response_hdr


@pytest.mark.vlbi_only
@pytest.mark.name("VLBI filter response")
async def test_filter_response(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_bandwidth: float,
    pass_channels: slice,
) -> None:
    """Test frequency response of VLBI narrowband DDC filter.

    Verification method
    -------------------
    Verified by means of test. A range of channels is sampled by using
    a tone as the signal. The tones are placed in the centres of the
    channel bins to eliminate spectral leakage from the results.
    Check that the passband ripple is at most 0.1 dB and that no alias in the
    passband exceeds -80 dB.

    The test is performed for 1712 MHz input bandwidth, 107 MHz output
    bandwidth and 64 MHz passband.
    """
    receiver = receive_baseline_correlation_products

    # Limit number of samples to take - otherwise the test becomes too slow.
    step = max(1, receiver.n_chans // 64)
    # Sample a range covering twice the bandwidth, to determine response to
    # aliases.
    channels = np.arange(-receiver.n_chans // 2 + step // 2, receiver.n_chans * 3 // 2, step)
    # Determine channel to read out response from i.e. into which the input tone
    # will alias.
    out_channels = channels % receiver.n_chans

    pdf_report.step(f"Collect response to {len(channels)} frequencies.")
    data = await sample_tone_response_hdr(
        cbf=cbf,
        receiver=receiver,
        pdf_report=pdf_report,
        amplitude=0.99,
        rel_freqs=channels,
    )

    pdf_report.step("Analyse results.")

    data /= np.max(data)  # Normalise
    # data contains the complete spectral response to every input, but we're
    # only interested in the response in the corresponding channel.
    data = data[np.arange(len(channels)), out_channels]
    # Make relative to centre frequency
    freqs = receiver.channel_frequency(channels) - receiver.center_freq
    # The maximum is to avoid errors when data is 0
    with np.errstate(divide="ignore"):  # Avoid warnings when taking log of 0
        data_db = 10 * np.log10(data)

    # Turn pass_channels (which slices the full set of channels) into
    # a mask over the channels collected.
    pass_select = (pass_channels.start <= channels) & (channels < pass_channels.stop)
    ripple = np.max(data_db[pass_select]) - np.min(data_db[pass_select])
    pdf_report.detail(f"Passband ripple is {ripple:.6f} dB.")
    with check:
        assert ripple < 0.1

    # Mask for channels that alias into the passband
    alias_select = (pass_channels.start <= out_channels) & (out_channels < pass_channels.stop)
    alias_select &= ~pass_select  # Ignore the passband itself
    max_alias = np.max(data_db[alias_select])
    pdf_report.detail(f"Maximum alias into the passband is {max_alias:.3f} dB.")
    with check:
        assert max_alias < -80.0

    stop_bandwidth = 2 * receiver.bandwidth - pass_bandwidth
    fig = Figure()
    ax = fig.add_subplot()
    ax.axvspan(-stop_bandwidth / 2e6, -pass_bandwidth / 2e6, color="red", alpha=0.2, linewidth=0)
    ax.axvspan(pass_bandwidth / 2e6, stop_bandwidth / 2e6, color="red", alpha=0.2, linewidth=0)
    ax.axvline(-receiver.bandwidth / 2e6, linestyle="dashed", color="black")
    ax.axvline(receiver.bandwidth / 2e6, linestyle="dashed", color="black")
    ax.plot(freqs / 1e6, data_db)
    ax.set_title("DDC filter response")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("dB")
    ax.xaxis.set_major_locator(POTLocator())
    pdf_report.figure(fig)

    fig = Figure()
    ax = fig.add_subplot()
    ax.plot(freqs[pass_select] / 1e6, data_db[pass_select])
    ax.set_title("DDC filter response (passband)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("dB")
    ax.xaxis.set_major_locator(POTLocator())
    pdf_report.figure(fig)
