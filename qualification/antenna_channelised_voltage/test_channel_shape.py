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

"""Channel shape tests."""

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from katgpucbf import DIG_SAMPLE_BITS

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter
from . import sample_tone_response


def cutoff_bandwidth_half(data: np.ndarray, cutoff: float, step: float) -> float:
    """One-sided version of :func:`cutoff_bandwidth`.

    Here the first element of `data` must represent the centre.
    """
    assert data[0] > cutoff
    assert data[-1] < cutoff
    right = np.nonzero(data < cutoff)[0][0]
    left = right - 1
    interp = left + (data[left] - cutoff) / (data[left] - data[right])
    return interp * step


def cutoff_bandwidth(data: np.ndarray, cutoff: float, step: float) -> float:
    """Measure width of the response at a given power level.

    Estimate the width of the central portion where `data` is above `cutoff`,
    in units of channels. If there are sidelobes that rise about `cutoff`,
    they're included in the width.
    """
    assert len(data) % 2 == 1  # Must be symmetrical
    mid = len(data) // 2
    return cutoff_bandwidth_half(data[mid:], cutoff, step) + cutoff_bandwidth_half(data[mid::-1], cutoff, step)


async def test_channel_shape(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test the shape of the response to a single channel.

    Requirements verified:

    CBF-REQ-0126
        The CBF shall perform channelisation such that the 53 dB attenuation bandwidth
        is :math:`\le 2\times` (twice) the pass band width.
    """
    receiver = receive_baseline_correlation_products
    # Arbitrary channel, not too near the edges
    base_channel = correlator.n_chans // 3
    resolution = 128  # Number of samples per channel
    offsets = np.arange(resolution) / resolution - 0.5
    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)

    # Determine the ideal F-engine output leak at the peak
    target_voltage = 110  # Maximum is 127, but some headroom is good
    # We need to avoid saturating the signed 32-bit X-engine accumulation as
    # well (2e9 is comfortably less than 2^31).
    target_voltage = min(target_voltage, np.sqrt(2e9 / correlator.n_spectra_per_acc))
    # The PFB is scaled for fixed incoherent gain, but we need to be concerned
    # about coherent gain to avoid overflowing the F-engine output. Coherent gain
    # scales approximately with np.sqrt(correlator.n_chans / 2).
    dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
    gain = target_voltage / (amplitude * dig_max * np.sqrt(correlator.n_chans / 2))

    async def sample(offsets: ArrayLike) -> np.ndarray:
        """Measure response when frequency is offset from channel centre.

        Parameters
        ----------
        offset
            Offset of the frequency, in units of channels
        """
        rel_freq = base_channel - np.asarray(offsets)
        data = await sample_tone_response(rel_freq, amplitude, correlator, receiver)
        # Flatten to 1D (Fortran order so that offset is fastest-varying axis)
        data = data.ravel(order="F")
        # Slice out 5 channels, centred on the chosen one
        data = data[(base_channel - 2) * resolution : (base_channel + 3) * resolution + 1]
        return data

    pdf_report.step("Measure channel shape.")
    gain_step = 100.0
    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.
    for i in range(3):
        pdf_report.detail(f"Set gain to {gain}.")
        await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)
        pdf_report.detail(f"Collect power measurements ({resolution} per channel).")
        data = await sample(offsets)
        data = data.astype(np.float64)
        if i == 0:
            peak = np.max(data)
            hdr_data = data
        else:
            power_scale = gain_step ** (i * 2)
            hdr_data = np.where(hdr_data >= peak / power_scale, hdr_data, data / power_scale)
        gain *= gain_step

    rms_voltage = np.sqrt(peak / correlator.n_spectra_per_acc)
    pdf_report.detail(f"Peak power is {int(peak)} (RMS voltage {rms_voltage:.3f}).")

    # The maximum is to avoid errors when data is 0
    db = 10 * np.log10(np.maximum(hdr_data, 1e-100) / peak)
    x = np.linspace(-2.5, 2.5, len(hdr_data))

    for xticks, ymin, title in [
        (np.arange(-2.5, 2.6, 0.5), -100, "Channel response"),
        (np.arange(-0.5, 0.55, 0.1), -1.5, "Channel response (zoomed)"),
    ]:
        fig = Figure()
        ax = fig.subplots()
        # pgfplots seems to struggle if data is too far outside ylim
        ax.plot(x, np.maximum(db, ymin - 10))
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

    pdf_report.step("Check attenuation bandwidth.")
    width_3db = cutoff_bandwidth(db, -3, 1 / resolution)
    width_53db = cutoff_bandwidth(db, -53, 1 / resolution)
    pdf_report.detail(f"-3 dB bandwidth is {width_3db:.3f} channels.")
    pdf_report.detail(
        f"-53 dB bandwidth is {width_53db:.3f} channels ({width_53db / width_3db:.3f}x the pass bandwidth)."
    )
    assert width_53db <= 2 * width_3db
