################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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
import pytest
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from pytest_check import check

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter
from . import sample_tone_response_hdr


def _cutoff_interp(x0: float, y0: float, x1: float, y1: float, cutoff: float) -> float:
    """Compute x value where a line crosses a y value (by linear interpolation)."""
    return x0 + (x1 - x0) * (cutoff - y0) / (y1 - y0)


def cutoff_bandwidth(data: np.ndarray, cutoff: float, step: float) -> float:
    """Measure width of the response at a given power level.

    Estimate the width of the region where `data` is above `cutoff`. If the
    cutoff is crossed multiple times, the distance between the two most extreme
    values is used.

    The return value is in unit of channels, but the `data` may have sub-channel
    resolution, with a step of `step` channels between samples.
    """
    above = np.nonzero(data >= cutoff)[0]
    assert len(above) > 0
    # Minimum and maximum index that contain data above the cutoff
    left = above[0]
    right = above[-1]
    # Use linear interpolation to find fractional index values where the
    # cutoff is crossed.
    if left > 0:
        left = _cutoff_interp(left - 1, data[left - 1], left, data[left], cutoff)
    if right < len(data) - 1:
        right = _cutoff_interp(right + 1, data[right + 1], right, data[right], cutoff)
    return (right - left) * step  # Scale from indices to channels


@pytest.mark.requirements("CBF-REQ-0126")
async def test_channel_shape(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test the shape of the response to a single channel.

    Verification method
    -------------------
    Verification by means of test. This test selects a base frequency and generates multiple independent
    spectra based on the base frequency with offsets above and below the base frequency. The frequency
    deviations produce differing channel amplitudes in frequency domain which when viewed collectively
    and in series illustrate a channel shape. These meaurements are used to compute a -3dB and -53dB
    channel bandwidth.
    """
    receiver = receive_baseline_correlation_products
    # Arbitrary channel, not too near the edges
    base_channel = receiver.n_chans // 3
    resolution = 128  # Number of samples per channel
    offsets = np.arange(resolution) / resolution - 0.5
    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)

    async def sample(offsets: ArrayLike) -> np.ndarray:
        """Measure response when frequency is offset from channel centre.

        Parameters
        ----------
        offset
            Offset of the frequency, in units of channels
        """
        rel_freq = base_channel - np.asarray(offsets)
        pdf_report.detail(f"Collect power measurements ({resolution} per channel).")
        hdr_data = await sample_tone_response_hdr(
            correlator=correlator,
            receiver=receiver,
            pdf_report=pdf_report,
            amplitude=amplitude,
            rel_freqs=rel_freq,
        )

        # Flatten to 1D (Fortran order so that offset is fastest-varying axis)
        hdr_data = hdr_data.ravel(order="F")
        return hdr_data

    pdf_report.step("Measure channel shape.")
    hdr_data = await sample(offsets)

    peak = np.max(hdr_data)
    rms_voltage = np.sqrt(peak / receiver.n_spectra_per_acc)
    pdf_report.detail(f"Peak power is {int(peak)} (RMS voltage {rms_voltage:.3f}).")

    with np.errstate(divide="ignore"):  # Avoid warnings when taking log of 0
        db = 10 * np.log10(hdr_data / peak)
    # Slice out 5 channels, centred on the chosen one
    db_plot = db[(base_channel - 2) * resolution : (base_channel + 3) * resolution + 1]
    x = np.linspace(-2.5, 2.5, len(db_plot))

    for xticks, ymin, title in [
        (np.arange(-2.5, 2.6, 0.5), -150, "Channel response"),
        (np.arange(-0.5, 0.55, 0.1), -1.5, "Channel response (zoomed)"),
    ]:
        fig = Figure()
        ax = fig.subplots()
        ax.plot(x, db_plot)
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("dB")
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(ymin, -0.05 * ymin)

        for y in [-3, -53]:
            if ymin < y:
                ax.axhline(y, dashes=(1, 1), color="black")
                ax.annotate(
                    f"{y} dB",
                    (xticks[-1], y),
                    xytext=(-3, -3),
                    textcoords="offset points",
                    horizontalalignment="right",
                    verticalalignment="top",
                )
        pdf_report.figure(fig)

    pdf_report.step("Check attenuation bandwidth.")
    width_3db = cutoff_bandwidth(db, -3, 1 / resolution)
    width_53db = cutoff_bandwidth(db, -53, 1 / resolution)
    pdf_report.detail(f"-3 dB bandwidth is {width_3db:.3f} channels.")
    pdf_report.detail(
        f"-53 dB bandwidth is {width_53db:.3f} channels ({width_53db / width_3db:.3f}x the pass bandwidth)."
    )
    with check:
        assert width_53db <= 2 * width_3db
