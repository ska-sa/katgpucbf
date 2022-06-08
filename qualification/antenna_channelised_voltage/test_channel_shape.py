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


async def test_channel_shape(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """TODO."""
    receiver = receive_baseline_correlation_products
    # Arbitrary channel, not too near the edges
    base_channel = correlator.n_chans // 3
    resolution = 128  # Number of samples per channel
    offsets = np.arange(resolution) / resolution - 0.5
    amplitude = 0.99  # dsim amplitude, relative to the maximum (<1.0 to avoid clipping after dithering)

    # Determine the ideal F-engine output leak at the peak
    target_voltage = 110  # Maximum is 127, but some headroom is good
    # We need to avoid overflowing the signed 32-bit X-engine accumulation as
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
    # We also need to avoid overflowing the X-engine accumulation
    gain_step = 100.0
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
    x = np.linspace(-2, 2, len(hdr_data))

    for xmax, ymin, title in [(2, -100, "Channel response"), (0.5, -1.5, "Channel response (zoomed)")]:
        fig = Figure()
        ax = fig.subplots()
        # pgfplots seems to struggle if data is too far out ylim
        ax.plot(x, np.maximum(db, ymin - 10))
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("dB")
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(ymin, 0)
        # tikzplotlib.clean_figure doesn't like data outside the ylim
        pdf_report.figure(fig, clean_figure=False)
