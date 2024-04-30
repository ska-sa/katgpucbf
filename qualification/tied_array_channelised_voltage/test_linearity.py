################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Test linearity of gains and weights."""

from typing import Awaitable, Callable

import aiokatcp
import numpy as np
import pytest
from matplotlib.figure import Figure

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


async def _test_linearity(
    cbf: CBFRemoteControl,
    receiver: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    fixed: str,
    variable: str,
    set_fixed: Callable[[aiokatcp.Client, str, float], Awaitable],
    set_variable: Callable[[aiokatcp.Client, str, float], Awaitable],
) -> None:
    client = cbf.product_controller_client
    beam_name = receiver.stream_names[0]
    n_sources = len(receiver.source_indices[0])

    period = receiver.n_spectra_per_heap * receiver.n_samples_between_spectra
    # Small amplitude so that we don't saturate in the time domain.
    await cbf.dsim_gaussian(16.0, pdf_report, period=period)

    pdf_report.step(f"Set {fixed} to 1/antennas")
    await set_fixed(client, beam_name, 1.0 / n_sources)
    pdf_report.detail(f"Set {fixed} on {beam_name} to 1/{n_sources}")

    pdf_report.step("Measure total power responses")
    scales = np.logspace(-2.0, 2.0, 41)  # Note: n is chosen to ensure 1.0 is included
    middle = np.searchsorted(scales, 1.0)
    assert scales[middle] == pytest.approx(1.0)
    powers = np.zeros_like(scales)
    for i, scale in enumerate(scales):
        await set_variable(client, beam_name, scale)
        _, data = await receiver.next_complete_chunk()
        data = data[0]  # Use only the first beam
        powers[i] = np.sum(np.square(data.astype(np.float64)))
        pdf_report.detail(f"Set {variable} to {scale}; power is {powers[i]}")

    # Normalise power
    powers /= powers[middle]

    fig = Figure()
    ax = fig.add_subplot()
    ax.set_title(f"Normalised power relative to {variable}")
    ax.set_xlabel(f"{variable} (dB)")
    ax.set_ylabel("power (dB)")
    scales_db = 20 * np.log10(scales)
    powers_db = 10 * np.log10(powers)
    ax.plot(scales_db, scales_db, label="Reference")
    ax.plot(scales_db, powers_db, label="Measured")
    ax.legend()
    pdf_report.figure(fig)


@pytest.mark.requirements("CBF-REQ-0123")
async def test_weight_linearity(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test linearity of the weight coefficients.

    Verification method
    -------------------
    Verification by means of test. Configure the dsim with Gaussian noise with
    a period of one heap. Set the weights for all inputs to a range of values,
    and measure the total power (for one beam) in each case.

    Large weights are expected to have non-linear response due to saturation,
    and small weights are expected to have non-linear response due to
    quantisation.
    """
    receiver = receive_tied_array_channelised_voltage
    n_sources = len(receiver.source_indices[0])
    await _test_linearity(
        cbf,
        receiver,
        pdf_report,
        "quantisation gain",
        "weight",
        lambda client, stream, gain: client.request("beam-quant-gains", stream, gain),
        lambda client, stream, weight: client.request("beam-weights", stream, *([weight] * n_sources)),
    )


@pytest.mark.requirements("CBF-REQ-0117")
async def test_gain_linearity(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test linearity of the quantisation gains.

    Verification method
    -------------------
    Verification by means of test. Configure the dsim with Gaussian noise with
    a period of one heap. Set the gain to a range of values, and measure the
    total power (for one beam) in each case.

    Large gains are expected to have non-linear response due to saturation,
    and small gains are expected to have non-linear response due to
    quantisation.
    """
    receiver = receive_tied_array_channelised_voltage
    n_sources = len(receiver.source_indices[0])
    await _test_linearity(
        cbf,
        receiver,
        pdf_report,
        "weight",
        "quantisation gain",
        lambda client, stream, weight: client.request("beam-weights", stream, *([weight] * n_sources)),
        lambda client, stream, gain: client.request("beam-quant-gains", stream, gain),
    )
