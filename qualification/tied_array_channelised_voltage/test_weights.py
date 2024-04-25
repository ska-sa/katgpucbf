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

"""Weights test."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0123")
async def test_weight_mapping(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that each weight coefficient applies to the correct input.

    Verification method
    -------------------
    Verification by means of test. Configure the dsim with a tone in a random
    channel for each polarisation. For each input:

    - Set the ``?gain`` to 1 for that input and to zero for all other inputs.
    - Set one beam to use only that input, and all other beams to sum the remaining
      inputs.
    - Check that the tone is found in the correct channel of the chosen beam.
    - Check that the other beams all output zero.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    rng = np.random.default_rng()
    channels = rng.integers(1, receiver.n_chans, size=2)
    channel_width = receiver.bandwidth / receiver.n_chans
    freqs = receiver.center_freq + (channels - receiver.n_chans / 2) * channel_width
    amplitude = 0.5  # Will probably saturate the PFB, but we don't care
    signals = f"cw({amplitude}, {freqs[0]}); cw({amplitude}, {freqs[1]});"

    pdf_report.step("Configure dsim with tones.")
    await cbf.dsim_clients[0].request("signals", signals)
    pdf_report.detail(f"Set dsim signals to {signals!r}.")

    rng = np.random.default_rng()
    for input_idx in range(receiver.n_inputs):
        input_label = receiver.input_labels[input_idx]
        channel = channels[input_idx % 2]
        pdf_report.step(f"Testing input {input_idx} ({input_label}).")
        await client.request("gain-all", "antenna-channelised-voltage", 0.0)
        await client.request("gain", "antenna-channelised-voltage", input_label, 1.0)
        pdf_report.detail(f"Set gain on {input_label} to 1.0, all others to 0.0.")
        candidate_beams = [i for i, source_indices in enumerate(receiver.source_indices) if input_idx in source_indices]
        assert candidate_beams, "No beam includes this input"
        test_beam = rng.choice(candidate_beams)
        test_beam_name = receiver.stream_names[test_beam]
        pdf_report.detail(f"Using beam {test_beam_name} as the test beam.")

        source_indices = receiver.source_indices[test_beam]
        input_pos = source_indices.index(input_idx)
        weights = [0.0] * len(source_indices)
        weights[input_pos] = 1.0
        inv_weights = [1.0 - w for w in weights]
        await client.request("beam-weights", test_beam_name, *weights)
        pdf_report.detail(f"Set beam-weights for {test_beam_name} to {weights}")
        for stream_name in receiver.stream_names:
            if stream_name != test_beam_name:
                await client.request("beam-weights", stream_name, *inv_weights)
                pdf_report.detail(f"Set beam-weights for {stream_name} to {inv_weights}")

        timestamp, data = await receiver.next_complete_chunk()
        pdf_report.detail(f"Received chunk with timestamp {timestamp}.")
        data = data.astype(np.float32).view(np.complex64)[..., 0]  # Convert to complex64
        with check:
            # Should be much larger than 20.0, but this is enough to reject
            # noise and spectral leakage.
            assert np.all(np.abs(data[test_beam, channel]) > 20.0)
            pdf_report.detail(f"Tone found in channel {channel}.")
        # Zero out the tone so that we can check that everything else is empty.
        data[test_beam, channel] = 0

        with check:
            assert np.all(np.abs(data[test_beam, channel]) < 2.0)
            pdf_report.detail("All other data is close to zero.")


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
    client = cbf.product_controller_client
    beam_name = receiver.stream_names[0]
    n_sources = len(receiver.source_indices[0])

    # Small amplitude so that we don't saturate in the time domain.
    period = receiver.n_spectra_per_heap * receiver.n_samples_between_spectra
    await cbf.dsim_gaussian(16.0, pdf_report, period=period)

    pdf_report.step("Set quantiser gain to 1/antennas")
    await client.request("beam-quant-gains", beam_name, 1.0 / n_sources)
    pdf_report.detail(f"Set beam-quant-gains on {beam_name} to 1/{n_sources}")

    pdf_report.step("Measure total power responses")
    weights = np.logspace(-2.0, 2.0, 41)  # Note: n is chosen to ensure 1.0 is included
    middle = np.searchsorted(weights, 1.0)
    assert weights[middle] == pytest.approx(1.0)
    powers = np.zeros_like(weights)
    for i, weight in enumerate(weights):
        await client.request("beam-weights", beam_name, *([weight] * n_sources))
        _, data = await receiver.next_complete_chunk()
        powers[i] = np.sum(np.square(data[0].astype(np.float64)))
        pdf_report.detail(f"Set weights to {weight}; power is {powers[i]}")

    # Normalise power
    powers /= powers[middle]

    fig = Figure()
    ax = fig.add_subplot()
    ax.set_title("Normalised power relative to weight")
    ax.set_xlabel("Weight (dB)")
    ax.set_ylabel("Power (dB)")
    weights_db = 20 * np.log10(weights)
    powers_db = 10 * np.log10(powers)
    ax.plot(weights_db, weights_db, label="Reference")
    ax.plot(weights_db, powers_db, label="Measured")
    ax.legend()
    pdf_report.figure(fig)
