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

import asyncio

import numpy as np
import pytest
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayChannelisedVoltageReceiver
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
    Verification by means of test. Configure the dsim with Gaussian noise.
    For each input, set eq gains that zero all but a single channel, using
    a different channel for each input. Then, for each input:

    - Set one beam to use only that input, and all other beams to sum the remaining
      inputs.
    - Check that signal is found in the correct channel of the chosen beam.
    - Check that the other beams all output zero for that channel.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    assert receiver.n_inputs < receiver.n_chans, "Test assumes a unique channel per input"
    rng = np.random.default_rng()
    # Pick a random channel for each input
    channels = rng.choice(receiver.n_chans, receiver.n_inputs, replace=False)
    signals = "common=wgn(0.2); common; common;"

    pdf_report.step("Configure dsim with Gaussian noise.")
    await cbf.dsim_clients[0].request("signals", signals)
    pdf_report.detail(f"Set dsim signals to {signals!r}.")

    pdf_report.step("Set eq gains to select one channel per input.")
    async with asyncio.TaskGroup() as tg:
        for input_label, channel in zip(receiver.input_labels, channels):
            gains = [0.0] * receiver.n_chans
            gains[channel] = 1.0
            tg.create_task(client.request("gain", "antenna-channelised-voltage", input_label, *gains))
            pdf_report.detail(f"Set input {input_label} to pass through only channel {channel}.")

    pdf_report.step("Test all inputs.")
    for input_idx, (input_label, channel) in enumerate(zip(receiver.input_labels, channels)):
        candidate_beams = [i for i, source_indices in enumerate(receiver.source_indices) if input_idx in source_indices]
        assert candidate_beams, "No beam includes this input"
        test_beam = rng.choice(candidate_beams)
        test_beam_name = receiver.stream_names[test_beam]
        pdf_report.detail(f"Test input {input_idx} ({input_label}) with channel {channel} and beam {test_beam_name}.")

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
        with check:
            assert np.sum(np.abs(data[test_beam, channel])) > 0
            pdf_report.detail(f"Signal found in channel {channel} of the test beam.")
        # Zero out the tone so that we can check that everything else is empty.
        data[test_beam, channel] = 0

        with check:
            assert np.all(data[test_beam, channel] == 0)
            pdf_report.detail("All other beams have no signal in the channel.")
