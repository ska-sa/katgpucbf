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

"""Delay test."""

import numpy as np
import pytest
from pytest_check import check

from katgpucbf import DIG_SAMPLE_BITS

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0220")
async def test_delay(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test beam steering delay application.

    Verification method
    -------------------
    Verification by means of test. Set a delay on one input and form a beam
    from it with a compensating delay. Use a different input with no delay
    to form a reference beam. Check that the results are consistent to within 1
    ULP.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    pdf_report.step("Configure the D-sim with Gaussian noise.")
    dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
    # Small so that we don't saturate, as delaying and undelaying a saturated
    # value won't round-trip properly
    amplitude = 16 / dig_max
    await cbf.dsim_clients[0].request("signals", f"common=nodither(wgn({amplitude}));common;common;")
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}.")

    pdf_report.step("Choose random inputs for delay and reference beams and set weights.")
    rng = np.random.default_rng(seed=123)
    delay_beam = 0
    delay_input_idx = rng.integers(len(receiver.source_indices[delay_beam]))
    delay_input = receiver.source_indices[delay_beam][delay_input_idx]
    pdf_report.detail(f"Using input {delay_input} for delay beam.")

    ref_beam = 1
    ref_input_idx = rng.integers(len(receiver.source_indices[ref_beam]))
    ref_input = receiver.source_indices[ref_beam][ref_input_idx]
    pdf_report.detail(f"Using input {ref_input} for reference beam.")
    # Should never happen because they're different polarisations
    assert delay_input != ref_input

    delay_weights = [0.0] * len(receiver.source_indices[delay_beam])
    delay_weights[delay_input_idx] = 1.0
    await client.request("beam-weights", receiver.stream_names[delay_beam], *delay_weights)
    pdf_report.detail(f"Set weights on {receiver.stream_names[delay_beam]} to {delay_weights}")
    ref_weights = [0.0] * len(receiver.source_indices[ref_beam])
    ref_weights[ref_input_idx] = 1.0
    await client.request("beam-weights", receiver.stream_names[ref_beam], *ref_weights)
    pdf_report.detail(f"Set weights on {receiver.stream_names[ref_beam]} to {delay_weights}")

    # TODO: need the final version of the requirements to know what values to test
    delays = [-509e-9, 509e-9]
    for delay in delays:
        pdf_report.step(f"Test with delay {delay * 1e12} ps.")
        # Ensure load time is in the past, so that it is already applied when we
        # receive data.
        load_time = await cbf.dsim_time() - 5.0
        input_delays = ["0,0:0,0"] * receiver.n_inputs
        input_delays[delay_input] = f"{delay},0:0,0"
        await client.request("delays", "antenna-channelised-voltage", load_time, *input_delays)
        pdf_report.detail(f"Set input delays to {input_delays}")
        beam_delays = ["0:0"] * len(receiver.source_indices[delay_beam])
        beam_delays[delay_input_idx] = f"{-delay}:0"
        await client.request("beam-delays", receiver.stream_names[delay_beam], *beam_delays)
        pdf_report.detail(f"Set beam {delay_beam} delays to {beam_delays}")
        timestamp, data = await receiver.next_complete_chunk()
        pdf_report.detail(f"Received chunk with timestamp {timestamp}")
        # Need more precision to avoid overflows when subtracting
        data = data.astype(np.int16)
        max_error = np.max(np.abs(data[delay_beam] - data[ref_beam]))
        with check:
            assert max_error <= 1
        pdf_report.detail(f"Maximum difference is {max_error} ULP")
