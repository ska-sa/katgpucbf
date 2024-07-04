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

"""Test that beamformer output has required precision."""

import numpy as np
import pytest
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0118")
async def test_precision(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test tied array beam quantisation bits.

    Verification method
    -------------------
    Verification by means of test. Set an input Gaussian noise signal that
    will cause all the output bits to be used. Check that the full range -127
    to 127 is present.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    await cbf.dsim_gaussian(64.0, pdf_report)

    pdf_report.step("Set weights to select a single input.")
    weights = [0.0] * len(receiver.source_indices[0])
    weights[0] = 1.0
    await client.request("beam-weights", receiver.stream_names[0], *weights)
    pdf_report.detail(f"Weights for {receiver.stream_names[0]} set to {weights}.")

    pdf_report.step("Collect and validate a chunk.")
    timestamp, data = await receiver.next_complete_chunk()
    pdf_report.detail(f"Received chunk with timestamp {timestamp}.")
    unique_values = np.unique(data)
    pdf_report.detail(f"{len(unique_values)} unique values observed.")
    with check:
        assert len(unique_values) == 255
        assert unique_values[0] == -127 and unique_values[-1] == 127
