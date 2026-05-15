################################################################################
# Copyright (c) 2025-2026, National Research Foundation (SARAO)
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

"""Test operation for VLBI mode."""

import pytest

from katgpucbf.pytest_plugins.reporter import Reporter
from qualification.recv import TiedArrayChannelisedVoltageReceiver

from ..cbf import CBFRemoteControl


@pytest.mark.vlbi_only
@pytest.mark.name("VLBI configuration")
async def test_configuration(
    cbf_mode_config: dict,
    cbf: CBFRemoteControl,
    start_tied_array_resampled_voltage_stream: bool,
    receiver: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test configuration of tied-array-resampled-voltage stream.

    Verification method
    -------------------
    Verified by means of test. Create a product via the product controller client with no errors.
    """
    pdf_report.step("Verify configuration of tied-array-resampled-voltage stream.")
    pcc = cbf.product_controller_client
    dsim_timestamp = await pcc.sensor_value(f"{cbf.dsim_names[0]}.steady-state-timestamp", int)

    _, data = await receiver.next_complete_chunk(min_timestamp=dsim_timestamp)
    assert start_tied_array_resampled_voltage_stream
    for beam in range(cbf_mode_config["beams"]):
        assert cbf.init_sensors[f"tied-array-resampled-voltage-{beam}.n-chans"].value == 2
        pdf_report.detail(
            f"Tied-array-resampled-voltage-{beam} has "
            + f"{cbf.init_sensors[f'tied-array-resampled-voltage-{beam}.n-chans'].value} channels."
        )
