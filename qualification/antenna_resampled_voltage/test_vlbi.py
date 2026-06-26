################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Sample test for tied-array-resampled-voltage stream."""

import pytest
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter

from ..recv import TiedArrayResampledVoltageReceiver


@pytest.mark.name("VLBI VDIF output")
async def test_vlbi_vdif(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
) -> None:
    """Test VDIF frame output.

    Verification method
    -------------------
    Verified by means of test.
    Collect a valid VDIF frameset.
    """
    receiver = receive_tied_array_resampled_voltage
    pdf_report.step("Collect a valid VDIF frame.")
    frameset, _ = await receiver.get_frameset()
    pdf_report.detail("Verify we have `n_chans * len(pol_ordering)` threads in the set.")
    with check:
        assert len(frameset) == receiver.n_chans * len(receiver.pol_ordering)
