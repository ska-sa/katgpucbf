################################################################################
# Copyright (c)2026, National Research Foundation (SARAO)
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

import numpy as np
import pytest
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayResampledVoltageReceiver


@pytest.mark.name("VLBI VDIF output")
async def test_vlbi_vdif(
    cbf: CBFRemoteControl,
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
) -> None:
    """Test VDIF frame output.

    Verification method
    -------------------
    Verified by means of test. Collect a valid VDIF frame and verify that the
    frame is valid.
    """
    receiver = receive_tied_array_resampled_voltage
    pdf_report.step("Collect a valid VDIF frame.")
    _, frameset = await receiver.get_frameset()
    pdf_report.step("Verify we have log_2(1) channels.")
    with check:
        assert frameset.header0.nchan == 1
    pdf_report.detail(f"VDIF frame max value: {np.max(frameset.data)}")
    pdf_report.detail(f"VDIF frame min value: {np.min(frameset.data)}")
    pdf_report.detail(f"VDIF frame mean value: {np.mean(frameset.data)}")
    pdf_report.detail(f"VDIF frame std value: {np.std(frameset.data)}")
    pdf_report.detail(f"VDIF frame shape: {frameset.data.shape}")
