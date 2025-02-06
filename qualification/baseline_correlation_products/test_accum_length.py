################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

"""Accumulation length test."""

import numpy as np
import pytest
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0096")
async def test_accum_length(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that accumulations are set to the correct length.

    Verification method
    -------------------
    Verify by testing that the accumulation interval is within specification
    when set to 500ms.
    """
    receiver = receive_baseline_correlation_products
    pdf_report.step("Retrieve the reported accumulation time and check it.")
    pdf_report.detail(f"Integration time is {receiver.int_time * 1000:.3f} ms.")
    # Requirement doesn't list an upper bound, but assume a symmetric limit
    with check:
        assert 0.48 <= receiver.int_time <= 0.52


@pytest.mark.requirements("CBF-REQ-0096")
async def test_accum_power(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    """
    Test that the actual accumulation length matches the reported length.

    Verification method
    -------------------
    Inject a noise signal and check that the measured signal has the expected
    power for the number of accumulations.
    """
    receiver = receive_baseline_correlation_products
    pdf_report.step("Inject a white noise signal.")
    level = 32  # Expected magnitude of F-engine outputs
    input_std = level / 511  # dsim will scale up by 511 to fill [-511, 511] range
    await cbf.dsim_clients[0].request("signals", f"common=wgn({input_std});common;common;")

    pdf_report.step("Collect two dumps and check the timestamp difference.")
    chunks = await receiver.consecutive_chunks(2)
    pdf_report.detail(f"Timestamps are {chunks[0][0]}, {chunks[1][0]}.")
    delta = chunks[1][0] - chunks[0][0]
    delta_s = delta / receiver.scale_factor_timestamp
    pdf_report.detail(f"Difference is {delta} samples, {delta_s * 1000:.3f} ms.")
    with check:
        # pytest.approx just to allow for floating-point rounding
        assert delta_s == pytest.approx(receiver.int_time, rel=1e-15)

    pdf_report.step("Compare power against expected value.")
    # Sum over channels, but use only one baseline and real part because
    # the input signals are the same for all antennas.
    assert isinstance(chunks[1][1].data, np.ndarray)
    total_power = np.sum(chunks[1][1].data[pass_channels, 0, 0], dtype=np.int64)
    acc_len = round(
        receiver.int_time * receiver.scale_factor_timestamp / (2 * receiver.n_chans * receiver.decimation_factor)
    )
    expected_power = acc_len * (pass_channels.stop - pass_channels.start) * (level * level)
    pdf_report.detail(f"Total power: {total_power}; expected: {expected_power}.")
    # Statistical analysis of total_power is quite tricky because there is
    # quantisation and saturation in both the time and frequency domain, and
    # the values are not fully independent in either time or channel. But 1%
    # variation seems safe.
    with check:
        assert total_power == pytest.approx(expected_power, rel=0.01)
    pdf_report.detail("Power agrees to within 1%.")
