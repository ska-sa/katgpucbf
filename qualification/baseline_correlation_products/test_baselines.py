################################################################################
# Copyright (c) 2022-2026, National Research Foundation (SARAO)
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

"""Baseline verification tests."""

import numpy as np
import pytest
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0087,CBF-REQ-0104")
async def test_baseline_correlation_products(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that the baseline ordering indicated in the sensor matches the output data.

    Verification method
    -------------------
    Verification by means of test. Verify by testing all correlation product
    combinations. Use gain correction after channelisation to turn the input
    signal on or off. Iterate through all combinations and verify that the
    expected output appears in the correct baseline product.
    """
    receiver = receive_baseline_correlation_products  # Just to reduce typing
    pcc = cbf.product_controller_client

    pdf_report.step("Configure the D-sim with Gaussian noise.")

    amplitude = 0.2
    await pcc.request("dsim-signals", cbf.dsim_names[0], f"common=wgn({amplitude});common;common;")
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude} on both pols.")

    # Look up input index by label
    input_index = {input_label: i for i, input_label in enumerate(receiver.input_labels)}
    # First and second elements of each baseline, as indices into input_labels
    a_idx = [input_index[bl[0]] for bl in receiver.bls_ordering]
    b_idx = [input_index[bl[1]] for bl in receiver.bls_ordering]
    for start_idx in range(0, receiver.n_bls, receiver.n_chans - 1):
        end_idx = min(start_idx + receiver.n_chans - 1, receiver.n_bls)
        # The last block may be smaller
        pdf_report.step(f"Check baselines {start_idx} to {end_idx - 1}.")

        input_gains = np.zeros((len(input_index), receiver.n_chans), np.float32)

        # determine input gains.
        await pcc.request("gain-all", "antenna-channelised-voltage", "0")
        pdf_report.detail("Compute gains to enable at least one baseline per channel.")
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1  # Avoid channel 0, which is DC so a bit odd
            input_gains[a_idx[i], channel] = 1.0
            input_gains[b_idx[i], channel] = 1.0

        pdf_report.detail("Set gains.")
        for i, channel_gains in enumerate(input_gains.tolist()):
            await pcc.request("gain", "antenna-channelised-voltage", receiver.input_labels[i], *channel_gains)

        # Compute which visibilities have non-zero total gain
        gain_non_zero = input_gains > 0
        expected_non_zero = gain_non_zero[a_idx, :] & gain_non_zero[b_idx, :]

        _, data = await receiver.next_complete_chunk()
        assert data.shape == (receiver.n_chans, receiver.n_bls, 2)

        # confirm the signals are in baselines as expected
        with check:
            pdf_report.detail(
                "Compare output nonzero correlation values to expected antenna gain configuration for this range."
            )
            np.testing.assert_array_equal(
                data[1:, :, 0] > 0,
                expected_non_zero.T[1:, :],
                err_msg="output nonzero correlation values doesn't match the "
                + f"expected antenna gain configuration for channels 1 to {receiver.n_chans}",
            )
