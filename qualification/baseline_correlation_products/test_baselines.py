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

"""Baseline verification tests."""

import itertools

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
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude} on both poles.")

    antennas = dict[str, int]()
    for _, bl in enumerate(receiver.bls_ordering):
        for ant in bl:
            if ant not in antennas:
                antennas[ant] = len(antennas)

    antenna_index: dict[int, str] = {i: ant for ant, i in antennas.items()}
    baseline_index_from_antennas = dict[tuple[str, str], int]()
    for i, bl in enumerate(receiver.bls_ordering):
        baseline_index_from_antennas[bl] = i

    for start_idx in range(0, receiver.n_bls, receiver.n_chans - 1):  # what are we ranging?
        # = n_bls if n_bls < n_chans and there is only one block,
        # The last block may also be smaller
        end_idx = min(start_idx + receiver.n_chans - 1, receiver.n_bls)
        pdf_report.step(f"Check baselines {start_idx} to {end_idx - 1}.")

        antenna_gains = np.zeros((len(antennas), receiver.n_chans), np.float32)

        # determine antenna gains.
        await pcc.request("gain-all", "antenna-channelised-voltage", "0")
        pdf_report.detail("Compute gains to enable one baseline per channel.")
        expected_loud_bls_channels = np.zeros((receiver.n_chans, receiver.n_bls), np.float32)
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1  # Avoid channel 0, which is DC so a bit odd
            antenna_gains[antennas[receiver.bls_ordering[i][0]], channel] = 1.0
            antenna_gains[antennas[receiver.bls_ordering[i][1]], channel] = 1.0

        # determine baseline matches
        for channel in range(receiver.n_chans):
            # for the nonzero antenna gains, set the expected loud baselines
            assert antenna_gains[:, channel].shape == (len(antennas),)
            nonzero_antennas = np.nonzero(antenna_gains[:, channel])[0]
            # the baselines are all combinations of the nonzero antennas
            expected_antenna_indexed_baselines = list(itertools.permutations(nonzero_antennas, 2))
            print("expected_baseline_indexes:", expected_antenna_indexed_baselines)
            print("autocorrelated baselines:", [(ant, ant) for ant in nonzero_antennas])
            expected_antenna_indexed_baselines.extend([(ant, ant) for ant in nonzero_antennas])
            for baseline_by_antenna_index in expected_antenna_indexed_baselines:
                baseline_index = baseline_index_from_antennas[
                    antenna_index[baseline_by_antenna_index[0]], antenna_index[baseline_by_antenna_index[1]]
                ]
                expected_loud_bls_channels[channel, baseline_index] = 1.0

        pdf_report.detail("Set gains.")
        for i, channel_gains in enumerate(antenna_gains.tolist()):
            await pcc.request("gain", "antenna-channelised-voltage", antenna_index[i], *channel_gains)

        _, data = await receiver.next_complete_chunk()
        assert data.shape == (receiver.n_chans, receiver.n_bls, 2)

        # confirm the signals are in baselines as expected
        with check:
            for i in range(1, receiver.n_chans):
                # which baselines are loud on this channel?
                loud_bln_index = np.nonzero(data[i, :, 0])
                expected_loud_bls_index = np.nonzero(expected_loud_bls_channels[i, :])

                print("channel:", i)
                print("baselines on channel should be:", expected_loud_bls_channels[i, :])
                print("baselines on channel are:", data[i, :, 0])
                print("antenna gains on channel are:", antenna_gains[:, i])
                np.testing.assert_array_equal(
                    loud_bln_index,
                    expected_loud_bls_index,
                    err_msg="output nonzero values doesn't match the "
                    + f"expected antenna gain configuration for channel {i}",
                )
