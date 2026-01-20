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


@pytest.fixture
def input_index(receive_baseline_correlation_products: BaselineCorrelationProductsReceiver) -> dict[str, int]:
    """A dictionary mapping input names to input indices."""
    inputs = dict[str, int]()
    for _, bl in enumerate(receive_baseline_correlation_products.bls_ordering):
        for ant in bl:
            if ant not in inputs:
                inputs[ant] = len(inputs)
    return inputs


@pytest.fixture
def input_reverse_lookup(input_index: dict[str, int]) -> dict[int, str]:
    """A dictionary mapping input indices to input names."""
    input_reverse_lookup: dict[int, str] = {i: ant for ant, i in input_index.items()}
    return input_reverse_lookup


@pytest.fixture
def baseline_order_lookup(
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
) -> dict[tuple[str, str], int]:
    """A dictionary mapping input pairs to baseline indices."""
    baseline_order_lookup = dict[tuple[str, str], int]()
    for i, bl in enumerate(receive_baseline_correlation_products.bls_ordering):
        baseline_order_lookup[bl] = i
    return baseline_order_lookup


def determine_expected_loud_baselines(
    n_chans: int,
    n_bls: int,
    antenna_gains: np.ndarray,
    input_reverse_lookup: dict[int, str],
    baseline_order_lookup: dict[tuple[str, str], int],
) -> np.ndarray:
    """Determine expected baselines to be loud based on antenna gains.

    Args:
        n_chans: Number of channels
        n_bls: Number of baselines
        antenna_gains: Array of shape (n_inputs, n_chans) with antenna gains
        input_reverse_lookup: Dictionary mapping input index to antenna name
        baseline_order_lookup: Dictionary mapping antenna tuple to baseline index

    Returns
    -------
        Array of shape (n_chans, n_bls) with 1.0 for expected loud baselines
    """
    expected_loud_bls_channels = np.zeros((n_chans, n_bls), np.float32)

    # determine baseline matches
    for channel in range(n_chans):
        # for the nonzero antenna gains, set the expected loud baselines
        assert antenna_gains[:, channel].shape == (len(input_reverse_lookup),)
        nonzero_antennas = np.nonzero(antenna_gains[:, channel])[0]
        # the baselines are all combinations of the nonzero inputs on the same channel
        expected_antenna_indexed_baselines = list(itertools.permutations(nonzero_antennas, 2))
        expected_antenna_indexed_baselines.extend([(ant, ant) for ant in nonzero_antennas])
        for baseline_by_antenna_index in expected_antenna_indexed_baselines:
            antenna_tuple = (
                input_reverse_lookup[baseline_by_antenna_index[0]],
                input_reverse_lookup[baseline_by_antenna_index[1]],
            )
            # we don't permutate all combinations, only antenna pairs on the same antenna are permutated completely
            # ie: ('m800v', 'm800h') and ('m800h', 'm800v') but not ('m800h', 'm801h') and ('m801h', 'm800h')
            if baseline_order_lookup.get(antenna_tuple) is None:
                # if the baseline is not in the dictionary, we need to check the reverse order
                if baseline_order_lookup.get(antenna_tuple[::-1]) is not None:
                    # if the reverse order is in the dictionary, we can use it
                    baseline_index = baseline_order_lookup[
                        input_reverse_lookup[baseline_by_antenna_index[1]],
                        input_reverse_lookup[baseline_by_antenna_index[0]],
                    ]
                    expected_loud_bls_channels[channel, baseline_index] = 1.0
            else:
                baseline_index = baseline_order_lookup[
                    input_reverse_lookup[baseline_by_antenna_index[0]],
                    input_reverse_lookup[baseline_by_antenna_index[1]],
                ]
                expected_loud_bls_channels[channel, baseline_index] = 1.0

    return expected_loud_bls_channels


@pytest.mark.requirements("CBF-REQ-0087,CBF-REQ-0104")
async def test_baseline_correlation_products(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    input_index: dict[str, int],
    input_reverse_lookup: dict[int, str],
    baseline_order_lookup: dict[tuple[str, str], int],
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

    for start_idx in range(0, receiver.n_bls, receiver.n_chans - 1):
        end_idx = min(start_idx + receiver.n_chans - 1, receiver.n_bls)
        # The last block may be smaller
        pdf_report.step(f"Check baselines {start_idx} to {end_idx - 1}.")

        antenna_gains = np.zeros((len(input_index), receiver.n_chans), np.float32)

        # determine antenna gains.
        await pcc.request("gain-all", "antenna-channelised-voltage", "0")
        pdf_report.detail("Compute gains to enable atleast one baseline per channel.")
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1  # Avoid channel 0, which is DC so a bit odd
            antenna_gains[input_index[receiver.bls_ordering[i][0]], channel] = 1.0
            antenna_gains[input_index[receiver.bls_ordering[i][1]], channel] = 1.0

        pdf_report.detail("Set gains.")
        for i, channel_gains in enumerate(antenna_gains.tolist()):
            await pcc.request("gain", "antenna-channelised-voltage", input_reverse_lookup[i], *channel_gains)

        expected_loud_bls_channels = determine_expected_loud_baselines(
            receiver.n_chans,
            receiver.n_bls,
            antenna_gains,
            input_reverse_lookup,
            baseline_order_lookup,
        )

        _, data = await receiver.next_complete_chunk()
        assert data.shape == (receiver.n_chans, receiver.n_bls, 2)

        # confirm the signals are in baselines as expected
        with check:
            loud_bln_index = np.nonzero(data[1:, :, 0])
            expected_loud_bls_index = np.nonzero(expected_loud_bls_channels[1:, :])

            np.testing.assert_array_equal(
                loud_bln_index,
                expected_loud_bls_index,
                err_msg="output nonzero correlation values doesn't match the "
                + f"expected antenna gain configuration for channels 1 to {receiver.n_chans}",
            )
