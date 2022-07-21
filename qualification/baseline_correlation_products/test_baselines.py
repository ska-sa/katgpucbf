################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

from typing import Tuple

import numpy as np
import pytest

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0087,CBF-REQ-0104")
async def test_baseline_correlation_products(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
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
    pdf_report.step("Connect to correlator's product controller to retrieve configuration for the running correlator.")
    pc_client = correlator.product_controller_client

    pdf_report.step("Configure the D-sim with Gaussian noise.")

    amplitude = 0.2
    await correlator.dsim_clients[0].request("signals", f"common=wgn({amplitude});common;common;")
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude} on both pols.")

    for start_idx in range(0, receiver.n_bls, receiver.n_chans - 1):
        end_idx = min(start_idx + receiver.n_chans - 1, receiver.n_bls)
        pdf_report.step(f"Check baselines {start_idx} to {end_idx - 1}.")
        await pc_client.request("gain-all", "antenna_channelised_voltage", "0")
        pdf_report.detail("Compute gains to enable one baseline per channel.")
        gains = {}
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1  # Avoid channel 0, which is DC so a bit odd
            for inp in receiver.bls_ordering[i]:
                if inp not in gains:
                    gains[inp] = np.zeros(receiver.n_chans, np.float32)
                gains[inp][channel] = 1.0
        pdf_report.detail("Set gains.")
        for inp, g in gains.items():
            await pc_client.request("gain", "antenna_channelised_voltage", inp, *g)

        _, data = await receiver.next_complete_chunk()
        everything_is_awesome = True
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1
            bl = receiver.bls_ordering[i]
            loud_bls = np.nonzero(data[channel, :, 0])[0]
            # Check that the baseline actually appears in the list.
            appears = i in loud_bls
            expect(appears, f"{bl} ({i}) doesn't show up in the list ({loud_bls})!")
            # Check that no unexpected baselines have signal.
            no_unexpected = all(
                is_signal_expected_in_baseline(bl, receiver.bls_ordering[loud_bl]) for loud_bl in loud_bls
            )
            expect(no_unexpected, "Signal found in unexpected baseline.")
            if not (appears and no_unexpected):
                everything_is_awesome = False
        pdf_report.detail(
            "All baselines in this range correct." if everything_is_awesome else "Errors detected in this range."
        )


def is_signal_expected_in_baseline(expected_bl: Tuple[str, str], loud_bl: Tuple[str, str]) -> bool:
    """Check whether signal is expected in the loud baseline, given which one had a test signal injected.

    It isn't possible in the general case to get signal in only a single
    baseline. There will be auto-correlations, and the conjugate correlations
    which will show signal as well.

    Parameters
    ----------
    expected_bl
        A tuple of the form ("m801h", "m802v") indicating which baseline we are
        checking.
    loud_bl
        A baseline where signal has been detected.

    Returns
    -------
    bool
        Indication of whether signal is expected, i.e. whether the test can pass.
    """
    return loud_bl[0] in expected_bl and loud_bl[1] in expected_bl
