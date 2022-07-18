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

    # Some helper functions:
    async def zero_all_gains():
        pdf_report.detail("Setting all F-engine gains to zero.")
        await pc_client.request("gain-all", "antenna_channelised_voltage", "0")

    async def unzero_a_baseline(baseline_tuple: Tuple[str, str]):
        pdf_report.detail(f"Unzeroing gain on {baseline_tuple}")
        for inp in baseline_tuple:
            await pc_client.request("gain", "antenna_channelised_voltage", inp, "1")

    for start_idx in range(0, receiver.n_bls, receiver.n_chans - 1):
        end_idx = min(start_idx + receiver.n_chans - 1, receiver.n_bls)
        pdf_report.step(f"Check baselines {start_idx} to {end_idx - 1}.")
        await zero_all_gains()
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
        for i in range(start_idx, end_idx):
            channel = i - start_idx + 1
            bl = receiver.bls_ordering[i]
            loud_bls = np.nonzero(data[channel, :, 0])[0]
            pdf_report.detail(
                f"Checking {bl}: {len(loud_bls)} baseline{'s' if len(loud_bls) != 1 else ''} "
                f"had signal in {'them' if len(loud_bls) != 1 else 'it'}: {loud_bls}"
            )
            expect(
                i in loud_bls,
                f"{bl} ({i}) doesn't show up in the list ({loud_bls})!",
            )
            for loud_bl in loud_bls:
                expect(
                    is_signal_expected_in_baseline(bl, receiver.bls_ordering[loud_bl], pdf_report),
                    "Signal found in unexpected baseline.",
                )


def is_signal_expected_in_baseline(
    expected_bl: Tuple[str, str], loud_bl: Tuple[str, str], pdf_report: Reporter
) -> bool:
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
    if loud_bl == expected_bl:
        pdf_report.detail(f"Signal confirmed in bl {expected_bl} where expected")
        return True
    elif loud_bl == (expected_bl[0], expected_bl[0]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's ant0's autocorrelation.")
        return True
    elif loud_bl == (expected_bl[1], expected_bl[1]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's ant1's autocorrelation.")
        return True
    elif loud_bl == (expected_bl[1], expected_bl[0]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's the conjugate of what we expect.")
        return True
    else:
        pdf_report.detail(f"Signal injected into bl {expected_bl} wasn't expected to show up in {loud_bl}!")
        return False
