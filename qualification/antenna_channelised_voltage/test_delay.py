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

"""Delay and phase compensation tests."""

import math
from typing import List

import aiokatcp
import numpy as np
import pytest

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter


async def test_delay_application_time(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    """Test that delay/phase changes are applied at the correct time.

    Requirements verified:

    CBF-REQ-0077
        The CBF shall delay execution of Continuous Parameter Control commands until
        a UTC time, as received on the CAM interface, with an execution time
        accuracy of <= 10 ms, provided the command is received at least 200ms
        before the execution time, and the execution time delay is no more than
        2 seconds.

    Verification method:

    Verification by means of test. A 90 degree phase change is loaded for one
    polarisation at a chosen time. The accumulation spanning the load time is
    checked to ensure that it contains the appropriate ratio of real and
    imaginary component.
    """
    receiver = receive_baseline_correlation_products

    pdf_report.step("Inject correlated white noise signal")
    pdf_report.detail("Setting signal")
    await correlator.dsim_clients[0].request("signal", "common=wgn(0.1); common; common;")
    pdf_report.detail("Waiting for updated signal to propagate through the pipeline")
    _, chunk = await receiver.next_complete_chunk()
    receiver.stream.add_free_chunk(chunk)

    attempts = 5
    advance = 0.2
    accs: List[np.ndarray] = []
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    for attempt in range(attempts):
        pdf_report.step(f"Set delay {advance * 1000:d}ms in the future (attempt {attempt + 1} / {attempts})")
        pdf_report.detail("Get current time according to the dsim")
        now = aiokatcp.decode(float, (await correlator.dsim_clients[0].request("time"))[0][0])
        target = now + advance
        delays = ["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants
        pdf_report.detail("Set delays")
        await correlator.product_controller_client.request("delays", target, *delays)
        pdf_report.step("Receive data for the corresponding dump plus one either side")
        target_ts = receiver.unix_to_timestamp(target)
        target_acc_ts = target_ts // receiver.timestamp_step * receiver.timestamp_step
        accs = []
        async for timestamp, chunk in receiver.complete_chunks(max_delay=0):
            if target_acc_ts - receiver.timestamp_step <= timestamp <= target_acc_ts + receiver.timestamp_step:
                assert isinstance(chunk.data, np.ndarray)
                accs.append(np.sum(chunk.data[:, bl_idx, :], axis=0))  # Sum over channels
            receiver.stream.add_free_chunk(chunk)
            if timestamp > target_acc_ts + receiver.timestamp_step:
                break
        if len(accs) == 3:
            break

        pdf_report.detail("Didn't receive all the expected chunks, reseting delay and trying again")
        delays = ["0,0:0,0", "0,0:0,0"] * receiver.n_ants
        await correlator.product_controller_client.request("delays", 0, *delays)
    else:
        pytest.fail(f"Giving up after {attempts} attempts")

    pdf_report.step("Check the received data")
    # First chunk won't be perfectly real due to uncorrelated dithering.
    pdf_report.detail("Check that accumulation prior to load time had no phase compensation")
    expect(abs(accs[0][1]) < 1e-5 * abs(accs[0][0]))
    # Similarly last chunk should be almost but not exactly imaginary.
    pdf_report.detail("Check that accumulation after load time had phase compensation")
    expect(abs(accs[2][0]) < 1e-5 * abs(accs[2][1]))
    # Estimate time at which delay was applied based on real:imaginary
    total = np.sum(np.abs(accs[1]))
    load_frac = abs(accs[1][1]) / total  # Load time as fraction of the accumulation
    load_time = receiver.timestamp_to_unix(target_acc_ts) + load_frac * receiver.int_time
    delta = load_time - target
    pdf_report.detail(f"Estimated load time error: {delta * 1000:.3f}ms")
    expect(delta < 0.01)
