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
from typing import Optional

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
    polarisation at a chosen time. The actual application time is estimated by
    checking the ratio of real to imaginary components in the corresponding
    accumulation.
    """
    receiver = receive_baseline_correlation_products

    pdf_report.step("Inject correlated white noise signal")
    pdf_report.detail("Setting signal")
    await correlator.dsim_clients[0].request("signals", "common=wgn(0.1); common; common;")
    pdf_report.detail("Waiting for updated signal to propagate through the pipeline")
    _, chunk = await receiver.next_complete_chunk()
    receiver.stream.add_free_chunk(chunk)

    attempts = 5
    advance = 0.2
    acc: Optional[np.ndarray] = None
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    for attempt in range(attempts):
        pdf_report.step(f"Set delay {advance * 1000:.0f}ms in the future (attempt {attempt + 1} / {attempts})")
        pdf_report.detail("Get current time according to the dsim")
        now = aiokatcp.decode(float, (await correlator.dsim_clients[0].request("time"))[0][0])
        target = now + advance
        delays = ["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants
        pdf_report.detail("Set delays")
        await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", target, *delays)
        pdf_report.step("Receive data for the corresponding dump")
        target_ts = receiver.unix_to_timestamp(target)
        target_acc_ts = target_ts // receiver.timestamp_step * receiver.timestamp_step
        acc = None
        async for timestamp, chunk in receiver.complete_chunks(max_delay=0):
            pdf_report.detail(f"Received chunk with timestamp {timestamp}, target is {target_acc_ts}")
            assert isinstance(chunk.data, np.ndarray)  # Keeps mypy happy
            total = np.sum(chunk.data[:, bl_idx, :], axis=0)  # Sum over channels
            receiver.stream.add_free_chunk(chunk)
            if timestamp == target_acc_ts:
                acc = total
            if timestamp >= target_acc_ts:
                break
        if acc is not None:
            break

        pdf_report.detail("Didn't receive all the expected chunks, reseting delay and trying again")
        delays = ["0,0:0,0", "0,0:0,0"] * receiver.n_ants
        await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", 0, *delays)
    else:
        pytest.fail(f"Giving up after {attempts} attempts")

    pdf_report.step("Check the received data")
    # Estimate time at which delay was applied based on real:imaginary
    total = np.sum(np.abs(acc))
    load_frac = abs(acc[0]) / total  # Load time as fraction of the accumulation
    load_time = receiver.timestamp_to_unix(target_acc_ts) + load_frac * receiver.int_time
    delta = load_time - target
    pdf_report.detail(f"Estimated load time error: {delta * 1000:.3f}ms")
    expect(delta < 0.01)
