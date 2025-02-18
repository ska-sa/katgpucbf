################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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

"""Test capture-start following state change requests."""

import asyncio
from typing import Awaitable, Callable

import numpy as np
import pytest

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


async def _test_capture_start(
    cbf: CBFRemoteControl,
    receiver: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    prepare: Callable[[], Awaitable],
) -> np.ndarray:
    """Implementation for tests of capture-start sequencing.

    Each test provides a callback that issues a request to the product
    controller. The received data is returned for the test to do
    verification.
    """
    pcc = cbf.product_controller_client

    # Note: because the beam quant gains are not adjusted, this will easily
    # saturate with large numbers of antennas, but that doesn't affect the
    # validity of any of the tests that use this function.
    pdf_report.step("Inject white noise signal.")
    signals = "common = nodither(wgn(0.1)); common; common;"
    await pcc.request("dsim-signals", cbf.dsim_names[0], signals)
    dsim_timestamp = await pcc.sensor_value(f"{cbf.dsim_names[0]}.steady-state-timestamp", int)
    pdf_report.detail(f"Set dsim signals to {signals}, starting with timestamp {dsim_timestamp}.")

    pdf_report.step("Wait for injected signal to reach XB-engines Tx.")
    # Only need to query one stream, since it's the same engine backing
    # all of them.
    stream_name = receiver.stream_names[0]
    for _ in range(30):
        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i in range(receiver.n_bengs):
                tasks.append(tg.create_task(pcc.sensor_value(f"{stream_name}.{i}.tx.next-timestamp")))
        min_timestamp = min(task.result() for task in tasks)
        pdf_report.detail(f"minimum tx.next-timestamp = {min_timestamp}.")
        if min_timestamp > dsim_timestamp:
            break
        else:
            pdf_report.detail("Sleep for 0.5s.")
            await asyncio.sleep(0.5)
    else:
        pytest.fail("Digitiser signal did not reach XB-engines Tx in time.")

    await prepare()

    pdf_report.step("Capture and verify output")
    async with asyncio.TaskGroup() as tg:
        for stream in receiver.stream_names:
            tg.create_task(pcc.request("capture-start", stream))
    # We use dsim_timestamp as a minimum to ensure that we're not receiving
    # data from a *previous* capture-start/stop.
    _, data = await receiver.next_complete_chunk(min_timestamp=dsim_timestamp)
    return data


@pytest.mark.name("Ordering of beam-quant-gains and capture-start")
@pytest.mark.no_capture_start
async def test_beam_quant_gains_capture_start(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that beam-quant-gains issued before capture-start is not delayed.

    Verification method
    -------------------
    Verified by test. Issue a request, then immediately issue a capture-start
    request. Verify that the received data reflects the change.
    """
    receiver = receive_tied_array_channelised_voltage

    async def prepare() -> None:
        pdf_report.step("Send request.")
        pdf_report.detail("Set beam-quant-gains to 0 on first beam.")
        await cbf.product_controller_client.request("beam-quant-gains", receiver.stream_names[0], 0.0)

    data = await _test_capture_start(cbf, receiver, pdf_report, prepare)
    assert np.all(data[0] == 0)
    assert np.sum(data[1] != 0) >= data[1].size // 2  # Should be mostly non-zero
    pdf_report.detail("Output reflects effects of beam-quant-gains.")


@pytest.mark.name("Ordering of beam-weights and capture-start")
@pytest.mark.no_capture_start
async def test_beam_weights_capture_start(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that beam-weights issued before capture-start is not delayed.

    Verification method
    -------------------
    Verified by test. Issue a request, then immediately issue a capture-start
    request. Verify that the received data reflects the change.
    """
    receiver = receive_tied_array_channelised_voltage

    async def prepare() -> None:
        pdf_report.step("Send request.")
        pdf_report.detail("Set beam-weights to 0 on first beam.")
        weights = [0.0] * len(receiver.source_indices[0])
        await cbf.product_controller_client.request("beam-weights", receiver.stream_names[0], *weights)

    data = await _test_capture_start(cbf, receiver, pdf_report, prepare)
    assert np.all(data[0] == 0)
    assert np.sum(data[1] != 0) >= data[1].size // 2  # Should be mostly non-zero
    pdf_report.detail("Output reflects effects of beam-weights.")


@pytest.mark.name("Ordering of beam-delays and capture-start")
@pytest.mark.no_capture_start
async def test_beam_delays_capture_start(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that beam-delays issued before capture-start is not delayed.

    Verification method
    -------------------
    Verified by test. Issue a request, then immediately issue a capture-start
    request. Verify that the received data reflects the change.
    """
    receiver = receive_tied_array_channelised_voltage

    async def prepare() -> None:
        pdf_report.step("Send request.")
        pdf_report.detail("Set beam-delays to phase π on first beam.")
        delays = [f"0:{np.pi}"] * len(receiver.source_indices[0])
        await cbf.product_controller_client.request("beam-delays", receiver.stream_names[0], *delays)

    data = await _test_capture_start(cbf, receiver, pdf_report, prepare)
    assert np.sum(data[0] != 0) >= data[0].size // 2  # Should be mostly non-zero
    # We use data[2] instead of data[1], because data[1] is the other
    # polarisation and so experiences different F-engine dithering. The
    # tolerance allows for some rounding error plus dithered quantisation.
    np.testing.assert_allclose(data[0], -data[2], atol=2)
    pdf_report.detail("Output reflects effects of beam-delays.")
