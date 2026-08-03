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
from collections.abc import Awaitable, Callable

import numpy as np
import pytest

from katgpucbf.pytest_plugins.reporter import Reporter

from ..cbf import CBFRemoteControl
from ..recv import TiedArrayChannelisedVoltageReceiver


@pytest.fixture
def streams_in_use(
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver, capture_stop_streams: list[str]
) -> list[str]:
    """The streams that are both in the receiver and in `capture_stop_streams`."""
    return sorted(set(receive_tied_array_channelised_voltage.stream_names) & set(capture_stop_streams))


@pytest.fixture
def stream_index_under_test(
    streams_in_use: list[str],
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
) -> int:
    """Index of the first stream in `streams_in_use` in the receiver."""
    return receive_tied_array_channelised_voltage.stream_names.index(streams_in_use[0])


async def _test_capture_start(
    cbf: CBFRemoteControl,
    receiver: TiedArrayChannelisedVoltageReceiver,
    streams_in_use: list[str],
    pdf_report: Reporter,
    prepare: Callable[[], Awaitable],
) -> np.ndarray:
    """Implement tests of capture-start sequencing.

    Each test provides a callback that issues a request to the product
    controller. The received data is returned for the test to do
    verification.

    capture-start requests cannot be issued to tied-array-channelised-voltage
    streams that feed tied-array-resampled-voltage (VLBI) streams. `streams_in_use`
    is filtered to exclude such streams.

    The tests are run with the assumption that the first stream in `streams_in_use`
    is the stream under test.
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
    stream_name = streams_in_use[0]
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
        for stream in streams_in_use:
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
    streams_in_use: list[str],
    stream_index_under_test: int,
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
        pdf_report.detail(f"Set beam-quant-gains to 0 on beam {streams_in_use[0]}.")
        await cbf.product_controller_client.request("beam-quant-gains", streams_in_use[0], 0.0)

    data = await _test_capture_start(cbf, receiver, streams_in_use, pdf_report, prepare)
    assert np.all(data[stream_index_under_test] == 0)
    assert (
        np.sum(data[stream_index_under_test + 1] != 0) >= data[stream_index_under_test + 1].size // 2
    )  # Should be mostly non-zero
    pdf_report.detail("Output reflects effects of beam-quant-gains.")


@pytest.mark.name("Ordering of beam-weights and capture-start")
@pytest.mark.no_capture_start
async def test_beam_weights_capture_start(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    streams_in_use: list[str],
    stream_index_under_test: int,
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
        pdf_report.detail(f"Set beam-weights to 0 on beam {streams_in_use[0]}.")
        weights = [0.0] * len(receiver.source_indices[0])
        await cbf.product_controller_client.request("beam-weights", streams_in_use[0], *weights)

    data = await _test_capture_start(cbf, receiver, streams_in_use, pdf_report, prepare)
    assert np.all(data[stream_index_under_test] == 0)
    assert (
        np.sum(data[stream_index_under_test + 1] != 0) >= data[stream_index_under_test + 1].size // 2
    )  # Should be mostly non-zero
    pdf_report.detail("Output reflects effects of beam-weights.")


@pytest.mark.name("Ordering of beam-delays and capture-start")
@pytest.mark.no_capture_start
async def test_beam_delays_capture_start(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    streams_in_use: list[str],
    stream_index_under_test: int,
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
        pdf_report.detail(f"Set beam-delays to phase π on beam {streams_in_use[0]}.")
        delays = [f"0:{np.pi}"] * len(receiver.source_indices[0])
        await cbf.product_controller_client.request("beam-delays", streams_in_use[0], *delays)

    data = await _test_capture_start(cbf, receiver, streams_in_use, pdf_report, prepare)
    assert (
        np.sum(data[stream_index_under_test] != 0) >= data[stream_index_under_test].size // 2
    )  # Should be mostly non-zero
    # We use a data subset two indices over because the immediate neighbour
    # is the other polarisation and so experiences different F-engine dithering.
    reference_stream_index = stream_index_under_test + 2
    if stream_index_under_test >= 2:
        # NOTE: The `receiver` only requires that we have 2 dual-pol beams (4 tacv streams).
        # If streams 0 and 1 are used for VLBI, we might index off the end of the array.
        reference_stream_index = stream_index_under_test - 2
    # The tolerance allows for some rounding error plus dithered quantisation.
    np.testing.assert_allclose(data[stream_index_under_test], -data[reference_stream_index], atol=2)
    pdf_report.detail("Output reflects effects of beam-delays.")
