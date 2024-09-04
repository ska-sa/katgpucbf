################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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

"""Test that controlling a correlator doesn't cause data to be lost."""

import asyncio
import math
import time
from collections.abc import AsyncGenerator, Awaitable

import aiokatcp
import numpy as np
import pytest
from pytest_check import check

from katgpucbf.fgpu.delay import wrap_angle

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver, TiedArrayChannelisedVoltageReceiver, XBReceiver
from ..reporter import Reporter

#: Time to run the test for (in seconds)
TEST_TIME = 30.0
#: Tolerance for amount of data received, compared to TEST_TIME (seconds)
TEST_TIME_TOL = 10.0
# These magic values are from MK+ Delay Tracking Requirements draft revision E
DELAY_INTERVAL = 6.0
MAX_DELAY = 80e-6
MAX_DELAY_RATE = 2.56e-9
MAX_PHASE_RATE = 49.22
# From M0000-0000-096 rev 1
BEAM_DELAY_INTERVAL = 3.4
BEAM_MAX_DELAY = 509e-9


async def measure(awaitable: Awaitable) -> float:
    """Time how long an awaitable takes to await."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    await awaitable
    return loop.time() - start


async def periodically(rng: np.random.Generator, interval: float) -> AsyncGenerator[float, None]:
    """Yield periodically.

    The first yield happens after a random delay in [0, `interval`) seconds,
    after which each subsequent yields happens every `interval` seconds.

    This raises an :exc:`AssertionError` if the next scheduled time is missed
    (already in the past).
    """
    loop = asyncio.get_event_loop()
    await asyncio.sleep(rng.uniform(0, interval))
    start = loop.time()
    target = start
    while True:
        yield target
        target += interval
        now = loop.time()
        assert now <= target
        await asyncio.sleep(target - now)


async def consume_chunks(receiver: XBReceiver, timestamps: list[int]) -> None:
    """Consume chunks from a receiver until it is closed.

    The timestamps of the complete chunks are appended to `timestamps`.
    """
    max_delay = math.ceil(MAX_DELAY * receiver.scale_factor_timestamp)
    receiver.start()
    async for timestamp, chunk in receiver.complete_chunks(max_delay=max_delay):
        with chunk:
            timestamps.append(timestamp)


async def control_acv_delays(rng: np.random.Generator, cbf: CBFRemoteControl, pdf_report: Reporter, name: str) -> None:
    """Periodically change the delays of an antenna-channelised-voltage stream.

    The delays are set DELAY_INTERVAL/2 in advance, and are chosen to avoid
    large step changes at the boundaries.

    Raises
    ------
    AssertionError
        if it takes more than 1s to set the delays
    """
    pcc = cbf.product_controller_client
    n_inputs = len(cbf.config["outputs"][name]["input_labels"])
    delay = rng.uniform(0.0, MAX_DELAY, n_inputs)
    delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE, n_inputs)
    phase = rng.uniform(-np.pi, np.pi, n_inputs)
    phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE, n_inputs)
    target_time = time.time() + DELAY_INTERVAL * 0.5
    all_elapsed = []
    try:
        async for _ in periodically(rng, DELAY_INTERVAL):
            coeff = [f"{d},{dr}:{p},{pr}" for d, dr, p, pr in zip(delay, delay_rate, phase, phase_rate)]
            elapsed = await measure(pcc.request("delays", name, target_time, *coeff))
            pdf_report.detail(f"Set delays for {name} in {elapsed:.3f} s.")
            with check:
                assert elapsed < 1.0
            all_elapsed.append(elapsed)
            # Set next set of delays to be consistent with the old delay and delay_rate
            delay += delay_rate * DELAY_INTERVAL
            phase += phase_rate * DELAY_INTERVAL
            delay = np.clip(delay, 0.0, MAX_DELAY)
            phase = wrap_angle(phase)
            delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE, n_inputs)
            phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE, n_inputs)
    except asyncio.CancelledError:
        mean_elapsed = np.mean(all_elapsed)
        pdf_report.detail(f"Average delay-setting time for {name}: {mean_elapsed:.3f} s.")
        raise


async def control_tacv_delays(rng: np.random.Generator, cbf: CBFRemoteControl, pdf_report: Reporter, name: str) -> None:
    """Periodically change the delays of a tied-array-channelised-voltagestream.

    Unlike :func:`control_tacv_delays`, there is no attempt to smooth the
    delays. The beamformer doesn't use coarse delays so it is not sensitive
    to large changes.

    Raises
    ------
    AssertionError
        if it takes more than 1s to set the delays
    """
    pcc = cbf.product_controller_client
    src_stream = cbf.config["outputs"][name]["src_streams"][0]
    n_inputs: int = len(cbf.config["outputs"][src_stream]["input_labels"]) // 2
    all_elapsed = []
    try:
        async for _ in periodically(rng, BEAM_DELAY_INTERVAL):
            delay = rng.uniform(-BEAM_MAX_DELAY, BEAM_MAX_DELAY, n_inputs)
            phase = rng.uniform(-np.pi, np.pi, n_inputs)
            coeff = [f"{d}:{p}" for d, p in zip(delay, phase)]
            elapsed = await measure(pcc.request("beam-delays", name, *coeff))
            pdf_report.detail(f"Set delays for {name} in {elapsed:.3f} s.")
            with check:
                assert elapsed < 1.0
            all_elapsed.append(elapsed)
    except asyncio.CancelledError:
        mean_elapsed = np.mean(all_elapsed)
        pdf_report.detail(f"Average delay-setting time for {name}: {mean_elapsed:.3f} s.")
        raise


@pytest.fixture
async def sensor_watcher(cbf: CBFRemoteControl) -> AsyncGenerator[aiokatcp.SensorWatcher, None]:
    """Establish a secondary connection to the product controller with a sensor watcher.

    The yielded sensor watcher is not yet synchronised.
    """
    # aiokatcp doesn't currently handle adding watchers after the connection
    # is already established; SensorWatcher is also somewhat expensive. So
    # instead we create a separate connection for monitoring sensors.
    secondary = aiokatcp.Client(*cbf.product_controller_endpoint)
    sensor_watcher = aiokatcp.SensorWatcher(secondary)
    secondary.add_sensor_watcher(sensor_watcher)

    yield sensor_watcher

    secondary.close()
    await secondary.wait_closed()


def check_timestamps(
    name: str,
    receiver: XBReceiver,
    pdf_report: Reporter,
    timestamps: list[int],
):
    """Validate the timestamps generated by :func:`consume_chunks`.

    The timestamps must span at least TEST_TIME - TEST_TIME_TOL seconds, and
    must have at most two missing chunks.
    """
    assert timestamps, f"No {name} chunks received"
    elapsed = (timestamps[-1] - timestamps[0]) / receiver.scale_factor_timestamp
    pdf_report.detail(f"{name}: received data over {elapsed:.3f}s.")
    min_time = TEST_TIME - TEST_TIME_TOL
    with check:
        assert elapsed >= min_time, f"Less than {min_time}s of data received for {name}"
    expected = (timestamps[-1] - timestamps[0]) // receiver.timestamp_step + 1
    missing = expected - len(timestamps)
    pdf_report.detail(f"{name}: missed {missing} of {expected} chunks.")
    if missing > 0:
        last_missing = -1
        ptr = 0
        for i in range(expected):
            t = timestamps[0] + i * receiver.timestamp_step
            while ptr < len(timestamps) and timestamps[ptr] < t:
                ptr += 1
            if ptr == len(timestamps) or timestamps[ptr] > t:
                last_missing = i
        last_missing_time = last_missing * receiver.timestamp_step / receiver.scale_factor_timestamp
        pdf_report.detail(f"Last missing chunk is #{last_missing} ({last_missing_time:.6f} s after start).")
    with check:
        assert missing <= 2, f"{missing} of {expected} chunks missing for {name}"


# TODO: once requirements spec is finalised, note which requirements
# this corresponds to.
@pytest.mark.xfail(reason="Not 100% reliable yet (NGC-1265)")
async def test_control(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products_manual_start: BaselineCorrelationProductsReceiver,
    receive_tied_array_channelised_voltage_manual_start: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    sensor_watcher: aiokatcp.SensorWatcher,
) -> None:
    # The docstring is re-assigned after the function so that it can use an
    # f-string.
    pdf_report.step("Subscribe to sensors.")
    await sensor_watcher.synced.wait()
    pdf_report.detail(f"Subscribed to {len(sensor_watcher.sensors)} sensors.")

    pdf_report.step("Set up periodic control.")
    # We don't actually care about the gains, but setting it updates the
    # steady-state timestamp, and ensures that consume_chunks sees only
    # "fresh" data.
    await cbf.product_controller_client.request("gain-all", "antenna-channelised-voltage", "default")
    rng = np.random.default_rng(seed=123)
    async with asyncio.TaskGroup() as tg:
        tasks = []
        for name, output in cbf.config["outputs"].items():
            match output["type"]:
                case "gpucbf.antenna_channelised_voltage":
                    tasks.append(tg.create_task(control_acv_delays(rng, cbf, pdf_report, name)))
                case "gpucbf.tied_array_channelised_voltage":
                    tasks.append(tg.create_task(control_tacv_delays(rng, cbf, pdf_report, name)))
        timestamps_bcp: list[int] = []
        timestamps_tacv: list[int] = []
        tasks.append(tg.create_task(consume_chunks(receive_baseline_correlation_products_manual_start, timestamps_bcp)))
        tasks.append(
            tg.create_task(consume_chunks(receive_tied_array_channelised_voltage_manual_start, timestamps_tacv))
        )
        pdf_report.step(f"Run correlator for {TEST_TIME}s.")
        await asyncio.sleep(TEST_TIME)
        pdf_report.detail("Stop asynchronous tasks.")
        for task in tasks:
            task.cancel()

    pdf_report.step("Check timestamps of received chunks")
    check_timestamps(
        "baseline_correlation_products", receive_baseline_correlation_products_manual_start, pdf_report, timestamps_bcp
    )
    check_timestamps(
        "tied_array_channelised_voltage",
        receive_tied_array_channelised_voltage_manual_start,
        pdf_report,
        timestamps_tacv,
    )


test_control.__doc__ = f"""Test that controlling a correlator doesn't cause data to be lost.

    Verification method
    -------------------
    Verified by means of test. Issue requests to the configured streams as follows:

    - antenna_channelised_voltage: ``?delays`` every 6 seconds, with 3 seconds lead time.
    - tied_array_channelised_voltage: ``?beam-delays`` every 3.4 seconds.

    Check that each request completes in at most 1 second. Receive data from
    the streams over {TEST_TIME}s and check that the received chunks span at
    least {TEST_TIME - TEST_TIME_TOL} seconds and have no more than two chunks
    lost.

    Additionally, subscribe to all sensor updates to emulate CAM's load.
    """
