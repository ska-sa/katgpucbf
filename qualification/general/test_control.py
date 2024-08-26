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


async def consume_chunks(receiver: XBReceiver, timestamps: list[int]):
    async for timestamp, chunk in receiver.complete_chunks():
        with chunk:
            timestamps.append(timestamp)


async def control_acv_delays(rng: np.random.Generator, cbf: CBFRemoteControl, pdf_report: Reporter, name: str) -> None:
    pcc = cbf.product_controller_client
    n_inputs = len(cbf.config["outputs"][name]["input_labels"])
    delay = rng.uniform(0.0, MAX_DELAY, n_inputs)
    delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE, n_inputs)
    phase = rng.uniform(-np.pi, np.pi, n_inputs)
    phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE, n_inputs)
    target_time = time.time() + DELAY_INTERVAL * 0.5
    async for _ in periodically(rng, DELAY_INTERVAL):
        coeff = [f"{d},{dr}:{p},{pr}" for d, dr, p, pr in zip(delay, delay_rate, phase, phase_rate)]
        elapsed = await measure(pcc.request("delays", name, target_time, *coeff))
        pdf_report.detail(f"Set delays for {name} in {elapsed:.3f} s.")
        with check:
            assert elapsed < 1.0
        # Set next set of delays to be consistent with the old delay and delay_rate
        delay += delay_rate * DELAY_INTERVAL
        phase += phase_rate * DELAY_INTERVAL
        delay = np.clip(delay, 0.0, MAX_DELAY)
        phase = wrap_angle(phase)
        delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE, n_inputs)
        phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE, n_inputs)


async def control_tacv_delays(rng: np.random.Generator, cbf: CBFRemoteControl, pdf_report: Reporter, name: str) -> None:
    pcc = cbf.product_controller_client
    src_stream = cbf.config["outputs"][name]["src_streams"][0]
    n_inputs: int = len(cbf.config["outputs"][src_stream]["input_labels"]) // 2
    async for _ in periodically(rng, BEAM_DELAY_INTERVAL):
        delay = rng.uniform(-BEAM_MAX_DELAY, BEAM_MAX_DELAY, n_inputs)
        phase = rng.uniform(-np.pi, np.pi, n_inputs)
        coeff = [f"{d}:{p}" for d, p in zip(delay, phase)]
        elapsed = await measure(pcc.request("beam-delays", name, *coeff))
        pdf_report.detail(f"Set delays for {name} in {elapsed:.3f} s.")
        with check:
            assert elapsed < 1.0


@pytest.fixture
async def sensor_watcher(cbf: CBFRemoteControl) -> AsyncGenerator[aiokatcp.SensorWatcher, None]:
    # aiokatcp doesn't currently handle adding watchers after the connection
    # is already established; SensorWatcher is also somewhat expensive. So
    # instead we create a separate connection for monitoring sensors.
    secondary = aiokatcp.Client(*cbf.product_controller_endpoint)
    sensor_watcher = aiokatcp.SensorWatcher(secondary)
    secondary.add_sensor_watcher(sensor_watcher)
    await sensor_watcher.synced.wait()
    yield sensor_watcher
    secondary.close()
    await secondary.wait_closed()


# TODO: once requirements spec is finalised, note which requirements
# this corresponds to.
async def test_control(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    sensor_watcher: aiokatcp.SensorWatcher,
) -> None:
    """Test that controlling a correlator doesn't cause data to be lost.

    Verification method
    -------------------
    Verified by means of test. Issue requests to the configured streams as follows:

    - antenna_channelised_voltage: ``?delays`` every 6 seconds, with 3 seconds lead time.
    - tied_array_channelised_voltage: ``?beam-delays`` every 3.4 seconds.

    Check that each request completes in at most 1 second. Receive data from
    the streams over 60s and check that the received chunks span at least 55
    seconds and have no more than one chunk lost.

    Additionally, subscribe to all sensor updates to emulate CAM's load.
    """
    pcc = cbf.product_controller_client
    pdf_report.step("Subscribe to sensors.")
    pdf_report.detail(f"Subscribed to {len(sensor_watcher.sensors)} sensors.")

    pdf_report.step("Set up periodic control.")
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
        tasks.append(tg.create_task(consume_chunks(receive_baseline_correlation_products, timestamps_bcp)))
        tasks.append(tg.create_task(consume_chunks(receive_tied_array_channelised_voltage, timestamps_tacv)))
        pdf_report.step("Run correlator for 60s.")
        await asyncio.sleep(60.0)
        pdf_report.detail("Stop asynchronous tasks.")
        for task in tasks:
            task.cancel()

    pdf_report.step("Check timestamps of received chunks")
    # TODO
