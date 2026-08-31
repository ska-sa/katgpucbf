################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Test for tied-array-resampled-voltage stream."""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime

import aiokatcp
import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter, plot_focus
from katgpucbf.utils import TimeConverter
from qualification.cbf import CBFRemoteControl

from ..recv import TiedArrayChannelisedVoltageReceiver, TiedArrayResampledVoltageReceiver


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


async def max_retry_test(
    test_procedure: Callable[[int], Awaitable[bool]], max_retries: int, retry_interval: float
) -> tuple[bool, int]:
    """
    Test a subroutine with a maximum number of retries and a retry interval.

    Parameters
    ----------
        test_procedure
            A subroutine that takes an integer indicating the attempt number (starting from 1)
            and returns a boolean `True` if the test passed.
        max_retries
            The maximum number of retries.
        retry_interval
           The retry interval in seconds.

    Returns
    -------
         A tuple containing a boolean indicating whether the test passed and the number of attempts.
    """
    for attempt_nr in range(1, max_retries + 1):
        start_time = time.time()
        if await test_procedure(attempt_nr):
            return True, attempt_nr
        sleep_period = retry_interval - (time.time() - start_time)
        await asyncio.sleep(sleep_period)
    return False, attempt_nr


@pytest.mark.name("VLBI mean power")
async def test_mean_power(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver | None,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    cbf: CBFRemoteControl,
    sensor_watcher: aiokatcp.SensorWatcher,
    pass_channels: slice,
) -> None:
    """Test mean-power sensor values against tied-array channelised voltage.

    Verification method
    -------------------
    Verified by means of test. Inject a white noise signal and set beam weights
    so that a single antenna contributes. Wait until each ``mean-power`` sensor
    timestamp is after the system steady-state timestamp (plus one
    ``power-int-time`` so the averaging window is entirely post-steady-state).
    Measure mean power from the tied-array channelised voltage stream over the
    passband channels, and compare against each ``mean-power`` sensor.
    """
    assert receive_tied_array_resampled_voltage is not None
    receiver = receive_tied_array_resampled_voltage
    pcc = cbf.product_controller_client

    pdf_report.step("Set beam weights.")
    async with asyncio.TaskGroup() as tg:
        for i, name in enumerate(receive_tied_array_channelised_voltage.stream_names):
            weights = [0.0] * len(receive_tied_array_channelised_voltage.source_indices[i])
            weights[0] = 1.0
            tg.create_task(pcc.request("beam-weights", name, *weights))

    pdf_report.detail("Beam weights set to use antenna 0.")

    pdf_report.step("Inject white noise signal.")
    dsim_signals = "common=wgn(0.02);common;common;"
    async with asyncio.TaskGroup() as tg:
        for dsim_name in cbf.dsim_names:
            tg.create_task(pcc.request("dsim-signals", dsim_name, dsim_signals))
    pdf_report.detail(f"Set dsim signals to {dsim_signals}.")

    pdf_report.step("Wait for mean-power sensors to reach steady state.")
    await sensor_watcher.synced.wait()  # Implicitly waits for connection too
    time_converter = TimeConverter(receiver.sync_time, receiver.scale_factor_timestamp)
    steady_state_unix = time_converter.adc_to_unix(await cbf.steady_state_timestamp())
    # TODO: See NGC-2099: Because the data is zero for the first several seconds,
    # for now just retry with .2 second intervals (sample rate of 5) since steady state is
    # not known beforehand.
    min_sensor_time = steady_state_unix + receiver.power_int_time

    sensor_names = [
        f"{receiver.stream_names[0]}.{pol}{chan}.mean-power"
        for pol in receiver.pol_ordering
        for chan in range(receiver.n_chans)
    ]

    pdf_report.step("Measure power from tied-array channelised voltage.")
    _, tacv_data = await receive_tied_array_channelised_voltage.next_complete_chunk()
    tacv_data = tacv_data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
    # Only use the pass channels for beam zero for the power calculation.
    tacv_data = tacv_data[0][pass_channels]
    tacv_power = (np.square(tacv_data.real) + np.square(tacv_data.imag)).mean()
    pdf_report.detail(f"Mean TACV power over passband channels: {tacv_power}.")

    sample_rate = 5  # TODO: NGC-2099 These are arbitrary values to get data for plotting mean power values.
    samples = int(1e3 * sample_rate)
    mean_power_sensor_readings = np.zeros(shape=(len(sensor_names), samples, 2), dtype=np.float64)

    async def wait_mean_power_steady_state(j: int) -> bool:
        sample_index = j - 1
        for i, name in enumerate(sensor_names):
            reading = sensor_watcher.sensors[name].reading
            mean_power_sensor_readings[i, sample_index, 0] = reading.timestamp
            mean_power_sensor_readings[i, sample_index, 1] = reading.value

        return bool(
            np.all(mean_power_sensor_readings[:, sample_index, 0] >= min_sensor_time)
            and np.all(mean_power_sensor_readings[:, sample_index, 1] == pytest.approx(tacv_power, rel=5e-3))
        )

    pdf_report.step("Compare mean-power sensors against TACV power.")
    test_passed, total_retries = await max_retry_test(wait_mean_power_steady_state, samples, 1 / sample_rate)
    with check:
        assert test_passed, f"Power does not agree to within 0.5% after {samples} retries."
        assert tacv_power > 0.0

    pdf_report.detail(
        f"Mean power sensor readings from {datetime.fromtimestamp(np.min(mean_power_sensor_readings[:, 0, 0]), UTC)}"
        f" to {datetime.fromtimestamp(np.max(mean_power_sensor_readings[:, total_retries, 0]), UTC)}"
        f" in {total_retries} steps."
    )
    mean_power_sensor_readings[:, :, 0] = mean_power_sensor_readings[:, :, 0] - mean_power_sensor_readings[:, :1, 0]

    fig = Figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel("Mean Power (dBmV)")
    ax.set_title("Mean Power Sensor Values")
    for i, name in enumerate(sensor_names):
        plot_focus(
            ax,
            slice(0, total_retries),
            mean_power_sensor_readings[i, :total_retries, 0],
            mean_power_sensor_readings[i, :total_retries, 1],
            label=name,
        )
    ax.legend()
    pdf_report.figure(fig)
