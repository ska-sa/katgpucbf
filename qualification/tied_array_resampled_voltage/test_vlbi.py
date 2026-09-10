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
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter, plot_focus
from katgpucbf.utils import TimeConverter
from qualification.cbf import CBFRemoteControl

from ..recv import TiedArrayChannelisedVoltageReceiver, TiedArrayResampledVoltageReceiver


async def max_retry_test(
    test_procedure: Callable[[int], Awaitable[bool]], max_attempts: int, retry_interval: float
) -> tuple[bool, int]:
    """
    Test a subroutine with a maximum number of attempts and a retry interval.

    Parameters
    ----------
        test_procedure
            An asynchronous subroutine that takes an integer indicating the attempt number
            and returns a boolean `True` if the test passed or `False` if the test failed.
        max_attempts
            The maximum number of attempts.
        retry_interval
           The retry interval in seconds.

    Returns
    -------
        bool
            Whether the test passed.
        int
            Zero-indexed attempt index upon completion.
    """
    loop = asyncio.get_running_loop()
    for attempt_num in range(max_attempts):
        start_time = loop.time()
        if await test_procedure(attempt_num):
            return True, attempt_num
        sleep_period = retry_interval - (loop.time() - start_time)
        await asyncio.sleep(sleep_period)
    return False, attempt_num


@pytest.mark.name("VLBI mean power")
async def test_mean_power(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    cbf: CBFRemoteControl,
    pass_channels: slice,
) -> None:
    """Test mean-power sensor values against tied-array channelised voltage.

    Verification method
    -------------------
    Verified by means of test. Inject a white noise signal and set beam weights
    so that a single antenna contributes to the resampled voltages. Wait until each
    ``mean-power`` sensor timestamp is after the system steady-state timestamp
    (plus one ``power-int-time`` so the averaging window is entirely post-steady-state).
    Measure mean power from the tied-array channelised voltage stream over the
    passband channels, and compare against each ``mean-power`` sensor. The values
    must agree to within 0.5%.
    """
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
    time_converter = TimeConverter(receiver.sync_time, receiver.scale_factor_timestamp)
    steady_state_unix = time_converter.adc_to_unix(await cbf.steady_state_timestamp())
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

    # TODO: See NGC-2099 & 1689: Because the data is zero for the first several seconds,
    # for now just retry with .2 second intervals (sample rate of 5) since steady state is
    # not known beforehand.
    sample_rate = 5  # TODO: NGC-2099 These are arbitrary values to get data for plotting mean power values.
    samples = int(1e2 * sample_rate)
    mean_power_sensor_readings = np.zeros(shape=(len(sensor_names), samples, 2), dtype=np.float64)

    async def wait_mean_power_steady_state(j: int) -> bool:
        for i, name in enumerate(sensor_names):
            reading = await pcc.sensor_reading(name, float)
            mean_power_sensor_readings[i, j, 0] = reading.timestamp
            if reading.status.valid_value():
                mean_power_sensor_readings[i, j, 1] = reading.value
            else:
                mean_power_sensor_readings[i, j, 1] = np.nan

        return bool(
            np.all(mean_power_sensor_readings[:, j, 0] >= min_sensor_time)
            and np.all(mean_power_sensor_readings[:, j, 1] == pytest.approx(tacv_power, rel=5e-3))
        )

    pdf_report.step("Compare mean-power sensors against TACV power.")
    test_passed, total_retries = await max_retry_test(wait_mean_power_steady_state, samples, 1 / sample_rate)
    with check:
        assert test_passed, f"Power does not agree to within 0.5% after {samples} retries."
        assert tacv_power > 0.0

    pdf_report.detail(
        f"Mean power sensor readings from {datetime.fromtimestamp(np.min(mean_power_sensor_readings[:, 0, 0]), UTC)}"
        f" to {datetime.fromtimestamp(np.max(mean_power_sensor_readings[:, total_retries, 0]), UTC)}"
        f" in {total_retries + 1} steps."
    )
    mean_power_sensor_readings[:, :, 0] = mean_power_sensor_readings[:, :, 0] - mean_power_sensor_readings[:, :1, 0]

    fig = Figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel("Mean Power (dB)")
    ax.set_title("Mean Power Sensor Values")
    for i, name in enumerate(sensor_names):
        plot_focus(
            ax,
            slice(0, total_retries + 1),
            mean_power_sensor_readings[i, : total_retries + 1, 0],
            mean_power_sensor_readings[i, : total_retries + 1, 1],
            label=name,
        )
    ax.legend()
    pdf_report.figure(fig)
