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

"""Sample test for tied-array-resampled-voltage stream."""

import asyncio
from collections.abc import AsyncGenerator

import aiokatcp
import numpy as np
import pytest
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter
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


@pytest.mark.name("VLBI VDIF output")
async def test_vlbi_vdif(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver | None,
) -> None:
    """Test VDIF frame output.

    Verification method
    -------------------
    Verified by means of test.
    Collect a valid VDIF frameset and check that it contains one thread per
    channel and polarisation.
    """
    assert receive_tied_array_resampled_voltage is not None
    receiver = receive_tied_array_resampled_voltage
    pdf_report.step("Collect a valid VDIF frameset.")
    frameset, _ = await receiver.next_complete_frameset()
    expected_threads = receiver.n_chans * len(receiver.pol_ordering)
    pdf_report.detail(f"Received frameset with {len(frameset.seq_ids)} threads; expected {expected_threads}.")
    with check:
        assert len(frameset.seq_ids) == expected_threads
    pdf_report.detail("Frameset contains `n_chans * len(pol_ordering)` threads.")


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

    pdf_report.step("Setup signal generator and gains.")
    async with asyncio.TaskGroup() as tg:
        for i, name in enumerate(receive_tied_array_channelised_voltage.stream_names):
            gains = [0.0] * len(receive_tied_array_channelised_voltage.source_indices[i])
            gains[0] = 1.0
            tg.create_task(pcc.request("beam-weights", name, *gains))
    pdf_report.detail(f"Set beam weights to {gains}.")

    async with asyncio.TaskGroup() as tg:
        for dsim_name in cbf.dsim_names:
            tg.create_task(pcc.request("dsim-signals", dsim_name, "common=wgn(0.02);common;common;"))
    pdf_report.detail("Set dsim signals white noise.")

    pdf_report.step("Wait for mean-power sensors to reach steady state.")
    await sensor_watcher.synced.wait()  # Implicitly waits for connection too
    time_converter = TimeConverter(receiver.sync_time, receiver.scale_factor_timestamp)
    steady_state_unix = time_converter.adc_to_unix(await cbf.steady_state_timestamp())
    # Require a full power-int-time of data after steady state so the sensor
    # average does not include pre-change samples.
    min_sensor_time = steady_state_unix + receiver.power_int_time

    sensor_names = [
        f"{receiver.stream_names[0]}.{pol}{chan}.mean-power"
        for pol in receiver.pol_ordering
        for chan in range(receiver.n_chans)
    ]

    async def wait_mean_power_steady_state() -> None:
        while True:
            timestamps = [sensor_watcher.sensors[name].timestamp for name in sensor_names]
            earliest = min(timestamps)
            if earliest >= min_sensor_time:
                pdf_report.detail("Mean-power sensors reached steady state timestamp.")
                break
            await asyncio.sleep(0.5)

    await asyncio.wait_for(asyncio.create_task(wait_mean_power_steady_state()), timeout=15.0)

    pdf_report.step("Measure power from tied-array channelised voltage.")
    _, tacv_data = await receive_tied_array_channelised_voltage.next_complete_chunk()
    tacv_data = tacv_data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
    # Only use the pass channels for beam zero for the power calculation.
    tacv_data = tacv_data[0][pass_channels]
    tacv_power = (np.square(tacv_data.real) + np.square(tacv_data.imag)).mean()
    pdf_report.detail(f"Mean TACV power over passband channels: {tacv_power}.")
    # Test that we aren't accidentally testing zero values:
    assert tacv_power > 0.0

    pdf_report.step("Compare mean-power sensors against TACV power.")
    for sensor_name in sensor_names:
        sensor = sensor_watcher.sensors[sensor_name]
        with check:
            assert sensor.timestamp >= min_sensor_time
            assert sensor.value == pytest.approx(tacv_power, rel=5e-3), (
                f"TACV power ^2: {tacv_power} does not match total theta^2: {sensor.value}"
                + f" for sensor {sensor_name}"
            )
    pdf_report.detail("Power agrees to within 0.5%.")
