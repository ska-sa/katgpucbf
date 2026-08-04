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
import logging
from collections.abc import AsyncGenerator

import aiokatcp
import numpy as np
import pytest
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter
from qualification.cbf import CBFRemoteControl

from ..recv import TiedArrayChannelisedVoltageReceiver, TiedArrayResampledVoltageReceiver

logger = logging.getLogger(__name__)


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


@pytest.mark.name("VLBI VDIF output")
async def test_vlbi_vdif(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver | None,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    cbf: CBFRemoteControl,
    sensor_watcher: aiokatcp.SensorWatcher,
    pass_channels: slice,
) -> None:
    """Test VDIF frame output.

    Verification method
    -------------------
    Verified by means of test.
    Collect a valid VDIF frameset.
    """
    assert receive_tied_array_resampled_voltage is not None
    pcc = cbf.product_controller_client

    async with asyncio.TaskGroup() as tg:
        for i, name in enumerate(receive_tied_array_channelised_voltage.stream_names):
            gains = [0.0] * len(receive_tied_array_channelised_voltage.source_indices[i])
            gains[0] = 1.0
            tg.create_task(pcc.request("beam-weights", name, *gains))

    pdf_report.step("Setup signal generator and gains.")
    pdf_report.detail(f"Set beam weights to {gains}.")

    async with asyncio.TaskGroup() as tg:
        for dsim_name in cbf.dsim_names:
            tg.create_task(pcc.request("dsim-signals", dsim_name, "common=wgn(0.0165);common;common;"))
    pdf_report.detail("Set dsim signals white noise.")

    receiver = receive_tied_array_resampled_voltage
    pdf_report.step("Collect a valid VDIF frameset.")
    frameset, _ = await receiver.next_complete_frameset()
    pdf_report.detail("Verify we have `n_chans * len(pol_ordering)` threads in the set.")
    with check:
        assert len(frameset.seq_ids) == receiver.n_chans * len(receiver.pol_ordering)

    pdf_report.step("Verify power average calculation.")
    # wait for steady state
    int_time = receiver.power_int_time
    await asyncio.sleep(int_time + 1 + 5)
    await sensor_watcher.synced.wait()  # Implicitly waits for connection too
    _, tacv_data = await receive_tied_array_channelised_voltage.next_complete_chunk()
    logger.error(f"TACV data retrieved with shape: {tacv_data.shape}")
    tacv_data = tacv_data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
    # Only use the pass channels for beam zero for the power calculation.
    tacv_data = tacv_data[0][pass_channels]
    logger.error(f"TACV data with pass channels retrieved with shape: {tacv_data.shape}")
    sum_tacv_power = (np.square(tacv_data.real) + np.square(tacv_data.imag)).mean()
    logger.error(f"Sum TACV power: {sum_tacv_power}")

    # """Get the mean power for a given polarization in volts^2."""
    for pol_ordering in receiver.pol_ordering:
        for chan in range(receiver.n_chans):
            o2 = sensor_watcher.sensors[f"{receiver.stream_names[0]}.{pol_ordering}{chan}.mean-power"].value
            with check:
                assert pytest.approx(sum_tacv_power, rel=5e-3) == o2, (
                    f"TACV power ^2: {sum_tacv_power} does not match total theta^2: {o2}"
                    + f" for polarization {pol_ordering} and channel {chan}"
                )
    # Test that we aren't accidentally testing zero values:
    assert sum_tacv_power > 0.0
