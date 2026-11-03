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
import io
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf.meerkat import BANDS
from katgpucbf.pytest_plugins.reporter import Reporter, plot_focus
from katgpucbf.utils import TimeConverter
from qualification.cbf import CBFRemoteControl
from qualification.types import AsyncRunner
from test.qualification.test_recv import BasebandVDIFFrame

from ..recv import TiedArrayChannelisedVoltageReceiver, TiedArrayResampledVoltageReceiver

FRAMESETS_TO_CAPTURE = 10
TONE_SPACING = 100e6


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
        Zero-indexed attempt (whether successful or not).
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
    run_async: AsyncRunner,
) -> None:
    """Test mean-power sensor values against tied-array channelised voltage.

    Verification method
    -------------------
    Verified by means of test. Inject a white noise signal on the v polarity dsim stream
    and set beam weights so that a single antenna contributes to the resampled voltages.
    Wait until each ``mean-power`` sensor timestamp is after the system steady-state timestamp
    (plus one ``power-int-time`` so the averaging window is entirely post-steady-state).
    Measure mean power from the tied-array channelised voltage stream v polarity over the
    passband channels, and compare against x polarity ``mean-power`` sensor.
    The values must agree to within 0.5%. During the test the mean power for the y polarity
    is verified to be zero.
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
    dsim_signals = "0;wgn(0.02);"  # [h, v] pol ordering
    async with asyncio.TaskGroup() as tg:
        for dsim_name in cbf.dsim_names:
            tg.create_task(pcc.request("dsim-signals", dsim_name, dsim_signals))
    pdf_report.detail(f"Set dsim signals to {dsim_signals}.")

    pdf_report.step("Wait for mean-power sensors to reach steady state.")
    time_converter = TimeConverter(receiver.sync_time, receiver.scale_factor_timestamp)
    steady_state_unix = time_converter.adc_to_unix(await cbf.steady_state_timestamp())
    min_sensor_time = steady_state_unix + receiver.power_int_time

    sensor_names = [f"{receiver.stream_name}.x{chan}.mean-power" for chan in range(receiver.n_chans)]

    pdf_report.step("Measure power from tied-array channelised voltage.")
    _, tacv_data = await receive_tied_array_channelised_voltage.next_complete_chunk_data()
    tacv_data = tacv_data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
    # Only use the passband channels for beam zero for the power calculation and the first beam's `v polarity` data.
    tacv_data = tacv_data[1][pass_channels]
    tacv_power = (np.square(tacv_data.real) + np.square(tacv_data.imag)).mean()
    pdf_report.detail(f"Mean TACV power over passband channels: {tacv_power}.")

    # TODO: See NGC-2099 & 1689: Because the data is zero for the first several seconds,
    # for now just retry with .2 second intervals (sample rate of 5) since steady state is
    # not known beforehand.
    sample_rate = 5
    seconds = 5
    samples = seconds * sample_rate
    mean_power_sensor_timestamps = np.zeros((len(sensor_names), samples), float)
    mean_power_sensor_values = np.zeros_like(mean_power_sensor_timestamps)

    async def wait_mean_power_steady_state(j: int) -> bool:
        for i, name in enumerate(sensor_names):
            reading = await pcc.sensor_reading(name, float)
            mean_power_sensor_timestamps[i, j] = reading.timestamp
            if reading.status.valid_value():
                mean_power_sensor_values[i, j] = reading.value
            else:
                mean_power_sensor_values[i, j] = np.nan

        return bool(
            np.all(mean_power_sensor_timestamps[:, j] >= min_sensor_time)
            and np.all(mean_power_sensor_values[:, j] == pytest.approx(tacv_power, rel=5e-3))
        )

    pdf_report.step("Compare mean-power sensors against TACV power.")
    test_passed, last_attempt = await max_retry_test(wait_mean_power_steady_state, samples, 1 / sample_rate)
    with check:
        assert test_passed, (
            f"X polarity mean power does not agree to within 0.5% of TACV v polarity power after {samples} retries."
        )
        assert tacv_power > 0.0

    pdf_report.step("Y polarity mean power sensors should be zero.")
    for chan in range(receiver.n_chans):
        sensor_name = f"{receiver.stream_name}.y{chan}.mean-power"
        reading = await pcc.sensor_reading(sensor_name, float)
        with check:
            assert reading.value == pytest.approx(0.0, rel=5e-3), (
                f"Y polarity mean power for channel {chan} is not zero: {reading.value}"
            )

    pdf_report.detail(
        f"Mean power sensor readings from {datetime.fromtimestamp(np.min(mean_power_sensor_timestamps[:, 0]), UTC)}"
        f" to {datetime.fromtimestamp(np.max(mean_power_sensor_timestamps[:, last_attempt]), UTC)}"
        f" in {last_attempt + 1} steps."
    )

    # Subtract time of first reading to make time relative to first reading
    mean_power_sensor_timestamps -= mean_power_sensor_timestamps[:, :1]
    fig = Figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel("Mean Power (dB)")
    ax.set_title("Mean Power Sensor Values")
    for i, name in enumerate(sensor_names):
        plot_focus(
            ax,
            slice(0, last_attempt + 1),
            mean_power_sensor_timestamps[i, : last_attempt + 1],
            mean_power_sensor_values[i, : last_attempt + 1],
            label=name,
        )
    ax.legend()
    await run_async(pdf_report.figure, fig)


@pytest.mark.name("VLBI VDIF output")
async def test_vlbi_vdif(
    pdf_report: Reporter,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
) -> None:
    """Test VDIF frame output.

    Verification method
    -------------------
    Verified by means of test. Collect ``FRAMESETS_TO_CAPTURE`` valid VDIF framesets.
    Verify the number of threads in the framesets are correct.
    """
    receiver = receive_tied_array_resampled_voltage
    pdf_report.step("Collect a valid VDIF frameset.")
    # capture a few valid framesets
    framesets = np.empty(FRAMESETS_TO_CAPTURE, dtype=object)
    frameset_count = 0
    async for frameset in receiver.complete_framesets():
        framesets[frameset_count] = frameset
        frameset_count += 1
        if frameset_count == FRAMESETS_TO_CAPTURE:
            break

    pdf_report.detail("Verify we have `n_chans * len(pol_ordering)` threads in the set.")
    with check:
        for frameset in framesets:
            assert len(frameset.frames) == receiver.n_chans * len(receiver.pol_ordering)


def sideband_tone_frequency(sky_freq: float, adc_sample_rate: float, output_sample_rate: float) -> float:
    """Frequency of a sky tone in the real V-engine output.

    Convert to the vlbi sideband tone frequency.
    """
    aliased = abs(sky_freq) % adc_sample_rate
    if aliased > adc_sample_rate / 2:
        aliased = adc_sample_rate - aliased
    shifted = abs(aliased - adc_sample_rate / 4) % output_sample_rate
    if shifted > output_sample_rate / 2:
        shifted = output_sample_rate - shifted
    return shifted


@pytest.mark.name("VLBI group delay compensation")
async def test_group_delay_compensation(
    pdf_report: Reporter,
    cbf: CBFRemoteControl,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
    band: str,
    run_async: AsyncRunner,
) -> None:
    """Test group delay compensation.

    Verification method
    -------------------
    Verified by means of test. Set input frequency tones seperated by 100 MHz, Collect a
    valid VDIF frameset, and determine the phase difference between the tones.
    The phase difference should be linear and within 5 degrees of zero after compensation.
    """
    receiver = receive_tied_array_resampled_voltage
    b = BANDS[band]
    start_frequency = b.centre_frequency - b.adc_sample_rate / 2

    low_frequencies = np.arange(start_frequency, b.centre_frequency, TONE_SPACING, dtype=np.float64)
    high_frequencies = np.arange(
        b.centre_frequency, b.centre_frequency + b.adc_sample_rate / 2, TONE_SPACING, dtype=np.float64
    )
    search_frequencies = np.append(
        low_frequencies,
        high_frequencies,
    )

    pcc = cbf.product_controller_client
    dsim_names = cbf.dsim_names
    for dsim_name in dsim_names:
        await pcc.request(
            "dsim-signals",
            dsim_name,
            f"multicw({len(search_frequencies)}, 0.16, 0, {start_frequency}, {TONE_SPACING});"
            + f"multicw({len(search_frequencies)}, 0.16, 0, {start_frequency}, {TONE_SPACING});",
        )

    steady_state_timestamp = await cbf.steady_state_timestamp()
    pdf_report.step("Collect a valid VDIF frameset.")
    frameset = await receiver.next_complete_frameset(min_timestamp=steady_state_timestamp)

    x_indices = [i for i, pol in enumerate(receiver.pol_ordering) if pol == "x"]
    assert len(x_indices) == 1
    x_indices = [x_indices[0], x_indices[0] + receiver.n_chans]
    y_indices = [i for i, pol in enumerate(receiver.pol_ordering) if pol == "y"]
    assert len(y_indices) == 1
    y_indices = [y_indices[0], y_indices[0] + receiver.n_chans]
    reciever_sample_rate = float(receiver.frame_rate * receiver.n_samples_per_frame)

    # channel zero is high frequency, channel one is low frequency
    channels = [
        [sideband_tone_frequency(freq, b.adc_sample_rate, reciever_sample_rate) for freq in high_frequencies],
        [sideband_tone_frequency(freq, b.adc_sample_rate, reciever_sample_rate) for freq in low_frequencies],
    ]
    channels_keys = ["high", "low"]
    assert receiver.n_chans == len(channels)
    time = np.arange(receiver.n_samples_per_frame) / (receiver.frame_rate * receiver.n_samples_per_frame)
    x_phases = np.zeros((len(channels_keys), len(search_frequencies)))
    y_phases = np.zeros((len(channels_keys), len(search_frequencies)))
    fig = Figure(tight_layout=True)
    axs = fig.subplots(
        len(channels_keys),
        len(receiver.pol_ordering),
        sharex="col",
        squeeze=False,
    )
    x_frameset_data = [BasebandVDIFFrame.fromfile(io.BytesIO(frameset.frames[index].raw_frame)) for index in x_indices]
    y_frameset_data = [BasebandVDIFFrame.fromfile(io.BytesIO(frameset.frames[index].raw_frame)) for index in y_indices]

    for i in range(len(channels_keys)):
        pdf_report.step(f"Calculate the delays of the tones for the {channels_keys[i]} channel.")
        for j, tone in enumerate(channels[i]):
            ref_sin = np.sin(2 * np.pi * tone * time)
            ref_cos = np.cos(2 * np.pi * tone * time)

            def calculate_phase(data: np.ndarray) -> float:
                nonlocal ref_sin, ref_cos
                q = data * ref_sin  # noqa: B023
                i = data * ref_cos  # noqa: B023
                return np.atan2(np.sum(i), np.sum(q))

            x_phases[i, j] = calculate_phase(x_frameset_data[i].data)
            y_phases[i, j] = calculate_phase(y_frameset_data[i].data)

        x_phase_differences = np.diff(x_phases[i, :])
        y_phase_differences = np.diff(y_phases[i, :])
        for j in range(len(x_phase_differences)):
            with check:
                assert np.degrees(x_phase_differences[j]) == pytest.approx(0, abs=5)
                assert np.degrees(y_phase_differences[j]) == pytest.approx(0, abs=5)

        xplt = axs[i][0]
        xplt.plot(
            (search_frequencies + (TONE_SPACING / 2))[: len(search_frequencies) - 1],
            x_phase_differences,
        )
        xplt.set_title(f"X polarity {channels_keys[i]} channel delay")
        xplt.legend()

        yplt = axs[i][1]
        yplt.plot(
            (search_frequencies + (TONE_SPACING / 2))[: len(search_frequencies) - 1],
            y_phase_differences,
        )
        yplt.set_title(f"Y polarity {channels_keys[i]} channel delay")
        yplt.legend()

    await run_async(pdf_report.figure, fig)
