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
import contextlib
import io
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
from astropy.time import Time
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf.pytest_plugins.reporter import Reporter, plot_focus
from katgpucbf.utils import TimeConverter
from qualification.cbf import CBFRemoteControl
from qualification.types import AsyncRunner
from test.qualification.test_recv import BasebandVDIFFrame

from ..recv import TiedArrayChannelisedVoltageReceiver, TiedArrayResampledVoltageReceiver, VDIFFrameset

FRAMESETS_TO_CAPTURE = 5000
SEARCH_TONES = 8


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


def sideband_tone_frequency(baseband_freq: float, centre_frequency: float, output_sample_rate: float) -> float:
    """Frequency of a tone in one VLBI sideband.

    A tone above Nyquist aliases to ``adc_sample_rate - centre_frequency``.
    """
    shifted = abs(baseband_freq - centre_frequency) % output_sample_rate
    if shifted > output_sample_rate / 2:
        shifted = output_sample_rate - shifted
    return shifted


def seek_thread_samples(framesets: list[VDIFFrameset], start_sample: int, n_samples: int) -> list[np.ndarray]:
    """Seek to the start sample for a thread and return the samples from the framesets.

    Parameters
    ----------
    framesets
        The list of framesets to seek in.
    start_sample
        The start sample to skip to.
    n_samples
        The number of samples to return.

    Returns
    -------
    list[np.ndarray]
        The samples from the frameset after seeking split by thread.
    """
    per_thread_id: list = [[] for _ in range(len(framesets[0].frames))]
    for frameset in framesets:
        for k, vdif_frame in enumerate(frameset.frames):
            frame = BasebandVDIFFrame.fromfile(io.BytesIO(vdif_frame.raw_frame))
            per_thread_id[k].append(np.reshape(frame.data, -1))
    return [np.concatenate(parts)[start_sample : start_sample + n_samples] for parts in per_thread_id]


def analyze_signal(data: np.ndarray, tone: float, time: np.ndarray) -> np.ndarray:
    """Analyze a signal at a given tone frequency."""
    samples = data.reshape(-1)
    assert len(samples) == len(time)
    analytical_signal = samples * np.exp(1j * 2 * np.pi * tone * time)
    return analytical_signal


@pytest.mark.name("VLBI group delay compensation")
@pytest.mark.no_capture_start("gpucbf.tied_array_resampled_voltage")
async def test_group_delay_compensation(  # noqa: D103
    pdf_report: Reporter,
    cbf: CBFRemoteControl,
    receive_tied_array_resampled_voltage: TiedArrayResampledVoltageReceiver,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    run_async: AsyncRunner,
) -> None:
    # The docstring is re-assigned after the function so that it can use an f-string.
    receiver = receive_tied_array_resampled_voltage
    # TODO(NGC-2186): Refactor `compute_tone_gain` away from the receiver to a utility function.
    b_receiver = receive_tied_array_channelised_voltage
    pcc = cbf.product_controller_client
    # filter-group-delay is in ADC samples; vlbi-delay is in seconds.
    pdf_report.step("compensate for filter-group-delay.")
    delay_samples = cbf.init_sensors["antenna-channelised-voltage.filter-group-delay"].value
    delay = delay_samples / b_receiver.adc_sample_rate
    delay_tolerance = 4e-6
    pdf_report.detail(f"group-delay: {delay * 1e6:.3f} µs.")
    await pcc.request("vlbi-delay", receiver.stream_name, -delay)

    # Use flat response bandwidth reported from the receiver rather than `pass_bandwidth` from conftest.
    tone_spacing = 1 / (2 * delay_tolerance)
    # Integer count, so a float endpoint cannot add an extra tone at sky_centre.
    tone_offsets = b_receiver.bandwidth / 4 + (np.arange(SEARCH_TONES) - SEARCH_TONES // 2) * tone_spacing
    low_frequencies = b_receiver.centre_frequency - tone_offsets[::-1]
    high_frequencies = b_receiver.centre_frequency + tone_offsets
    assert len(low_frequencies) == len(high_frequencies), "the number of low and high frequencies must be equal"
    n_injected = SEARCH_TONES * 2

    n_samples = FRAMESETS_TO_CAPTURE * receiver.n_samples_per_frame
    receiver_sample_rate = receiver.frame_rate * receiver.n_samples_per_frame

    assert len(cbf.dsim_names) == 1
    pdf_report.step("Set Antenna Channelised Voltage gains.")
    amplitude = 0.9 / n_injected
    gain = b_receiver.compute_tone_gain(amplitude, 100)  # assuming single tone per channel maximum
    pdf_report.detail(
        f"Setting gain for {len(b_receiver.input_labels)} antenna-channelised-voltage streams to {gain:.3f}."
    )
    await cbf.product_controller_client.request("gain-all", "antenna-channelised-voltage", 0)
    await cbf.product_controller_client.request("gain-all", "wideband-antenna-channelised-voltage", 0)

    await pcc.request("gain", "antenna-channelised-voltage", b_receiver.input_labels[0], gain)

    pdf_report.step("Set dsim signals to search for phase in frequencies.")
    pdf_report.detail(f"Searching for phase in {n_injected} frequencies every {tone_spacing * 1e3:.3f} KHz.")
    dsim_names = cbf.dsim_names
    low_signal = f"multicw({len(low_frequencies)}, {amplitude}, 0, {low_frequencies[0]}, {tone_spacing})"
    high_signal = f"multicw({len(high_frequencies)}, {amplitude}, 0, {high_frequencies[0]}, {tone_spacing})"
    for dsim_name in dsim_names:
        await pcc.request(
            "dsim-signals",
            dsim_name,
            f"signal={low_signal} + {high_signal}; signal;signal;",
        )
    await pcc.request("capture-start", receiver.stream_name)

    # The dsim waveform repeats every `dsim_period` ADC samples, with phase 0 at
    # each multiple. The signal request uses the default period, which is max-period.
    dsim_period = int(cbf.init_sensors[f"{dsim_names[0]}.max-period"].value)
    pdf_report.step("Collect VDIF framesets covering a dsim phase-zero epoch.")
    framesets: list[VDIFFrameset] = []
    start_sample = 0
    n_framesets = 0
    sync = Time(receiver.sync_time, format="unix", scale="utc")

    async with contextlib.aclosing(receiver.complete_framesets()) as framesets_generator:
        async for frameset in framesets_generator:
            if framesets and frameset.timestamp.linear != framesets[-1].timestamp.linear + 1:
                framesets.clear()
            if not framesets:
                first_timestamp = (frameset.timestamp.timestamp - sync).sec * receiver.scale_factor_timestamp
                # Same epoch as test_delay.compute_delay: the dsim period boundary at or
                # before this frame. Step forward one period when that boundary is in the past.
                phase_zero_timestamp = first_timestamp // dsim_period * dsim_period
                if phase_zero_timestamp < first_timestamp:
                    phase_zero_timestamp += dsim_period
                start_sample = (
                    (phase_zero_timestamp - first_timestamp) * receiver_sample_rate / receiver.scale_factor_timestamp
                )

                n_framesets = math.ceil((start_sample + n_samples) / receiver.n_samples_per_frame)
            framesets.append(frameset)
            if len(framesets) == n_framesets:
                break
    pdf_report.detail(f"Skipping {start_sample} samples to start at a phase-zero epoch.")
    assert len(framesets) == n_framesets

    channels_keys = ["high", "low"]
    n_pols = len(receiver.pol_ordering)

    frequencies = xr.DataArray(
        np.stack([high_frequencies, low_frequencies]),
        dims=("channel", "tone"),
        coords={"channel": channels_keys},
    )
    tone_channels = xr.DataArray(
        [
            [
                [
                    sideband_tone_frequency(freq, b_receiver.centre_frequency, receiver_sample_rate)
                    for freq in frequencies.sel(channel=channel).values
                ]
                for channel in channels_keys
            ]
            for _ in receiver.pol_ordering
        ],
        dims=("pol", "channel", "tone"),
        coords={"pol": receiver.pol_ordering, "channel": channels_keys},
    )

    assert receiver.n_chans == frequencies.sizes["channel"]
    time = np.arange(n_samples) / receiver_sample_rate
    phases = np.zeros((n_pols * receiver.n_chans, SEARCH_TONES))

    fig = Figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Basebandfrequency (MHz)")
    ax.set_ylabel("Delay (µs)")
    ax.set_title("Group delay")

    pdf_report.step("Calculate the delays of the tones per polarisation-channel thread.")
    # Each channel's polarisations are adjacent, build the index for each polarisation to thread index.
    # example: [0, 2] for X, [1, 3] for Y.
    pol_channel_indices = [
        [base + channel_i * n_pols for channel_i in range(len(channels_keys))] for base in range(n_pols)
    ]
    phase_zero_samples = seek_thread_samples(framesets, math.floor(start_sample), n_samples)
    pol_samples = [[phase_zero_samples[index] for index in indices] for indices in pol_channel_indices]
    for chan_i, channel in enumerate(channels_keys):
        pdf_report.detail(
            f"Reference frequency for {channel} channel: {frequencies.sel(channel=channel)[0].item() * 1e-6:.3f} MHz."
        )
        for pol_i, pol in enumerate(tone_channels.pol.values):
            order = chan_i * n_pols + pol_i
            for j, tone in enumerate(tone_channels.sel(channel=channel, pol=pol).values):
                phases[order, j] = np.angle(np.sum(analyze_signal(pol_samples[pol_i][chan_i], tone, time)))

            # Phases from atan2 are wrapped to ±π. Unwrap along the tone comb
            # before subtracting the reference, so a slope past ±π stays continuous.
            phase_differences = np.unwrap(phases[order])[1:] - phases[order, 0]
            tau = -phase_differences / (2 * np.pi * tone_spacing * (1 + np.arange(len(phase_differences))))
            ax.plot(frequencies.sel(channel=channel)[1:] * 1e-6, tau * 1e6, label=f"{str(pol).upper()} {channel}")

            with check:
                np.testing.assert_allclose(
                    tau,
                    0.0,
                    atol=delay_tolerance,
                    err_msg=f"Delay for channel {channel}, {pol} polarisation does not agree to "
                    + f"within {delay_tolerance * 1e6:.3f} µs of zero.",
                )
                pdf_report.detail(
                    f"Delays are within {delay_tolerance * 1e6:.3f} µs of zero for {channel} channel {pol} pols."
                )

    ax.legend()
    await run_async(pdf_report.figure, fig)


test_group_delay_compensation.__doc__ = f"""Test group delay compensation.

    Verification method
    -------------------
    Verified by means of test. Generate {SEARCH_TONES} tones in the passband separated evenly within the linear
    passband. Compensate for the filter-group-delay by setting the VLBI delay to the filter-group-delay.
    Collect valid VDIF framesets, and determine the phase difference between each tone and the
    reference tone for each thread. The phase difference should be linear and near zero with 4µs tolerance.
    """
