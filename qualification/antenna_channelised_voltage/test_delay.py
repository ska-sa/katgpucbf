################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

"""Delay and phase compensation tests."""

import asyncio
import math
from ast import literal_eval
from collections.abc import Callable

import numpy as np
import pytest
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf import BYTE_BITS, COMPLEX, N_POLS
from katgpucbf.fgpu.delay import wrap_angle

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver, TiedArrayChannelisedVoltageReceiver
from ..reporter import POTLocator, Reporter, plot_focus

MAX_DELAY = 79.53e-6  # seconds
MAX_DELAY_RATE = 2.56e-9
MAX_PHASE = math.pi  # rad
MAX_PHASE_RATE = 49.22  # rad/second


@pytest.mark.requirements("CBF-REQ-0077")
async def test_delay_application_time(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that delay/phase changes are applied at the correct time.

    Verification method
    -------------------
    Verification by means of test. A 90 degree phase change is loaded for one
    polarisation at a chosen time. The actual application time is estimated by
    checking the ratio of real to imaginary components in the corresponding
    accumulation.
    """
    receiver = receive_baseline_correlation_products
    pcc = cbf.product_controller_client

    pdf_report.step("Inject correlated white noise signal.")
    await pcc.request("dsim-signals", cbf.dsim_names[0], "common=nodither(wgn(0.1)); common; common;")
    pdf_report.detail("Wait for updated signal to propagate through the pipeline.")
    await receiver.next_complete_chunk()

    attempts = 5
    advance = 0.2
    acc: np.ndarray | None = None
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    for attempt in range(attempts):
        pdf_report.step(f"Set delay {advance * 1000:.0f}ms in the future (attempt {attempt + 1} / {attempts}).")
        pdf_report.detail("Get current time according to the dsim.")
        now = await cbf.dsim_time()
        target = now + advance
        delays = ["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants
        pdf_report.detail("Set delays.")
        await cbf.product_controller_client.request("delays", "antenna-channelised-voltage", target, *delays)
        pdf_report.step("Receive data for the corresponding dump.")
        target_ts = round(receiver.time_converter.unix_to_adc(target))
        target_acc_ts = target_ts // receiver.timestamp_step * receiver.timestamp_step
        acc = None
        async for timestamp, chunk in receiver.complete_chunks(min_timestamp=target_acc_ts, time_limit=10.0):
            with chunk:
                pdf_report.detail(f"Received chunk with timestamp {timestamp}, target is {target_acc_ts}.")
                total = np.sum(chunk.data[:, bl_idx, :], axis=0)  # Sum over channels
                if timestamp == target_acc_ts:
                    acc = total
                if timestamp >= target_acc_ts:
                    break
        else:
            pdf_report.detail("Did not reach the target timestamp within 10s.")
        if acc is not None:
            break

        pdf_report.detail("Did not receive all the expected chunks; reset delay and try again.")
        delays = ["0,0:0,0", "0,0:0,0"] * receiver.n_ants
        await cbf.product_controller_client.request(
            "delays", "antenna-channelised-voltage", receiver.sync_time, *delays
        )
    else:
        pytest.fail(f"Give up after {attempts} attempts.")

    pdf_report.step("Check the received data.")
    # Estimate time at which delay was applied based on real:imaginary
    total = np.sum(np.abs(acc))
    load_frac = abs(acc[0]) / total  # Load time as fraction of the accumulation
    load_time = receiver.time_converter.adc_to_unix(target_acc_ts) + load_frac * receiver.int_time
    delta = load_time - target
    pdf_report.detail(f"Estimated load time error: {delta * 1e6:.3f}µs.")
    with check:
        assert delta < 0.01


@pytest.mark.name("Delay Enable/Disable")
@pytest.mark.requirements("CBF-REQ-0066,CBF-REQ-0110,CBF-REQ-0187,CBF-REQ-0188")
async def test_delay_enable_disable(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that delay and phase compensation can be enabled and disabled.

    Additionally, verify that changes to delays/phase can be achieved at the
    required rate.

    Verification method
    -------------------
    Verified by means of test. Insert a signal with a tone. Enable delay/phase
    compensation and check that it is applied, then disable and check again.
    Check that all requests complete within the required time.
    """

    async def measure_phase() -> float:
        """Retrieve the phase of the chosen channel from the next chunk."""
        _, data = await receiver.next_complete_chunk()
        value = data[channel, bl_idx, :]
        phase = np.arctan2(value[1], value[0])
        return phase

    async def set_delays(delays: list[str]) -> None:
        start = asyncio.get_running_loop().time()
        await cbf.product_controller_client.request(
            "delays", "antenna-channelised-voltage", receiver.sync_time, *delays
        )
        finish = asyncio.get_running_loop().time()
        elapsed.append(finish - start)

    receiver = receive_baseline_correlation_products
    pcc = cbf.product_controller_client

    channel = 3 * receiver.n_chans // 4
    freq = receiver.channel_frequency(channel)
    signal = f"cw(0.1, {freq})"
    gain = receiver.compute_tone_gain(0.1, 100)
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    elapsed: list[float] = []

    pdf_report.step("Inject tone.")
    pdf_report.detail(f"Set signal to {signal} on both pols.")
    await pcc.request("dsim-signals", cbf.dsim_names[0], f"common={signal}; common; common;")
    pdf_report.detail(f"Set gain to {gain} for all inputs.")
    await cbf.product_controller_client.request("gain-all", "antenna-channelised-voltage", gain)

    pdf_report.step("Check that phase compensation can be enabled.")
    pdf_report.detail("Apply 90 degree phase to one pol")
    await set_delays(["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants)
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    with check:
        assert phase == pytest.approx(-math.pi / 2, abs=np.deg2rad(1))

    pdf_report.step("Check that delay compensation can be enabled.")
    pdf_report.detail("Apply 1/2 cycle delay to one pol.")
    await set_delays(["0,0:0,0", f"{0.5 / freq},0:0,0"] * receiver.n_ants)
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    # One might expect it to be pi radians, but that ignores the implicit
    # phase adjustment that ensures the centre channel has zero phase.
    with check:
        assert phase == pytest.approx(math.pi * (freq - receiver.center_freq) / freq, abs=np.deg2rad(1))

    pdf_report.step("Check that compensation can be disabled.")
    await set_delays(["0,0:0,0"] * (2 * receiver.n_ants))
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    with check:
        assert phase == pytest.approx(0, abs=np.deg2rad(1))

    pdf_report.step("Check update time.")
    max_elapsed = max(elapsed)
    pdf_report.detail(f"Maximum time for ?delays request is {max_elapsed:.3f}s.")
    with check:
        assert max_elapsed < 1 / 0.167


@pytest.mark.requirements("CBF-REQ-0187,CBF-REQ-0188")
@pytest.mark.xfail(
    reason="requirement cannot be met for all modes as delays cannot be updated more than once per spectrum"
)
async def test_delay_application_rate(cbf: CBFRemoteControl, pdf_report: Reporter) -> None:
    """Test that delay and phase polynomials are applied at the required rate.

    Verification method
    -------------------
    Verified by analysis. The delay and phase are calculated separately for
    every spectrum. Thus, it is sufficient for the rate of spectra to be high
    enough.
    """
    pdf_report.step("Query rate of spectra.")
    n_samples_between_spectra = cbf.init_sensors["antenna-channelised-voltage.n-samples-between-spectra"].value
    scale_factor_timestamp = cbf.init_sensors["antenna-channelised-voltage.scale-factor-timestamp"].value
    rate = scale_factor_timestamp / n_samples_between_spectra
    pdf_report.detail(f"There are {rate:.3f} spectra per second.")
    assert rate >= 2500.0


async def test_delay_sensors(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that delay sensors work correctly.

    Verification method
    -------------------
    Verified by test. Load a set of random delays with a load time in the
    future. Once that time arrives, check that the sensors report the correct
    values.
    """
    pcc = cbf.product_controller_client
    receiver = receive_baseline_correlation_products
    delay_tuples = []  # Expected sensor values
    delay_strs = []  # Strings to load
    rng = np.random.default_rng(seed=31)
    now = await cbf.dsim_time()
    load_time = now + 2.0
    load_ts = round(receiver.time_converter.unix_to_adc(load_time))
    for _ in range(receiver.n_inputs):
        delay = rng.uniform(-MAX_DELAY, MAX_DELAY)
        delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE)
        phase = rng.uniform(-np.pi, np.pi)
        phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE)
        delay_strs.append(f"{delay},{delay_rate}:{phase},{phase_rate}")
        delay_tuples.append((load_ts, delay, delay_rate, phase, phase_rate))

    async def delay_sensor_value(label: str) -> tuple:
        return literal_eval(await pcc.sensor_value(f"antenna-channelised-voltage.{label}.delay", str))

    pdf_report.step("Load delays.")
    pdf_report.detail(f"Set delays to load at {load_time} (timestamp {load_ts}).")
    await cbf.product_controller_client.request("delays", "antenna-channelised-voltage", load_time, *delay_strs)
    await asyncio.sleep(0.1)  # Allow time for any invalid sensor updates to propagate
    pdf_report.detail("Check that sensors do not reflect the future.")
    for label in receiver.input_labels:
        value = await delay_sensor_value(label)
        with check:
            assert value[1:] == (0.0, 0.0, 0.0, 0.0)
    pdf_report.step("Wait for load time and check sensors.")
    pdf_report.detail(f"Wait for an accumulation with timestamp >= {load_ts}.")
    await receiver.next_complete_chunk(min_timestamp=load_ts)
    for expected, label in zip(delay_tuples, receiver.input_labels, strict=True):
        value = await delay_sensor_value(label)
        pdf_report.detail(f"Input {label} has delay sensor {value}, expected value {expected}.")
        with check:
            assert value == pytest.approx(expected, rel=1e-9), f"Delay sensor for {label} has incorrect value"


def check_phases(
    pdf_report: Reporter,
    actual: np.ndarray,
    expected: np.ndarray,
    pass_channels: slice,
    caption: str,
    tolerance_deg: float = 1,
) -> None:
    """Compare expected and actual phases to ensure they're within 1°.

    The error in phase is also plotted.
    """
    n_chans = len(actual)
    delta = wrap_angle(actual - expected)
    pass_delta = delta[pass_channels]
    max_error = np.max(np.abs(pass_delta))
    rms_error = np.sqrt(np.mean(np.square(pass_delta)))
    pdf_report.detail(f"Maximum error is {np.rad2deg(max_error):.3f}°.")
    pdf_report.detail(f"RMS error over channels is {np.rad2deg(rms_error):.5f}°.")
    with check:
        assert np.rad2deg(max_error) <= tolerance_deg, f"Maximum error is more than {tolerance_deg}°"

    fig = Figure(tight_layout=True)
    ax, ax_err = fig.subplots(2)
    x = range(0, n_chans)

    ax.set_title(f"Phase with {caption}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Phase (degrees)")
    ax.xaxis.set_major_locator(POTLocator())
    plot_focus(ax, pass_channels, x, np.rad2deg(wrap_angle(actual)), label="Actual")
    plot_focus(ax, pass_channels, x, np.rad2deg(wrap_angle(expected)), label="Expected")
    ax.legend()

    ax_err.set_title(f"Phase error with {caption}")
    ax_err.set_xlabel("Channel")
    ax_err.set_ylabel("Error (degrees)")
    ax_err.xaxis.set_major_locator(POTLocator())
    plot_focus(ax_err, pass_channels, x, np.rad2deg(delta))

    pdf_report.figure(fig)


def delay_phase(receiver: BaselineCorrelationProductsReceiver, delay_samples: float) -> np.ndarray:
    """Calculate expected phase for a given delay.

    The return value is appropriate if the sample signal is provided on both
    inputs, but the first input in the correlation is configured with a delay
    of `delay_samples` samples, and no phase compensation.
    """
    n_chans = receiver.n_chans
    return np.arange(-n_chans // 2, n_chans // 2) / n_chans / receiver.decimation_factor * np.pi * -delay_samples


async def _test_delay_phase_fixed(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
    delay_phases: list[tuple[float, float]],
    caption_cb: Callable[[float, float], str],
    report_residual: bool,
) -> None:
    """Test performance of delay or phase compensation with a fixed value.

    This is the implementation for both :func:`test_delay` and
    :func:`test_delay_phase`.

    Parameters
    ----------
    cbf, receive_baseline_correlation_products, pdf_report, pass_channels
        Fixtures
    delay_phases
        Pairs of (delay, phase) to test
    caption_cb
        Callback to generate a figure caption from a delay and phase
    report_residual
        If true, report the residual delay between the delay applied to the
        signal and the delay compensation
    """
    receiver = receive_baseline_correlation_products
    pcc = cbf.product_controller_client
    n_dsims = len(cbf.dsim_names)
    assert N_POLS * n_dsims > len(delay_phases)  # > rather than >= because we need a reference

    pdf_report.step("Set input signals and delays.")
    base_signal = "wgn(0.05, 1)"
    signals = [f"nodither({base_signal});"] * (N_POLS * n_dsims)
    delay_spec = ["0,0:0,0"] * receiver.n_inputs
    delay_samples = []
    for i, (delay, phase) in enumerate(delay_phases):
        # It's more efficient for the dsim to delay by a multiple of 8 samples
        delay_samples.append(round(delay * receiver.scale_factor_timestamp / BYTE_BITS) * BYTE_BITS)
        signals[i] = f"nodither(delay({base_signal}, {-delay_samples[-1]}));"
        delay_spec[i] = f"{delay},0:{phase},0"

    async with asyncio.TaskGroup() as tg:
        for i, dsim_name in enumerate(cbf.dsim_names):
            signal_spec = "".join(signals[i * N_POLS : (i + 1) * N_POLS])
            pdf_report.detail(f"Set signal to {signal_spec!r} on dsim {i}.")
            tg.create_task(pcc.request("dsim-signals", dsim_name, signal_spec))
    for i in range(len(delay_phases)):
        pdf_report.detail(f"Set delay model to {delay_spec[i]} on input {i}")
    await cbf.product_controller_client.request(
        "delays", "antenna-channelised-voltage", receiver.sync_time, *delay_spec
    )

    pdf_report.step("Verify results")
    pdf_report.detail("Receive an accumulation")
    _, chunk_data = await receiver.next_complete_chunk()
    actual = np.arctan2(chunk_data[..., 1], chunk_data[..., 0])

    for i, (delay, phase) in enumerate(delay_phases):
        caption = caption_cb(delay, phase)
        # The delay is mostly cancelling out the delay applied in the dsim, but
        # there will be fine delay left over.
        residual = delay * receiver.scale_factor_timestamp - delay_samples[i]
        if report_residual:
            residual_msg = f" (residual delay {residual:.6f} samples)"
        else:
            residual_msg = ""
        pdf_report.detail(f"Testing {caption}{residual_msg}")
        input1 = receiver.input_labels[i]
        input2 = receiver.input_labels[-1]
        bl_idx = receiver.bls_ordering.index((input1, input2))
        expected = delay_phase(receiver, residual) + phase
        # The delay in the dsim will affect the phase of the centre frequency,
        # which the delay compensation won't correct.
        expected += 2 * np.pi * delay_samples[i] / receiver.scale_factor_timestamp * receiver.center_freq
        check_phases(pdf_report, actual[:, bl_idx], expected, pass_channels, caption)


async def _test_delay_phase_rate(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
    rates: list[tuple[float, float]],
    caption_cb: Callable[[float, float], str],
) -> None:
    """Test performance of delay or phase compensation with a rate of change.

    This is the implementation for both :func:`test_delay_rate` and
    :func:`test_phase_rate`.

    Parameters
    ----------
    cbf, receive_baseline_correlation_products, pdf_report, pass_channels
        Fixtures
    rates
        Pairs of (delay_rate, phase_rate) to test
    caption_cb
        Callback to generate a figure caption from a delay rate and phase rate
    """
    receiver = receive_baseline_correlation_products
    pcc = cbf.product_controller_client
    # Minimum, maximum, resolution step
    n_dsims = len(cbf.dsim_names)
    assert N_POLS * n_dsims > len(rates)  # > rather than >= because we need a reference

    pdf_report.step("Set input signals and delays.")
    signal = "common = nodither(wgn(0.05, 1)); common; common;"
    max_period: int = cbf.init_sensors[f"{cbf.dsim_names[0]}.max-period"].value
    # Choose a period that makes all accumulations the same, so that we can
    # compare accumulations without extraneous noise.
    period = math.gcd(max_period, receiver.n_samples_between_spectra * receiver.n_spectra_per_acc)
    pdf_report.detail(f"Set signal to {signal!r} on all dsims.")
    async with asyncio.TaskGroup() as tg:
        for dsim_name in cbf.dsim_names:
            tg.create_task(pcc.request("dsim-signals", dsim_name, signal, period))
    delay_spec = ["0,0:0,0"] * receiver.n_inputs
    for i, (delay_rate, phase_rate) in enumerate(rates):
        delay_spec[i] = f"0,{delay_rate}:0,{phase_rate}"
        pdf_report.detail(f"Set delay model to {delay_spec[i]} on input {i}")
    now = await cbf.dsim_time()
    await cbf.product_controller_client.request("delays", "antenna-channelised-voltage", now, *delay_spec)

    pdf_report.step("Collect two consecutive accumulations.")
    timestamps = []
    phases = []
    for timestamp, chunk in await receiver.consecutive_chunks(2):
        with chunk:
            timestamps.append(timestamp)
            phases.append(np.arctan2(chunk.data[..., 1], chunk.data[..., 0]))
    elapsed = timestamps[1] - timestamps[0]
    elapsed_s = elapsed / receiver.scale_factor_timestamp
    pdf_report.detail(f"Timestamps are {timestamps[0]}, {timestamps[1]} with difference {elapsed} ({elapsed_s:.3f} s).")

    pdf_report.step("Verify results.")
    for i, (delay_rate, phase_rate) in enumerate(rates):
        caption = caption_cb(delay_rate, phase_rate)
        pdf_report.detail(f"Testing {caption}")
        input1 = receiver.input_labels[i]
        input2 = receiver.input_labels[-1]
        bl_idx = receiver.bls_ordering.index((input1, input2))
        actual = phases[1][:, bl_idx] - phases[0][:, bl_idx]
        expected = delay_phase(receiver, delay_rate * elapsed) + phase_rate * elapsed_s
        # Allow 2° rather than 1° because we're taking the difference between
        # two phases which each have a 1° tolerance.
        check_phases(pdf_report, actual, expected, pass_channels, caption, tolerance_deg=2)


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0185")
async def test_delay(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    r"""Test performance of delay compensation with a fixed delay.

    Verification method
    -------------------
    Verified by test. Set a variety of delays on different inputs. Delay the
    corresponding dsim signal by the same amount, rounded to the nearest 8
    samples. Check that the resulting phases are within :math:`\ang{1}`
    of the expected value.
    """
    receiver = receive_baseline_correlation_products
    # Minimum, maximum, resolution step, and a small coarse delay
    delays = [0.0, MAX_DELAY, -MAX_DELAY, 2.5e-12, 2.75 / receiver.scale_factor_timestamp]
    await _test_delay_phase_fixed(
        cbf,
        receive_baseline_correlation_products,
        pdf_report,
        pass_channels,
        [(delay, 0.0) for delay in delays],
        lambda delay, phase: f"delay {delay * 1e12:.2f}ps",
        True,
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0185")
async def test_delay_rate(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    r"""Test performance of delay compensation with a delay rate.

    Verification method
    -------------------
    Verified by test. Set identical signals on all dsims. Set different delay
    rates on several inputs (all other coefficients being zero). Collect two
    successive accumulations and measure the change in phase between them,
    checking that it is within :math:`\ang{1}` of the expected step.
    """
    # Minimum, maximum, resolution step
    rates = [-MAX_DELAY_RATE, MAX_DELAY_RATE, 2.5e-12]
    await _test_delay_phase_rate(
        cbf,
        receive_baseline_correlation_products,
        pdf_report,
        pass_channels,
        [(delay_rate, 0.0) for delay_rate in rates],
        lambda delay_rate, phase_rate: f"delay rate {delay_rate}",
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0112")
async def test_delay_phase(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    r"""Test performance of delay tracking with a fixed phase.

    Verification method
    -------------------
    Verified by test. Set a variety of phase corrections on different inputs
    (with other coefficients being zero). Check that the resulting phases are
    within :math:`\ang{1}` of the expected value.
    """
    # Min, max, large non-multiple of pi/2, and resolution
    phases = [-MAX_PHASE, MAX_PHASE, 2 * math.pi / 3, 0.01]
    await _test_delay_phase_fixed(
        cbf,
        receive_baseline_correlation_products,
        pdf_report,
        pass_channels,
        [(0.0, phase) for phase in phases],
        lambda delay, phase: f"phase {phase:.4f} rad ({np.rad2deg(phase):.2f}°)",
        False,
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0112")
@pytest.mark.xfail(reason="requirements unclear")
async def test_phase_rate(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
) -> None:
    r"""Test performance of delay tracking with a phase rate.

    Verification method
    -------------------
    Verified by test. Set identical signals on all dsims. Set different phase
    rates on several inputs (all other coefficients being zero). Collect two
    successive accumulations and measure the change in phase between them,
    checking that it is within :math:`\ang{1}` of the expected step.
    """
    # Minimum, maximum, resolution step
    rates = [-MAX_PHASE_RATE, MAX_PHASE_RATE, 0.044]
    await _test_delay_phase_rate(
        cbf,
        receive_baseline_correlation_products,
        pdf_report,
        pass_channels,
        [(0.0, phase_rate) for phase_rate in rates],
        lambda delay_rate, phase_rate: f"phase rate {phase_rate}",
    )


async def test_group_delay(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
    pass_channels: slice,
    narrowband_vlbi: bool,
) -> None:
    r"""Test the ``filter-group-delay`` sensor.

    Verification method
    -------------------
    Verified by means of a test.

    Group delay is defined as the negative of the derivative of phase shift
    with respect to angular frequency :math:`\omega`. To estimate the
    derivative by finite differences, the dsim is configured with two sine
    waves (one in each polarisation) with a small difference in frequency. The
    difference needs to be carefully selected. If it too small, the effect is
    dominated by quantisation effects. However, the group delay can only be
    determined modulo the period of the beat frequency, and hence the
    frequencies cannot be too far apart.

    Rather than try to pick a single difference, we use two: firstly the
    minimum difference supported by the dsim, to localise the delay, and
    then a much larger difference to refine it.

    Additionally, a phase rate is set in the F-engine delay model. This
    cancels out when measuring phase differences, but ensures that there is
    variation in the phases and thus helps prevent systematic bias in the
    quantisation errors.

    The test is replicated across many channels to increase the number of
    samples and hence improve signal-to-noise. The channels are divided amongst
    the simulated antennas then combined by beamforming, which allows for
    larger amplitudes without saturation in the time domain, and also speeds up
    the test.
    """
    receiver = receive_tied_array_channelised_voltage
    pcc = cbf.product_controller_client

    pdf_report.step("Choose channels.")
    # In VLBI mode we don't have a PFB, and there is too much leakage
    # between channels to be able to multiplex multiple channels onto
    # one signal.
    n_dsims = len(cbf.dsim_names)
    if narrowband_vlbi:
        channel_step = 1
        channels_slice = slice(
            pass_channels.start,
            min(pass_channels.stop, pass_channels.start + n_dsims),
        )
    else:
        # Pick channels to test, keeping some distance from the edges.
        # The channel step is set to keep a fixed number of channels
        # between tones in the same dsim, because that determines the
        # spectral leakage between tones.
        channel_step = math.ceil(256 / n_dsims)
        channels_slice = slice(
            pass_channels.start + channel_step // 2,
            pass_channels.stop - channel_step // 2,
            channel_step,
        )
    freq_step = receiver.bandwidth / receiver.n_chans * channel_step
    channels = np.arange(receiver.n_chans, dtype=int)[channels_slice]
    n_channels = len(channels)
    cfreqs = receiver.channel_frequency(channels)
    pdf_report.detail(f"Using {n_channels} channels separated by {channel_step} channels ({freq_step} Hz).")

    # Distribute the channels amongst the dsims (round-robin)
    cfreqs_by_dsim: list[list[float]] = [[] for _ in range(n_dsims)]
    for i, freq in enumerate(cfreqs):
        cfreqs_by_dsim[i % n_dsims].append(freq)
    freq_step_dsim = freq_step * n_dsims

    pdf_report.step("Determine dsim frequency resolution.")
    dsim_period: int = cbf.init_sensors[f"{cbf.dsim_names[0]}.max-period"].value
    dsim_resolution = receiver.adc_sample_rate / dsim_period
    pdf_report.detail(f"Resolution is {dsim_resolution:.6f} Hz.")

    # We set per-channel gains so that each F-engine only outputs non-zero
    # values for the channels where it has signal. This suppresses any spectral
    # leakage that would otherwise contaminate the signal from other F-engines
    # when the results are added in the beamformer.
    pdf_report.step("Set F-engine gains.")
    amplitude = 0.99 / len(cfreqs_by_dsim[0])
    gain = receiver.compute_tone_gain(amplitude, 100)
    gains = [[0.0] * receiver.n_chans for _ in range(n_dsims)]
    for i, channel in enumerate(channels):
        gains[i % n_dsims][channel] = gain
    async with asyncio.TaskGroup() as tg:
        for i, g in enumerate(gains):
            for j in range(2):
                input_label = receiver.input_labels[2 * i + j]
                tg.create_task(pcc.request("gain", "antenna-channelised-voltage", input_label, *g))
    pdf_report.detail(f"Set gain to {gain} on chosen antenna/channel pairs.")

    # Collect about 2^27 samples, to improve SNR. However, for VLBI mode this
    # would take far too long because we're only able to test one channel per
    # antenna. It's a trade-off between test time and the strength of the
    # test.
    if narrowband_vlbi:
        n_spectra = 2**23 // n_channels
    else:
        n_spectra = 2**27 // n_channels
    n_chunks = math.ceil(n_spectra / receiver.n_spectra_per_heap)
    n_spectra = n_chunks * receiver.n_spectra_per_heap
    chunk_timestamp_step = receiver.n_spectra_per_heap * receiver.n_samples_between_spectra
    acc_time = n_chunks * chunk_timestamp_step / receiver.scale_factor_timestamp

    pdf_report.step("Set F-engine phase rate.")
    # Swing phase through 2pi radians over the collection time. Start away from
    # phase 0 where quantisation effects are particularly bad.
    delay_model = f"0,0:0.1,{2 * np.pi / acc_time}"
    delay_models = [delay_model] * receiver.n_inputs
    await pcc.request("delays", "antenna-channelised-voltage", receiver.sync_time, *delay_models)
    pdf_report.detail(f"Set delays to {delay_model} for all inputs.")

    async def measure_once(rel_freqs: tuple[float, float]) -> tuple[float, float, float]:
        """Estimate group delay for a single pair of frequency offsets.

        For each channel centre frequency `cfreq` in `cfreqs`, tones are
        injected at frequencies `cfreq + rel_freqs[0]` and
        `cfreq + rel_freqs[1]` and the difference in phase between these is
        examined to determine the delay.

        The delay is ambiguous and could actually be any value of the form
        :samp:`{delay} + {i} * {period}`.

        Returns
        -------
        delay
            Estimated delay in samples.
        period
            Step between possible delay values.
        std
            Standard deviation in the delay estimate. Note that the distribution is
            non-Gaussian and is due to quantisation error (which is bounded), but
            the estimate is taken from the mean of a large number of samples and so
            the Central Limit Theorem applies.
        """
        async with asyncio.TaskGroup() as tg:
            for dsim_name, dsim_freqs in zip(cbf.dsim_names, cfreqs_by_dsim, strict=True):
                if dsim_freqs:
                    signal = " ".join(
                        f"multicw({len(dsim_freqs)}, {amplitude}, 0.0, {dsim_freqs[0] + rfreq}, {freq_step_dsim});"
                        for rfreq in rel_freqs
                    )
                else:
                    signal = "0;0;"
                tg.create_task(pcc.request("dsim-signals", dsim_name, signal, dsim_period))
                pdf_report.detail(f"Set dsim signal to {signal}.")
        pdf_report.detail("All dsim signals set.")

        pdf_report.detail(f"Receive {n_chunks} chunks ({acc_time:.3f} s) of contiguous data.")
        i = 0
        attempts = 0
        first_timestamp = -1
        # First axis corresponds to the 2 signals we're comparing.
        raw_data = np.ones((2, n_channels, n_spectra, COMPLEX), np.int8)
        try:
            async with asyncio.timeout(acc_time * 3 + 10.0):
                async for timestamp, chunk in receiver.complete_chunks():
                    with chunk:
                        if i == 0 or timestamp != first_timestamp + i * chunk_timestamp_step:
                            first_timestamp = timestamp
                            i = 0  # If we had a gap, start from the beginning again
                            attempts += 1
                        start_spectrum = i * receiver.n_spectra_per_heap
                        end_spectrum = (i + 1) * receiver.n_spectra_per_heap
                        raw_data[:, :, start_spectrum:end_spectrum] = chunk.data[:2, channels_slice]
                    i += 1
                    if i == n_chunks:
                        pdf_report.detail(f"Received all chunks after {attempts} attempt(s).")
                        break
        except TimeoutError:
            pytest.fail("Timed out.")

        # Convert Gaussian integers to complex128
        data = raw_data.astype(np.float64).view(np.complex128)[..., 0]
        phase = np.angle(data[1] * data[0].conj())  # Takes the difference between phases
        del data  # Free up some memory
        # Pick a reference timestamp at which we know the dsim will be outputting
        # both signals with phase 0.
        ref_timestamp = first_timestamp // dsim_period * dsim_period
        # Phase-rotate everything to be referenced to ref_timestamp
        rel_timestamps = np.arange(n_spectra) * receiver.n_samples_between_spectra + (first_timestamp - ref_timestamp)
        rel_times = rel_timestamps / receiver.scale_factor_timestamp
        delta = rel_freqs[1] - rel_freqs[0]
        phase -= 2 * np.pi * delta * rel_times[np.newaxis, :]
        # The phases should all be similar, but we need to avoid steps of 2pi.
        # np.unwrap is problematic since it accumulates rounding errors. The
        # following is good enough as long as all phases are within pi of each
        # other.
        phase = wrap_angle(phase - phase[0, 0]) + phase[0, 0]
        mean_phase = wrap_angle(np.mean(phase))
        std_phase = np.std(phase) / np.sqrt(phase.size - 1)
        scale = receiver.scale_factor_timestamp / (2 * np.pi * delta)
        delay = -mean_phase * scale
        period = 2 * np.pi * scale
        std = std_phase * scale
        pdf_report.detail(f"Delay is {delay} + k*{period} ± {std} samples.")
        return delay, period, std

    pdf_report.step("Compare two tones that differ as little as possible.")
    delay1, period1, std1 = await measure_once((0.0, dsim_resolution))

    pdf_report.step("Compare two tones that differ by 2/3 of a channel.")
    channel_width = receiver.bandwidth / receiver.n_chans
    steps = round(channel_width / 3 / dsim_resolution)
    delay2, period2, std2 = await measure_once((-steps * dsim_resolution, steps * dsim_resolution))

    pdf_report.step("Analyse results.")
    assert 5 * std1 < period2  # Ensure the first test localised things sufficiently
    # Solve (approximately) delay1 = delay2 + k * period2
    k = round((delay1 - delay2) / period2)
    delay = delay2 + k * period2
    # Could do some error propagation to combine std1 and std2, but std2 will
    # be orders of magnitude smaller so it is not worth worrying about.
    std = std2
    pdf_report.detail(f"Measured delay is {delay} ± {std} samples.")

    reported_delay = await pcc.sensor_value("antenna-channelised-voltage.filter-group-delay", float)
    pdf_report.detail(f"Reported delay is {reported_delay}.")
    assert abs(reported_delay - delay) < 5 * std
    pdf_report.detail("Measured value agrees with reported value to within 5 sigma.")
