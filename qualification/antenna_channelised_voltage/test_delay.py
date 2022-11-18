################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

from katgpucbf import BYTE_BITS, N_POLS
from katgpucbf.fgpu.delay import wrap_angle

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl, get_sensor_val
from ..reporter import POTLocator, Reporter
from . import compute_tone_gain

MAX_DELAY = 79.53e-6  # seconds
MAX_DELAY_RATE = 2.56e-9
MAX_PHASE = math.pi  # rad
MAX_PHASE_RATE = 49.22  # rad/second


@pytest.mark.requirements("CBF-REQ-0077")
async def test_delay_application_time(
    correlator: CorrelatorRemoteControl,
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

    pdf_report.step("Inject correlated white noise signal.")
    await correlator.dsim_clients[0].request("signals", "common=nodither(wgn(0.1)); common; common;")
    pdf_report.detail("Wait for updated signal to propagate through the pipeline.")
    await receiver.next_complete_chunk()

    attempts = 5
    advance = 0.2
    acc: np.ndarray | None = None
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    for attempt in range(attempts):
        pdf_report.step(f"Set delay {advance * 1000:.0f}ms in the future (attempt {attempt + 1} / {attempts}).")
        pdf_report.detail("Get current time according to the dsim.")
        now = await correlator.dsim_time()
        target = now + advance
        delays = ["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants
        pdf_report.detail("Set delays.")
        await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", target, *delays)
        pdf_report.step("Receive data for the corresponding dump.")
        target_ts = round(receiver.time_converter.unix_to_adc(target))
        target_acc_ts = target_ts // receiver.timestamp_step * receiver.timestamp_step
        acc = None
        async for timestamp, chunk in receiver.complete_chunks(max_delay=0):
            with chunk:
                pdf_report.detail(f"Received chunk with timestamp {timestamp}, target is {target_acc_ts}.")
                total = np.sum(chunk.data[:, bl_idx, :], axis=0)  # Sum over channels
                if timestamp == target_acc_ts:
                    acc = total
                if timestamp >= target_acc_ts:
                    break
        if acc is not None:
            break

        pdf_report.detail("Did not receive all the expected chunks; reset delay and try again.")
        delays = ["0,0:0,0", "0,0:0,0"] * receiver.n_ants
        await correlator.product_controller_client.request(
            "delays", "antenna_channelised_voltage", receiver.sync_time, *delays
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
    correlator: CorrelatorRemoteControl,
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
        await correlator.product_controller_client.request(
            "delays", "antenna_channelised_voltage", receiver.sync_time, *delays
        )
        finish = asyncio.get_running_loop().time()
        elapsed.append(finish - start)

    receiver = receive_baseline_correlation_products
    channel = 3 * receiver.n_chans // 4
    freq = 3 * receiver.bandwidth / 4
    signal = f"cw(0.1, {freq})"
    gain = compute_tone_gain(receiver, 0.1, 100)
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    elapsed: list[float] = []

    pdf_report.step("Inject tone.")
    pdf_report.detail(f"Set signal to {signal} on both pols.")
    await correlator.dsim_clients[0].request("signals", f"common={signal}; common; common;")
    pdf_report.detail(f"Set gain to {gain} for all inputs.")
    await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

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
        assert phase == pytest.approx(math.pi / 3, abs=np.deg2rad(1))

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
async def test_delay_application_rate(correlator: CorrelatorRemoteControl, pdf_report: Reporter) -> None:
    """Test that delay and phase polynomials are applied at the required rate.

    Verification method
    -------------------
    Verified by analysis. The delay and phase are calculated separately for
    every spectrum. Thus, it is sufficient for the rate of spectra to be high
    enough.
    """
    pdf_report.step("Query rate of spectra.")
    n_samples_between_spectra = correlator.sensors["antenna_channelised_voltage-n-samples-between-spectra"].value
    scale_factor_timestamp = correlator.sensors["antenna_channelised_voltage-scale-factor-timestamp"].value
    rate = scale_factor_timestamp / n_samples_between_spectra
    pdf_report.detail(f"There are {rate:.3f} spectra per second.")
    assert rate >= 2500.0


async def test_delay_sensors(
    correlator: CorrelatorRemoteControl,
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
    receiver = receive_baseline_correlation_products
    delay_tuples = []  # Expected sensor values
    delay_strs = []  # Strings to load
    rng = np.random.default_rng(seed=31)
    now = await correlator.dsim_time()
    load_time = now + 2.0
    load_ts = round(receiver.time_converter.unix_to_adc(load_time))
    for _ in range(receiver.n_inputs):
        delay = rng.uniform(-MAX_DELAY, MAX_DELAY)
        delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE)
        phase = rng.uniform(-np.pi, np.pi)
        phase_rate = rng.uniform(-MAX_PHASE_RATE, MAX_PHASE_RATE)
        delay_strs.append(f"{delay},{delay_rate}:{phase},{phase_rate}")
        delay_tuples.append((load_ts, delay, delay_rate, phase, phase_rate))

    def delay_sensor_value(label: str) -> tuple:
        return literal_eval(correlator.sensors[f"antenna_channelised_voltage-{label}-delay"].value.decode())

    pdf_report.step("Load delays.")
    pdf_report.detail(f"Set delays to load at {load_time} (timestamp {load_ts}).")
    await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", load_time, *delay_strs)
    await asyncio.sleep(0.1)  # Allow time for any invalid sensor updates to propagate
    pdf_report.detail("Check that sensors do not reflect the future.")
    for label in receiver.input_labels:
        value = delay_sensor_value(label)
        with check:
            assert value[1:] == (0.0, 0.0, 0.0, 0.0)
    pdf_report.step("Wait for load time and check sensors.")
    pdf_report.detail(f"Wait for an accumulation with timestamp >= {load_ts}.")
    await receiver.next_complete_chunk(min_timestamp=load_ts)
    for expected, label in zip(delay_tuples, receiver.input_labels):
        value = delay_sensor_value(label)
        pdf_report.detail(f"Input {label} has delay sensor {value}, expected value {expected}.")
        with check:
            assert value == pytest.approx(expected, rel=1e-9), f"Delay sensor for {label} has incorrect value"


def check_phases(
    pdf_report: Reporter,
    actual: np.ndarray,
    expected: np.ndarray,
    caption: str,
    tolerance_deg: float = 1,
) -> None:
    """Compare expected and actual phases to ensure they're within 1°.

    The error in phase is also plotted.
    """
    n_chans = len(actual)
    # Exclude DC component, because it always has zero phase
    actual = actual[1:]
    expected = expected[1:]
    delta = wrap_angle(actual - expected)
    max_error = np.max(delta)
    rms_error = np.sqrt(np.mean(np.square(delta)))
    pdf_report.detail(f"Maximum error is {np.rad2deg(max_error):.3f}°.")
    pdf_report.detail(f"RMS error over channels is {np.rad2deg(rms_error):.5f}°.")
    with check:
        assert np.rad2deg(max_error) <= tolerance_deg, f"Maximum error is more than {tolerance_deg}°"

    fig = Figure(tight_layout=True)
    ax, ax_err = fig.subplots(2)
    x = range(1, n_chans)

    ax.set_title(f"Phase with {caption}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Phase (degrees)")
    ax.xaxis.set_major_locator(POTLocator())
    # It's very noisy, and a thinner linewidth allows more detail to be seen
    ax.plot(x, np.rad2deg(wrap_angle(actual)), linewidth=0.3, label="Actual")
    ax.plot(x, np.rad2deg(wrap_angle(expected)), linewidth=0.3, label="Expected")
    ax.legend()

    ax_err.set_title(f"Phase error with {caption}")
    ax_err.set_xlabel("Channel")
    ax_err.set_ylabel("Error (degrees)")
    ax_err.xaxis.set_major_locator(POTLocator())
    ax_err.plot(x, np.rad2deg(delta), linewidth=0.3)

    pdf_report.figure(fig)


def delay_phase(n_chans: int, delay_samples: float) -> np.ndarray:
    """Calculate expected phase for a given delay.

    The return value is appropriate if the sample signal is provided on both
    inputs, but the first input in the correlation is configured with a delay
    of `delay_samples` samples, and no phase compensation.
    """
    return np.arange(-n_chans // 2, n_chans // 2) / n_chans * np.pi * -delay_samples


async def _test_delay_phase_fixed(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    delay_phases: list[tuple[float, float]],
    caption_cb: Callable[[float, float], str],
    report_residual: bool,
) -> None:
    """Test performance of delay or phase compensation with a fixed value.

    This is the implementation for both :func:`test_delay` and
    :func:`test_delay_phase`.

    Parameters
    ----------
    correlator, receive_baseline_correlation_products, pdf_report
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
    # Minimum, maximum, resolution step, and a small coarse delay
    n_dsims = len(correlator.dsim_clients)
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

    futures = []
    for i, client in enumerate(correlator.dsim_clients):
        signal_spec = "".join(signals[i * N_POLS : (i + 1) * N_POLS])
        pdf_report.detail(f"Set signal to {signal_spec!r} on dsim {i}.")
        futures.append(asyncio.create_task(client.request("signals", signal_spec)))
    await asyncio.gather(*futures)
    for i in range(len(delay_phases)):
        pdf_report.detail(f"Set delay model to {delay_spec[i]} on input {i}")
    await correlator.product_controller_client.request(
        "delays", "antenna_channelised_voltage", receiver.sync_time, *delay_spec
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
        expected = delay_phase(receiver.n_chans, residual) + phase
        check_phases(pdf_report, actual[:, bl_idx], expected, caption)


async def _test_delay_phase_rate(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    rates: list[tuple[float, float]],
    caption_cb: Callable[[float, float], str],
) -> None:
    """Test performance of delay or phase compensation with a rate of change.

    This is the implementation for both :func:`test_delay_rate` and
    :func:`test_phase_rate`.

    Parameters
    ----------
    correlator, receive_baseline_correlation_products, pdf_report
        Fixtures
    rates
        Pairs of (delay_rate, phase_rate) to test
    caption_cb
        Callback to generate a figure caption from a delay rate and phase rate
    """
    receiver = receive_baseline_correlation_products
    # Minimum, maximum, resolution step
    n_dsims = len(correlator.dsim_clients)
    assert N_POLS * n_dsims > len(rates)  # > rather than >= because we need a reference

    pdf_report.step("Set input signals and delays.")
    signal = "common = nodither(wgn(0.05, 1)); common; common;"
    max_period = await get_sensor_val(correlator.dsim_clients[0], "max-period")
    # Choose a period that makes all accumulations the same, so that we can
    # compare accumulations without extraneous noise.
    period = math.gcd(max_period, receiver.n_samples_between_spectra * receiver.n_spectra_per_acc)
    pdf_report.detail(f"Set signal to {signal!r} on all dsims.")
    await asyncio.gather(*[client.request("signals", signal, period) for client in correlator.dsim_clients])
    delay_spec = ["0,0:0,0"] * receiver.n_inputs
    for i, (delay_rate, phase_rate) in enumerate(rates):
        delay_spec[i] = f"0,{delay_rate}:0,{phase_rate}"
        pdf_report.detail(f"Set delay model to {delay_spec[i]} on input {i}")
    now = await correlator.dsim_time()
    await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", now, *delay_spec)

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
        expected = delay_phase(receiver.n_chans, delay_rate * elapsed) + phase_rate * elapsed_s
        # Allow 2° rather than 1° because we're taking the difference between
        # two phases which each have a 1° tolerance.
        check_phases(pdf_report, actual, expected, caption, tolerance_deg=2)


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0185")
async def test_delay(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
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
        correlator,
        receive_baseline_correlation_products,
        pdf_report,
        [(delay, 0.0) for delay in delays],
        lambda delay, phase: f"delay {delay * 1e12:.2f}ps",
        True,
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0185")
async def test_delay_rate(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
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
        correlator,
        receive_baseline_correlation_products,
        pdf_report,
        [(delay_rate, 0.0) for delay_rate in rates],
        lambda delay_rate, phase_rate: f"delay rate {delay_rate}",
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0112")
async def test_delay_phase(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
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
        correlator,
        receive_baseline_correlation_products,
        pdf_report,
        [(0.0, phase) for phase in phases],
        lambda delay, phase: f"phase {phase:.4f} rad ({np.rad2deg(phase):.2f}°)",
        False,
    )


@pytest.mark.requirements("CBF-REQ-0128,CBF-REQ-0112")
async def test_phase_rate(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
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
        correlator,
        receive_baseline_correlation_products,
        pdf_report,
        [(0.0, phase_rate) for phase_rate in rates],
        lambda delay_rate, phase_rate: f"phase rate {phase_rate}",
    )
