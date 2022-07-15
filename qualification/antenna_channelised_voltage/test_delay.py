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
from typing import List, Optional, cast

import aiokatcp
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from katgpucbf import BYTE_BITS, N_POLS
from katgpucbf.fgpu.delay import wrap_angle

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter

MAX_DELAY = 75e-6  # seconds
MAX_DELAY_RATE = 1.9e-9
MAX_PHASE_RATE = 186.13  # rad/second


async def test_delay_application_time(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    """Test that delay/phase changes are applied at the correct time.

    Requirements verified:

    CBF-REQ-0077
        The CBF shall delay execution of Continuous Parameter Control commands until
        a UTC time, as received on the CAM interface, with an execution time
        accuracy of <= 10 ms, provided the command is received at least 200ms
        before the execution time, and the execution time delay is no more than
        2 seconds.

    Verification method:

    Verification by means of test. A 90 degree phase change is loaded for one
    polarisation at a chosen time. The actual application time is estimated by
    checking the ratio of real to imaginary components in the corresponding
    accumulation.
    """
    receiver = receive_baseline_correlation_products

    pdf_report.step("Inject correlated white noise signal.")
    await correlator.dsim_clients[0].request("signals", "common=wgn(0.1); common; common;")
    pdf_report.detail("Wait for updated signal to propagate through the pipeline.")
    _, chunk = await receiver.next_complete_chunk()
    receiver.stream.add_free_chunk(chunk)

    attempts = 5
    advance = 0.2
    acc: Optional[np.ndarray] = None
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    for attempt in range(attempts):
        pdf_report.step(f"Set delay {advance * 1000:.0f}ms in the future (attempt {attempt + 1} / {attempts}).")
        pdf_report.detail("Get current time according to the dsim.")
        now = aiokatcp.decode(float, (await correlator.dsim_clients[0].request("time"))[0][0])
        target = now + advance
        delays = ["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants
        pdf_report.detail("Set delays.")
        await correlator.product_controller_client.request("delays", "antenna_channelised_voltage", target, *delays)
        pdf_report.step("Receive data for the corresponding dump.")
        target_ts = receiver.unix_to_timestamp(target)
        target_acc_ts = target_ts // receiver.timestamp_step * receiver.timestamp_step
        acc = None
        async for timestamp, chunk in receiver.complete_chunks(max_delay=0):
            pdf_report.detail(f"Received chunk with timestamp {timestamp}, target is {target_acc_ts}.")
            assert isinstance(chunk.data, np.ndarray)  # Keeps mypy happy
            total = np.sum(chunk.data[:, bl_idx, :], axis=0)  # Sum over channels
            receiver.stream.add_free_chunk(chunk)
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
    load_time = receiver.timestamp_to_unix(target_acc_ts) + load_frac * receiver.int_time
    delta = load_time - target
    pdf_report.detail(f"Estimated load time error: {delta * 1000:.3f}ms.")
    expect(delta < 0.01)


async def test_delay_enable_disable(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    """Test that delay and phase compensation can be enabled and disabled.

    Requirements verified:

    CBF-REQ-0066
        The CBF shall, on request via the CAM interface, enable or disable
        delay compensation.

    CBF-REQ-0110
        The CBF shall, on request via the CAM interface, enable or disable
        phase compensation.

    CBF-REQ-0200
        The CBF shall receive and apply a complete set of phase-up coefficients
        from SP at a rate of up to once a second.

    Verification method:

    Verified by means of test. Insert a signal with a tone. Enable delay/phase
    compensation and check that it is applied, then disable and check again.
    Check that all requests complete within 1s.
    """

    async def measure_phase() -> float:
        """Retrieve the phase of the chosen channel from the next chunk."""
        _, chunk = await receiver.next_complete_chunk()
        assert isinstance(chunk.data, np.ndarray)
        value = chunk.data[channel, bl_idx, :]
        phase = np.arctan2(value[1], value[0])
        receiver.stream.add_free_chunk(chunk)
        return phase

    async def set_delays(delays: List[str]) -> None:
        start = asyncio.get_running_loop().time()
        await correlator.product_controller_client.request(
            "delays", "antenna_channelised_voltage", receiver.sync_time, *delays
        )
        finish = asyncio.get_running_loop().time()
        elapsed.append(finish - start)

    receiver = receive_baseline_correlation_products
    channel = receiver.n_chans // 2
    freq = receiver.bandwidth / 2
    signal = f"cw(0.01, {freq})"
    bl_idx = receiver.bls_ordering.index((receiver.input_labels[0], receiver.input_labels[1]))
    elapsed: List[float] = []

    pdf_report.step("Inject tone.")
    pdf_report.detail(f"Set signal to {signal} on both pols.")
    await correlator.dsim_clients[0].request("signals", f"common={signal}; common; common;")

    pdf_report.step("Check that phase compensation can be enabled.")
    pdf_report.detail("Apply 90 degree phase to one pol")
    await set_delays(["0,0:0,0", f"0,0:{math.pi / 2},0"] * receiver.n_ants)
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    expect(phase == pytest.approx(-math.pi / 2, np.deg2rad(1)))

    pdf_report.step("Check that delay compensation can be enabled.")
    pdf_report.detail("Apply 1/4 cycle delay to one pol.")
    await set_delays(["0,0:0,0", f"{0.25 / freq},0:0,0"] * receiver.n_ants)
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    expect(phase == pytest.approx(-math.pi / 2, np.deg2rad(1)))

    pdf_report.step("Check that compensation can be disabled.")
    await set_delays(["0,0:0,0"] * (2 * receiver.n_ants))
    phase = await measure_phase()
    pdf_report.detail(f"Phase is {np.rad2deg(phase):.3f} degrees.")
    expect(phase == pytest.approx(0, np.deg2rad(1)))

    pdf_report.step("Check update time.")
    max_elapsed = max(elapsed)
    pdf_report.detail(f"Maximum time for ?delays request is {max_elapsed:.3f}s.")
    expect(max_elapsed < 1.0)


async def test_delay_sensors(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    r"""Test that delay sensors work correctly.

    Requirements verified: none

    Verification method:

    Verified by test. Load a set of random delays with a load time in the
    future. Once that time arrives, check that the sensors report the correct
    values.
    """
    receiver = receive_baseline_correlation_products
    delay_tuples = []  # Expected sensor values
    delay_strs = []  # Strings to load
    rng = np.random.default_rng(seed=31)
    now = aiokatcp.decode(float, (await correlator.dsim_clients[0].request("time"))[0][0])
    load_time = now + 2.0
    load_ts = receiver.unix_to_timestamp(load_time)
    for _ in range(receiver.n_inputs):
        delay = rng.uniform(0.0, MAX_DELAY)
        delay_rate = rng.uniform(-MAX_DELAY_RATE, MAX_DELAY_RATE)
        phase = rng.uniform(-np.pi, np.pi)
        phase_rate = rng.uniform(0.0, MAX_PHASE_RATE)
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
        expect(value[1:] == (0.0, 0.0, 0.0, 0.0))
    pdf_report.step("Wait for load time and check sensors.")
    pdf_report.detail(f"Wait for an accumulation with timestamp >= {load_ts}.")
    _, chunk = await receiver.next_complete_chunk(min_timestamp=load_ts)
    receiver.stream.add_free_chunk(chunk)
    for expected, label in zip(delay_tuples, receiver.input_labels):
        value = delay_sensor_value(label)
        pdf_report.detail(f"Input {label} has delay sensor {value}, expected value {expected}.")
        expect(value == pytest.approx(expected, rel=1e-9), f"Delay sensor for {label} has incorrect value")


async def test_delay(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    r"""Test performance of delay compensation.

    Requirements verified:

    CBF-REQ-0128
        The CBF shall have an overall per-antenna phase error of
        :math:`\le \ang{1}` RMS for correlation products, including
        quantisation effects and imperfect phase tracking.

    CBF-REQ-0185
        The CBF shall apply the delay polynomial, as provided via the CAM
        interface, with the following criteria:

        1. range of delay: 0 to :math:`\ge \SI{75}{\micro\second}`.
        2. resolution of delay: :math:`\le \SI{2.5}{\pico\second}`.
        3. range of rate of change of delay:
           :math:`\le \SI{500}{\pico\second\per\second}` to
           :math:`\ge \SI{1.9}{\nano\second\per\second}`.
        4. resolution of rate of change of delay:
           :math:`\le \SI{2.5}{\pico\second\per\second}`.

    Verification method:

    Verified by test. Set a variety of delays on different inputs. Delay the
    corresponding dsim signal by the same amount, rounded to the nearest 8
    samples. Check that the resulting phases are within :math:`\ang{1}` degree
    of the expected value.
    """
    receiver = receive_baseline_correlation_products
    # Minimum, maximum, resolution step, and a small coarse delay
    delays = [0.0, MAX_DELAY, 2.5e-12, 2.75 / receiver.scale_factor_timestamp]
    n_dsims = len(correlator.dsim_clients)
    assert N_POLS * n_dsims > len(delays)  # > rather than >= because we need a reference

    pdf_report.step("Set input signals and delays.")
    base_signal = "wgn(0.05, 1)"
    signals = [base_signal + ";"] * (N_POLS * n_dsims)
    delay_spec = ["0,0:0,0"] * receiver.n_inputs
    delay_samples = []
    for i, delay in enumerate(delays):
        # It's more efficient for the dsim to delay by a multiple of 8 samples
        delay_samples.append(round(delay * receiver.scale_factor_timestamp / BYTE_BITS) * BYTE_BITS)
        signals[i] = f"delay({base_signal}, {-delay_samples[-1]});"
        delay_spec[i] = f"{delay},0:0,0"

    futures = []
    for i, client in enumerate(correlator.dsim_clients):
        signal_spec = "".join(signals[i * N_POLS : (i + 1) * N_POLS])
        pdf_report.detail(f"Set signal to {signal_spec!r} on dsim {i}.")
        futures.append(asyncio.create_task(client.request("signals", signal_spec)))
    await asyncio.gather(*futures)
    pdf_report.detail(f"Set delays: {delays}.")
    await correlator.product_controller_client.request(
        "delays", "antenna_channelised_voltage", receiver.sync_time, *delay_spec
    )

    pdf_report.step("Verify results")
    pdf_report.detail("Receive an accumulation")
    _, chunk = await receiver.next_complete_chunk()
    assert isinstance(chunk.data, np.ndarray)
    # Convert to floating-point complex values for easier analysis
    data = chunk.data.astype(np.float64).view(np.complex128)[..., 0]
    # cast to work around https://github.com/numpy/numpy/issues/21972
    phase = cast(NDArray[np.float64], np.angle(data))
    receiver.stream.add_free_chunk(chunk)

    for i, delay in enumerate(delays):
        # The delay is mostly cancelling out the delay applied in the dsim, but
        # there will be fine delay left over
        residual = delay_samples[i] - delay * receiver.scale_factor_timestamp
        pdf_report.detail(f"Testing delay {delay * 1e12:.2f}ps (residual delay {residual:.6f} samples)")
        input1 = receiver.input_labels[i]
        input2 = receiver.input_labels[-1]
        bl_idx = receiver.bls_ordering.index((input1, input2))
        actual = phase[:, bl_idx]
        expected = np.arange(-receiver.n_chans // 2, receiver.n_chans // 2) / receiver.n_chans * np.pi * residual
        # Exclude DC component, because it always has zero phase
        actual = actual[1:]
        expected = expected[1:]
        delta = wrap_angle(actual - expected)
        max_error = np.max(delta)
        rms_error = np.sqrt(np.mean(np.square(delta)))
        pdf_report.detail(f"Maximum error is {np.rad2deg(max_error):.3f} degrees.")
        pdf_report.detail(f"RMS error is {np.rad2deg(rms_error):.5f} degrees.")
        expect(np.rad2deg(max_error) <= 1.0, "Maximum error is more than 1 degree")

        fig = Figure()
        ax = fig.subplots()
        ax.set_title(f"Phase error with delay={delay * 1e12:.2f}ps")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Error (degrees)")
        x = np.arange(1, receiver.n_chans)
        ax.plot(x, np.rad2deg(delta))
        pdf_report.figure(
            fig,
            tikzplotlib_kwargs=dict(
                extra_axis_parameters=[
                    "scaled ticks=false",  # Prevent common factor being extracted
                    "yticklabel style={/pgf/number format/fixed}",  # Prevent scientific notation
                ]
            ),
        )
