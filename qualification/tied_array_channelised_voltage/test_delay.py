################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Delay test."""

import time
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytest_check import check

from katgpucbf.fgpu.delay import wrap_angle

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import POTLocator, Reporter


@pytest.mark.requirements("CBF-REQ-0220")
async def test_delay_small(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test beam steering delay application, for small delays.

    Verification method
    -------------------
    Verification by means of test. Set a delay on one input and form a beam
    from it with a compensating delay. Use a different input with no delay
    to form a reference beam. Check that the results are consistent to within 2
    ULP.

    This test is only valid for delays of less than half a sample. For larger
    delays, the F-engine delay is done partially in the time domain, while the
    compensating beam delay is purely a phase correction, and so they aren't
    expected to cancel out. The delays tested are a spread of positive and
    negative values in the picosecond range.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    # Small amplitude so that we don't saturate, as delaying and undelaying a
    # saturated value won't round-trip properly
    await cbf.dsim_gaussian(16.0, pdf_report)

    pdf_report.step("Choose random inputs for delay and reference beams and set weights.")
    rng = np.random.default_rng(seed=123)
    delay_beam = 0
    delay_input_idx = rng.integers(len(receiver.source_indices[delay_beam]))
    delay_input = receiver.source_indices[delay_beam][delay_input_idx]
    pdf_report.detail(f"Using input {delay_input} for delay beam.")

    ref_beam = 1
    ref_input_idx = rng.integers(len(receiver.source_indices[ref_beam]))
    ref_input = receiver.source_indices[ref_beam][ref_input_idx]
    pdf_report.detail(f"Using input {ref_input} for reference beam.")
    # Should never happen because they're different polarisations
    assert delay_input != ref_input

    delay_weights = [0.0] * len(receiver.source_indices[delay_beam])
    delay_weights[delay_input_idx] = 1.0
    await client.request("beam-weights", receiver.stream_names[delay_beam], *delay_weights)
    pdf_report.detail(f"Set weights on {receiver.stream_names[delay_beam]} to {delay_weights}")
    ref_weights = [0.0] * len(receiver.source_indices[ref_beam])
    ref_weights[ref_input_idx] = 1.0
    await client.request("beam-weights", receiver.stream_names[ref_beam], *ref_weights)
    pdf_report.detail(f"Set weights on {receiver.stream_names[ref_beam]} to {delay_weights}")

    # TODO: need the final version of the requirements to know what values to test
    max_delay = 0.5 / receiver.adc_sample_rate
    delay_phases = [
        (-200e-12, -np.pi / 2),
        (-10e-12, -1.0),
        (0.0, 0.35),
        (16e-12, 0.0),
        (400e-12, np.pi / 2),
    ]
    for delay, phase in delay_phases:
        pdf_report.step(f"Test with delay {delay * 1e12} ps.")
        if abs(delay) > max_delay:
            pdf_report.detail(f"Skipping because delay > max_delay ({max_delay * 1e12:.1} ps).")
            continue
        # Ensure load time is in the past, so that it is already applied when we
        # receive data.
        load_time = await cbf.dsim_time() - 5.0
        input_delays = ["0,0:0,0"] * receiver.n_inputs
        input_delays[delay_input] = f"{delay},0:{phase},0"
        await client.request("delays", "antenna-channelised-voltage", load_time, *input_delays)
        pdf_report.detail(f"Set input delays to {input_delays}")
        beam_delays = ["0:0"] * len(receiver.source_indices[delay_beam])
        beam_delays[delay_input_idx] = f"{-delay}:{-phase}"
        await client.request("beam-delays", receiver.stream_names[delay_beam], *beam_delays)
        pdf_report.detail(f"Set beam {delay_beam} delays to {beam_delays}")
        timestamp, data = await receiver.next_complete_chunk()
        pdf_report.detail(f"Received chunk with timestamp {timestamp}")
        # Need more precision to avoid overflows when subtracting
        data = data.astype(np.int16)
        max_error = np.max(np.abs(data[delay_beam] - data[ref_beam]))
        with check:
            assert max_error <= 2
        pdf_report.detail(f"Maximum difference is {max_error} ULP")


@pytest.mark.requirements("CBF-REQ-0220")
async def test_delay(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test beam steering delay application.

    Verification method
    -------------------
    Verification by means of test. Set a delay on one beam (for all inputs)
    and no delay on another. Check that the results on the test (delayed)
    beam match expectations computed from the reference (no delay) beam
    within 2.9 ULP. This allows for 1 ULP tolerance in both the real and
    imaginary components of both beams.

    Correlate the beams and check that the angle of the correlation product
    matches expectations to within 1°.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    # Small amplitude so that we don't saturate.
    await cbf.dsim_gaussian(16.0, pdf_report)

    delay_beam = 0
    delay_name = receiver.stream_names[delay_beam]
    ref_beam = 1
    ref_name = receiver.stream_names[ref_beam]
    n_indices = len(receiver.source_indices[delay_beam])
    quant_gain = 1.0 / n_indices
    pdf_report.step("Set beam gains.")
    await client.request("beam-quant-gains", delay_name, quant_gain)
    pdf_report.detail(f"Set gain for {delay_name} to {quant_gain}.")
    await client.request("beam-quant-gains", ref_name, quant_gain)
    pdf_report.detail(f"Set gain for {ref_name} to {quant_gain}.")

    channel_width = receiver.bandwidth / receiver.n_chans
    # Per-channel coefficient for delay -> phase mapping
    phase_scale = -2 * np.pi * (np.arange(receiver.n_chans) - receiver.n_chans / 2) * channel_width

    # TODO: need requirements to know what values to test
    delay_phases = [
        (-509e-9, 1.0),
        (-100e-9, np.pi / 2),
        (15.9e-12, 0.0),
        (509e-9, -np.pi / 2),
    ]
    for delay, phase in delay_phases:
        pdf_report.step(f"Test with delay {delay * 1e12} ps and phase {phase}.")
        await client.request("beam-delays", delay_name, *([f"{delay}:{phase}"] * n_indices))
        pdf_report.detail(f"Set beam delays on {delay_name} to {delay}:{phase} on all inputs.")
        timestamp, data = await receiver.next_complete_chunk()
        pdf_report.detail(f"Received chunk with timestamp {timestamp}.")

        data = data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
        expected_phase = phase_scale * delay + phase
        rotate = np.cos(expected_phase) + 1j * np.sin(expected_phase)
        expected = data[ref_beam] * rotate[:, np.newaxis]
        max_error = np.max(np.abs(data[delay_beam] - expected))
        pdf_report.detail(f"Maximum difference from expected is {max_error:.3f} ULP.")
        with check:
            assert max_error <= 2.9  # A bit more than 2*sqrt(2)

        corr = np.sum(data[delay_beam] * data[ref_beam].conj(), axis=1)
        # Collect more chunks so that quantisation effects average out
        n_chunks = 10
        for _ in range(n_chunks - 1):
            timestamp, data = await receiver.next_complete_chunk()
            data = data.astype(np.float64).view(np.complex128)[..., 0]  # Convert to complex128
            corr += np.sum(data[delay_beam] * data[ref_beam].conj(), axis=1)
        pdf_report.detail(f"Correlated delay and reference beams over {n_chunks} chunks.")

        corr_phase = np.angle(corr)
        fig = Figure(tight_layout=True)
        # matplotlib's typing doesn't specialise for Nx1 case
        ax, ax_err = cast(Sequence[Axes], fig.subplots(2))
        x = range(receiver.n_chans)
        delta = wrap_angle(corr_phase - expected_phase)
        max_error_deg = np.max(np.abs(np.rad2deg(delta)))
        pdf_report.detail(f"Maximum phase error is {max_error_deg:.3f}°.")
        with check:
            assert max_error_deg < 1.0

        ax.set_title(f"Phase with delay {delay}:{phase}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Phase (degrees)")
        ax.xaxis.set_major_locator(POTLocator())
        ax.plot(x, np.rad2deg(corr_phase), label="Actual")
        ax.plot(x, np.rad2deg(wrap_angle(expected_phase)), label="Expected")
        ax.legend()

        ax_err.set_title(f"Phase error with delay {delay}:{phase}")
        ax_err.set_xlabel("Channel")
        ax_err.set_ylabel("Error (degrees)")
        ax_err.xaxis.set_major_locator(POTLocator())
        ax_err.plot(x, np.rad2deg(delta))

        pdf_report.figure(fig)


@pytest.mark.requirements("CBF-REQ-0076")
async def test_delay_update_time(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that delay updates are fast enough.

    Verification method
    -------------------
    Verified by means of test. Measure the time taken to issue
    ``?beam-delays`` requests for all beams.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    # Use random values just because they'll have lots of significant digits
    # and hence take longer to format/parse.
    rng = np.random.default_rng(1)
    params = [
        [f"{rng.uniform(-100e-9, 100e-9)}:{rng.uniform(-np.pi, np.pi)}" for _ in source_indices]
        for source_indices in receiver.source_indices
    ]

    pdf_report.step("Measure delay setting time")
    start = time.monotonic()
    pdf_report.detail(f"Start time is {start:.6f}.")
    for stream_name, stream_params in zip(receiver.stream_names, params):
        await client.request("beam-delays", stream_name, *stream_params)
        pdf_report.detail(f"Set delays on {stream_name} to {stream_params}")
    stop = time.monotonic()
    pdf_report.detail(f"Stop time is {stop:.6f}.")
    elapsed = stop - start
    pdf_report.detail(f"Elapsed time is {elapsed:.6f}.")
    assert elapsed < 5.0
