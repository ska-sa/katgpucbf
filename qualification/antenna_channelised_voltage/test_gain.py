################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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

"""Gain tests."""

import asyncio

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest_check import check

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0119")
async def test_gains(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test that gains can be applied.

    Verification method
    -------------------
    Verified by test. A repeating white noise signal is injected and the
    response measured with the default gains. The gain is then changed to a
    random set of gains, and the response is compared to this.
    """

    async def next_chunk_data() -> NDArray[np.complex128]:
        _, chunk_data = await receiver.next_complete_chunk()
        # Turn 2-element axis into complex number
        data = chunk_data.astype(np.float64).view(np.complex128)[..., 0]
        return data

    receiver = receive_baseline_correlation_products

    pdf_report.step("Inject white noise with fixed seed.")
    scale = 0.03
    signals = f"common=wgn({scale}, 1); common; common;"
    # Compute repeat period guaranteed to divide into accumulation length.
    max_period = await cbf.dsim_clients[0].sensor_value("max-period", int)
    period = receiver.n_samples_between_spectra * receiver.n_spectra_per_heap
    period = min(period, max_period)
    pdf_report.detail(f"Set white Gaussian noise with scale {scale}, period {period} samples.")
    await asyncio.gather(*[client.request("signals", signals, period) for client in cbf.dsim_clients])

    pdf_report.step("Measure response with default gain.")
    orig = await next_chunk_data()

    pdf_report.step("Set random gains.")
    shape = (receiver.n_inputs, receiver.n_chans)
    rng = np.random.default_rng(seed=2)
    mag = rng.uniform(0.5, 2.0, size=shape)
    phase = rng.uniform(-np.pi, np.pi, size=shape)
    gains = mag * np.exp(1j * phase)
    gains_text = [[f"{gain.real}{gain.imag:+}j" for gain in input_gain] for input_gain in gains]

    loop = asyncio.get_running_loop()
    start_time = loop.time()
    for input_gain, input_label in zip(gains_text, receiver.input_labels):
        await cbf.product_controller_client.request(
            "gain",
            "antenna-channelised-voltage",
            input_label,
            *input_gain,
        )
    end_time = loop.time()
    elapsed = end_time - start_time
    pdf_report.detail(f"Gains set in {elapsed:.3f}s")
    # MKAT-ECP-277 specifies 300s between gain updates, but we'd like to be
    # able to set gains faster than that. This is an arbitrary number.
    with check:
        assert elapsed < 30.0, "Took too long to set gains"

    pdf_report.step("Collect and compare results.")
    data = await next_chunk_data()
    max_rel_error = 0.0
    for i, bl in enumerate(receiver.bls_ordering):
        a = receiver.input_labels.index(bl[0])
        b = receiver.input_labels.index(bl[1])
        expected_gain = gains[a] * gains[b].conj()
        expected = orig[:, i] * expected_gain
        rel_error = np.abs(data[:, i] - expected) / np.abs(expected)
        max_rel_error = max(max_rel_error, np.max(rel_error))
    pdf_report.detail(f"Maximum relative error: {max_rel_error}.")
    # 10^-0.05 ~= 0.9, so a relative error of <0.1 implies that the gain is
    # accurate to 0.5 dB (in power - there is no spec for phase), and hence
    # must have a resolution of 1 dB or better.
    # In reality the limit on accuracy is likely to be the F-engine output
    # quantisation, since gain is handled as single-precision float.
    with check:
        assert max_rel_error < 0.1, "Maximum error exceeds 0.5 dB"


@pytest.mark.name("Ordering of gains and capture-start")
@pytest.mark.no_capture_start("baseline-correlation-products")
async def test_gains_capture_start(
    cbf: CBFRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Test that gains applied before capture-start are not delayed.

    Verification methods
    --------------------
    Verified by test. Change the gains, then immediately issue a capture-start
    request. Verify that the received data reflects the change in gains.
    """
    receiver = receive_baseline_correlation_products
    pcc = cbf.product_controller_client

    pdf_report.step("Inject white noise on first antenna.")
    signals = "common = wgn(0.1); common; common;"
    await cbf.dsim_clients[0].request("signals", signals)
    dsim_timestamp = await cbf.dsim_clients[0].sensor_value("steady-state-timestamp", int)
    pdf_report.detail(f"Set dsim signals to {signals}, starting with timestamp {dsim_timestamp}.")

    pdf_report.step("Wait for injected signal to reach F-engine.")
    label = receiver.input_labels[0]
    for _ in range(10):
        rx_timestamp = await pcc.sensor_value(f"antenna-channelised-voltage.{label}.rx.timestamp", int)
        pdf_report.detail(f"rx.timestamp = {rx_timestamp}")
        if rx_timestamp >= dsim_timestamp:
            break
        else:
            pdf_report.detail("Sleep for 0.5s.")
            await asyncio.sleep(0.5)
    else:
        pytest.fail("Digitiser signal did not reach F-engine.")

    pdf_report.step("Set gains on input 0")
    gains = np.ones(receiver.n_chans)
    cut = receiver.n_chans // 2
    gains[cut:] = 0  # Zero out the upper half of the band
    await pcc.request("gain", "antenna-channelised-voltage", label, *gains)
    pdf_report.detail(f"Upper half of band on {label} set to zero gain.")

    pdf_report.step("Capture and verify output")
    await pcc.request("capture-start", "baseline-correlation-products")
    _, data = await receiver.next_complete_chunk(min_timestamp=0)
    bls_idx = receiver.bls_ordering.index((label, label))
    data = data[:, bls_idx, 0]  # 0 to take just the real part (these are auto-correlations)
    assert np.max(data[cut:]) == 0
    # It's random, so technically it's possible for any of the values to be
    # zero, but exceedingly unlikely.
    assert np.min(data[:cut]) > 0
    pdf_report.detail("Output reflects effects of gains.")
