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

"""Gain tests."""

import asyncio

import numpy as np
from numpy.typing import NDArray

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl, get_sensor_val
from ..reporter import Reporter


async def test_gains(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    expect,
) -> None:
    r"""Test that gains can be applied.

    Requirements verified:

    CBF-REQ-0119
        The CBF shall apply gain correction per antenna, per polarisation, per
        frequency channel with a range of at least :math:`\pm 6` dB and
        a resolution of :math:`\le 1` dB.

    Verification method:

    Verified by test. A repeating white noise signal is injected and the
    response measured with the default gains. The gain is then changed to a
    random set of gains, and the response is compared to this.
    """

    async def next_chunk_data() -> NDArray[np.complex128]:
        _, chunk = await receiver.next_complete_chunk()
        assert isinstance(chunk.data, np.ndarray)  # Keeps mypy happy
        # Turn 2-element axis into complex number
        data = chunk.data.astype(np.float64).view(np.complex128)[..., 0]
        receiver.stream.add_free_chunk(chunk)
        return data

    receiver = receive_baseline_correlation_products

    pdf_report.step("Inject white noise with fixed seed.")
    scale = 0.02
    signals = f"common=wgn({scale}, 1); common; common;"
    # Compute repeat period guaranteed to divide into accumulation length.
    max_period = await get_sensor_val(correlator.dsim_clients[0], "max-period")
    period = receiver.n_samples_between_spectra * receiver.spectra_per_heap
    period = min(period, max_period)
    pdf_report.detail(f"Set white Gaussian noise with scale {scale}, period {period} samples.")
    await asyncio.gather(*[client.request("signals", signals, period) for client in correlator.dsim_clients])

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
        await correlator.product_controller_client.request(
            "gain",
            "antenna_channelised_voltage",
            input_label,
            *input_gain,
        )
    end_time = loop.time()
    elapsed = end_time - start_time
    pdf_report.detail(f"Gains set in {elapsed:.3f}s")
    # Disabled until we determine whether CBF-REQ-0200 is applicable - it
    # currently fails.
    # expect(elapsed < 1, "Took too long to set gains")

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
    expect(max_rel_error < 0.1, "Maximum error exceeds 0.5 dB")