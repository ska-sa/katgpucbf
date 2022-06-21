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

"""Baseline and accumulation consistency tests."""

import matplotlib.figure
import numpy as np

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl, get_sensor_val
from ..reporter import Reporter


async def test_consistency(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that repeated calculations give consistent results.

    Verification method:

    Provide the same simulated digitiser data to every antenna, with a
    period that divides into the accumulation length. Verify that every
    baseline produces the same bit-wise output, and that two successive
    accumulations produce the same bit-wise output.
    """
    receiver = receive_baseline_correlation_products  # Just to reduce typing

    pdf_report.step("Configure the D-sim with Gaussian noise.")
    amplitude = 0.2
    max_period = await get_sensor_val(correlator.dsim_clients[0], "max-period")
    period = receiver.n_samples_between_spectra * receiver.spectra_per_heap
    period = min(period, max_period)
    await correlator.dsim_clients[0].request("signals", f"common=wgn({amplitude});common;common;", period)
    pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}, period {period} on both pols.")

    pdf_report.step("Collect two accumulations of data.")
    data = []
    for _ in range(2):
        timestamp, chunk = await receiver.next_complete_chunk()
        pdf_report.detail(f"Received accumulation with timestamp {timestamp}")
        assert isinstance(chunk.data, np.ndarray)
        # Separate axes into antenna baseline and Jones term. The dsim uses
        # different random dithers for the pols so we don't expect consistency
        # between Jones terms.
        chunk_data = chunk.data.reshape(chunk.data.shape[0], -1, 4, 2)
        data.append(chunk_data.copy())  # Copy since we've returning the chunk
        receiver.stream.add_free_chunk(chunk)

    pdf_report.step("Verify consistency")
    expected = data[0][:, 0:1, ...].repeat(data[0].shape[1], axis=1)
    np.testing.assert_array_equal(data[0], expected)
    pdf_report.detail("All baselines are equal in the first accumulation")
    np.testing.assert_array_equal(data[0], data[1])
    pdf_report.detail("Accumulations are equal")

    pdf_report.step("Plot spectrum")
    fig = matplotlib.figure.Figure()
    ax = fig.subplots()
    ax.plot(10 * np.log10(data[0][:, 0, 0, 0] / (2**31 - 1)))
    ax.set_title("Power")
    ax.set_xlabel("Channel")
    ax.set_ylabel("dBfs")
    pdf_report.figure(fig)
