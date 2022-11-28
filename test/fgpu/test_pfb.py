################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""Unit tests for PFB, for numerical correctness."""

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf import BYTE_BITS
from katgpucbf.fgpu import SAMPLE_BITS, pfb

from .. import unpackbits

pytestmark = [pytest.mark.cuda_only]


def pfb_fir_host(data, channels, unzip_factor, weights):
    """Apply a PFB-FIR filter to a set of data on the host."""
    step = 2 * channels
    assert len(weights) % step == 0
    taps = len(weights) // step
    decoded = unpackbits(data)
    window_size = 2 * channels * taps
    out = np.empty((len(decoded) // step - taps + 1, step), np.float32)
    for i in range(0, len(out)):
        windowed = decoded[i * step : i * step + window_size] * weights
        out[i] = np.sum(windowed.reshape(-1, step), axis=0)
    # Unzip
    out = out.reshape(-1, channels // unzip_factor, unzip_factor, 2)
    out = out.swapaxes(1, 2)
    out = out.reshape(-1, step)
    total_power = np.sum(np.square(decoded[step * (taps - 1) :].astype(np.int64)))
    return out, total_power


@pytest.mark.parametrize("unzip_factor", [1, 2, 4])
def test_pfb_fir(context: AbstractContext, command_queue: AbstractCommandQueue, unzip_factor: int) -> None:
    """Test the GPU PFB-FIR for numerical correctness."""
    taps = 16
    spectra = 3123
    channels = 4096
    samples = 2 * channels * (spectra + taps - 1)
    rng = np.random.default_rng(seed=1)
    h_in = rng.integers(0, 256, samples * SAMPLE_BITS // BYTE_BITS, np.uint8)
    weights = rng.uniform(-1.0, 1.0, (2 * channels * taps,)).astype(np.float32)
    expected, expected_total_power = pfb_fir_host(h_in, channels, unzip_factor, weights)

    template = pfb.PFBFIRTemplate(context, taps, channels, unzip_factor)
    fn = template.instantiate(command_queue, samples, spectra)
    fn.ensure_all_bound()
    fn.buffer("in").set(command_queue, h_in)
    fn.buffer("weights").set(command_queue, weights)
    fn.buffer("total_power").zero(command_queue)
    # Split into two parts to test the offsetting
    fn.in_offset = 0
    fn.out_offset = 0
    fn.spectra = 1003
    fn()
    fn.in_offset = fn.spectra * 2 * channels
    fn.out_offset = fn.spectra
    fn.spectra = spectra - fn.spectra
    fn()
    h_out = fn.buffer("out").get(command_queue)
    h_total_power = fn.buffer("total_power").get(command_queue)[()]
    np.testing.assert_allclose(h_out, expected, rtol=1e-5, atol=1e-3)
    assert h_total_power == expected_total_power
