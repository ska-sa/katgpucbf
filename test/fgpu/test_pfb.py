################################################################################
# Copyright (c) 2020-2021, 2023 National Research Foundation (SARAO)
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

from katgpucbf import BYTE_BITS, N_POLS
from katgpucbf.fgpu import DIG_SAMPLE_BITS_VALID, pfb

from .. import unpackbits

pytestmark = [pytest.mark.cuda_only]
TAPS = 16
SPECTRA = 3123
CHANNELS = 4096


def pfb_fir_host_real(data, channels, input_sample_bits, unzip_factor, weights):
    """Apply a PFB-FIR filter to a set of packed real data on the host."""
    step = 2 * channels
    assert len(weights) % step == 0
    taps = len(weights) // step
    n_pols = data.shape[0]
    decoded = np.stack([unpackbits(pol_data, input_sample_bits) for pol_data in data])
    window_size = 2 * channels * taps
    out = np.empty((n_pols, decoded.shape[1] // step - taps + 1, step), np.float32)
    for i in range(0, out.shape[1]):
        windowed = decoded[:, i * step : i * step + window_size] * weights
        out[:, i] = np.sum(windowed.reshape(n_pols, taps, step), axis=1)
    # Unzip
    out = out.reshape(n_pols, -1, channels // unzip_factor, unzip_factor, 2)
    out = out.swapaxes(2, 3)
    out = out.reshape(n_pols, -1, step)
    total_power = np.sum(np.square(decoded[:, step * (taps - 1) :].astype(np.int64)), axis=1)
    return out, total_power


def pfb_fir_host_complex(data, channels, unzip_factor, weights):
    """Apply a PFB-FIR filter to a set of complex data on the host."""
    assert len(weights) % channels == 0
    assert data.shape[1] % channels == 0
    n_pols = data.shape[0]
    taps = len(weights) // channels
    window_size = channels * taps
    out = np.empty((n_pols, data.shape[1] // channels - taps + 1, channels), np.complex64)
    for i in range(0, out.shape[1]):
        windowed = data[:, i * channels : i * channels + window_size] * weights
        out[:, i] = np.sum(windowed.reshape(n_pols, taps, channels), axis=1)
    # Unzip
    out = out.reshape(n_pols, -1, channels // unzip_factor, unzip_factor)
    out = out.swapaxes(2, 3)
    out = out.reshape(n_pols, -1, channels)
    return out


def _pfb_fir(fn: pfb.PFBFIR, h_in: np.ndarray, weights: np.ndarray, step: int) -> np.ndarray:
    """Run common parts of the different PFB tests and return output on the host."""
    command_queue = fn.command_queue
    fn.buffer("in").set(command_queue, h_in)
    fn.buffer("weights").set(command_queue, weights)
    # Split into two parts to test the offsetting
    fn.in_offset[:] = 0
    fn.out_offset = 0
    fn.spectra = 1003
    fn()
    fn.in_offset[:] = fn.spectra * step
    fn.out_offset = fn.spectra
    fn.spectra = SPECTRA - fn.spectra
    fn()
    return fn.buffer("out").get(command_queue)


@pytest.mark.combinations(
    "input_sample_bits,unzip_factor",
    DIG_SAMPLE_BITS_VALID,
    [1, 2, 4],
)
def test_pfb_fir_real(
    context: AbstractContext, command_queue: AbstractCommandQueue, input_sample_bits: int, unzip_factor: int
) -> None:
    """Test the real GPU PFB-FIR for numerical correctness."""
    samples = 2 * CHANNELS * (SPECTRA + TAPS - 1)
    rng = np.random.default_rng(seed=1)
    h_in = rng.integers(0, 256, (N_POLS, samples * input_sample_bits // BYTE_BITS), np.uint8)
    weights = rng.uniform(-1.0, 1.0, (2 * CHANNELS * TAPS,)).astype(np.float32)
    expected_out, expected_total_power = pfb_fir_host_real(h_in, CHANNELS, input_sample_bits, unzip_factor, weights)

    template = pfb.PFBFIRTemplate(context, TAPS, CHANNELS, input_sample_bits, unzip_factor, n_pols=N_POLS)
    fn = template.instantiate(command_queue, samples, SPECTRA)
    fn.ensure_all_bound()
    fn.buffer("total_power").zero(command_queue)
    h_out = _pfb_fir(fn, h_in, weights, 2 * CHANNELS)
    h_total_power = fn.buffer("total_power").get(command_queue)
    np.testing.assert_allclose(h_out, expected_out, rtol=1e-5, atol=1e-6 * 2**input_sample_bits)
    np.testing.assert_equal(h_total_power, expected_total_power)


@pytest.mark.parametrize("unzip_factor", [1, 2, 4])
def test_pfb_fir_complex(context: AbstractContext, command_queue: AbstractCommandQueue, unzip_factor: int) -> None:
    samples = CHANNELS * (SPECTRA + TAPS - 1)
    shape = (N_POLS, samples)
    rng = np.random.default_rng(seed=1)
    h_in = rng.normal(0.0, 128.0, size=shape) + 1j * rng.normal(0.0, 128.0, size=shape)
    h_in = h_in.astype(np.complex64)
    weights = rng.uniform(-1.0, 1.0, (CHANNELS * TAPS,)).astype(np.float32)
    expected_out = pfb_fir_host_complex(h_in, CHANNELS, unzip_factor, weights)

    template = pfb.PFBFIRTemplate(context, TAPS, CHANNELS, 32, unzip_factor, n_pols=N_POLS, complex_input=True)
    fn = template.instantiate(command_queue, samples, SPECTRA)
    fn.ensure_all_bound()
    h_out = _pfb_fir(fn, h_in, weights, CHANNELS)
    np.testing.assert_allclose(h_out, expected_out, rtol=1e-5, atol=1e-3)
