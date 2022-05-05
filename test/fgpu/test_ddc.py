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

"""Unit tests for digital down conversion."""

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf import BYTE_BITS
from katgpucbf.fgpu import SAMPLE_BITS
from katgpucbf.fgpu.ddc import DDCTemplate

from .test_pfb import decode_10bit_host


def ddc_host(samples: np.ndarray, weights: np.ndarray, decimation: int, mix_frequency: float) -> np.ndarray:
    """Implement the DDC calculation simply in numpy."""
    # Calculation is done in double precision for better accuracy
    samples = decode_10bit_host(samples).astype(np.complex128)
    mix_angle = 2 * np.pi * mix_frequency * np.arange(0, len(samples), dtype=np.float64)
    mix = np.cos(mix_angle) + 1j * np.sin(mix_angle)
    samples *= mix
    # weights is reversed to make it a standard convolution instead of a correlation
    filtered = np.convolve(samples, weights[::-1], mode="valid")
    decimated = filtered[::decimation]
    return decimated.astype(np.complex64)


@pytest.mark.parametrize(
    "taps,decimation,samples",
    [
        (256, 16, 256),
        (256, 16, 1024 * 1024),
        (256, 16, 1234568),
        (256, 8, 1234568),
        (256, 4, 1234568),
        (256, 32, 1234568),
        (256, 64, 1234568),
        (32, 32, 1234568),
    ],
)
def test_ddc(
    context: AbstractContext, command_queue: AbstractCommandQueue, taps: int, decimation: int, samples: int
) -> None:
    """Test DDC kernel."""
    rng = np.random.default_rng(seed=1)
    h_in = rng.integers(0, 256, samples * SAMPLE_BITS // BYTE_BITS, np.uint8)
    weights = rng.uniform(-1.0, 1.0, (taps,)).astype(np.float32)
    mix_frequency = 0.375
    expected = ddc_host(h_in, weights, decimation, mix_frequency)

    template = DDCTemplate(context, taps=taps, decimation=decimation)
    fn = template.instantiate(command_queue, samples)
    fn.mix_frequency = mix_frequency
    fn.ensure_all_bound()
    fn.buffer("in").set(command_queue, h_in)
    fn.buffer("weights").set(command_queue, weights)
    fn()
    h_out = fn.buffer("out").get(command_queue)
    # atol has to be quite large because the calculation is fundamentally
    # numerically unstable.
    np.testing.assert_allclose(h_out, expected, atol=2e-2)
    # RMS error should be an order of magnitude lower
    err = h_out - expected
    rms = np.sqrt(np.vdot(err, err) / err.size)
    assert rms < 2e-3
