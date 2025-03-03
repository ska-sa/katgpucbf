################################################################################
# Copyright (c) 2022-2023, 2025, National Research Foundation (SARAO)
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

from fractions import Fraction

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf import BYTE_BITS, N_POLS
from katgpucbf.fgpu.ddc import DDCTemplate, _TuningDict

from .. import unpackbits


def ddc_host(
    samples: np.ndarray, weights: np.ndarray, subsampling: int, input_sample_bits: int, mix_frequency: Fraction
) -> np.ndarray:
    """Implement the DDC calculation simply in numpy."""
    # Calculation is done in double precision for better accuracy
    samples = np.stack([unpackbits(pol_samples, input_sample_bits).astype(np.complex128) for pol_samples in samples])
    mix_cycles_scaled = (
        mix_frequency.numerator * np.arange(0, samples.shape[1], dtype=np.int64) % mix_frequency.denominator
    )
    mix_cycles = mix_cycles_scaled.astype(np.float64) / mix_frequency.denominator
    mix_angle = 2 * np.pi * mix_cycles
    mix = np.cos(mix_angle) + 1j * np.sin(mix_angle)
    samples *= mix
    # weights is reversed to make it a standard convolution instead of a correlation
    filtered = np.stack([np.convolve(pol_samples, weights[::-1], mode="valid") for pol_samples in samples])
    subsampled = filtered[:, ::subsampling]
    return subsampled.astype(np.complex64)


@pytest.mark.parametrize(
    "n_pols,taps,subsampling,samples,input_sample_bits,tuning",
    [
        (2, 256, 16, 256, 10, None),
        (1, 256, 16, 1024 * 1024, 12, None),
        (2, 256, 16, 1234568, 13, None),
        (1, 256, 8, 1234568, 16, None),
        (2, 255, 4, 1234568, 32, None),
        (1, 257, 32, 1234568, 8, None),
        (2, 256, 64, 1234568, 5, None),
        (1, 32, 32, 1234568, 7, None),
        (2, 55, 5, 123464, 10, None),
        (1, 256, 16, 256 * 1024, 10, {"wgs": 96, "unroll": 4}),
    ],
)
def test_ddc(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    n_pols: int,
    taps: int,
    subsampling: int,
    samples: int,
    input_sample_bits: int,
    tuning: _TuningDict | None,
) -> None:
    """Test DDC kernel."""
    rng = np.random.default_rng(seed=1)
    h_in = rng.integers(0, 256, (n_pols, samples * input_sample_bits // BYTE_BITS), np.uint8)
    weights = rng.uniform(-1.0, 1.0, (taps,)).astype(np.float32)
    mix_frequency = Fraction("0.21")

    template = DDCTemplate(
        context, taps=taps, subsampling=subsampling, input_sample_bits=input_sample_bits, tuning=tuning
    )
    fn = template.instantiate(command_queue, samples, n_pols)
    fn.configure(mix_frequency, weights)
    fn.ensure_all_bound()
    fn.buffer("in").set(command_queue, h_in)
    fn()
    h_out = fn.buffer("out").get(command_queue)

    assert fn.mix_frequency == mix_frequency

    # atol has to be quite large because the calculation is fundamentally
    # numerically unstable.
    expected = ddc_host(h_in, weights, subsampling, input_sample_bits, fn.mix_frequency)
    np.testing.assert_allclose(h_out, expected, atol=2e-5 * 2**input_sample_bits)
    # RMS error should be an order of magnitude lower
    err = h_out - expected
    rms = np.sqrt(np.vdot(err, err) / err.size)
    assert rms < 2e-6 * 2**input_sample_bits


@pytest.mark.parametrize(
    "taps,subsampling,input_sample_bits",
    [
        (0, 64, 10),  # <= 0
        (8, -1, 12),  # <= 0
        (64, 8, 33),  # Too large sample_bits
    ],
)
def test_bad_template_parameters(context: AbstractContext, taps: int, subsampling: int, input_sample_bits: int) -> None:
    """Test that :class:`DDCTemplate` raises ValueError when given invalid parameters."""
    with pytest.raises(ValueError):
        DDCTemplate(context, taps=taps, subsampling=subsampling, input_sample_bits=input_sample_bits)


def test_bad_tuning(context: AbstractContext) -> None:
    """Test that :class:`DDCTemplate` raises ValueError when given bad tuning parameters."""
    with pytest.raises(ValueError, match="unroll must be a multiple of 16"):
        DDCTemplate(context, taps=255, subsampling=5, input_sample_bits=10, tuning={"wgs": 32, "unroll": 5})


def test_too_few_samples(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    """Test that :class:`DDC` raises ValueError when `samples` is too small."""
    template = DDCTemplate(context, taps=256, input_sample_bits=10, subsampling=16)
    with pytest.raises(ValueError):
        template.instantiate(command_queue, 255, N_POLS)


@pytest.mark.parametrize(
    "taps,subsampling,input_sample_bits",
    [
        (256, 16, 10),
        (96, 8, 10),
        (256, 5, 32),
        (96, 16, 5),
    ],
)
@pytest.mark.force_autotune
def test_autotune(context: AbstractContext, taps: int, subsampling: int, input_sample_bits: int) -> None:
    """Test that autotuner runs successfully."""
    DDCTemplate(context, taps=taps, subsampling=subsampling, input_sample_bits=input_sample_bits)
