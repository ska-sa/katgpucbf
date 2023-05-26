################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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

from katgpucbf import BYTE_BITS, DIG_SAMPLE_BITS
from katgpucbf.fgpu.ddc import DDCTemplate, _TuningDict

from .. import unpackbits


def ddc_host(samples: np.ndarray, weights: np.ndarray, subsampling: int, mix_frequency: float) -> np.ndarray:
    """Implement the DDC calculation simply in numpy."""
    # Calculation is done in double precision for better accuracy
    samples = unpackbits(samples).astype(np.complex128)
    mix_angle = 2 * np.pi * mix_frequency * np.arange(0, len(samples), dtype=np.float64)
    mix = np.cos(mix_angle) + 1j * np.sin(mix_angle)
    samples *= mix
    # weights is reversed to make it a standard convolution instead of a correlation
    filtered = np.convolve(samples, weights[::-1], mode="valid")
    subsampled = filtered[::subsampling]
    return subsampled.astype(np.complex64)


@pytest.mark.parametrize(
    "taps,subsampling,samples,tuning",
    [
        (256, 16, 256, None),
        (256, 16, 1024 * 1024, None),
        (256, 16, 1234568, None),
        (256, 8, 1234568, None),
        (256, 4, 1234568, None),
        (256, 32, 1234568, None),
        (256, 64, 1234568, None),
        (32, 32, 1234568, None),
        (55, 5, 123464, None),
        (256, 16, 256 * 1024, {"wgs": 96, "unroll": 4}),
    ],
)
def test_ddc(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    taps: int,
    subsampling: int,
    samples: int,
    tuning: _TuningDict | None,
) -> None:
    """Test DDC kernel."""
    rng = np.random.default_rng(seed=1)
    h_in = rng.integers(0, 256, samples * DIG_SAMPLE_BITS // BYTE_BITS, np.uint8)
    weights = rng.uniform(-1.0, 1.0, (taps,)).astype(np.float32)
    mix_frequency = 0.21
    expected = ddc_host(h_in, weights, subsampling, mix_frequency)

    template = DDCTemplate(context, taps=taps, subsampling=subsampling, tuning=tuning)
    fn = template.instantiate(command_queue, samples)
    fn.configure(mix_frequency, weights)
    fn.ensure_all_bound()
    fn.buffer("in").set(command_queue, h_in)
    fn()
    h_out = fn.buffer("out").get(command_queue)
    # atol has to be quite large because the calculation is fundamentally
    # numerically unstable.
    np.testing.assert_allclose(h_out, expected, atol=2e-2)
    # RMS error should be an order of magnitude lower
    err = h_out - expected
    rms = np.sqrt(np.vdot(err, err) / err.size)
    assert rms < 2e-3


@pytest.mark.parametrize(
    "taps,subsampling",
    [
        (123, 12),  # Not a multiple
        (0, 64),  # <= 0
        (8, -1),  # <= 0
    ],
)
def test_bad_template_parameters(context: AbstractContext, taps: int, subsampling: int) -> None:
    """Test that :class:`DDCTemplate` raises ValueError when given invalid parameters."""
    with pytest.raises(ValueError):
        DDCTemplate(context, taps=taps, subsampling=subsampling)


def test_bad_tuning(context: AbstractContext) -> None:
    with pytest.raises(ValueError, match="unroll must be a multiple of 16"):
        DDCTemplate(context, taps=255, subsampling=5, tuning={"wgs": 32, "unroll": 5})


def test_too_few_samples(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    """Test that :class:`DDC` raises ValueError when `samples` is too small."""
    template = DDCTemplate(context, taps=256, subsampling=16)
    with pytest.raises(ValueError):
        template.instantiate(command_queue, 255)


@pytest.mark.force_autotune
def test_autotune(context: AbstractContext) -> None:
    """Test that autotuner runs successfully."""
    DDCTemplate(context, taps=128, subsampling=8)
