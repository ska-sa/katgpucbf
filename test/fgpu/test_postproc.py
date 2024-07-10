################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

"""Unit tests for Postproc class."""

from collections.abc import Callable

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext
from numpy.typing import DTypeLike

from katgpucbf import N_POLS
from katgpucbf.fgpu import postproc

from .. import unpack_complex

pytestmark = [pytest.mark.cuda_only]


def postproc_host_pol(
    data: np.ndarray,
    spectra: int,
    spectra_per_heap_out: int,
    channels: int,
    unzip_factor: int,
    complex_pfb: bool,
    out_channels: tuple[int, int],
    out_bits: int,
    fine_delay: np.ndarray,
    fringe_phase: np.ndarray,
    gains: np.ndarray,
):
    """Calculate postproc steps on the host CPU for a single polarisation."""
    out_s = np.s_[out_channels[0] : out_channels[1]]
    n_out_channels = out_channels[1] - out_channels[0]
    # Fix up unzipped complex-to-complex transform into a full-size
    # real-to-complex or complex-to-complex transform. Rather than doing this
    # directly, go back to the time domain and do a fresh transform, to ensure
    # correctness rather than efficiency.
    data_time = np.fft.ifft(data.reshape(-1, unzip_factor, channels // unzip_factor), axis=-1)
    data_time = data_time.swapaxes(-1, -2).reshape(-1, channels)
    assert data_time.dtype == np.complex128  # numpy only does double-precision FFTs
    if complex_pfb:
        data = np.fft.fftshift(np.fft.fft(data_time, axis=-1).astype(np.complex64), axes=-1)
    else:
        data_rfft = np.fft.rfft(data_time.view(np.float64), axis=-1)
        # Throw out last channel (Nyquist frequency)
        data = data_rfft.astype(np.complex64)[:, :channels]
    data = data[:, out_s]
    # Compute delay phases
    channel_idx = np.arange(channels, dtype=np.float32)[np.newaxis, out_s] - channels / 2
    m2jpi = np.complex64(-2j * np.pi)
    phase = np.exp(m2jpi * fine_delay[:, np.newaxis] * channel_idx / (2 * channels) + 1j * fringe_phase[:, np.newaxis])
    assert phase.dtype == np.complex64
    # Apply delay, phase and gain
    corrected: np.ndarray  # mypy seems to get confused about the dtype; this makes it Any
    corrected = data * phase.astype(np.complex64) * gains[np.newaxis, :].astype(np.complex64)
    # Split complex into real, imaginary
    corrected = corrected.view(np.float32).reshape(spectra, n_out_channels, 2)
    # Count saturation per heap. Dithering makes this uncertain, so we compute
    # a range.
    qmax = 2 ** (out_bits - 1) - 1
    saturated_low = np.sum(np.any(np.abs(corrected) > qmax + 0.5, axis=2), axis=1, dtype=np.uint32)
    saturated_low = np.sum(saturated_low.reshape(-1, spectra_per_heap_out), axis=1)
    saturated_high = np.sum(np.any(np.abs(corrected) > qmax - 0.5, axis=2), axis=1, dtype=np.uint32)
    saturated_high = np.sum(saturated_high.reshape(-1, spectra_per_heap_out), axis=1)
    # Convert to integral and saturate (still a real dtype though)
    corrected = np.rint(corrected)
    corrected = np.minimum(np.maximum(corrected, -qmax), qmax)
    # Recombine real and imaginary
    assert corrected.dtype == np.float32
    corrected = corrected.view(np.complex64)[..., -1]
    # Partial transpose
    reshaped = corrected.reshape(-1, spectra_per_heap_out, n_out_channels)
    return reshaped.transpose(0, 2, 1), saturated_low, saturated_high


def postproc_host(
    in_: np.ndarray,
    spectra_per_heap_out: int,
    spectra: int,
    channels: int,
    unzip_factor: int,
    complex_pfb: bool,
    out_channels: tuple[int, int],
    out_bits: int,
    fine_delay: np.ndarray,
    fringe_phase: np.ndarray,
    gains: np.ndarray,
):
    """Aggregate both polarisation's postproc on the host CPU."""
    out = []
    saturated_low = []
    saturated_high = []
    for pol in range(N_POLS):
        pol_out, pol_saturated_low, pol_saturated_high = postproc_host_pol(
            in_[pol],
            spectra_per_heap_out,
            spectra,
            channels,
            unzip_factor,
            complex_pfb,
            out_channels,
            out_bits,
            fine_delay[:, pol],
            fringe_phase[:, pol],
            gains[:, pol],
        )
        out.append(pol_out)
        saturated_low.append(pol_saturated_low)
        saturated_high.append(pol_saturated_high)
    return np.stack(out, axis=3), np.stack(saturated_low, axis=1), np.stack(saturated_high, axis=1)


def _make_complex(func: Callable[[], np.ndarray], dtype: DTypeLike = np.complex64) -> np.ndarray:
    """Build an array of complex random numbers.

    The `func` must return an array of real numbers. It is called twice: once
    for the real component and once for the imaginary component. The calls
    should return arrays of the same shape and dtype.

    Note that by default it returns complex64 rather than complex128.
    """
    return (func() + func() * 1j).astype(dtype)


@pytest.mark.parametrize("unzip_factor", [1, 2, 4])
@pytest.mark.parametrize("complex_pfb", [False, True])
@pytest.mark.parametrize("out_channels", [(0, 4096), (1024, 3072), (123, 3456)])
@pytest.mark.parametrize("out_bits", [4, 8])
def test_postproc(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    unzip_factor: int,
    complex_pfb: bool,
    out_channels: tuple[int, int],
    out_bits: int,
) -> None:
    """Test GPU Postproc for numerical correctness."""
    channels = 4096
    spectra_per_heap_out = 256
    spectra = 512
    rng = np.random.default_rng(seed=1)
    in_shape = (N_POLS, spectra, unzip_factor, channels // unzip_factor)
    h_in = _make_complex(lambda: rng.uniform(-512, 512, in_shape))
    h_fine_delay = rng.uniform(0.0, 2.0, (spectra, N_POLS)).astype(np.float32)
    h_phase = rng.uniform(0.0, np.pi / 2, (spectra, N_POLS)).astype(np.float32)
    h_gains = _make_complex(lambda: rng.uniform(-1.5, 1.5, (out_channels[1] - out_channels[0], N_POLS)))

    expected, saturated_low, saturated_high = postproc_host(
        h_in,
        spectra,
        spectra_per_heap_out,
        channels,
        unzip_factor,
        complex_pfb,
        out_channels,
        out_bits,
        h_fine_delay,
        h_phase,
        h_gains,
    )

    template = postproc.PostprocTemplate(
        context, channels, unzip_factor, complex_pfb=complex_pfb, out_channels=out_channels, out_bits=out_bits
    )
    fn = template.instantiate(command_queue, spectra, spectra_per_heap_out, seed=123, sequence_first=456)
    fn.ensure_all_bound()
    fn.buffer("in").set(command_queue, h_in)
    fn.buffer("fine_delay").set(command_queue, h_fine_delay)
    fn.buffer("phase").set(command_queue, h_phase)
    fn.buffer("gains").set(command_queue, h_gains)
    fn.buffer("out").zero(command_queue)
    fn()
    h_out = fn.buffer("out").get(command_queue)
    h_saturated = fn.buffer("saturated").get(command_queue)

    h_out = unpack_complex(h_out)
    # Tolerance of 1.5 allows for error of 1 in each of real and imaginary
    np.testing.assert_allclose(h_out, expected, atol=1.5)
    assert np.all(saturated_low <= h_saturated)
    assert np.all(saturated_high >= h_saturated)
    # Ensure most negative value gets clamped
    qmax = 2 ** (out_bits - 1) - 1
    assert np.min(h_out.real) == -qmax
    assert np.min(h_out.imag) == -qmax
