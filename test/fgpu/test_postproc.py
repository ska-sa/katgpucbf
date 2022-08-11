################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

from typing import Callable

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext
from numpy.typing import DTypeLike

from katgpucbf import N_POLS
from katgpucbf.fgpu import postproc

pytestmark = [pytest.mark.cuda_only]


def postproc_host_pol(data, spectra, spectra_per_heap_out, channels, unzip_factor, fine_delay, fringe_phase, gains):
    """Calculate postproc steps on the host CPU for a single polarisation."""
    # Fix up unzipped complex-to-complex transform into a real-to-complex
    # transform. Rather than doing this directly, go back to the time domain
    # and do a fresh transform, to ensure correctness rather than efficiency.
    data_time = np.fft.ifft(data.reshape(-1, unzip_factor, channels // unzip_factor), axis=-1)
    data_time = data_time.swapaxes(-1, -2).reshape(-1, channels)
    assert data_time.dtype == np.complex128  # numpy only does double-precision FFTs
    data_rfft = np.fft.rfft(data_time.view(np.float64), axis=-1)
    # Throw out last channel (Nyquist frequency)
    data = data_rfft.astype(np.complex64)[:, :channels]
    # Compute delay phases
    channel_idx = np.arange(channels, dtype=np.float32)[np.newaxis, :]
    m2jpi = np.complex64(-2j * np.pi)
    phase = np.exp(m2jpi * fine_delay[:, np.newaxis] * channel_idx / (2 * channels) + 1j * fringe_phase[:, np.newaxis])
    assert phase.dtype == np.complex64
    # Apply delay, phase and gain
    corrected = data * phase.astype(np.complex64) * gains[np.newaxis, :].astype(np.complex64)
    # Split complex into real, imaginary
    corrected = corrected.view(np.float32).reshape(spectra, channels, 2)
    # Convert to integer
    corrected = np.rint(corrected)
    # Cast to integer with saturation
    corrected = np.minimum(np.maximum(corrected, -127), 127)
    corrected = corrected.astype(np.int8)
    # Partial transpose
    reshaped = corrected.reshape(-1, spectra_per_heap_out, channels, 2)
    return reshaped.transpose(0, 2, 1, 3)


def postproc_host(in0, in1, channels, unzip_factor, spectra_per_heap_out, spectra, fine_delay, fringe_phase, gains):
    """Aggregate both polarisation's postproc on the host CPU."""
    out0 = postproc_host_pol(
        in0, channels, unzip_factor, spectra_per_heap_out, spectra, fine_delay[:, 0], fringe_phase[:, 0], gains[:, 0]
    )
    out1 = postproc_host_pol(
        in1, channels, unzip_factor, spectra_per_heap_out, spectra, fine_delay[:, 1], fringe_phase[:, 1], gains[:, 1]
    )
    return np.stack([out0, out1], axis=3)


def _make_complex(func: Callable[[], np.ndarray], dtype: DTypeLike = np.complex64) -> np.ndarray:
    """Build an array of complex random numbers.

    The `func` must return an array of real numbers. It is called twice: once
    for the real component and once for the imaginary component. The calls
    should return arrays of the same shape and dtype.

    Note that by default it returns complex64 rather than complex128.
    """
    return (func() + func() * 1j).astype(dtype)


@pytest.mark.parametrize("unzip_factor", [1, 2, 4])
def test_postproc(context: AbstractContext, command_queue: AbstractCommandQueue, unzip_factor: int) -> None:
    """Test GPU Postproc for numerical correctness."""
    channels = 4096
    spectra_per_heap_out = 256
    spectra = 512
    rng = np.random.default_rng(seed=1)
    in_shape = (spectra, unzip_factor, channels // unzip_factor)
    h_in0 = _make_complex(lambda: rng.uniform(-512, 512, in_shape))
    h_in1 = _make_complex(lambda: rng.uniform(-512, 512, in_shape))
    h_fine_delay = rng.uniform(0.0, 2.0, (spectra, N_POLS)).astype(np.float32)
    h_phase = rng.uniform(0.0, np.pi / 2, (spectra, N_POLS)).astype(np.float32)
    h_gains = _make_complex(lambda: rng.uniform(-1.5, 1.5, (channels, N_POLS)))

    expected = postproc_host(
        h_in0, h_in1, spectra, spectra_per_heap_out, channels, unzip_factor, h_fine_delay, h_phase, h_gains
    )

    template = postproc.PostprocTemplate(context, channels, unzip_factor)
    fn = template.instantiate(command_queue, spectra, spectra_per_heap_out)
    fn.ensure_all_bound()
    fn.buffer("in0").set(command_queue, h_in0)
    fn.buffer("in1").set(command_queue, h_in1)
    fn.buffer("fine_delay").set(command_queue, h_fine_delay)
    fn.buffer("phase").set(command_queue, h_phase / np.pi)
    fn.buffer("gains").set(command_queue, h_gains)
    fn.buffer("out").zero(command_queue)
    fn()
    h_out = fn.buffer("out").get(command_queue)

    np.testing.assert_allclose(h_out, expected, atol=1)
    assert np.min(h_out) == -127  # Ensure -128 gets clamped to -127
