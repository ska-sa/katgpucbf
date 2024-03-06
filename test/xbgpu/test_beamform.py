################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Test the :mod:`katgpucbf.xbgpu.beamform` module."""

import katsdpsigproc
import numba
import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.curand_helpers import RandomStateBuilder
from katgpucbf.xbgpu.beamform import BeamformTemplate


@numba.njit
def quant(x):
    """Round to integer and clamp to [-127, 127]."""
    return np.fmin(np.fmax(np.rint(x), -127), 127)


@numba.njit
def beamform_host(in_: np.ndarray, weights: np.ndarray, delays: np.ndarray, beam_pols: np.ndarray) -> np.ndarray:
    """Implement the beamforming operation on the CPU.

    The input arrays have the same shape and meaning as in :mod:`katgpucbf.xbgpu.beamform`.
    """
    n_frames = in_.shape[0]
    n_channels = in_.shape[2]
    n_times = in_.shape[3]
    n_beams = len(beam_pols)
    out = np.zeros((n_frames, n_beams, n_channels, n_times, COMPLEX), np.int8)
    for frame in range(n_frames):
        for channel in range(n_channels):
            in_c = in_[frame, :, channel, ..., 0] + np.complex64(1j) * in_[frame, :, channel, ..., 1]
            for beam in range(n_beams):
                p = beam_pols[beam]
                w = weights[:, beam]
                w = w * np.exp(delays[:, beam] * channel * np.complex64(1j) * np.float32(np.pi))
                for time in range(n_times):
                    # .copy() is added because otherwise we get a warning that np.dot is
                    # more efficient on C-contiguous arrays.
                    accum = np.dot(in_c[:, time, p].copy(), w)
                    out[frame, beam, channel, time, 0] = quant(accum.real)
                    out[frame, beam, channel, time, 1] = quant(accum.imag)
    return out


@pytest.mark.combinations(
    "n_frames, n_channels, n_times, n_antennas",
    [1, 5],
    [1, 128, 200, 1025],
    [1, 128, 256, 321],
    [1, 4, 19, 80],
)
def test_beamform(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    n_frames: int,
    n_channels: int,
    n_times: int,
    n_antennas: int,
) -> None:
    """Test :class:`.Beamform` by comparing it to a CPU implementation."""
    beam_pols = [0, 1, 0, 1, 1]
    n_beams = len(beam_pols)

    template = BeamformTemplate(context, beam_pols)
    fn = template.instantiate(command_queue, n_frames, n_antennas, n_channels, n_times)

    # Initialise the random states
    assert isinstance(context, katsdpsigproc.cuda.Context)
    assert isinstance(fn.slots["rand_states"], katsdpsigproc.accel.IOSlot)  # keep mypy happy
    builder = RandomStateBuilder(context)
    fn.bind(rand_states=builder.make_states(fn.slots["rand_states"].shape, seed=321, sequence_first=0))

    fn.ensure_all_bound()
    h_in = fn.buffer("in").empty_like()
    h_out = fn.buffer("out").empty_like()
    h_weights = fn.buffer("weights").empty_like()
    h_delays = fn.buffer("delays").empty_like()
    assert h_in.shape == (n_frames, n_antennas, n_channels, n_times, N_POLS, COMPLEX)
    assert h_out.shape == (n_frames, n_beams, n_channels, n_times, COMPLEX)
    assert h_weights.shape == (n_antennas, n_beams)
    assert h_delays.shape == (n_antennas, n_beams)

    rng = np.random.default_rng(seed=123)
    h_in[:] = rng.integers(-127, 127, h_in.shape)
    # Scale is chosen to have some (but not all) values saturate. The sqrt
    # scale factor is because that's how the standard deviation grows when
    # adding independent normal random variables.
    scale = 2.0 / np.sqrt(n_antennas)
    h_weights[:] = rng.uniform(-scale, scale, h_weights.shape) + 1j * rng.uniform(-scale, scale, h_weights.shape)
    # Delay value is phase step per channel. We want delays to wrap a little
    # across the whole band, but not an excessive number of times.
    h_delays[:] = rng.uniform(-100.0 / n_channels, 100.0 / n_channels, h_delays.shape)
    h_out.fill(0)
    expected = beamform_host(h_in, h_weights, h_delays, np.array(beam_pols))

    fn.buffer("out").zero(command_queue)
    fn.buffer("in").set(command_queue, h_in)
    fn.buffer("weights").set(command_queue, h_weights)
    fn.buffer("delays").set(command_queue, h_delays)
    fn()
    fn.buffer("out").get(command_queue, h_out)

    np.testing.assert_allclose(h_out, expected, atol=1)
    # Ensure that the scale factor on the weights causes some clamping, but
    # not too much. But only do it for larger test cases; in small tests cases
    # it is hard to guarantee this.
    if n_antennas > 1 and n_frames * n_channels * n_times > 1000:
        clamped = np.sum(np.abs(h_out) == 127, axis=(0, 1, 2, 3)) / h_out[..., 0].size
        assert 0.1 < clamped[0] < 0.9  # real
        assert 0.1 < clamped[1] < 0.9  # imag
