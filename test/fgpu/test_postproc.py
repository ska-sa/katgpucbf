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

"""Unit tests for Postproc class."""
import numpy as np
import pytest
from katsdpsigproc import accel

from katgpucbf import N_POLS
from katgpucbf.fgpu import postproc

pytestmark = [pytest.mark.cuda_only]


def postproc_host_pol(data, spectra, spectra_per_heap_out, channels, fine_delay, fringe_phase, quant_gain):
    """Calculate postproc steps on the host CPU for a single polarisation."""
    # Throw out last channel (Nyquist frequency)
    data = data[:, :channels]
    # Compute delay phases
    channel_idx = np.arange(channels, dtype=np.float32)[np.newaxis, :]
    m2ipi = np.complex64(-2j * np.pi)
    phase = np.exp(m2ipi * fine_delay[:, np.newaxis] * channel_idx / (2 * channels) + 1j * fringe_phase[:, np.newaxis])
    assert phase.dtype == np.complex64
    corrected = data * phase.astype(np.complex64)
    # Split complex into real, imaginary
    corrected = corrected.view(np.float32).reshape(spectra, channels, 2)
    # Convert to integer
    corrected = np.rint(corrected * quant_gain)
    # Cast to integer with saturation
    corrected = np.minimum(np.maximum(corrected, -128), 127)
    corrected = corrected.astype(np.int8)
    # Partial transpose
    reshaped = corrected.reshape(-1, spectra_per_heap_out, channels, 2)
    return reshaped.transpose(0, 2, 1, 3)


def postproc_host(in0, in1, channels, spectra_per_heap_out, spectra, fine_delay, fringe_phase, quant_gain):
    """Aggregate both polarisation's postproc on the host CPU."""
    out0 = postproc_host_pol(
        in0, channels, spectra_per_heap_out, spectra, fine_delay[:, 0], fringe_phase[:, 0], quant_gain
    )
    out1 = postproc_host_pol(
        in1, channels, spectra_per_heap_out, spectra, fine_delay[:, 1], fringe_phase[:, 1], quant_gain
    )
    return np.stack([out0, out1], axis=3)


def test_postproc(context, command_queue, repeat=1):
    """Test GPU Postproc for numerical correctness."""
    channels = 4096
    spectra_per_heap_out = 256
    spectra = 512
    quant_gain = 0.1
    # TODO: make properly complex
    rng = np.random.default_rng(seed=1)
    h_in0 = rng.uniform(-512, 512, (spectra, channels + 1)).astype(np.complex64)
    h_in1 = rng.uniform(-512, 512, (spectra, channels + 1)).astype(np.complex64)
    h_fine_delay = rng.uniform(0.0, 2.0, (spectra, N_POLS)).astype(np.float32)
    h_phase = rng.uniform(0.0, np.pi / 2, (spectra, N_POLS)).astype(np.float32)
    expected = postproc_host(h_in0, h_in1, spectra, spectra_per_heap_out, channels, h_fine_delay, h_phase, quant_gain)

    template = postproc.PostprocTemplate(context)
    fn = template.instantiate(command_queue, spectra, spectra_per_heap_out, channels)
    fn.ensure_all_bound()
    fn.buffer("in0").set(command_queue, h_in0)
    fn.buffer("in1").set(command_queue, h_in1)
    fn.buffer("fine_delay").set(command_queue, h_fine_delay)
    fn.buffer("phase").set(command_queue, h_phase / np.pi)
    fn.quant_gain = quant_gain
    for _ in range(repeat):
        fn()
    h_out = fn.buffer("out").get(command_queue)

    np.testing.assert_allclose(h_out, expected, atol=1)


if __name__ == "__main__":
    ctx = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    queue = ctx.create_command_queue()
    test_postproc(ctx, queue, repeat=100)
