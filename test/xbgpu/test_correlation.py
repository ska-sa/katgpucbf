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

"""Unit tests for the Tensor Core correlation kernel."""
import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext
from katsdpsigproc.accel import DeviceArray
from numba import njit, prange

from katgpucbf.xbgpu.correlation import Correlation, CorrelationTemplate, device_filter

from . import test_parameters

pytestmark = [pytest.mark.device_filter.with_args(device_filter)]

get_baseline_index = njit(Correlation.get_baseline_index)


@njit(parallel=True)
def correlate_host(input_array: np.ndarray) -> np.ndarray:
    """Calculate correlation products on the host CPU.

    Parameters
    ----------
    input_array
        Dataset to be correlated. Required shape:
        (n_batches, n_ants, n_chans, n_spectra, n_pols, complexity)

    Returns
    -------
    np.ndarray
        Correlation products or visibilities. Shape:
        (n_batches, n_chans, n_baselines, complexity)
    """
    n_ants = input_array.shape[1]
    n_chans = input_array.shape[2]
    n_pols = input_array.shape[4]
    complexity = input_array.shape[5]
    n_baselines = n_ants * (n_ants + 1) * 2
    output_array = np.zeros(shape=(n_chans, n_baselines, complexity), dtype=np.int64)
    for c in prange(n_chans):
        for a2 in range(n_ants):
            for a1 in range(a2 + 1):
                for p1 in range(n_pols):
                    input1 = input_array[:, a1, c, :, p1]
                    for p2 in range(n_pols):
                        bl_idx = get_baseline_index(a1, a2) * 4 + p1 + 2 * p2
                        input2 = input_array[:, a2, c, :, p2]
                        r_sum = np.int64(0)
                        i_sum = np.int64(0)
                        for i in range(input1.shape[0]):
                            for j in range(input1.shape[1]):
                                r1 = np.int64(input1[i, j, 0])
                                i1 = np.int64(input1[i, j, 1])
                                r2 = np.int64(input2[i, j, 0])
                                i2 = np.int64(input2[i, j, 1])
                                r_sum += r1 * r2 + i1 * i2
                                i_sum += r2 * i1 - r1 * i2
                        output_array[c, bl_idx, 0] = r_sum
                        output_array[c, bl_idx, 1] = i_sum

    return output_array


def fill_random(rng: np.random.Generator, buf: DeviceArray, command_queue: AbstractCommandQueue) -> None:
    """Fill a device buffer with random values.

    The buffer must have an integer dtype.
    """
    host_buf = buf.empty_like()
    host_buf[:] = rng.integers(
        low=np.iinfo(host_buf.dtype).min,
        high=np.iinfo(host_buf.dtype).max,
        size=host_buf.shape,
        dtype=host_buf.dtype,
        endpoint=True,
    )
    buf.set(command_queue, host_buf)


@pytest.mark.combinations(
    "num_ants, num_channels, num_spectra_per_heap",
    test_parameters.array_size,
    test_parameters.num_channels,
    test_parameters.num_spectra_per_heap,
)
def test_correlator(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    num_ants: int,
    num_spectra_per_heap: int,
    num_channels: int,
) -> None:
    """Test the Tensor Core correlation kernel for correctness."""
    n_chans_per_stream = num_channels // num_ants
    n_batches = 7
    batch_ranges = [(1, 5), (3, 4), (0, 7)]

    template = CorrelationTemplate(
        context,
        n_ants=num_ants,
        n_channels=n_chans_per_stream,
        n_spectra_per_heap=num_spectra_per_heap,
        sample_bitwidth=8,
    )

    correlation = template.instantiate(command_queue, n_batches)
    correlation.ensure_all_bound()

    buf_samples_device = correlation.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()

    rng = np.random.default_rng(seed=2021)
    buf_samples_host[:] = rng.integers(
        # The Tensor Core correlator can't manage the maximum negative value,
        # due to the asymmetry of signed integers, so we adjust the lower bound
        # up by 1.
        low=np.iinfo(buf_samples_host.dtype).min + 1,
        high=np.iinfo(buf_samples_host.dtype).max,
        size=buf_samples_host.shape,
        dtype=buf_samples_host.dtype,
        endpoint=True,  # We don't need to exclude the maximum positive value though.
    )

    buf_visibilities_device = correlation.buffer("out_visibilities")
    buf_visibilities_host = buf_visibilities_device.empty_like()
    # Fill the buffers with garbage, to ensure that the result does not depend
    # on the initial values.
    fill_random(rng, correlation.buffer("mid_visibilities"), command_queue)
    fill_random(rng, buf_visibilities_device, command_queue)

    # Calculate expected values
    calculated_visibilities_host = np.zeros(buf_visibilities_host.shape, buf_visibilities_host.dtype)
    for first_batch, last_batch in batch_ranges:
        calculated_visibilities_host += correlate_host(buf_samples_host[first_batch:last_batch])

    # Calculate using the kernel
    buf_samples_device.set(command_queue, buf_samples_host)
    correlation.zero_visibilities()
    for first_batch, last_batch in batch_ranges:
        correlation.first_batch = first_batch
        correlation.last_batch = last_batch
        correlation()
    correlation.reduce()
    buf_visibilities_device.get(command_queue, buf_visibilities_host)

    np.testing.assert_equal(buf_visibilities_host, calculated_visibilities_host)


def test_saturation(context: AbstractContext, command_queue: AbstractCommandQueue) -> None:
    """Test that values that overflow are saturated."""
    template = CorrelationTemplate(context, n_ants=2, n_channels=4, n_spectra_per_heap=256, sample_bitwidth=8)
    correlation = template.instantiate(command_queue, 2)
    correlation.ensure_all_bound()

    # Fill the source with maximal values, to ensure saturation is reached quickly
    buf_samples_device = correlation.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()
    in_dtype = buf_samples_device.dtype
    high = np.iinfo(in_dtype).max
    rng = np.random.default_rng(seed=2021)
    buf_samples_host[:] = rng.choice(np.array([-high, high], dtype=in_dtype), size=buf_samples_host.shape)
    buf_samples_device.set(command_queue, buf_samples_host)

    buf_visibilities_device = correlation.buffer("out_visibilities")
    out_dtype = buf_visibilities_device.dtype
    # Experimentally, enough iterations to saturate some but not all values.
    # The test *may* need modification if the parameters change.
    iters = np.iinfo(out_dtype).max // (template.n_spectra_per_heap * high * high) * 10

    correlation.zero_visibilities()
    correlation.first_batch = 0
    correlation.last_batch = 1
    for _ in range(iters):
        correlation()
    # Do one iteration of the other batch, which will pull some values back
    # into range.
    correlation.first_batch = 1
    correlation.last_batch = 2
    correlation()
    correlation.reduce()
    buf_visibilities_host = buf_visibilities_device.get(command_queue)
    buf_saturated_host = correlation.buffer("out_saturated").get(command_queue)

    expected = correlate_host(buf_samples_host[0:1]) * iters + correlate_host(buf_samples_host[1:2])
    # Check that saturation does in fact occur, but not all the time
    assert np.sum(expected < np.iinfo(out_dtype).min) > 0
    assert np.sum(expected > np.iinfo(out_dtype).max) > 0
    assert np.sum(np.all(np.abs(expected) < np.iinfo(out_dtype).max, axis=-1)) > 0

    expected_sat = np.sum(np.any(np.abs(expected) > np.iinfo(out_dtype).max, axis=-1))
    expected = np.clip(expected, -np.iinfo(out_dtype).max, np.iinfo(out_dtype).max)
    np.testing.assert_equal(buf_visibilities_host, expected)
    assert buf_saturated_host[()] == expected_sat
