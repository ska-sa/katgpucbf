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

"""Unit tests for the Pre-correlation Reorder."""

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf.xbgpu.precorrelation_reorder import PrecorrelationReorderTemplate

from . import test_parameters

pytestmark = [pytest.mark.cuda_only]

POLS = 2
CPLX = 2
TPB = 16  # corresponding to times_per_block in the kernel.


@pytest.mark.combinations(
    "num_ants, num_channels, num_spectra_per_heap",
    test_parameters.array_size,
    test_parameters.num_channels,
    test_parameters.num_spectra_per_heap,
)
def test_precorr_reorder_parametrised(
    context: AbstractContext,
    command_queue: AbstractCommandQueue,
    num_ants: int,
    num_channels: int,
    num_spectra_per_heap: int,
) -> None:
    """
    Parametrised unit test of the Pre-correlation Reorder kernel.

    Parameters
    ----------
    context
        Device context provided by :mod:`katsdpsigproc.pytest_plugin`
    command_queue
        Device command queue provided by :mod:`katsdpsigproc.pytest_plugin`
    num_ants
        The number of antennas from which F-engine data is expected.
    num_channels
        The number of frequency channels in the F-engine data.

        .. attention::

          This is not the number of frequency channels per stream, but the
          total. The number of channels per stream is calculated from this
          value.

    num_spectra_per_heap
        The number of time samples per frequency channel.
    """
    # This integer division is so that when num_ants % num_channels !=0 then the remainder will be dropped.
    # - This will only occur in the MeerKAT Extension correlator.
    # TODO: Need to consider the case where we round up as some X-Engines will
    # need to do this to capture all the channels.
    n_channels_per_stream = num_channels // num_ants // 4

    n_batches = 3

    template = PrecorrelationReorderTemplate(
        context,
        n_ants=num_ants,
        n_channels=n_channels_per_stream,
        n_spectra_per_heap=num_spectra_per_heap,
        n_batches=n_batches,
    )
    pre_correlation_reorder = template.instantiate(command_queue)
    pre_correlation_reorder.ensure_all_bound()

    buf_samples_device = pre_correlation_reorder.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()

    buf_reordered_device = pre_correlation_reorder.buffer("out_reordered")
    buf_reordered_host = buf_reordered_device.empty_like()

    # We seed np's random-number-generator in order to ensure unit tests that
    # run the same way every time. The number is selected arbitrarily.
    rng = np.random.default_rng(seed=2021)

    # We use `np.iinfo` to determine dynamically the min and max values that the
    # array can contain. If the dtype changes in the kernel, the unit test
    # should still work.
    buf_samples_host[:] = rng.uniform(
        np.iinfo(buf_samples_host.dtype).min, np.iinfo(buf_samples_host.dtype).max, buf_samples_host.shape
    ).astype(buf_samples_host.dtype)

    buf_samples_device.set(command_queue, buf_samples_host)
    pre_correlation_reorder()
    buf_reordered_device.get(command_queue, buf_reordered_host)

    reordered_reference_array_host = np.empty_like(buf_reordered_host)
    # Numpy's reshape and transpose work together to move the data around the
    # same way as the GPU-reorder does.
    reordered_reference_array_host[:] = buf_samples_host.reshape(
        n_batches, num_ants, n_channels_per_stream, num_spectra_per_heap // TPB, TPB, POLS, CPLX
    ).transpose(0, 2, 3, 1, 5, 4, 6)

    np.testing.assert_equal(buf_reordered_host, reordered_reference_array_host)
