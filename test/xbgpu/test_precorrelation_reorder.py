"""Unit tests for the Pre-correlation Reorder."""

import numpy as np
import pytest
from katsdpsigproc import accel
from numba import njit

from katgpucbf.xbgpu.precorrelation_reorder import PreCorrelationReorderTemplate

from . import test_parameters


@njit
def precorrelation_reorder_host_naive(
    input_array,
    output_array,
    n_batches,
    n_antennas,
    n_channels,
    n_samples_per_channel,
    n_tpb=16,
    n_polarisations=2,
):
    """Reorder data for correlation in a naive fashion on the host CPU.

    This function uses simple for-loops to reorder the input data into the shape
    of the output array. The Python implementation makes no attempt to optimise
    things, that is taken care of by :mod:`!numba`.

    These simple for-loops are very easy to verify for correctness. Numba speeds
    the operation up by a factor of about 600 empirically, on larger dataset
    sizes.

    Parameters
    ----------
    input_array
        Simulated F-engine data, with shape
        (batches, antennas, channels, time, pols).
    output_array
        Re-ordered input data with shape
        (batches, channels, samples_per_chan//times_per_block, antennas, pols, times_per_block)
    n_batches
        Number of batches of data that will be reordered.
    n_antennas
        Number of antennas we expect to receive data from.
    n_channels
        Number of frequency channels in the F-engine data, per stream.
    n_samples_per_channel
        How many time-series we expect to get.
    n_tpb
        [Optional] time samples per block. Required by the tensor-core
        correlator to better make use of its architecture. The default reflects
        the correlator kernel's needed parameter.
    n_polarisations
        [Optional] number of polarisations. I don't see a reason for this to be
        anything other than the default 2.
    """
    for b in range(n_batches):
        for c in range(n_channels):
            for s in range(n_samples_per_channel):
                for a in range(n_antennas):
                    for p in range(n_polarisations):
                        output_array[b][c][s // n_tpb][a][p][s % n_tpb] = input_array[b][a][c][s][p]


@pytest.mark.combinations(
    "num_ants, num_channels, num_samples_per_channel",
    test_parameters.array_size,
    test_parameters.num_channels,
    test_parameters.num_samples_per_channel,
)
def test_precorr_reorder_parametrised(num_ants, num_channels, num_samples_per_channel):
    """
    Parametrised unit test of the Pre-correlation Reorder kernel.

    Parameters
    ----------
    num_ants: int
        The number of antennas from which F-engine data is expected.
    num_channels: int
        The number of frequency channels in the F-engine data.

        .. attention::

          This is not the number of frequency channels per stream, but the
          total. The number of channels per stream is calculated from this
          value.

    num_samples_per_channel: int
        The number of time samples per frequency channel.
    """
    # This integer division is so that when num_ants % num_channels !=0 then the remainder will be dropped.
    # - This will only occur in the MeerKAT Extension correlator.
    # TODO: Need to consider the case where we round up as some X-Engines will
    # need to do this to capture all the channels.
    n_channels_per_stream = num_channels // num_ants // 4

    n_batches = 3

    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = PreCorrelationReorderTemplate(
        ctx,
        n_ants=num_ants,
        n_channels=n_channels_per_stream,
        n_samples_per_channel=num_samples_per_channel,
        n_batches=n_batches,
    )
    pre_correlation_reorder = template.instantiate(queue)
    pre_correlation_reorder.ensure_all_bound()

    buf_samples_device = pre_correlation_reorder.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()

    buf_reordered_device = pre_correlation_reorder.buffer("out_reordered")
    buf_reordered_host = buf_reordered_device.empty_like()

    rng = np.random.default_rng(seed=2021)
    buf_samples_host[:] = rng.uniform(
        np.iinfo(buf_samples_host.dtype).min, np.iinfo(buf_samples_host.dtype).max, buf_samples_host.shape
    ).astype(buf_samples_host.dtype)

    buf_samples_device.set(queue, buf_samples_host)
    pre_correlation_reorder()
    buf_reordered_device.get(queue, buf_reordered_host)

    host_reference_array = np.empty_like(buf_reordered_host)
    precorrelation_reorder_host_naive(
        buf_samples_host,
        host_reference_array,
        template.n_batches,
        template.n_ants,
        template.n_channels,
        template.n_samples_per_channel,
    )
    np.testing.assert_equal(buf_reordered_host, host_reference_array)
