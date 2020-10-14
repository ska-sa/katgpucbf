"""
Module for performing unit tests on the Tensor core correlation kernel.

Contains two unit tests:
    1. @test_correlator_exhaustive - this test takes very long to run, but generates a high degree of certainty
    over the results.
    2. @test_correlator_quick - this test runs very quickly but is less exhaustive. It is used for quickly testing
    much larger array sizes.

TODO:
    1. Once the functionality has been added to the TensorCoreCorrelator class, add a unit test to verify that the
    kernel can be called multiple times without zeroing the visibility matrix.
"""
import pytest
import numpy as np
from katxgpu import tensorcore_correlator
from katsdpsigproc import accel

# Array specifying different array sizes that could potentially be used by MeerKAT
array_size = [4, 8, 16, 32, 64, 84, 192, 256]


def get_simple_test_ant_value(channel_index, ant_index):
    """Hashing function to generate a unique int8 number for a specific antenna channel combination.

    The value returned is between -127 and 127(inclusive) of type np.int8.
    """
    ant_value = np.uint8(channel_index * ant_index) - 128
    if ant_value == -128:
        ant_value += 1
    return np.int8(ant_value)


def generate_antpair_visibilities_host(bufSamples_host, channel_index, ant1, ant2, time_outer_range, time_inner_range):
    """
    Host side code to generate visibilities between two antennas.

    Will generate visibilities for a particular channel and antenna combination from an array of input samples that has
    the shape (channels, samples_per_channel // times_per_block, dual_pol_ants, polarizastions, times_per_block)
    required by the Tensor core correlation kernels.

    This is a naive implementation of the correlation algorithm and is computationally very slow.

    Returns visibilities hh, hv, vh and vv representing different interpolarisation products.
    """
    hh, hv, vh, vv = 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j
    for time_index1 in range(0, time_outer_range):
        for time_index2 in range(0, time_inner_range):
            ant1h = bufSamples_host[channel_index][time_index1][ant1][0][time_index2]
            ant1v = bufSamples_host[channel_index][time_index1][ant1][1][time_index2]
            ant2h = bufSamples_host[channel_index][time_index1][ant2][0][time_index2]
            ant2v = bufSamples_host[channel_index][time_index1][ant2][1][time_index2]

            hh += ant1h * np.conj(ant2h)
            vh += ant1h * np.conj(ant2v)
            hv += ant1v * np.conj(ant2h)
            vv += ant1v * np.conj(ant2v)

    return hh, hv, vh, vv


@pytest.mark.parametrize("num_ants", array_size)
def test_correlator_exhaustive(num_ants):
    """
    Intensive unit test of the Tensor core correlation algorithm.

    This unit test runs on random input data. The CPU side check for correctness is very slow - the input sample
    array size has been kept small to speed this up. For MeerKAT sizes, the input sample array will be much bigger.
    """
    # 1. Array parameters
    dual_pol_ants = num_ants
    channels = 2
    samples_per_channel = 16

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_correlator.TensorCoreCorrelatorTemplate(
        ctx, dual_pol_ants=dual_pol_ants, channels=channels, samples_per_channel=samples_per_channel
    )
    tensorCoreCorrelator = template.instantiate(queue)
    tensorCoreCorrelator.ensure_all_bound()

    bufSamples_device = tensorCoreCorrelator.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = tensorCoreCorrelator.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # 3. Generate random input data - need to modify the dtype and  shape of the array as numpy does not have a packet
    # 8-bit int complex type.

    bufSamplesInt16Shape = template.inputDataShape
    bufSamplesInt8Shape = list(bufSamplesInt16Shape)
    bufSamplesInt8Shape[-1] *= 2  # By converting from int16 to int8, the length of the last dimension doubles.
    bufSamplesInt8Shape = tuple(bufSamplesInt8Shape)

    bufSamples_host.dtype = np.int8
    bufSamples_host[:] = np.random.randint(
        -127,
        high=128,
        size=bufSamplesInt8Shape,
        dtype=np.int8,
    )
    bufSamples_host.dtype = np.int16

    # 4. Transfer input sample array to the GPU, run kernel, transfer output visibilities array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    tensorCoreCorrelator()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 5. Check that the GPU visibilities are correct.
    # 5.1 Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
    # complex type.
    bufSamples_host.dtype = np.int8
    bufSamples_host = bufSamples_host.astype(np.float32)
    bufSamples_host.dtype = np.csingle

    bufVisibilities_host.dtype = np.int32
    bufVisibilities_host = bufVisibilities_host.astype(np.float32)
    bufVisibilities_host.dtype = np.csingle

    bufCorrectVisibilities_host = np.empty_like(bufVisibilities_host)

    # 5.2 Generate the visibilities on the CPU - this is not a simple matrix operation due to the indexing of the input
    # samples, I am just going to do a naive brute force for now to be optomised later if needs be.

    time_outer_range = samples_per_channel // template._times_per_block
    time_inner_range = template._times_per_block

    for channel_index in range(0, channels):
        for ant1_index in range(0, dual_pol_ants):
            for ant2_index in range(0, ant1_index + 1):
                hh, hv, vh, vv = generate_antpair_visibilities_host(
                    bufSamples_host, channel_index, ant1_index, ant2_index, time_outer_range, time_inner_range
                )
                baseline_index = tensorcore_correlator.TensorCoreCorrelator.get_baseline_index(ant1_index, ant2_index)
                bufCorrectVisibilities_host[channel_index][baseline_index][0][0] = hh
                bufCorrectVisibilities_host[channel_index][baseline_index][1][0] = hv
                bufCorrectVisibilities_host[channel_index][baseline_index][0][1] = vh
                bufCorrectVisibilities_host[channel_index][baseline_index][1][1] = vv

    # 5.3 Check that the CPU version and GPU version are identical
    np.testing.assert_array_equal(bufCorrectVisibilities_host, bufVisibilities_host)


@pytest.mark.parametrize("num_ants", array_size)
def test_correlator_quick(num_ants):
    """
    Lightweight unit test of the Tensor core correlation algorithm.

    This unit test uses a hashing function(@get_simple_test_ant_value(...)) to determine the values in the input
    sample array for a specific antenna-channel combination. This value is kept constant over all time values. This
    allows the output visibility values to be calculated quickly on the CPU without performing the full correlation
    operation. This unit test runs much quicker than @test_correlator_exhaustive, while being much less exhaustive.
    It is used to perform a quick check for the correctness on much larger input sample array sizes.
    """
    # 1. Array parameters
    dual_pol_ants = num_ants
    channels = 64
    samples_per_channel = 3072

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_correlator.TensorCoreCorrelatorTemplate(
        ctx, dual_pol_ants=dual_pol_ants, channels=channels, samples_per_channel=samples_per_channel
    )
    correlator = template.instantiate(queue)
    correlator.ensure_all_bound()

    time_outer_range = samples_per_channel // template._times_per_block

    bufSamples_device = correlator.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = correlator.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # 3. Generate predictable input data. The time samples remain constant for every antenna-channel combination.
    bufSamples_host.dtype = np.int8
    for channel_index in range(channels):
        for ant_index in range(dual_pol_ants):
            sample_value = get_simple_test_ant_value(channel_index, ant_index)
            for time_outer_index in range(time_outer_range):
                bufSamples_host[channel_index][time_outer_index][ant_index] = np.full(
                    bufSamples_host.shape[3:], sample_value, dtype=np.int8
                )
    bufSamples_host.dtype = np.int16

    # 4. Transfer input sample array to the GPU, run kernel, transfer output visibilities array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    correlator()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 5. Check that the GPU visibilities are correct.
    # 5.1 Need to convert the output array to 32-bit float complex type as numpy has no  32-bit int complex type.

    bufVisibilities_host.dtype = np.int32
    bufVisibilities_host = bufVisibilities_host.astype(np.float32)
    bufVisibilities_host.dtype = np.csingle

    # 5.2 Generate the visibilities on the CPU and check they match the expected value. The output values are simple to
    # calculate and do not require performing the matrix multiple as it is performed on the GPU.
    for channel_index in range(0, channels):
        for ant1_index in range(0, dual_pol_ants):
            for ant2_index in range(0, ant1_index + 1):
                ant1_value = get_simple_test_ant_value(channel_index, ant1_index)
                ant1_value = ant1_value + ant1_value * 1j
                ant2_value = get_simple_test_ant_value(channel_index, ant2_index)
                ant2_value = ant2_value + ant2_value * 1j
                baseline_index = tensorcore_correlator.TensorCoreCorrelator.get_baseline_index(ant1_index, ant2_index)

                bufSamples_host.dtype = np.int8
                product = ant1_value * np.conj(ant2_value)
                productAccumulated = product * samples_per_channel

                assert bufVisibilities_host[channel_index][baseline_index][0][0] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][1][0] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][0][1] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][1][1] == productAccumulated
