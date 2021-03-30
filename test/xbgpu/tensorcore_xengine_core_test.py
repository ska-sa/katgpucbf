"""
Module for performing unit tests on the Tensor core correlation kernel.

Contains two unit tests:
    1. @test_correlator_exhaustive - this test is time-consuming, taking many hours to run for larger input sizes. It
    generates a high degree of certainty over the results.
    2. @test_correlator_quick - this test runs very quickly but is less exhaustive. It is used for quickly testing
    much larger array sizes.
"""
import pytest
import numpy as np
from katxgpu import tensorcore_xengine_core
from katsdpsigproc import accel

import test_parameters


def get_simple_test_ant_value(channel_index, ant_index):
    """Hashing function to generate a repeatable int8 number for a specific antenna channel combination.

    The value returned is between -127 and 127 (inclusive) of type np.int8.
    """
    ant_value = np.uint8(channel_index * ant_index) - 128
    if ant_value == -128:
        ant_value += 1
    return np.int8(ant_value)


def generate_antpair_visibilities_host(bufSamples_host, channel_index, ant1, ant2, time_outer_range, time_inner_range):
    """
    Host-side code to generate visibilities between two antennas.

    Will generate visibilities for a particular channel and antenna combination from an array of input samples that has
    the shape (n_channels, n_samples_per_channel // n_times_per_block, n_ants, n_polarizastions, n_times_per_block)
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


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
def test_correlator_exhaustive(num_ants):
    """
    Exhaustive unit test of the Tensor core correlation algorithm.

    This unit test runs on random input data. The CPU-side check for correctness is very slow - the input sample
    array size has been kept small to speed this up. For MeerKAT sizes, the input sample array will be much bigger.
    """
    # 1. Array parameters
    n_ants = num_ants
    n_channels = 2
    n_samples_per_channel = 16

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
        ctx, n_ants=n_ants, n_channels=n_channels, n_samples_per_channel=n_samples_per_channel
    )
    tensorCoreXEngineCore = template.instantiate(queue)
    tensorCoreXEngineCore.ensure_all_bound()

    bufSamples_device = tensorCoreXEngineCore.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = tensorCoreXEngineCore.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # 3. Generate random input data - need to modify the dtype and  shape of the array as numpy does not have a packet
    # 8-bit int complex type.

    bufSamplesInt16Shape = template.inputDataShape
    bufSamplesInt8Shape = list(bufSamplesInt16Shape)
    bufSamplesInt8Shape[-1] *= 2  # By converting from int16 to int8, the length of the last dimension doubles.
    bufSamplesInt8Shape = tuple(bufSamplesInt8Shape)

    bufSamples_host.dtype = np.int8
    bufSamples_host[:] = np.random.randint(
        low=-127,
        high=128,
        size=bufSamplesInt8Shape,
        dtype=np.int8,
    )
    bufSamples_host.dtype = np.int16

    # 4. Transfer input sample array to the GPU, run kernel, transfer output visibilities array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    tensorCoreXEngineCore()
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
    # samples, I am just going to do a naive brute-force for now to be optomised later if needs be.

    time_outer_range = n_samples_per_channel // template.n_times_per_block
    time_inner_range = template.n_times_per_block

    for channel_index in range(0, n_channels):
        for ant1_index in range(0, n_ants):
            for ant2_index in range(0, ant1_index + 1):
                hh, hv, vh, vv = generate_antpair_visibilities_host(
                    bufSamples_host, channel_index, ant1_index, ant2_index, time_outer_range, time_inner_range
                )
                baseline_index = tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(
                    ant1_index, ant2_index
                )
                bufCorrectVisibilities_host[channel_index][baseline_index][0][0] = hh
                bufCorrectVisibilities_host[channel_index][baseline_index][1][0] = hv
                bufCorrectVisibilities_host[channel_index][baseline_index][0][1] = vh
                bufCorrectVisibilities_host[channel_index][baseline_index][1][1] = vv

    # 5.3 Check that the CPU version and GPU version are identical
    np.testing.assert_array_equal(bufCorrectVisibilities_host, bufVisibilities_host)


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
def test_correlator_quick(num_ants, num_samples_per_channel, num_channels):
    """
    Lightweight unit test of the Tensor core correlation algorithm.

    This unit test uses a hashing function(@get_simple_test_ant_value(...)) to determine the values in the input
    sample array for a specific antenna-channel combination. This value is kept constant over all time values. This
    allows the output visibility values to be calculated quickly on the CPU without performing the full correlation
    operation. This unit test runs much quicker than @test_correlator_exhaustive, while being much less exhaustive.
    It is used to perform a quick check for the correctness on much larger input sample array sizes.
    """
    # 1. Array parameters
    n_ants = num_ants

    # This integer division is so that when n_ants % num_channels !=0 then the remainder will be dropped. This will
    # only occur in the MeerKAT Extension correlator. Technically we will also need to consider the case where we round
    # up as some X-Engines will need to do this to capture all the channels, however that is not done in this test.
    n_channels_per_stream = num_channels // n_ants // 4
    n_samples_per_channel = num_samples_per_channel

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
        ctx, n_ants=n_ants, n_channels=n_channels_per_stream, n_samples_per_channel=n_samples_per_channel
    )
    correlator = template.instantiate(queue)
    correlator.ensure_all_bound()

    time_outer_range = n_samples_per_channel // template.n_times_per_block

    bufSamples_device = correlator.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = correlator.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # 3. Generate predictable input data. The time samples remain constant for every antenna-channel combination.
    bufSamples_host.dtype = np.int8
    for channel_index in range(n_channels_per_stream):
        for ant_index in range(n_ants):
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
    for channel_index in range(0, n_channels_per_stream):
        for ant1_index in range(0, n_ants):
            for ant2_index in range(0, ant1_index + 1):
                ant1_value = get_simple_test_ant_value(channel_index, ant1_index)
                ant1_value = ant1_value + ant1_value * 1j
                ant2_value = get_simple_test_ant_value(channel_index, ant2_index)
                ant2_value = ant2_value + ant2_value * 1j
                baseline_index = tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(
                    ant1_index, ant2_index
                )

                bufSamples_host.dtype = np.int8
                product = ant1_value * np.conj(ant2_value)
                productAccumulated = product * n_samples_per_channel

                assert bufVisibilities_host[channel_index][baseline_index][0][0] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][1][0] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][0][1] == productAccumulated
                assert bufVisibilities_host[channel_index][baseline_index][1][1] == productAccumulated


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
def test_multikernel_accumulation(num_ants):
    """
    Unit test that checks that the Tensor correlation algorithm can accumulate over a number of kernel calls.

    This unit test sets all the input samples to the same value. The output visibilities values are then all the same.
    This dramatically reduces the time taken to check that the multikernel accumulation works correctly. It is not
    required that these values all be random, as that is tested in the @test_correlator_exhaustive function.
    """
    # 1. Array parameters
    n_ants = num_ants
    n_channels = 16
    n_samples_per_channel = 16
    n_kernel_launches = 10

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
        ctx, n_ants=n_ants, n_channels=n_channels, n_samples_per_channel=n_samples_per_channel
    )
    tensorCoreXEngineCore = template.instantiate(queue)
    tensorCoreXEngineCore.ensure_all_bound()

    bufSamples_device = tensorCoreXEngineCore.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = tensorCoreXEngineCore.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # 3. Populate sample buffer so that all real and complex values samples are the same. The real and complex 8-bit
    # values are combined into a single 16-bit integer. This is easier to duplicate across the entire buffer.
    sample_value_i8 = 8
    sample_value_i16 = ((sample_value_i8 << 8) + sample_value_i8) & 0xFFFF
    bufSamples_host = np.full(bufSamples_host.shape, sample_value_i16, dtype=np.int16)

    # 4. Transfer input sample array to the GPU, run kernel, transfer output visibilities array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    tensorCoreXEngineCore()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 5. Test that the data on the output array is not zero, so that the next tests are meaningful.
    #
    # For each multiplication, if each input real and imaginary value has the same value "x", then the result should be:
    # (x + jx)(x - jx) = x^2 + jx^2 - jx^2 + x^2 = 2x^2
    # The real value is 2x^2 and the imaginary value is 0. A single kernel launch will accumulate n_samples_per_channel
    # times, giving an output real visibility value of n_samples_per_channel * 2 * 2x^2.
    #
    # The dtype of the array is int64, the real value being packed in the bottom 32 bits and the imaginary value being
    # packed in the top 32 bits.
    expected_output_real = sample_value_i8 * sample_value_i8 + sample_value_i8 * sample_value_i8
    expected_output_imaginary = sample_value_i8 * sample_value_i8 - sample_value_i8 * sample_value_i8
    output_value_i64 = (expected_output_imaginary * n_samples_per_channel) << 32
    output_value_i64 += expected_output_real * n_samples_per_channel
    np.testing.assert_equal(bufVisibilities_host, output_value_i64)

    # 6. Zero the visibilities on the GPU, transfer the visibilities data back the host, and confirm that it is
    # actually zero.
    tensorCoreXEngineCore.zero_visibilities()
    bufVisibilities_device.get(queue, bufVisibilities_host)
    np.testing.assert_equal(bufVisibilities_host, 0)

    # 7. Run kernel on the same input data, transfer output visibilities array to the CPU. Zeroing is not necessary,
    # as its done above - function is left in to make it clear that the matrix needs to be zeroed at the start of a new
    # accumulation.
    tensorCoreXEngineCore.zero_visibilities()
    bufSamples_device.set(queue, bufSamples_host)
    for _ in range(n_kernel_launches):
        tensorCoreXEngineCore()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 8. Check that multikernel accumulation produces the correct results
    output_value_i64 = (expected_output_imaginary * n_samples_per_channel * n_kernel_launches) << 32
    output_value_i64 += expected_output_real * n_samples_per_channel * n_kernel_launches
    np.testing.assert_equal(bufVisibilities_host, output_value_i64)
