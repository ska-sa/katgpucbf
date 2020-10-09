"""TODO: Write docstring."""

import pytest
import numpy as np
import tensorcore_correlator
from katsdpsigproc import accel

# Different array sizes that could potentially be used by MeerKAT
array_size = [4, 8, 16, 32, 64, 84, 192, 256]


def get_simple_test_ant_value(channel_index, ant_index):
    """TODO: Write docstring."""
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

    Returns visibilities hh, hv, vh and vv representing different interpolarisation products.

    This is a naive implementation of the algorithm and is very slow
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
    """TODO: Write docstring."""
    dual_pol_ants = num_ants
    channels = 2
    samples_per_channel = 16

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

    bufSamples_device.set(queue, bufSamples_host)
    tensorCoreCorrelator()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 8. Check that the GPU visibilities are correct.

    # 8.1 Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
    # complex type.
    bufSamples_host.dtype = np.int8
    bufSamples_host = bufSamples_host.astype(np.float32)
    bufSamples_host.dtype = np.csingle

    bufVisibilities_host.dtype = np.int32
    bufVisibilities_host = bufVisibilities_host.astype(np.float32)
    bufVisibilities_host.dtype = np.csingle

    bufCorrectVisibilities_host = np.empty_like(bufVisibilities_host)
    # 8.2 Generate the visibilities on the CPU - this is not a simple matrix operation due to the indexing of the input
    # samples, I am just going to do a naive brute force for now to be optomised later if needs be.

    time_outer_range = samples_per_channel // template._times_per_block
    time_inner_range = template._times_per_block

    for channel_index in range(0, channels):
        print(channel_index)
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

                # if(bufVisibilities_host[channel_index][baseline_index][0][0] != hh or
                #    bufVisibilities_host[channel_index][baseline_index][1][0] != hv or
                #    bufVisibilities_host[channel_index][baseline_index][0][1] != vh or
                #    bufVisibilities_host[channel_index][baseline_index][1][1] != vv):
                #    print("What",channel_index,ant1_index,ant2_index)
                #    print(hh, bufVisibilities_host[channel_index][baseline_index][0][0])
                #    print(hv, bufVisibilities_host[channel_index][baseline_index][1][0])
                #    print(vh, bufVisibilities_host[channel_index][baseline_index][0][1])
                #    print(vv, bufVisibilities_host[channel_index][baseline_index][1][1])
                #    temp += 1

    # print(temp,baselines,channels)

    # 8.3 Check that the CPU version and GPU version are identical
    np.testing.assert_array_equal(bufCorrectVisibilities_host, bufVisibilities_host)


@pytest.mark.parametrize("num_ants", array_size)
def test_correlator_quick(num_ants):
    """TODO: Write docstring."""
    dual_pol_ants = num_ants
    channels = 16
    samples_per_channel = 256

    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_correlator.TensorCoreCorrelatorTemplate(
        ctx, dual_pol_ants=dual_pol_ants, channels=channels, samples_per_channel=samples_per_channel
    )
    correlator = template.instantiate(queue)
    correlator.ensure_all_bound()

    time_outer_range = samples_per_channel // template._times_per_block
    # time_inner_range = template._times_per_block

    bufSamples_device = correlator.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufVisibilities_device = correlator.buffer("outVisibilities")
    bufVisibilities_host = bufVisibilities_device.empty_like()

    # Generate predictable values
    bufSamples_host.dtype = np.int8
    for channel_index in range(channels):
        for ant_index in range(dual_pol_ants):
            sample_value = get_simple_test_ant_value(channel_index, ant_index)
            for time_outer_index in range(time_outer_range):
                bufSamples_host[channel_index][time_outer_index][ant_index] = np.full(
                    bufSamples_host.shape[3:], sample_value, dtype=np.int8
                )
    print(bufSamples_host.shape[3:])
    bufSamples_host.dtype = np.int16
    print(bufSamples_host.shape[3:])

    bufSamples_device.set(queue, bufSamples_host)
    correlator()
    bufVisibilities_device.get(queue, bufVisibilities_host)

    # 8. Check that the GPU visibilities are correct.

    # 8.1 Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
    # complex type.

    bufVisibilities_host.dtype = np.int32
    bufVisibilities_host = bufVisibilities_host.astype(np.float32)
    bufVisibilities_host.dtype = np.csingle

    for channel_index in range(0, channels):
        print(channel_index)
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
