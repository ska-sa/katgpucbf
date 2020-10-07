"""Tensor core correlation kernel class."""

# TODO: Make kernel work for very long integration times
# TODO: Fix the pycuda "extern "C" {}"" issue
# TODO: Make not that -128 is a big no go

# 1.imports
import pkg_resources
import numpy as np
from katsdpsigproc import accel

# 2. key parameters
dual_pol_ants = 64
ants_per_block = 64
sample_bitwidth = 8
channels = 8
polarizastions = 2
samples_per_channel = 256
baselines = int(dual_pol_ants * (dual_pol_ants + 1) // 2)
times_per_block = 128 // sample_bitwidth

inputDataShape = (channels, samples_per_channel // times_per_block, dual_pol_ants, polarizastions, times_per_block)
outputDataShape = (channels, baselines, polarizastions, polarizastions)

# At the moment we have hard coded ants_per_block to be 64, however 48 is also an option and may give some performance
# benefits for N = 84, I dont see it helping too much for the other use cases.

# 3. Sanitize key paramters

if ants_per_block == 48:
    nrBlocks = int(
        ((dual_pol_ants + ants_per_block - 1) // ants_per_block)
        * ((dual_pol_ants + ants_per_block - 1) // ants_per_block + 1)
        // 2
    )
elif ants_per_block == 64:
    nrBlocks = int(
        ((dual_pol_ants + ants_per_block - 1) // ants_per_block)
        * ((dual_pol_ants + ants_per_block - 1) // ants_per_block)
    )
else:
    raise ValueError("ants_per_block must equal either 64 or 48, currently equal to {0}.".format(ants_per_block))


if samples_per_channel % times_per_block != 0:
    raise ValueError("samples_per_channel must be devisible by {0}.".format(times_per_block))

# 4. Set up GPU
ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)

bufSamples_device = accel.DeviceArray(ctx, inputDataShape, np.int16)
bufSamples_host = bufSamples_device.empty_like()

bufVisibilities_device = accel.DeviceArray(ctx, outputDataShape, np.int64)
bufVisibilities_host = bufVisibilities_device.empty_like()

queue = ctx.create_command_queue()

program = accel.build(
    ctx,
    "kernels/tensor_core_correlation_kernel.cu",
    {
        "ants_per_block": ants_per_block,
        "dual_pol_ants": dual_pol_ants,
        "sample_bitwidth": sample_bitwidth,
        "channels": channels,
        "polarizastions": polarizastions,
        "samples_per_channel": samples_per_channel,
        "baselines": baselines,
        "times_per_block": times_per_block,
    },
    extra_dirs=[pkg_resources.resource_filename(__name__, "")],
)
kernel = program.get_kernel("correlate")

# 5. Generate sample data
bufSamples_host.dtype = np.int8
bufSamples_host[:] = np.random.randint(
    -127,
    high=128,
    size=(channels, samples_per_channel // times_per_block, dual_pol_ants, polarizastions, times_per_block * 2),
    dtype=np.int8,
)  # np.ones(bufSamples_host.shape, dtype=np.int8)
bufSamples_host.dtype = np.int16
bufSamples_device.set(queue, bufSamples_host)

# 6. Run Kernel
queue.enqueue_kernel(
    kernel,
    [bufVisibilities_device.buffer, bufSamples_device.buffer],
    # Even though we are using CUDA, we follow OpenCLs  grid/block conventions, as such we need to multiply the number
    # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
    global_size=(32 * nrBlocks, 2 * channels, 2 * 1),
    local_size=(32, 2, 2),
)

# 7. Retrieve visibilities
bufVisibilities_device.get(queue, bufVisibilities_host)

# 8. Check that the GPU visibilities are correct.

# 8.1 Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
# complex type.
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host.dtype = np.int8
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host = bufSamples_host.astype(np.float32)
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host.dtype = np.csingle
print(bufSamples_host.shape, bufSamples_host.nbytes)

print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host.dtype = np.int32
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host = bufVisibilities_host.astype(np.float32)
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host.dtype = np.csingle
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)

bufCorrectVisibilities_host = np.empty_like(bufVisibilities_host)
print(bufCorrectVisibilities_host.shape, bufCorrectVisibilities_host.nbytes)

# 8.2 Generate the visibilities on the CPU - this is not a simple matrix operation due to the indexing of the input
# samples, I am just going to do a naive brute force for now to be optomised later if needs be.


def calculate_baseline_index(ant1, ant2):
    """
    Return the index in the visibilities output matrix of the visibility produced by ant1 and ant2.

    The visibilities matrix indexing is as follows:
        ant1 = 0  1  2  3  4
             +---------------
    ant2 = 0 | 00 01 03 06 10
           1 |    02 04 07 11
           2 |       05 08 12
           3 |          09 13
           4 |             14

    This function requires that ant1>=ant2
    """
    if ant2 > ant1:
        raise ValueError("It is required that ant1 >= ant2 in all cases")
    return ant1 * (ant1 + 1) // 2 + ant2


def generate_visibilities_host(bufSamples_host, channel_index, ant1, ant2):
    """
    Host side code to generate visibility matrix.

    Will generate visibilities for a particular channel and antenna combination from an array of input samples that has
    the shape (channels, samples_per_channel // times_per_block, dual_pol_ants, polarizastions, times_per_block)
    required by the Tensor core correlation kernels.
    """
    hh, hv, vh, vv = 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j
    for time_index1 in range(0, samples_per_channel // times_per_block):
        for time_index2 in range(0, times_per_block):
            ant1h = bufSamples_host[channel_index][time_index1][ant1][0][time_index2]
            ant1v = bufSamples_host[channel_index][time_index1][ant1][1][time_index2]
            ant2h = bufSamples_host[channel_index][time_index1][ant2][0][time_index2]
            ant2v = bufSamples_host[channel_index][time_index1][ant2][1][time_index2]

            hh += ant1h * np.conj(ant2h)
            vh += ant1h * np.conj(ant2v)
            hv += ant1v * np.conj(ant2h)
            vv += ant1v * np.conj(ant2v)

    return hh, hv, vh, vv


temp = 0

for channel_index in range(0, channels):
    print(channel_index)
    for ant1_index in range(0, dual_pol_ants):
        for ant2_index in range(0, ant1_index + 1):
            hh, hv, vh, vv = generate_visibilities_host(bufSamples_host, channel_index, ant1_index, ant2_index)
            baseline_index = calculate_baseline_index(ant1_index, ant2_index)
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
