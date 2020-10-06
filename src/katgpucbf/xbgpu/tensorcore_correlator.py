"""Tensor core correlation kernel class."""

import pkg_resources
import numpy as np
from katsdpsigproc import accel

# typedef char2   Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_STATIONS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
# typedef int2    Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

dual_pol_ants = 64
ants_per_block = 64
sample_bitwidth = 8
channels = 16
polarizastions = 2
samples_per_channel = 3072
baselines = int(dual_pol_ants * (dual_pol_ants + 1) / 2)

if ants_per_block == 48:
    nrBlocks = int(
        ((dual_pol_ants + ants_per_block - 1) / ants_per_block)
        * ((dual_pol_ants + ants_per_block - 1) / ants_per_block + 1)
        / 2
    )
elif ants_per_block == 64:
    nrBlocks = int(
        ((dual_pol_ants + ants_per_block - 1) / ants_per_block)
        * ((dual_pol_ants + ants_per_block - 1) / ants_per_block)
    )
else:
    raise ValueError("ants_per_block must equal either 64 or 48, currently equal to {0}.".format(ants_per_block))


if samples_per_channel % 16 != 0:
    raise ValueError("samples_per_channel must be devisible by 8.")

ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)

inputShape = (channels, samples_per_channel // 16, dual_pol_ants, polarizastions, 16)
outputShape = (channels, baselines, polarizastions, polarizastions)

bufSamples_device = accel.DeviceArray(ctx, inputShape, np.int32)
bufSamples_host = bufSamples_device.empty_like()

bufVisibilities_device = accel.DeviceArray(ctx, inputShape, np.int64)
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
    },
    extra_dirs=[pkg_resources.resource_filename(__name__, "")],
)
kernel = program.get_kernel("correlate")

bufSamples_host[:] = np.ones(bufSamples_host.shape, dtype=np.int16)
bufSamples_device.set(queue, bufSamples_host)

queue.enqueue_kernel(
    kernel,
    [bufVisibilities_device.buffer, bufSamples_device.buffer],
    # Even though we are using CUDA, we follow OpenCLs  grid/block conventions, as such we need to multiply the number
    # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
    global_size=(32 * nrBlocks, 2 * channels, 2 * 1),
    local_size=(32, 2, 2),
)

bufVisibilities_device.get(queue, bufVisibilities_host)

print("Total threads: ", 32 * nrBlocks * 2 * channels * 2 * 1)
print("Input Shape: ", inputShape, bufSamples_host.nbytes)
print("Output Shape: ", outputShape, bufVisibilities_host.nbytes)


# import pycuda.driver as cuda
# import pycuda.autoinit  # noqa: F401
# from pycuda.compiler import SourceModule

# module = SourceModule(open("/home/gcallanan/katxgpu/katxgpu/kernels/tensor_core_correlation_kernel.cu").read())
# kernel = module.get_function("kernel_vector_add")
