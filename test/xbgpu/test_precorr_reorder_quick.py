# This is a simple script to do a quick, ad-hoc test of the pre-correlation reorder
# as it is being developed/incorporated into katxgpu

import argparse
import numpy as np
from katxgpu import precorrelation_reorder_core
# from katxgpu import test_parameters
from katsdpsigproc import accel


if __name__ == '__main__':
    desc = """Initial test script for Pre-correlation Reorder kernel."""
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-a', '--ants', type=int, action='store', default=4,# choices=range(1, 64),
        help='Number of Antennas to execute for the kernel, between 1 and 64.')
    parser.add_argument(
        '-c', '--chans', type=int, action='store', default=4, # choices=range(1, 128),
        help='Number of Channels to execute for the kernel, typically 128.')
    parser.add_argument(
        '-s', '--samples_per_chan', type=int, action='store', default=32,# choices=range(1, 256),
        help='Number of Samples per Channel, typically 256.')
    parser.add_argument(
        '-b', '--batches', type=int, action='store', default=1,
        help='Number of batches of reorder to run.')

    args = parser.parse_args()

    # TODO: Error-check the inputs!

    # Now to create the actual PrecorrelationReorderCoreTemplate
    # 1. Array parameters
    # - Will be args.{ants, chans, samples_per_chan, batches}

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = precorrelation_reorder_core.PreCorrelationReorderCoreTemplate(
                ctx, n_ants=args.ants, n_channels=args.chans,
                n_samples_per_channel=args.samples_per_chan,
                n_batches=args.batches
                )
    preCorrelationReorderCore = template.instantiate(queue)
    preCorrelationReorderCore.ensure_all_bound()

    bufSamples_device = preCorrelationReorderCore.buffer("inSamples")
    bufSamples_host = bufSamples_device.empty_like()

    bufReordered_device = preCorrelationReorderCore.buffer("outReordered")
    bufReordered_host = bufReordered_device.empty_like()

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

    # 4. Transfer input sample array to the GPU, run kernel, transfer output Reordered array to the CPU.
    bufSamples_device.set(queue, bufSamples_host)
    preCorrelationReorderCore()
    bufReordered_device.get(queue, bufReordered_host)

    # 5. Verify the processed/returned result
    # 5.1. Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
    #      complex type.
    bufSamples_host.dtype = np.int8
    bufSamples_host = bufSamples_host.astype(np.float32)
    bufSamples_host.dtype = np.csingle

    bufReordered_host.dtype = np.int32
    bufReordered_host = bufReordered_host.astype(np.float32)
    bufReordered_host.dtype = np.csingle

    # bufCorrectReordered_host = np.empty_like(bufReordered_host)
    print(bufReordered_host)
    print("\n------------------------------------\n")

    for ant_index in range(0, args.ants):
        for chan_index in range(0, args.chans):
            for sample_index in range(0, args.samples_per_chan):
                print(bufReordered_host[ant_index][chan_index][sample_index] + " ")
            print("\n")
        print("\n------------------------------------\n")