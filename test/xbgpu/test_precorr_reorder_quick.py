"""
Simple script to do a quick test of the pre-correlation reorder as it is being developed for katxgpu.

The pre-correlation reorder operates on a block of data with the following dimensions:
    - uint16_t [n_antennas] [n_channels] [n_samples_per_channel] [polarizations]
      transposed to
      uint16_t [n_channels] [n_samples_per_channel//times_per_block]
            [n_antennas] [polarizations] [times_per_block]
    - Typical values for the dimensions
        - n_antennas (a) = 64
        - n_channels (c) = 128
        - n_samples_per_channel (t) = 256
        - polarisations (p) = 2, always
        - times_per_block = 16, always

Contains one test:
    1. Accepting parameters for the input dimensions via command-line arguments using argparse
    2. Populates a host-side array with random, numpy.int8 data ranging from -127 to 128.
    3. Instantiates the precorrelation_reorder_kernel and passes this input data to it.
    4. Grabs the output, reordered data.
    5. Verifies it relative to the input array.

TODO:
    - Implement batch(ed) operation!
"""

import argparse
import numpy as np
from katxgpu.precorrelation_reorder_core import PreCorrelationReorderCoreTemplate

# from katxgpu import test_parameters
from katsdpsigproc import accel


def verify_reorder(
    array_host: accel.Operation.buffer,
    arrayReordered_host: accel.Operation.buffer,
    template: PreCorrelationReorderCoreTemplate,
) -> bool:
    """
    Operation verificiation function for Pre-correlation Reorder data output by the kernel.

    This is done using a single for-loop, calculating the respective strides and indices on-the-fly.

    Parameters
    ----------
    array_host: accel.Operation.buffer
        The original, host-side buffer populated with data to be reordered by the kernel.
    arrayReordered_host: accel.Operation.buffer
        The reordered data output by the precorrelation_reorder_kernel, copied back to the host.
    template: PreCorrelationReorderCoreTemplate
        The template has all the attributes of the dimensions of data being processed
        - Antennas, Channels, Samples-per-channel, Polarisations and Times-per-block
    """
    # 1. Declare dimensional strides
    #    - These will be used as w
    # 1.1. For the original, input array
    antStride_orig = template.n_channels * template.n_samples_per_channel * template.n_polarisations
    chanStride_orig = template.n_samples_per_channel * template.n_polarisations
    timeStride_orig = template.n_polarisations
    # polStride_orig = 1

    # 1.2. For the output, reordered array
    chanStride_new = (
        (template.n_samples_per_channel // template.n_times_per_block)
        * template.n_ants
        * template.n_polarisations
        * template.n_times_per_block
    )
    sampleStride_new = template.n_ants * template.n_polarisations * template.n_times_per_block
    antStride_new = template.n_polarisations * template.n_times_per_block
    polStride_new = template.n_times_per_block

    # 2. Begin scrolling through the arrays and calculating relative indices on-the-fly
    # 2.1. Ultimately still calculating for each batch
    # for batchCounter in range(0, template.n_batches):

    #     # Calculate the outer-most matrix stride, based on which batch we are verifying
    #     matrix_stride = batchCounter * template.matrix_size # How convenient

    # 2.2. Now for the particular matrix of this batch
    for currIndex in range(0, template.matrix_size):

        # 2.2.1. Calculate the original, input indices
        antIndex_orig = currIndex // antStride_orig
        remIndex = currIndex % antStride_orig

        chanIndex_orig = remIndex // chanStride_orig
        remIndex = remIndex % chanStride_orig

        timeIndex_orig = remIndex // timeStride_orig
        remIndex = remIndex % timeStride_orig
        # 0 = Even = Pol-0, 1 = Odd = Pol-1
        polIndex_orig = remIndex

        # 2.2.2. Calculate the new, reordered indices
        chanOffset_new = chanIndex_orig * chanStride_new
        timeIndexOuter = timeIndex_orig // template.n_times_per_block
        timeOuterOffset = timeIndexOuter * sampleStride_new
        antOffset_new = antIndex_orig * antStride_new
        polOffset_new = polIndex_orig * polStride_new
        timeIndexInner = timeIndex_orig % template.n_times_per_block

        # 2.2.3. Un/Fortunately, the input buffers have to be accessed using the specific dimensions
        #        and not with a single indexing value.
        currData_orig = array_host[antIndex_orig][chanIndex_orig][timeIndex_orig][polIndex_orig]
        currData_new = arrayReordered_host[chanOffset_new][timeOuterOffset][antOffset_new][polOffset_new][
            timeIndexInner
        ]

        if currData_new != currData_orig:
            # Problem
            errmsg = (
                "Reordered: "
                + str(currData_new)
                + " at index "
                + str(currIndex)
                + " != Original: "
                + str(currData_orig)
                + "\n"
            )
            raise ValueError(errmsg)

    return True


if __name__ == "__main__":
    desc = """Initial test script for Pre-correlation Reorder kernel."""
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-a",
        "--ants",
        type=int,
        action="store",
        default=4,  # choices=range(1, 64),
        help="Number of Antennas to execute for the kernel, between 1 and 64.",
    )
    parser.add_argument(
        "-c",
        "--chans",
        type=int,
        action="store",
        default=4,  # choices=range(1, 128),
        help="Number of Channels to execute for the kernel, typically 128.",
    )
    parser.add_argument(
        "-s",
        "--samples_per_chan",
        type=int,
        action="store",
        default=32,  # choices=range(1, 256),
        help="Number of Samples per Channel, typically 256.",
    )
    parser.add_argument(
        "-b", "--batches", type=int, action="store", default=1, help="Number of batches of reorder to run."
    )

    args = parser.parse_args()

    # TODO: Error-check the inputs!

    # Now to create the actual PrecorrelationReorderCoreTemplate
    # 1. Array parameters
    # - Will be args.{ants, chans, samples_per_chan, batches}

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = PreCorrelationReorderCoreTemplate(
        ctx,
        n_ants=args.ants,
        n_channels=args.chans,
        n_samples_per_channel=args.samples_per_chan,
        n_batches=args.batches,
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
    bufSamplesInt8Shape = tuple(bufSamplesInt8Shape)  # type: ignore

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
    #    - Both the input and output data are ultimately of type np.int8
    # bufSamples_host.dtype = np.int16 # Shouldn't need this line..
    bufSamples_host = bufSamples_host.astype(np.int8)
    bufSamples_host.dtype = np.int8

    bufReordered_host.dtype = np.int16
    bufReordered_host = bufReordered_host.astype(np.int8)
    bufReordered_host.dtype = np.int8

    # bufCorrectReordered_host = np.empty_like(bufReordered_host)
    print("\n------------------------------------\n")
    print(bufReordered_host.shape)

    result = verify_reorder(bufSamples_host, bufReordered_host, template)
    print(result)
