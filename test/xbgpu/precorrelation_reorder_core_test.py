"""
Module for performing unit tests on the Pre-correlation Reorder as it is being developed for katxgpu.

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
    1. Using static parameters, for now.
    2. Populates a host-side array with random, numpy.int8 data ranging from -127 to 128.
    3. Instantiates the precorrelation_reorder_kernel and passes this input data to it.
    4. Grabs the output, reordered data.
    5. Verifies it relative to the input array.

TODO:
    - Parametrise the unit test(s).
"""

# import pytest
import numpy as np
from katxgpu.precorrelation_reorder_core import PreCorrelationReorderCoreTemplate
from katsdpsigproc import accel


def verify_reorder(
    array_host: accel.Operation.buffer,
    arrayReordered_host: accel.Operation.buffer,
    template: PreCorrelationReorderCoreTemplate,
) -> None:
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

    # 2. Begin scrolling through the arrays and calculating relative indices on-the-fly
    # 2.1. Ultimately still calculating for each batch
    for batchCounter in range(0, template.n_batches):

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
            #   - Turns out we can use most of the originals (which is nice),
            #     the only difference being with the Samples-per-channel/times-per-block strides
            timeIndexOuter = timeIndex_orig // template.n_times_per_block
            timeIndexInner = timeIndex_orig % template.n_times_per_block

            # 2.2.3. Un/Fortunately, the input buffers have to be accessed using the specific dimensions
            #        and not with a single indexing value.
            currData_orig = array_host[batchCounter][antIndex_orig][chanIndex_orig][timeIndex_orig][polIndex_orig]
            currData_new = arrayReordered_host[batchCounter][chanIndex_orig][timeIndexOuter][antIndex_orig][
                polIndex_orig
            ][timeIndexInner]

            errmsg = f"Reordered: {str(currData_new)} at index {str(currIndex)} != Original: {str(currData_orig)} \n"
            assert currData_new == currData_orig, errmsg


def test_precorr_reorder_quick():
    """
    First pytest-compatible unit test for the Pre-correlation Reorder kernel, with static parameters.

    Also invokes verification of reordered data.

    Using default values for the following:
    - Antennas = 64
    - Channels = 128
    - Samples per Channel = 256
    - Polarisations = 2
    - Batches = 2, because more batches take longer to verify...
    """
    # Now to create the actual PrecorrelationReorderCoreTemplate
    # 1. Array parameters
    # - Will be {ants, chans, samples_per_chan, batches}
    n_ants = 64
    n_channels = 128
    n_samples_per_channel = 256
    n_batches = 2

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = PreCorrelationReorderCoreTemplate(
        ctx,
        n_ants=n_ants,
        n_channels=n_channels,
        n_samples_per_channel=n_samples_per_channel,
        n_batches=n_batches,
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

    verify_reorder(bufSamples_host, bufReordered_host, template)
