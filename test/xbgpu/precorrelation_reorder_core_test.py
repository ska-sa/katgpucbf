"""
Module for performing unit tests on the Pre-correlation Reorder.

NOTE: These tests will take some time to complete altogether, we would recommend running them separately via:
    - pytest -v test/precorrelation_reorder_core_test.py::test_precorr_reorder_{parametrised, batched}

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

Contains two tests, one parametrised and one batched:
    1. The first test uses the list of values present in test/test_parameters.py to run the
        kernel through a range of value combinations.
        - This is limited to a batch of one, as the CPU-side verification takes some time to complete.
    2. The second test uses a static set of value combinations for the largest possible array size,
        but now testing for multiple batches.
        - Antennas = 84
        - Channels (from the F-Engine) = 4096
        - Samples per Channel = 256
        - Batches = 3

Ultimately both kernels:
    1. Populate a host-side array with random, numpy.int8 data ranging from -127 to 128.
    2. Instantiate the precorrelation_reorder_kernel and passes this input data to it.
    3. Grab the output, reordered data.
    4. Verify it relative to the input array.

"""

import os
import pytest
import test_parameters
import numpy as np
from ctypes import c_int  # Only need this from the library
from katxgpu.precorrelation_reorder_core import PreCorrelationReorderCoreTemplate
from katsdpsigproc import accel


def verify_reorder(
    array_host: accel.Operation.buffer,
    arrayReordered_host: accel.Operation.buffer,
    template: PreCorrelationReorderCoreTemplate,
) -> None:
    """
    Verificiation function for Pre-correlation Reorder data output by the kernel.

    This is done using a single for-loop, calculating the respective strides and indices on-the-fly.
    NOTE: This has been superceded by the C-based function in the shared library, functionlib.c.

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


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
def test_precorr_reorder_parametrised(num_ants, num_channels, num_samples_per_channel):
    """
    Parametrised unit test of the Pre-correlation Reorder kernel.

    This unit test runs the kernel on a combination of parameters indicated in test_parameters.py. The values parametrised
    are indicated in the parameter list, operating on a *single* batch. This unit test also invokes verification of the reordered data.
    However, due to the CPU-side verification taking quite long, the batch-mode test is done in a separate unit test using static parameters.

    Parameters
    ----------
    num_ants: int
        The number of antennas from which data will be received.
    num_channels: int
        The number of frequency channels out of the FFT.
        NB: This is not the number of FFT channels per stream.
        The number of channels per stream is calculated from this value.
    num_samples_per_channel: int
        The number of time samples per frequency channel.
    """
    # Now to create the actual PrecorrelationReorderCoreTemplate
    # 1. Array parameters
    # - Will be {ants, chans, samples_per_chan, batches}
    n_ants = num_ants

    # This integer division is so that when n_ants % num_channels !=0 then the remainder will be dropped. This will
    # only occur in the MeerKAT Extension correlator. Technically we will also need to consider the case where we round
    # up as some X-Engines will need to do this to capture all the channels, however that is not done in this test.
    n_channels_per_stream = num_channels // n_ants // 4
    n_samples_per_channel = num_samples_per_channel
    n_batches = 1  # Parametrised test is complex enough already!

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = PreCorrelationReorderCoreTemplate(
        ctx,
        n_ants=n_ants,
        n_channels=n_channels_per_stream,
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
    bufSamples_host = bufSamples_host.astype(np.int8)
    bufSamples_host.dtype = np.int8

    bufReordered_host.dtype = np.int16
    bufReordered_host = bufReordered_host.astype(np.int8)
    bufReordered_host.dtype = np.int8

    # 5.1. Now using the external C-library for a more efficient verification
    #   - TODO: Need to figure out a better way of indicating the loader_path
    #   - Using './test/' as this function is called from the parent (root) katxgpu directory
    functionlib_C = np.ctypeslib.load_library(libname="libfunc.so", loader_path=os.path.abspath("./test/"))
    verify_reorder_C = functionlib_C.verify_reorder

    # 5.1.1. Need to clarify these argument types
    #   - Ideally we want to pass Pointers to the array in here
    #   - As well as the array dimensions to calculate the strides
    #   - Fortunately, we can flatten numpy.ndarrays, so its shape is simply the total matrix size
    verify_reorder_C.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.int8, shape=(template.matrix_size * template.n_batches,), flags="C_CONTIGUOUS"  # Input data array
        ),
        np.ctypeslib.ndpointer(
            dtype=np.int8,  # Reordered, output data array
            shape=(template.matrix_size * template.n_batches,),
            flags="C_CONTIGUOUS",
        ),
        c_int,  # Batches
        c_int,  # Antennas
        c_int,  # Channels
        c_int,  # Samples-per-channel
        c_int,  # Polarisations
        c_int,  # Times-per-block
    ]

    #   - Function return type - 0/1: Fail/Success
    verify_reorder_C.restype = c_int

    # 5.1.2. Call the function with the required variables
    result = verify_reorder_C(
        bufSamples_host.flatten(order="C"),  # Flatten in row-major (C-style) order
        bufReordered_host.flatten(order="C"),
        template.n_batches,
        template.n_ants,
        template.n_channels,
        template.n_samples_per_channel,
        template.n_polarisations,
        template.n_times_per_block,
    )

    if result:
        # Great success!
        print("Reorder was successful!")
    else:
        print("Reorder failed...")


def test_precorr_reorder_batched():
    """
    Unit test for the Pre-correlation Reorder kernel in batched operation, with static parameters.

    Also invokes verification of reordered data. Note that the CPU-side verification will take around five minutes for a batch size of ten (10) :(.
    Using default values for the following:
    - Antennas = 84
    - F-Engine Channels = 4096
    - Samples per Channel = 256
    - Batches = 3
    """
    # Now to create the actual PrecorrelationReorderCoreTemplate
    # 1. Array parameters
    # - Will be {ants, chans, samples_per_chan, batches}
    n_ants = 84
    n_channels = 4096 // n_ants // 4
    n_samples_per_channel = 256
    n_batches = 10

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

    # These buffers are of type katsdpsigproc.accel.HostArray
    # - Which is further of type np.ndarray
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
    #   - Both the input and output data are ultimately of type np.int8
    bufSamples_host = bufSamples_host.astype(np.int8)
    bufSamples_host.dtype = np.int8

    bufReordered_host.dtype = np.int16
    bufReordered_host = bufReordered_host.astype(np.int8)
    bufReordered_host.dtype = np.int8

    # 5.1. Now using the external C-library for a more efficient verification
    #   - TODO: Need to figure out a better way of indicating the loader_path
    #   - Using './test/' as this function is called from the parent (root) katxgpu directory
    functionlib_C = np.ctypeslib.load_library(libname="libfunc.so", loader_path=os.path.abspath("./test/"))
    verify_reorder_C = functionlib_C.verify_reorder

    # 5.1.1. Need to clarify these argument types
    #   - Ideally we want to pass Pointers to the array in here
    #   - As well as the array dimensions to calculate the strides
    #   - Fortunately, we can flatten numpy.ndarrays, so its shape is simply the total matrix size
    verify_reorder_C.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.int8, shape=(template.matrix_size * template.n_batches,), flags="C_CONTIGUOUS"  # Input data array
        ),
        np.ctypeslib.ndpointer(
            dtype=np.int8,  # Reordered, output data array
            shape=(template.matrix_size * template.n_batches,),
            flags="C_CONTIGUOUS",
        ),
        c_int,  # Batches
        c_int,  # Antennas
        c_int,  # Channels
        c_int,  # Samples-per-channel
        c_int,  # Polarisations
        c_int,  # Times-per-block
    ]

    #   - Function return type - 0/1: Fail/Success
    verify_reorder_C.restype = c_int

    # 5.1.2. Call the function with the required variables
    result = verify_reorder_C(
        bufSamples_host.flatten(order="C"),  # Flatten in row-major (C-style) order
        bufReordered_host.flatten(order="C"),
        template.n_batches,
        template.n_ants,
        template.n_channels,
        template.n_samples_per_channel,
        template.n_polarisations,
        template.n_times_per_block,
    )

    if result:
        # Great success!
        print("Reorder was successful!")
    else:
        print("Reorder failed...")
