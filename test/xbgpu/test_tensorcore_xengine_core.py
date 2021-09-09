"""Module for performing unit tests on the Tensor core correlation kernel."""
import numpy as np
import pytest
from katsdpsigproc import accel
from numba import njit, prange

from katgpucbf.xbgpu import tensorcore_xengine_core

from . import test_parameters

get_baseline_index = njit(tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index)


@njit(parallel=True)
def correlate_host(input_array: np.ndarray) -> np.ndarray:
    """Calculate correlation products on the host CPU.

    Parameters
    ----------
    input_array
        Dataset to be correlated. Required shape:
        (n_chans, n_blocks, n_ants, n_pols, n_spectra_per_block, complexity)

    Returns
    -------
    np.ndarray
        Correlation products or visibilities. Shape:
        (n_chans, n_ants, n_pols, n_spectra_per_heap_in, complexity)
        where n_spectra_per_heap_in is equal to n_blocks * n_spectra_per_block
        in the input.
    """
    n_chans = input_array.shape[0]
    n_ants = input_array.shape[2]
    n_pols = input_array.shape[3]
    n_spectra_per_heap_in = input_array.shape[1] * input_array.shape[4]
    complexity = input_array.shape[5]
    # I think it's nicer to get the time-series all in one axis so that we can
    # just use np.sum() across that axis.
    # Numba can't work on non-contiguous arrays, so we have to copy this one in
    # order to be able to reshape it in the next step.
    ez_in = input_array.transpose(0, 2, 3, 1, 4, 5).copy()
    ez_in = ez_in.reshape((n_chans, n_ants, n_pols, n_spectra_per_heap_in, complexity)).astype(np.int32)
    n_baselines = n_ants * (n_ants + 1) * 2
    output_array = np.empty(shape=(n_chans, n_baselines, complexity), dtype=np.int32)
    for c in prange(n_chans):
        for a2 in range(n_ants):
            for a1 in range(a2 + 1):
                for p1 in range(n_pols):
                    r1 = ez_in[c, a1, p1, :, 0]
                    i1 = ez_in[c, a1, p1, :, 1]
                    for p2 in range(n_pols):
                        bl_idx = get_baseline_index(a1, a2) * 4 + p1 + 2 * p2
                        r2 = ez_in[c, a2, p2, :, 0]
                        i2 = ez_in[c, a2, p2, :, 1]

                        output_array[c, bl_idx, 0] = np.sum(r1 * r2 + i1 * i2)
                        output_array[c, bl_idx, 1] = np.sum(r2 * i1 - r1 * i2)

    return output_array


@pytest.mark.combinations(
    "num_ants, num_channels, num_spectra_per_heap_in",
    test_parameters.array_size,
    test_parameters.num_channels,
    test_parameters.num_spectra_per_heap_in,
)
def test_correlator(num_ants, num_spectra_per_heap_in, num_channels):
    """Parameterised unit test of the Tensor-Core correlation kernel."""
    # TODO: A lot of this is duplicated in other functions. It would be nice to
    # move it into a test fixture.
    n_chans_per_stream = num_channels // num_ants // 4
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
        ctx, n_ants=num_ants, n_channels=n_chans_per_stream, n_spectra_per_heap_in=num_spectra_per_heap_in
    )
    tensor_core_x_engine_core = template.instantiate(queue)
    tensor_core_x_engine_core.ensure_all_bound()

    buf_samples_device = tensor_core_x_engine_core.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()

    rng = np.random.default_rng(seed=2021)
    buf_samples_host[:] = rng.integers(
        # The Tensor-Core correlator can't manage the maximum negative value,
        # due to the asymmetry of signed integers, so we adjust the lower bound
        # up by 1.
        low=np.iinfo(buf_samples_host.dtype).min + 1,
        high=np.iinfo(buf_samples_host.dtype).max,
        size=buf_samples_host.shape,
        dtype=buf_samples_host.dtype,
        endpoint=True,  # We don't need to exclude the maximum positive value though.
    )

    buf_visibilities_device = tensor_core_x_engine_core.buffer("out_visibilities")
    buf_visibilities_host = buf_visibilities_device.empty_like()

    buf_samples_device.set(queue, buf_samples_host)
    tensor_core_x_engine_core()
    buf_visibilities_device.get(queue, buf_visibilities_host)

    calculated_visibilities_host = correlate_host(buf_samples_host)
    np.testing.assert_equal(buf_visibilities_host, calculated_visibilities_host)


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
    n_spectra_per_heap_in = 16
    n_kernel_launches = 10

    # 2. Initialise GPU kernels and buffers.
    ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    queue = ctx.create_command_queue()

    template = tensorcore_xengine_core.TensorCoreXEngineCoreTemplate(
        ctx, n_ants=n_ants, n_channels=n_channels, n_spectra_per_heap_in=n_spectra_per_heap_in
    )
    tensor_core_x_engine_core = template.instantiate(queue)
    tensor_core_x_engine_core.ensure_all_bound()

    buf_samples_device = tensor_core_x_engine_core.buffer("in_samples")
    buf_samples_host = buf_samples_device.empty_like()

    buf_visibilities_device = tensor_core_x_engine_core.buffer("out_visibilities")
    buf_visibilities_host = buf_visibilities_device.empty_like()

    # 3. Populate sample buffer so that all real and complex values samples are the same.
    sample_value_i8 = 8
    buf_samples_host[:] = sample_value_i8

    # 4. Transfer input sample array to the GPU, run kernel, transfer output visibilities array to the CPU.
    buf_samples_device.set(queue, buf_samples_host)
    tensor_core_x_engine_core()
    buf_visibilities_device.get(queue, buf_visibilities_host)

    # 5. Test that the data on the output array is not zero, so that the next tests are meaningful.
    #
    # For each multiplication, if each input real and imaginary value has the same value "x", then the result should be:
    # (x + jx)(x - jx) = x^2 + jx^2 - jx^2 + x^2 = 2x^2
    # The real value is 2x^2 and the imaginary value is 0. A single kernel launch will accumulate n_spectra_per_heap_in
    # times, giving an output real visibility value of n_spectra_per_heap_in * 2 * 2x^2.
    expected_output = np.zeros_like(buf_visibilities_host)
    expected_output[:, :, 0] = (
        sample_value_i8 * sample_value_i8 + sample_value_i8 * sample_value_i8
    ) * n_spectra_per_heap_in
    np.testing.assert_equal(buf_visibilities_host, expected_output)

    # 6. Zero the visibilities on the GPU, transfer the visibilities data back the host, and confirm that it is
    # actually zero.
    tensor_core_x_engine_core.zero_visibilities()
    buf_visibilities_device.get(queue, buf_visibilities_host)
    np.testing.assert_equal(buf_visibilities_host, 0)

    # 7. Run kernel on the same input data, transfer output visibilities array to the CPU. Zeroing is not necessary,
    # as its done above - function is left in to make it clear that the matrix needs to be zeroed at the start of a new
    # accumulation.
    tensor_core_x_engine_core.zero_visibilities()
    buf_samples_device.set(queue, buf_samples_host)
    for _ in range(n_kernel_launches):
        tensor_core_x_engine_core()
    buf_visibilities_device.get(queue, buf_visibilities_host)

    # 8. Check that multikernel accumulation produces the correct results
    expected_output *= n_kernel_launches
    np.testing.assert_equal(buf_visibilities_host, expected_output)
