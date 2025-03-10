################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


"""Module wrapping the ASTRON Tensor-Core Correlation Kernels in the MeerKAT katsdpsigproc framework.

.. todo::

    Eventually modify the classes to support 4 and 16 bit input samples. The
    kernel supports this, but it is not exposed to the reader. There is no use
    case for this at the moment, so this is a low priority.

"""

import importlib.resources

import numpy as np
from katsdpsigproc import accel, cuda
from katsdpsigproc.abc import AbstractContext, AbstractDevice

from .. import COMPLEX, N_POLS

#: Minimum CUDA compute capability needed for the kernel (with 8-bit samples)
MIN_COMPUTE_CAPABILITY = (7, 2)
#: Magic value indicating missing data
MISSING = np.array([-(2**31), 1], dtype=np.int32)


def device_filter(device: AbstractDevice) -> bool:
    """Determine whether a device is suitable for running the kernel."""
    return isinstance(device, cuda.Device) and device.compute_capability >= MIN_COMPUTE_CAPABILITY


class CorrelationTemplate:
    r"""Template class for the Tensor-Core correlation kernel.

    The template creates a :class:`Correlation` that will run the
    compiled kernel. The parameters are used to compile the kernel and by the
    :class:`Correlation` to specify the shape of the memory buffers
    connected to this kernel.

    The number of baselines calculated here is not the canonical way that it is
    done in radio astronomy:

    .. math::

        n_{baselines} = \frac{n_{ants} * (n_{ants} + 1)}{2}

    Because we have a dual-polarised telescope, we calculate four 'baselines'
    for each canonical baseline as calculated above, namely :math:`h_1 h_2`,
    :math:`h_1 v_2`, :math:`v_1 h_2`, and :math:`v_1 v_2`. So the list of
    baselines appears four times as long as you might expect.

    Parameters
    ----------
    n_ants
        The number of antennas that will be correlated. Each antennas is
        expected to produce two polarisations.
    n_channels_per_substream
        The number of frequency channels to be processed.
    n_spectra_per_heap
        The number of time samples to be processed per frequency channel.
    input_sample_bits
        The number of bits per input sample. Only 8 bits is supported at the moment.
    """

    def __init__(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
        input_sample_bits: int,
    ) -> None:
        self.n_ants = n_ants
        self.n_channels_per_substream = n_channels_per_substream
        self.n_spectra_per_heap = n_spectra_per_heap
        self.n_baselines = self.n_ants * (self.n_ants + 1) // 2

        self.input_sample_bits = input_sample_bits  # hardcoded to 8 upstream
        self._n_ants_per_block = 32  # Hardcoded to 32 for now, but can be set to 32/48/64.

        # This 128 is hardcoded in the original Tensor-Core kernel. It loads
        # each block as two int4's, which is 256 bits (the extra factor of 2
        # is because input_sample_bits only counts the real part of a complex
        # number).
        self.n_times_per_block = 128 // self.input_sample_bits

        valid_bitwidths = [4, 8, 16]
        if self.input_sample_bits not in valid_bitwidths:
            raise ValueError(
                f"input_sample_bits must equal either 4, 8 or 16, currently equal to {self.input_sample_bits}."
            )
        elif self.input_sample_bits == 4 or self.input_sample_bits == 16:
            raise ValueError(
                f"Sample bitwidth of {self.input_sample_bits} "
                "will eventually be supported but has not yet been implemented."
            )

        if self.n_spectra_per_heap % self.n_times_per_block != 0:
            raise ValueError(f"spectra_per_heap must be divisible by {self.n_times_per_block}.")

        n_blocks_1d = accel.divup(self.n_ants, self._n_ants_per_block)
        if self._n_ants_per_block in {32, 48}:
            self.n_blocks = n_blocks_1d * (n_blocks_1d + 1) // 2
        elif self._n_ants_per_block == 64:
            self.n_blocks = n_blocks_1d * n_blocks_1d
        else:
            raise ValueError(
                f"ants_per_block must equal either 32, 48 or 64, currently equal to {self._n_ants_per_block}."
            )

        source = (importlib.resources.files(__package__) / "kernels" / "tensor_core_correlation_kernel.cu").read_text()
        program = context.compile(
            source,
            [
                f"-DNR_RECEIVERS={self.n_ants}",
                f"-DNR_RECEIVERS_PER_BLOCK={self._n_ants_per_block}",
                f"-DNR_BITS={self.input_sample_bits}",
                f"-DNR_CHANNELS={self.n_channels_per_substream}",
                f"-DNR_SAMPLES_PER_CHANNEL={self.n_spectra_per_heap}",
                f"-DNR_POLARIZATIONS={N_POLS}",
                "-DCUSTOM_STORE_VISIBILITY=1",
                # Suppress "pointless comparison of unsigned integer with zero"
                "-Xcudafe",
                "--diag_suppress=186",
            ],
        )
        self.correlate_kernel = program.get_kernel("correlate")
        self.reduce_kernel = program.get_kernel("reduce")

    def instantiate(self, command_queue: accel.AbstractCommandQueue, n_batches: int) -> "Correlation":
        """Create a :class:`Correlation` using this template to build the kernel."""
        return Correlation(self, command_queue, n_batches)


class Correlation(accel.Operation):
    """Tensor-Core correlation kernel.

    Specifies the shape of the input sample and output visibility buffers
    required by the kernel. The parameters specified in the
    :class:`CorrelationTemplate` object are used to determine the
    shape of the buffers. There is an outer-most dimension called "batches",
    over which the operation is parallelised. Not all batches need to be
    processed every time: set the ``first_batch`` and ``last_batch``
    attributes to control which batches will be computed.

    The input sample buffer must have the shape:
    ``[n_batches][n_ants][channels][spectra_per_heap][polarisations]``

    There is an alignment requirement for ``spectra_per_heap`` due to the
    implementation details of the kernel. For 8-bit input mode,
    ``spectra_per_heap`` must be a multiple of 16.

    Each input element is a complex 8-bit integer sample. :mod:`.numpy` does not
    support 8-bit complex numbers, so the dimensionality is extended by 1, with
    the last dimension sized ``2`` to represent the complexity.

    With 8-bit input samples, the value -128i is not supported by the kernel as
    there is no 8-bit complex conjugate representation of this number. Passing
    ``-128i`` into the kernel will produce incorrect values at the output.

    The output visibility buffer must have the shape
    ``[channels][baselines][COMPLEX]``. In 8-bit mode, each element in this
    visibility matrix is a 32-bit integer value.

    Calling this object does not directly update the output. Instead, it
    updates an intermediate buffer (called ``mid_visibilities``). To produce
    the output, call :meth:`reduce`. This function can also flag data that
    was missing during the accumulation, by writing a special value. This
    is controlled by the ``present_baselines`` slot, which has one boolean
    entry per baseline (antenna pair).

    Currently only 8-bit input mode is supported.
    """

    def __init__(
        self, template: CorrelationTemplate, command_queue: accel.AbstractCommandQueue, n_batches: int
    ) -> None:
        super().__init__(command_queue)
        self.template = template

        # Determine how many accumulators to use. Fewer is better for both
        # memory usage and I/O throughput, but too few means there will not
        # be enough parallelism to saturate the GPU. Aim for 1024-2048
        # work-groups, while sticking to powers of 2 since that's likely to
        # give an even division of work across them.
        n_mid = 1
        while n_mid * self.template.n_channels_per_substream * self.template.n_blocks < 1024:
            n_mid *= 2

        input_data_dimensions = (
            accel.Dimension(n_batches),
            accel.Dimension(self.template.n_ants, exact=True),
            accel.Dimension(self.template.n_channels_per_substream, exact=True),
            accel.Dimension(self.template.n_spectra_per_heap, exact=True),
            accel.Dimension(N_POLS, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )
        mid_data_dimensions = (
            accel.Dimension(n_mid),
            accel.Dimension(self.template.n_channels_per_substream, exact=True),
            accel.Dimension(self.template.n_baselines * N_POLS * N_POLS, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )

        # TODO: NGC-1104 update this once 4-bit correlation is supported
        assert (
            self.template.input_sample_bits == 8
        ), f"{self.template.input_sample_bits}-bit mode not supported yet, only 8-bit."
        self.slots["in_samples"] = accel.IOSlot(dimensions=input_data_dimensions, dtype=np.int8)
        self.slots["mid_visibilities"] = accel.IOSlot(dimensions=mid_data_dimensions, dtype=np.int64)
        self.slots["out_visibilities"] = accel.IOSlot(dimensions=mid_data_dimensions[1:], dtype=np.int32)
        self.slots["out_saturated"] = accel.IOSlot(dimensions=(), dtype=np.uint32)
        self.slots["present_baselines"] = accel.IOSlot(dimensions=(self.template.n_baselines,), dtype=np.uint8)
        if n_batches * self.template.n_channels_per_substream * self.template.n_baselines * N_POLS * N_POLS >= 2**31:
            # Can probably go higher, but rather keep it low to reduce the risk
            # of indexing bugs.
            raise ValueError("2^31 or more visibilities are not currently supported")
        self.first_batch = 0
        self.last_batch = n_batches
        self.n_batches = n_batches

    def _run(self) -> None:
        """Run the correlation kernel and add the generated values to internal buffer."""
        if not 0 <= self.first_batch < self.last_batch <= self.n_batches:
            raise ValueError("Invalid batch range")
        in_samples_buffer = self.buffer("in_samples")
        mid_visibilities_buffer = self.buffer("mid_visibilities")
        n_z = mid_visibilities_buffer.shape[0]

        n_batches = self.last_batch - self.first_batch  # Number of batches for this launch
        n_time_blocks_per_batch = self.template.n_spectra_per_heap // self.template.n_times_per_block
        n_time_blocks = n_batches * n_time_blocks_per_batch
        n_time_blocks_per_z = accel.divup(n_time_blocks, n_z)
        # The rounding up of n_time_blocks_per_z may leave some z values with
        # no work. So recompute n_z to avoid launching them at all.
        n_z = accel.divup(n_time_blocks, n_time_blocks_per_z)
        first_time_block = self.first_batch * n_time_blocks_per_batch

        self.command_queue.enqueue_kernel(
            self.template.correlate_kernel,
            [
                mid_visibilities_buffer.buffer,
                in_samples_buffer.buffer,
                np.uint32(first_time_block),
                np.uint32(n_time_blocks),
                np.uint32(n_time_blocks_per_z),
            ],
            # NOTE: Even though we are using CUDA, we follow OpenCL's grid/block
            # conventions. As such we need to multiply the number of
            # blocks(global_size) by the block size(local_size) in order to
            # specify global threads not global blocks.
            global_size=(32 * self.template.n_blocks, 2 * self.template.n_channels_per_substream, 2 * n_z),
            local_size=(32, 2, 2),
        )

    def reduce(self) -> None:
        """Finalise computation of the output visibilities from the internal buffer."""
        self.ensure_all_bound()
        mid_visibilities_buffer = self.buffer("mid_visibilities")
        out_visibilities_buffer = self.buffer("out_visibilities")
        out_saturated_buffer = self.buffer("out_saturated")
        present_baselines_buffer = self.buffer("present_baselines")
        wgs = 128  # TODO: could be tuned. But this kernel costs a tiny amount
        out_saturated_buffer.zero(self.command_queue)
        self.command_queue.enqueue_kernel(
            self.template.reduce_kernel,
            [
                out_visibilities_buffer.buffer,
                out_saturated_buffer.buffer,
                mid_visibilities_buffer.buffer,
                present_baselines_buffer.buffer,
                np.uint32(mid_visibilities_buffer.shape[0]),
            ],
            global_size=(accel.roundup(int(np.prod(out_visibilities_buffer.shape)), wgs), 1, 1),
            local_size=(wgs, 1, 1),
        )

    def zero_visibilities(self) -> None:
        """Zero all the values in the internal buffer."""
        self.ensure_bound("mid_visibilities")
        self.buffer("mid_visibilities").zero(self.command_queue)

    @staticmethod
    def get_baseline_index(ant1: int, ant2: int) -> int:
        r"""Get index in the visibilities matrix for baseline (ant1, ant2).

        The visibilities matrix indexing is as follows:

        .. code::

                  ant2 = 0  1  2  3  4
                      +---------------
             ant1 = 0 | 00 01 03 06 10
                    1 |    02 04 07 11
                    2 |       05 08 12
                    3 |          09 13
                    4 |             14

        This function requires that :math:`ant2 \ge ant1`
        """
        if ant1 > ant2:
            raise ValueError("It is required that ant2 >= ant1 in all cases")
        return ant2 * (ant2 + 1) // 2 + ant1
