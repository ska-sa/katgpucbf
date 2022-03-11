################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

    - Fix the floating close brace in ``tensor_core_correlation_kernel.cu``. The
      reasons for it are described in that file.
    - Eventually modify the classes to support 4 and 16 bit input samples. The
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
    n_channels
        The number of frequency channels to be processed.
    n_spectra_per_heap
        The number of time samples to be processed per frequency channel.
    """

    def __init__(self, context: AbstractContext, n_ants: int, n_channels: int, n_spectra_per_heap: int) -> None:
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_spectra_per_heap = n_spectra_per_heap
        self.n_baselines = self.n_ants * (self.n_ants + 1) // 2

        self._sample_bitwidth = 8  # hardcoded to 8 for now, but 4 and 16 bits are also supported
        self._n_ants_per_block = 64  # Hardcoded to 64 for now, but can be set to 48. 32 is not supported yet.

        # This 128 is hardcoded in the original Tensor-Core kernel. The reason
        # it is set to this needs to be determined.
        n_times_per_block = 128 // self._sample_bitwidth

        valid_bitwidths = [4, 8, 16]
        if self._sample_bitwidth not in valid_bitwidths:
            raise ValueError(
                f"Sample_bitwidth must equal either 4, 8 or 16, currently equal to {self._sample_bitwidth}."
            )
        elif self._sample_bitwidth == 4 or self._sample_bitwidth == 16:
            raise ValueError(
                f"Sample bitwidth of {self._sample_bitwidth} "
                "will eventually be supported but has not yet been implemented."
            )

        if self.n_spectra_per_heap % n_times_per_block != 0:
            raise ValueError(f"spectra_per_heap must be divisible by {n_times_per_block}.")

        self.input_data_dimensions = (
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_spectra_per_heap, exact=True),
            accel.Dimension(N_POLS, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )

        self.output_data_dimensions = (
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_baselines * 4, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )

        if self._n_ants_per_block == 32:
            raise NotImplementedError(
                "32 antennas per thread-block is not supported yet - \
                Need to clarify the formula for thread-block calculation."
            )
        elif self._n_ants_per_block == 48:
            self.n_blocks = int(
                ((self.n_ants + self._n_ants_per_block - 1) // self._n_ants_per_block)
                * ((self.n_ants + self._n_ants_per_block - 1) // self._n_ants_per_block + 1)
                // 2
            )
        elif self._n_ants_per_block == 64:
            self.n_blocks = int(
                ((self.n_ants + self._n_ants_per_block - 1) // self._n_ants_per_block)
                * ((self.n_ants + self._n_ants_per_block - 1) // self._n_ants_per_block)
            )
        else:
            raise ValueError(
                "ants_per_block must equal either 64 or 48, currently equal to {0}.".format(self._n_ants_per_block)
            )

        with importlib.resources.path("katgpucbf.xbgpu", "kernels") as kernels:
            source = (kernels / "tensor_core_correlation_kernel.cu").read_text()
        program = context.compile(
            source,
            [
                f"-DNR_RECEIVERS={self.n_ants}",
                f"-DNR_RECEIVERS_PER_BLOCK={self._n_ants_per_block}",
                f"-DNR_BITS={self._sample_bitwidth}",
                f"-DNR_CHANNELS={self.n_channels}",
                f"-DNR_SAMPLES_PER_CHANNEL={self.n_spectra_per_heap}",
                f"-DNR_POLARIZATIONS={N_POLS}",
            ],
        )
        self.kernel = program.get_kernel("correlate")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "Correlation":
        """Create a :class:`Correlation` using this template to build the kernel."""
        return Correlation(self, command_queue)


class Correlation(accel.Operation):
    """Tensor-Core correlation kernel.

    Specifies the shape of the input sample and output visibility buffers
    required by the kernel. The parameters specified in the
    :class:`CorrelationTemplate` object are used to determine the
    shape of the buffers.

    The input sample buffer must have the shape:
    ``[n_ants][channels][spectra_per_heap][polarisations]``

    A complexity that is introduced by the Tensor-Core kernel is that the
    ``spectra_per_heap`` index is split over two different indices. The first
    index ranges from ``0`` to ``spectra_per_heap//times_per_block`` and the
    second index ranges from ``0`` to ``times_per_block``. Times per block is
    calculated by the :class:`CorrelationTemplate`. In 8-bit input mode,
    ``times_per_block`` is equal to 16.

    Each input element is a complex 8-bit integer sample. :mod:`.numpy` does not
    support 8-bit complex numbers, so the dimensionality is extended by 1, with
    the last dimension sized ``2`` to represent the complexity.

    With 8-bit input samples, the value -128i is not supported by the kernel as
    there is no 8-bit complex conjugate representation of this number. Passing
    ``-128i`` into the kernel will produce incorrect values at the output.

    The output visibility buffer must have the shape
    ``[channels][baselines][CPLX]``. In 8-bit mode, each element in this
    visibility matrix is a 32-bit integer value.

    Currently only 8-bit input mode is supported.
    """

    def __init__(self, template: CorrelationTemplate, command_queue: accel.AbstractCommandQueue) -> None:
        super().__init__(command_queue)
        self.template = template
        self.slots["in_samples"] = accel.IOSlot(
            dimensions=self.template.input_data_dimensions, dtype=np.int8
        )  # TODO: This must depend on input bitwidth
        self.slots["out_visibilities"] = accel.IOSlot(dimensions=self.template.output_data_dimensions, dtype=np.int32)

    def _run(self) -> None:
        """Run the correlation kernel and add the generated values to the out_visibilities buffer."""
        in_samples_buffer = self.buffer("in_samples")
        out_visibilities_buffer = self.buffer("out_visibilities")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [out_visibilities_buffer.buffer, in_samples_buffer.buffer],
            # NOTE: Even though we are using CUDA, we follow OpenCL's grid/block
            # conventions. As such we need to multiply the number of
            # blocks(global_size) by the block size(local_size) in order to
            # specify global threads not global blocks.
            global_size=(32 * self.template.n_blocks, 2 * self.template.n_channels, 2 * 1),
            local_size=(32, 2, 2),
        )

    def zero_visibilities(self) -> None:
        """Zero all the values in the out_visibilities buffer."""
        self.buffer("out_visibilities").zero(self.command_queue)

    @staticmethod
    def get_baseline_index(ant1, ant2) -> int:
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
