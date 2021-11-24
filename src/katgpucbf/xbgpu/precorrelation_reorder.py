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

"""
Module wrapping the pre-correlation reorder Kernel.

The pre-correlation reorder kernel operates on a set of data with dimensions
explained below (and in its _kernel.mako file).  It makes provision for batched
operations, i.e. reordering multiple sets of data (matrices) passed to the
kernel in a single array.

This module has two classes:
    1. PreCorrelationReorderTemplate
        - This class allows for multiple different compilations of the same
          kernel with different parameters to take place.
    2. PreCorrelationReorder
        - This class provides the interface to call the kernel created in a
          PreCorrelationReorderTemplate object.

TODO:
    1. Update naming conventions as necessary.

"""

from typing import Final

import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext

from .. import COMPLEX, N_POLS


class PrecorrelationReorderTemplate:
    """
    Template class to compile the pre-correlation reorder kernel.

    This template creates a :class:`PrecorrelationReorder` that will
    run the compiled kernel. The parameters are used to compile the
    kernel and by the :class:`PrecorrelationReorder` to specify the
    shape of the memory buffers connected to this kernel.

    Parameters
    ----------
    context
        The GPU device's context provided by katsdpsigproc's abstraction of
        PyCUDA.  A context is associated with a single device and 'owns'
        all memory allocations.  For the purposes of this python module,
        and its Tensor Core usage, the CUDA context is required.
    n_ants
        The number of antennas that will be correlated. Each antennas is
        expected to produce two polarisations.
    n_channels
        The number of frequency channels to be processed.
    n_spectra_per_heap
        The number of time samples to be processed per frequency channel.
    n_batches
        The number of matrices to be reordered, a single data matrix = one batch.
    """

    def __init__(
        self, context: AbstractContext, n_ants: int, n_channels: int, n_spectra_per_heap: int, n_batches: int
    ) -> None:
        # 1. Set member variables that are used to calculate indices for the
        # input and output buffers
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_spectra_per_heap = n_spectra_per_heap
        self.n_batches = n_batches

        # This is set to 8 for now, but must be updated to 4- and 16-bit
        # as and when the TensorCoreXEngine requires it.
        self._sample_bitwidth = 8

        # This 128 is hardcoded in the original Tensor Core kernel and
        # (probably) has to do with optimising the thread utilisation in Tensor
        # Cores - 128 = 4 x warps, where one warp = 32 threads.
        self.n_times_per_block = 128 // self._sample_bitwidth

        if self.n_spectra_per_heap % self.n_times_per_block != 0:
            raise ValueError(f"spectra_per_heap must be divisible by {self.n_times_per_block}.")

        # 3. Declare the input and output data shapes
        self.input_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_spectra_per_heap, exact=True),
            accel.Dimension(N_POLS, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )

        self.output_data_dimensions = (
            accel.Dimension(self.n_batches, exact=True),
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_spectra_per_heap // self.n_times_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(N_POLS, exact=True),
            accel.Dimension(self.n_times_per_block, exact=True),
            accel.Dimension(COMPLEX, exact=True),
        )

        # The size of a data matrix required to be reordered is the same for
        # Input or Output data shapes
        self.matrix_size = self.n_ants * self.n_channels * self.n_spectra_per_heap * N_POLS
        # Maximum number of threads per block, as per Section I of Nvidia's CUDA Programming Guide
        THREADS_PER_BLOCK: Final[int] = 1024  # noqa: N806

        # 4. Calculate the number of thread blocks to launch per kernel call
        # - This is in the x-dimension and remains constant for the lifetime of
        #   the object.
        # - TODO: Error-check these values (As in, bounds/values, not method).
        self.n_blocks_x = (self.matrix_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        # 5. Compile the kernel
        #   - The size of this kernel simply depends on the individual matrix size and the
        #     number of batches required to be reordered.
        program = accel.build(
            context,
            "kernels/precorrelation_reorder_kernel.mako",
            {
                "n_ants": self.n_ants,
                "n_channels": self.n_channels,
                "n_spectra_per_heap": self.n_spectra_per_heap,
                "n_polarisations": N_POLS,
                "n_times_per_block": self.n_times_per_block,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("precorrelation_reorder")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "PrecorrelationReorder":
        """Create a PreCorrelationReorder object using this template to build the kernel."""
        return PrecorrelationReorder(self, command_queue)


class PrecorrelationReorder(accel.Operation):
    """
    Class containing a pre-correlation reorder kernel compiled from a PreCorrelationReorderTemplate.

    This class specifies the shape of the input sample and output reordered
    buffers required by the kernel. The parameters specified in the
    PreCorrelationReorderTemplate object are used to determine the shape of the
    buffers.

    It is worth noting these matrices follow the C convention, with the
    fastest-changing dimension being the last on the list. The input sample
    buffer must have the shape:
    [batch][antennas][channels][spectra_per_heap][polarisations]

    The output sample buffer must have the shape:
    [batch][channels][spectra_per_heap//times_per_block][n_ants][polarisations][times_per_block]

    A complexity that is introduced by the pre-correlation reorder kernel is
    that the spectra_per_heap index is split over two different indices. The
    first index ranges from 0 to spectra_per_heap//times_per_block and the
    second index ranges from 0 to times_per_block. Times per block is
    calculated by the PreCorrelationReorderTemplate object.  In 8-bit input
    mode times_per_block is equal to 16.

    Each input element is a complex 8-bit integer sample. Numpy does not
    support 8-bit complex numbers, so the input sample array has dtype of
    np.int16 as a placeholder.
    """

    def __init__(self, template: PrecorrelationReorderTemplate, command_queue: accel.AbstractCommandQueue) -> None:
        """Initialise the PreCorrelationReorder object and specify the size of the memory buffers."""
        super().__init__(command_queue)
        self.template = template
        self.slots["in_samples"] = accel.IOSlot(
            dimensions=self.template.input_data_dimensions, dtype=np.int8
        )  # TODO: This must depend on input bitwidth
        self.slots["out_reordered"] = accel.IOSlot(dimensions=self.template.output_data_dimensions, dtype=np.int8)

    def _run(self) -> None:
        """Run the correlation kernel."""
        in_samples_buffer = self.buffer("in_samples")
        out_reordered_buffer = self.buffer("out_reordered")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [in_samples_buffer.buffer, out_reordered_buffer.buffer],
            # Even though we are using CUDA, we follow OpenCLs grid/block
            # conventions. As such we need to multiply the number of
            # blocks(global_size) by the block size(local_size) in order to
            # specify global threads not global blocks.
            # - Global size is across the x- and y-dimensions (for this
            #   application).
            global_size=(1024 * self.template.n_blocks_x, self.template.n_batches),
            local_size=(1024, 1),
        )
