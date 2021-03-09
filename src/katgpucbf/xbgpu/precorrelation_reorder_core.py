"""
Module wrapping the Pre-correlation Reorder Kernel in the MeerKAT katsdpsigproc framework.

This module has two classes:
    1. PreCorrelationReorderCoreTemplate
        - This class allows for multiple different compilations of the same kernel with parameters to take place.
    2. PreCorrelationReorderCore
        - This class provides the interface to call the kernel created in a PreCorrelationReorderTemplate object.

TODO:
    1. Actually test this.
    2. Update naming conventions as necessary.

"""

import pkg_resources
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import cuda


class PreCorrelationReorderCoreTemplate:
    """
    Template class for compiling different variations of the pre-correlation reorder kernel.

    This object will be used to create a PreCorrelationReorderCore object that will be able to run the created kernel.
    """

    def __init__(self, context: cuda.Context, n_ants: int, n_channels: int,
                    n_samples_per_channel: int, n_batches: int) -> None:
        """
        Initialise the PreCorrelationReorderCoreTemplate class and compile the pre-correlation reorder kernel.

        The parameters given to this function are used by this class to compile the kernel and by the
        PreCorrelationReorderCore to specify the shape of the memory buffers connected to this kernel.

        Parameters
        ----------
        n_ants: int
            The number of antennas that will be correlated. Each antennas is expected to produce two polarisations.
        n_channels: int
            The number of frequency channels to be processed.
        n_samples_per_channel: int
            The number of time samples to be processed per frequency channel.
        n_batches: int
            The number of reorders to complete.
        """
        # 1. Set accesible member functions that are used to calculate indices to the input and output buffers.
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.n_polarizations = 2  # Hardcoded to 2. No other values are supported
        self.n_batches = n_batches
        
        # 2. Determine kernel specific parameters
        self._sample_bitwidth = 8  # hardcoded to 8 for now, but 4 and 16 bits are also supported
        self._n_ants_per_block = 64  # Hardcoded to 64 for now, but can be set to 48 in the future

        # This 128 is hardcoded in the original Tensor Core kernel. The reason it is set to this needs to be determined.
        self.n_times_per_block = 128 // self._sample_bitwidth

        valid_bitwidths = [4, 8, 16]
        if self._sample_bitwidth not in valid_bitwidths:
            raise ValueError(
                f"Sample_bitwidth must equal either 4, 8 or 16, currently equal to {self._sample_bitwidth}."
            )
        elif self._sample_bitwidth == 4 or self._sample_bitwidth == 16:
            raise ValueError(
                f"Sample bitwidth of {self._sample_bitwidth} will eventually be supported but has not yet been implemented."
            )

        if self.n_samples_per_channel % self.n_times_per_block != 0:
            raise ValueError(f"samples_per_channel must be divisible by {self.n_times_per_block}.")

        # 3. Calculate the input and output data shape.
        self.inputDataShape = (self.n_ants, self.n_channels, self.n_samples_per_channel, self.n_polarizations)

        self.outputDataShape = (
            self.n_channels,
            self.n_samples_per_channel // self.n_times_per_block,
            self.n_ants,
            self.n_polarizations,
            self.n_times_per_block,
        )

        # Matrix size is the same for the Input and Output data shapes
        self.matrix_size = self.n_ants * self.n_channels * self.n_samples_per_channel * self.n_polarizastions
        # Seeing as we can't really define constants in Python
        THREADS_PER_BLOCK = 1024
        
        # 4. Calculate the number of thread blocks to launch per kernel call 
        # - This remains constant for the lifetime of the object.
        # - Unlike the Tensor Core Correlation kernel, the size of this kernel does not depend on the number of ants_per_block
        #   Rather, it simply depends on the individual matrix size and the number of batches.
        # - But also, how should I error-check this value? (As in, bounds/values, not method)
        self.n_blocks = (self.matrix_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

        # 5. Compile the kernel
        program = accel.build(
            context,
            "kernels/precorrelation_reorder_kernel.cu",
            {
                "n_ants": self.n_ants,
                "n_channels": self.n_channels,
                "n_samples_per_channel": self.n_samples_per_channel,
                "n_polarizastions": self.n_polarizations,
                "n_times_per_block": self.n_times_per_block,
                "n_batches": self.n_batches,
                # "sample_bitwidth": self._sample_bitwidth,
                # "n_ants_per_block": self._n_ants_per_block,
                # "n_baselines": self.n_baselines,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("reorder_naive")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "PreCorrelationReorderCore":
        """Create a PreCorrelationReorderCore object using this template to build the kernel."""
        return PreCorrelationReorderCore(self, command_queue)


class PreCorrelationReorderCore(accel.Operation):
    """
    Class containing a pre-correlation reorder kernel compiled from a PreCorrelationReorderCoreTemplate.

    This class specifies the shape of the input sample and output reordered buffers required by the kernel. The
    parameters specified in the PreCorrelationReorderCoreTemplate object are used to determine the shape of the buffers.

    The input sample buffer must have the shape:
    [antennas][channels][samples_per_channel][polarisations]

    The output sample buffer must have the shape:
    [channels][samples_per_channel//times_per_block][n_ants][polarizations][times_per_block]

    A complexity that is introduced by the pre-correlation reorder kernel is that the samples_per_channel index is split over two
    different indices. The first index ranges from 0 to samples_per_channel//times_per_block and the second index
    ranges from 0 to times_per_block. Times per block is calculated by the PreCorrelationReorderCoreTemplate object.
    In 8-bit input mode times_per_block is equal to 16.

    Each input element is n complex 8-bit integer sample. Numpy does not support 8-bit complex numbers, 
    so the input sample array has dtype of np.int16 as a placeholder.
    """

    def __init__(self, template: PreCorrelationReorderCoreTemplate, command_queue: accel.AbstractCommandQueue) -> None:
        """Initialise the PreCorrelationReorderCore object and specify the size of the memory buffers."""
        super().__init__(command_queue)
        self.template = template
        self.slots["inSamples"] = accel.IOSlot(
            dimensions=self.template.inputDataShape, dtype=np.int16
        )  # TODO: This must depend on input bitwidth
        self.slots["outReordered"] = accel.IOSlot(dimensions=self.template.outputDataShape, dtype=np.int64)

    def _run(self) -> None:
        """Run the correlation kernel."""
        inSamples_buffer = self.buffer("inSamples")
        outReordered_buffer = self.buffer("outReordered")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [outReordered_buffer.buffer, inSamples_buffer.buffer],
            # Even though we are using CUDA, we follow OpenCLs grid/block conventions. As such we need to multiply the number
            # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
            global_size=(32 * self.template.n_blocks, self.template.n_batches, 1),
            local_size=(32, 32, 1),
        )

    @staticmethod
    def get_baseline_index(ant1, ant2):
        """
        Return the index in the visibilities matrix of the visibility produced by ant1 and ant2.

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
