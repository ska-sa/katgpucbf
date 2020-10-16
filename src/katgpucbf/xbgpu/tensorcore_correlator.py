"""
Module wrapping the ASTRON Tensor Core Correlation Kernels in the MeerKAT katsdpsig proc framework.

This module has two classes:
    1. TensorCoreCorrelatorTemplate - This class allows for multiple different compilations of the same kernel with
    parameters to take place.
    2. TensorCoreCorrelator - This class provides the interface to call the kernel created in a
    TensorCoreCorrelatorTemplate object.

TODO:
    1. Modify the kernel so that the visibility matrix is not zeroed after every kernel call. This will allow for much
    longer integration times to occur without consuming all GPU memory.
    2. Fix the floating close brace in the tensor_core_correlation_kernel.cu file. The reasons for it are described in
    that file.
    3. Eventually modify the classes to support 4 and 16 bit input samples. The kernel supports this, but it is not
    exposed to the reader. There is no use case for this at the moment, so this is a low priority.
"""

import pkg_resources
import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import cuda


class TensorCoreCorrelatorTemplate:
    """
    Template class for compiling different variations of the Tensor core correlation kernel.

    This object will be used to create a TensorCoreCorrelator object that will be able to run the created kernel.
    """

    def __init__(
        self, context: cuda.Context, n_ants: int, n_channels: int, n_samples_per_channel: int
    ) -> None:
        """
        Initialise the TensorCoreCorrelatorTemplate class and compile the Tensor core correlation kernel.

        The parameters given to this function are used by this class to compile the kernel and by the
        TensorCoreCorrelator to specify the shape of the memory buffers connected to this kernel.

        Parameters
        ----------
        n_ants: int
            The number of antennas that will be correlated. Each antennas is expected to produce two polarisations.
        n_channels: int
            The number of frequency channels to be processed.
        n_samples_per_channel: int
            The number of time samples to be processed per frequency channel.
        """
        # 1. Set accesible member functions that are used to calculate indices to the input and output buffers.
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.n_polarizastions = 2  # Hardcoded to 2. No other values are supported
        self.n_baselines = self.n_ants * (self.n_ants + 1) // 2

        # 2. Determine kernel specific parameters
        self._sample_bitwidth = 8  # hardcoded to 8 for now, but 4 and 16 bits are also supported
        self._n_ants_per_block = 64  # Hardcoded to 64 for now, but can be set to 48 in the future
        
        # This 128 is hardcoded in the original tensor core kernel. The reason it is set to this needs to be determined.
        self._n_times_per_block = 128 // self._sample_bitwidth 

        valid_bitwidths = [4, 8, 16]
        if self._sample_bitwidth not in valid_bitwidths:
            raise ValueError(
                f"Sample_bitwidth must equal either 4, 8 or 16, currently equal to {self._sample_bitwidth}."
            )
        if self._sample_bitwidth == 4 or self._sample_bitwidth == 16:
            raise ValueError(
                f"Sample bitwidth of {self._sample_bitwidth} will eventually be supported but has not yet been implemented."
            )

        if self.n_samples_per_channel % self._n_times_per_block != 0:
            raise ValueError(f"samples_per_channel must be divisible by {self._n_times_per_block}.")

        # 3. Calculate the input and output data shape.
        self.inputDataShape = (
            self.n_channels,
            self.n_samples_per_channel // self._n_times_per_block,
            self.n_ants,
            self.n_polarizastions,
            self._n_times_per_block,
        )
        self.outputDataShape = (self.n_channels, self.n_baselines, self.n_polarizastions, self.n_polarizastions)

        # 4. Calculate the number of thread blocks to launch per kernel call - this remains constant for the lifetime
        # of the object.
        if self._n_ants_per_block == 48:
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

        # 5. Compile the kernel
        program = accel.build(
            context,
            "kernels/tensor_core_correlation_kernel.cu",
            {
                "n_ants_per_block": self._n_ants_per_block,
                "n_ants": self.n_ants,
                "sample_bitwidth": self._sample_bitwidth,
                "n_channels": self.n_channels,
                "n_polarizastions": self.n_polarizastions,
                "n_samples_per_channel": self.n_samples_per_channel,
                "n_baselines": self.n_baselines,
                "n_times_per_block": self._n_times_per_block,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("correlate")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "TensorCoreCorrelator":
        """Create a TensorCoreCorrelator class using this template to build the kernel."""
        return TensorCoreCorrelator(self, command_queue)


class TensorCoreCorrelator(accel.Operation):
    """
    Class containing a Tensor core kernel compiled from a TensorCoreCorrelatorTemplate.

    This class specifies the shape of the input sample and output visibility buffers required by the kernel. The
    parameters specified in the TensorCoreCorrelatorTemplate object are used to determine the shape of the buffers.

    The input sample buffer must have the shape:
    [channels][samples_per_channel//times_per_block][n_ants][polarizations][times_per_block]
    In 8-bit input mode times_per_block is equal to 16. Each element is an complex 8-bit integer sample. Numpy does
    not support 8-bit complex numbers, so the input sample array has dtype of np.int16 as a placeholder.
    With 8-bit input samples, the value -128i is not supported by the kernel as there is no 8-bit complex conjugate
    representation of this number. Passing -128i into the kernel will produce incorrect values at the output.

    The output visibility buffer must have the shape: [channels][baselines][polarisations][polarisations]
    In 8-bit mode, each element in this visibility matrix is a 32-bit integer complex value.

    Currently only 8-bit input sample mode is supported.
    """

    def __init__(self, template: TensorCoreCorrelatorTemplate, command_queue: accel.AbstractCommandQueue) -> None:
        """Initialise the TensorCoreCorrelator object and specify the size of the memory buffers."""
        super().__init__(command_queue)
        self.template = template
        self.slots["inSamples"] = accel.IOSlot(
            dimensions=self.template.inputDataShape, dtype=np.int16
        )  # TODO: This must depend on input bitwidth
        self.slots["outVisibilities"] = accel.IOSlot(dimensions=self.template.outputDataShape, dtype=np.int64)

    def _run(self) -> None:
        """Run the correlation kernel."""
        inSamples_buffer = self.buffer("inSamples")
        outVisibilities_buffer = self.buffer("outVisibilities")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [outVisibilities_buffer.buffer, inSamples_buffer.buffer],
            # Even though we are using CUDA, we follow OpenCLs grid/block conventions. As such we need to multiply the number
            # of blocks(global_size) by the block size(local_size) in order to specify global threads not global blocks.
            global_size=(32 * self.template.n_blocks, 2 * self.template.n_channels, 2 * 1),
            local_size=(32, 2, 2),
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
