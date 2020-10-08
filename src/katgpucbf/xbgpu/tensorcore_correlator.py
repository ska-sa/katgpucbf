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


class TensorCoreCorrelatorTemplate:
    """
    Template class for compiling different variations of the Tensor core correlation kernel.

    This object will be used to create a TensorCoreCorrelator object that will be able to run the created kernel.
    """

    def __init__(
        self, context: accel.AbstractContext, dual_pol_ants: int, channels: int, samples_per_channel: int
    ) -> None:
        """
        Initialise the TensorCoreCorrelatorTemplate class and compile the Tensor core correlation kernel.

        The parameters given to this function are used by this class to compile the kernel and by the
        TensorCoreCorrelator to specify the shape of the memory buffers connected to this kernel.

        Parameters
        ----------
        dual_pol_ants: int
            The number of antennas that will be correlated. Each antennas is expected to produce two polarisations.
        channels: int
            The number of frequency channels to be processed.
        samples_per_channel: int
            The number of time samples to be processed per frequency channel.
        """
        # 1. Set accesible member functions that are used to calculate indices to the input and output buffers.
        self.dual_pol_ants = dual_pol_ants
        self.channels = channels
        self.samples_per_channel = samples_per_channel
        self.polarizastions = 2  # Hardcoded to 2. No other values are supported
        self.baselines = int(self.dual_pol_ants * (self.dual_pol_ants + 1) // 2)

        # 2. Determine kernel specific parameters
        self._sample_bitwidth = 8  # hardcoded to 8 for now, but 4 and 16 bits are also supported
        self._ants_per_block = 64  # Hardcoded to 64 for now, but can be set to 48 in the future
        self._times_per_block = 128 // self._sample_bitwidth

        if self._sample_bitwidth != 4 and self._sample_bitwidth != 8 and self._sample_bitwidth != 16:
            raise ValueError(
                "sample_bitwidth must equal either 4, 8 or 16, currently equal to {0}.".format(self._sample_bitwidth)
            )

        if self.samples_per_channel % self._times_per_block != 0:
            raise ValueError("samples_per_channel must be divisible by {0}.".format(self._times_per_block))

        # 3. Calculate the input and output data shape.
        self.inputDataShape = (
            self.channels,
            self.samples_per_channel // self._times_per_block,
            self.dual_pol_ants,
            self.polarizastions,
            self._times_per_block,
        )
        self.outputDataShape = (self.channels, self.baselines, self.polarizastions, self.polarizastions)

        # 4. Calculate the number of blocks - this remains constant for the lifetime of the object
        if self._ants_per_block == 48:
            self.nrBlocks = int(
                ((self.dual_pol_ants + self._ants_per_block - 1) // self._ants_per_block)
                * ((self.dual_pol_ants + self._ants_per_block - 1) // self._ants_per_block + 1)
                // 2
            )
        elif self._ants_per_block == 64:
            self.nrBlocks = int(
                ((self.dual_pol_ants + self._ants_per_block - 1) // self._ants_per_block)
                * ((self.dual_pol_ants + self._ants_per_block - 1) // self._ants_per_block)
            )
        else:
            raise ValueError(
                "ants_per_block must equal either 64 or 48, currently equal to {0}.".format(self._ants_per_block)
            )

        # 5. Compile the kernel
        program = accel.build(
            context,
            "kernels/tensor_core_correlation_kernel.cu",
            {
                "ants_per_block": self._ants_per_block,
                "dual_pol_ants": self.dual_pol_ants,
                "sample_bitwidth": self._sample_bitwidth,
                "channels": self.channels,
                "polarizastions": self.polarizastions,
                "samples_per_channel": self.samples_per_channel,
                "baselines": self.baselines,
                "times_per_block": self._times_per_block,
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
    [channels][samples_per_channel//times_per_block][dual_pol_ants][polarizations][times_per_block]
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
            global_size=(32 * self.template.nrBlocks, 2 * self.template.channels, 2 * 1),
            local_size=(32, 2, 2),
        )

    @staticmethod
    def calculate_baseline_index(ant1, ant2):
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


dual_pol_ants = 64
channels = 8
samples_per_channel = 256

ctx = accel.create_some_context(device_filter=lambda x: x.is_cuda)
queue = ctx.create_command_queue()

template = TensorCoreCorrelatorTemplate(
    ctx, dual_pol_ants=dual_pol_ants, channels=channels, samples_per_channel=samples_per_channel
)
tensorCoreCorrelator = template.instantiate(queue)
tensorCoreCorrelator.ensure_all_bound()

bufSamples_device = tensorCoreCorrelator.buffer("inSamples")
bufSamples_host = bufSamples_device.empty_like()

bufVisibilities_device = tensorCoreCorrelator.buffer("outVisibilities")
bufVisibilities_host = bufVisibilities_device.empty_like()

bufSamples_device.set(queue, bufSamples_host)
tensorCoreCorrelator()
bufSamples_device.get(queue, bufVisibilities_host)

# 8. Check that the GPU visibilities are correct.

# 8.1 Need to convert the input and output arrays to 32-bit float complex type as numpy has no 8-bit or 32-bit int
# complex type.
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host.dtype = np.int8
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host = bufSamples_host.astype(np.float32)
print(bufSamples_host.shape, bufSamples_host.nbytes)
bufSamples_host.dtype = np.csingle
print(bufSamples_host.shape, bufSamples_host.nbytes)

print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host.dtype = np.int32
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host = bufVisibilities_host.astype(np.float32)
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)
bufVisibilities_host.dtype = np.csingle
print(bufVisibilities_host.shape, bufVisibilities_host.nbytes)

bufCorrectVisibilities_host = np.empty_like(bufVisibilities_host)
print(bufCorrectVisibilities_host.shape, bufCorrectVisibilities_host.nbytes)

# 8.2 Generate the visibilities on the CPU - this is not a simple matrix operation due to the indexing of the input
# samples, I am just going to do a naive brute force for now to be optomised later if needs be.


def generate_visibilities_host(bufSamples_host, channel_index, ant1, ant2):
    """
    Host side code to generate visibility matrix.

    Will generate visibilities for a particular channel and antenna combination from an array of input samples that has
    the shape (channels, samples_per_channel // times_per_block, dual_pol_ants, polarizastions, times_per_block)
    required by the Tensor core correlation kernels.

    Returns visibilities hh, hv, vh and vv representing different interpolarisation products.
    """
    hh, hv, vh, vv = 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j
    for time_index1 in range(0, samples_per_channel // template._times_per_block):
        for time_index2 in range(0, template._times_per_block):
            ant1h = bufSamples_host[channel_index][time_index1][ant1][0][time_index2]
            ant1v = bufSamples_host[channel_index][time_index1][ant1][1][time_index2]
            ant2h = bufSamples_host[channel_index][time_index1][ant2][0][time_index2]
            ant2v = bufSamples_host[channel_index][time_index1][ant2][1][time_index2]

            hh += ant1h * np.conj(ant2h)
            vh += ant1h * np.conj(ant2v)
            hv += ant1v * np.conj(ant2h)
            vv += ant1v * np.conj(ant2v)

    return hh, hv, vh, vv


for channel_index in range(0, channels):
    print(channel_index)
    for ant1_index in range(0, dual_pol_ants):
        for ant2_index in range(0, ant1_index + 1):
            hh, hv, vh, vv = generate_visibilities_host(bufSamples_host, channel_index, ant1_index, ant2_index)
            baseline_index = TensorCoreCorrelator.calculate_baseline_index(ant1_index, ant2_index)
            bufCorrectVisibilities_host[channel_index][baseline_index][0][0] = hh
            bufCorrectVisibilities_host[channel_index][baseline_index][1][0] = hv
            bufCorrectVisibilities_host[channel_index][baseline_index][0][1] = vh
            bufCorrectVisibilities_host[channel_index][baseline_index][1][1] = vv

            # if(bufVisibilities_host[channel_index][baseline_index][0][0] != hh or
            #    bufVisibilities_host[channel_index][baseline_index][1][0] != hv or
            #    bufVisibilities_host[channel_index][baseline_index][0][1] != vh or
            #    bufVisibilities_host[channel_index][baseline_index][1][1] != vv):
            #    print("What",channel_index,ant1_index,ant2_index)
            #    print(hh, bufVisibilities_host[channel_index][baseline_index][0][0])
            #    print(hv, bufVisibilities_host[channel_index][baseline_index][1][0])
            #    print(vh, bufVisibilities_host[channel_index][baseline_index][0][1])
            #    print(vv, bufVisibilities_host[channel_index][baseline_index][1][1])
            #    temp += 1

# print(temp,baselines,channels)

# 8.3 Check that the CPU version and GPU version are identical
np.testing.assert_array_equal(bufCorrectVisibilities_host, bufVisibilities_host)
