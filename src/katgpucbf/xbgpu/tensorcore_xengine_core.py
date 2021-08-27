"""Module wrapping the ASTRON Tensor-Core Correlation Kernels in the MeerKAT katsdpsigproc framework.

.. todo::

    - Fix the floating close brace in ``tensor_core_correlation_kernel.cu``. The
      reasons for it are described in that file.
    - Eventually modify the classes to support 4 and 16 bit input samples. The
      kernel supports this, but it is not exposed to the reader. There is no use
      case for this at the moment, so this is a low priority.

"""

import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext

complexity = 2


class TensorCoreXEngineCoreTemplate:
    r"""Template class for the Tensor-Core correlation kernel.

    The template creates a :class:`TensorCoreXEngineCore` that will run the
    compiled kernel. The parameters are used to compile the kernel and by the
    :class:`TensorCoreXEngineCore` to specify the shape of the memory buffers
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
    n_samples_per_channel
        The number of time samples to be processed per frequency channel.
    """

    def __init__(self, context: AbstractContext, n_ants: int, n_channels: int, n_samples_per_channel: int) -> None:
        # 1. Set accesible member functions that are used to calculate indices to the input and output buffers.
        self.n_ants = n_ants
        self.n_channels = n_channels
        self.n_samples_per_channel = n_samples_per_channel
        self.n_polarisations = 2  # Hardcoded to 2. No other values are supported
        self.n_baselines = self.n_ants * (self.n_ants + 1) // 2

        # 2. Determine kernel specific parameters
        self._sample_bitwidth = 8  # hardcoded to 8 for now, but 4 and 16 bits are also supported
        self._n_ants_per_block = 64  # Hardcoded to 64 for now, but can be set to 48. 32 is not supported yet.

        # This 128 is hardcoded in the original Tensor-Core kernel. The reason it is set to this needs to be determined.
        self.n_times_per_block = 128 // self._sample_bitwidth

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

        if self.n_samples_per_channel % self.n_times_per_block != 0:
            raise ValueError(f"samples_per_channel must be divisible by {self.n_times_per_block}.")

        # 3. Calculate the input and output data shape.
        self.input_data_dimensions = (
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_samples_per_channel // self.n_times_per_block, exact=True),
            accel.Dimension(self.n_ants, exact=True),
            accel.Dimension(self.n_polarisations, exact=True),
            accel.Dimension(self.n_times_per_block, exact=True),
            accel.Dimension(complexity, exact=True),
        )
        self.output_data_dimensions = (
            accel.Dimension(self.n_channels, exact=True),
            accel.Dimension(self.n_baselines * 4, exact=True),
            accel.Dimension(complexity, exact=True),
        )

        # 4. Calculate the number of thread blocks to launch per kernel call - this remains constant for the lifetime
        # of the object.
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

        # 5. Compile the kernel
        program = accel.build(
            context,
            "kernels/tensor_core_correlation_kernel.mako",
            {
                "n_ants_per_block": self._n_ants_per_block,
                "n_ants": self.n_ants,
                "sample_bitwidth": self._sample_bitwidth,
                "n_channels": self.n_channels,
                "n_polarisations": self.n_polarisations,
                "n_samples_per_channel": self.n_samples_per_channel,
                "n_baselines": self.n_baselines,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("correlate")

    def instantiate(self, command_queue: accel.AbstractCommandQueue) -> "TensorCoreXEngineCore":
        """Create a :class:`TensorCoreXEngineCore` using this template to build the kernel."""
        return TensorCoreXEngineCore(self, command_queue)


class TensorCoreXEngineCore(accel.Operation):
    """Tensor-Core correlation kernel.

    Specifies the shape of the input sample and output visibility buffers
    required by the kernel. The parameters specified in the
    :class:`TensorCoreXEngineCoreTemplate` object are used to determine the
    shape of the buffers.

    The input sample buffer must have the shape:
    ``[channels][samples_per_channel//times_per_block][n_ants][polarisations][times_per_block]``

    A complexity that is introduced by the Tensor-Core kernel is that the
    ``samples_per_channel`` index is split over two different indices. The first
    index ranges from ``0`` to ``samples_per_channel//times_per_block`` and the
    second index ranges from ``0`` to ``times_per_block``. Times per block is
    calculated by the :class:`TensorCoreXEngineCoreTemplate`. In 8-bit input mode,
    ``times_per_block`` is equal to 16.

    Each input element is a complex 8-bit integer sample. :mod:`.numpy` does not
    support 8-bit complex numbers, so the dimensionality is extended by 1, with
    the last dimension sized ``2`` to represent the complexity.

    With 8-bit input samples, the value -128i is not supported by the kernel as
    there is no 8-bit complex conjugate representation of this number. Passing
    ``-128i`` into the kernel will produce incorrect values at the output.

    The output visibility buffer must have the shape
    ``[channels][baselines][complexity]``. In 8-bit mode, each element in this
    visibility matrix is a 32-bit integer value.

    Currently only 8-bit input mode is supported.
    """

    def __init__(self, template: TensorCoreXEngineCoreTemplate, command_queue: accel.AbstractCommandQueue) -> None:
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
            # Even though we are using CUDA, we follow OpenCL's grid/block
            # conventions. As such we need to multiply the number of
            # blocks(global_size) by the block size(local_size) in order to
            # specify global threads not global blocks.
            global_size=(32 * self.template.n_blocks, 2 * self.template.n_channels, 2 * 1),
            local_size=(32, 2, 2),
        )

    def zero_visibilities(self):
        """Zero all the values in the out_visibilities buffer."""
        self.buffer("out_visibilities").zero(self.command_queue)

    @staticmethod
    def get_baseline_index(ant1, ant2):
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
        if ant2 > ant1:
            raise ValueError("It is required that ant2 >= ant1 in all cases")
        return ant2 * (ant2 + 1) // 2 + ant1
