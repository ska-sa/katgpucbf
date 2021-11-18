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

"""PFB module.

These classes handle the operation of the GPU in performing the PFB-FIR part
through a mako-templated kernel.
"""


import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from . import BYTE_BITS, SAMPLE_BITS


class PFBFIRTemplate:
    """Template for the PFB-FIR operation.

    Parameters
    ----------
    context: AbstractContext
        The GPU context that we'll operate in.
    taps: int
        The number of taps that you want the resulting PFB-FIRs to have.
    """

    def __init__(self, context: AbstractContext, taps: int) -> None:
        if taps <= 0:
            raise ValueError("taps must be at least 1")
        self.wgs = 128
        self.taps = taps
        program = accel.build(
            context,
            "kernels/pfb_fir.mako",
            {"wgs": self.wgs, "taps": self.taps},
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("pfb_fir")

    def instantiate(self, command_queue: AbstractCommandQueue, samples: int, spectra: int, channels: int) -> "PFBFIR":
        """Generate a :class:`PFBFIR` object based on the template."""
        return PFBFIR(self, command_queue, samples, spectra, channels)


class PFBFIR(accel.Operation):
    """The windowing FIR filters that form the first part of the PFB.

    The best place to look in order to understand how these work from a strictly
    DSP sense is Danny C. Price's paper :cite:`price2018spectrometers`.

    .. bibliography::

        price2018spectrometers

    .. rubric:: Slots

    **in**  : samples * SAMPLE_BITS // BYTE_BITS, uint8
        Input digitiser samples in a big chunk.
    **out** : spectra × 2*channels, float32
        FIR-filtered time data, ready to be processed by the FFT.
    **weights** : 2*channels*taps, float32
        The time-domain transfer function of the FIR filter to be applied.

    Raises
    ------
    ValueError
        If ``samples`` is not a multiple of 8, or if ``2*channels`` is not a
        multiple of the workgroup size (currently 128).

    Parameters
    ----------
    template
        Template for the PFB-FIR operation.
    command_queue
        The GPU command queue (typically this will be an instance of
        :class:`katsdpsigproc.cuda.CommandQueue` which wraps a CUDA Stream) on
        which actual processing operations are to be scheduled.
    samples
        Number of samples that will be processed each time the operation is run.
    spectra
        Number of spectra that we will get from each chunk of samples.
    channels
        Number of channels into which the input data will be decomposed.
    """

    def __init__(
        self, template: PFBFIRTemplate, command_queue: AbstractCommandQueue, samples: int, spectra: int, channels: int
    ) -> None:
        super().__init__(command_queue)
        if samples % BYTE_BITS != 0:
            raise ValueError(f"samples must be a multiple of {BYTE_BITS}")
        if (2 * channels) % template.wgs != 0:
            raise ValueError(f"2*channels must be a multiple of {template.wgs}")
        self.template = template
        self.samples = samples
        self.spectra = spectra  # Can be changed (TODO: documentation)
        self.channels = channels
        self.slots["in"] = accel.IOSlot((samples * SAMPLE_BITS // BYTE_BITS,), np.uint8)
        self.slots["out"] = accel.IOSlot((spectra, accel.Dimension(2 * channels, exact=True)), np.float32)
        self.slots["weights"] = accel.IOSlot((2 * channels * template.taps,), np.float32)
        self.in_offset = 0  # Number of samples to skip from the start of *in
        self.out_offset = 0  # Number of "spectra" to skip from the start of *out.

    def _run(self) -> None:
        if self.spectra == 0:
            return
        step = 2 * self.channels
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        if self.in_offset + step * (self.spectra + self.template.taps - 1) > in_buffer.shape[0]:
            raise IndexError("Input buffer does not contain sufficient samples")
        if self.out_offset + self.spectra > out_buffer.shape[0]:
            raise IndexError("Output buffer does not contain sufficient spectra")

        # We divide up the work so that there are sufficient threads to keep the GPU busy.
        # Aim for 256K workitems i.e. step * (out_n / stepy) == 256K
        # See pfb_fir.mako's comments for more info.

        # We might not need the entire capacity of the out slot,
        # e.g. if we are shutting down.
        out_n = step * self.spectra

        stepy = accel.roundup(accel.divup(step * out_n, 256 * 1024), step)
        groupsx = step // self.template.wgs
        groupsy = accel.divup(out_n, stepy)
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                self.buffer("out").buffer,
                self.buffer("in").buffer,
                self.buffer("weights").buffer,
                np.int32(out_n),
                np.int32(step),
                np.int32(stepy),
                np.int32(self.in_offset),
                np.int32(self.out_offset * step),  # Must be a multiple of step to make sense.
            ],
            global_size=(groupsx * self.template.wgs, groupsy),
            local_size=(self.template.wgs, 1),
        )
