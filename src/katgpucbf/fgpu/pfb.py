################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

from importlib import resources

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import BYTE_BITS


class PFBFIRTemplate:
    """Template for the PFB-FIR operation.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    taps
        The number of taps that you want the resulting PFB-FIRs to have.
    channels
        Number of channels into which the input data will be decomposed.
    dig_sample_bits
        Bits per digitiser sample.
    unzip_factor
        The output is reordered so that every unzip_factor'ith pair of
        outputs is placed contiguously.
    """

    def __init__(
        self, context: AbstractContext, taps: int, channels: int, dig_sample_bits: int, unzip_factor: int = 1
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be at least 1")
        self.wgs = 128
        self.taps = taps
        self.channels = channels
        self.dig_sample_bits = dig_sample_bits
        self.unzip_factor = unzip_factor
        if dig_sample_bits < 2 or (dig_sample_bits > 10 and dig_sample_bits not in {12, 16}):
            raise ValueError("dig_sample_bits must be 2-10, 12 or 16")
        if (2 * channels) % self.wgs != 0:
            raise ValueError(f"2*channels must be a multiple of {self.wgs}")
        if channels <= 1 or channels & (channels - 1):
            raise ValueError("channels must be an even power of 2")
        if channels % unzip_factor != 0:
            raise ValueError("channels must be a multiple of unzip_factor")
        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/pfb_fir.mako",
                {
                    "wgs": self.wgs,
                    "taps": self.taps,
                    "channels": channels,
                    "dig_sample_bits": dig_sample_bits,
                    "unzip_factor": unzip_factor,
                },
                extra_dirs=[str(resource_dir)],
            )
        self.kernel = program.get_kernel("pfb_fir")

    def instantiate(
        self,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
    ) -> "PFBFIR":
        """Generate a :class:`PFBFIR` object based on the template."""
        return PFBFIR(self, command_queue, samples, spectra)


class PFBFIR(accel.Operation):
    """The windowing FIR filters that form the first part of the PFB.

    The best place to look in order to understand how these work from a strictly
    DSP sense is Danny C. Price's paper :cite:`price2018spectrometers`.

    In general the operation can read some interval of the input slot and write
    to some interval of the output slot. The sizes of these slots need not be
    related. This can be useful to build up a larger output from smaller
    invocations that have different coarse delays.

    .. bibliography::

        price2018spectrometers

    .. rubric:: Slots

    **in** : samples * dig_sample_bits // BYTE_BITS, uint8
        Input digitiser samples in a big chunk.
    **out** : spectra × 2*channels, float32
        FIR-filtered time data, ready to be processed by the FFT.
    **weights** : 2*channels*taps, float32
        The time-domain transfer function of the FIR filter to be applied.
    **total_power** : uint64
        Sum of squares of input samples. This will not include every input
        sample. Rather, it will contain a specific tap from each PFB window
        (currently, the last tap, but that is an implementation detail).

        This is incremented rather than overwritten. It is the caller's
        responsibility to zero it when desired, or alternatively to track
        values before and after to measure the change.

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
    """

    def __init__(
        self,
        template: PFBFIRTemplate,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
    ) -> None:
        super().__init__(command_queue)
        if samples % BYTE_BITS != 0:
            raise ValueError(f"samples must be a multiple of {BYTE_BITS}")
        if samples > 2**29:
            # This ensures no overflow in samples_to_bytes in the kernel
            raise ValueError("at most 2**29 samples are supported")
        self.template = template
        self.samples = samples
        step = 2 * template.channels
        self.spectra = spectra  # Can be changed (TODO: documentation)
        # Some load operations can run past the end. Not all dig_sample_bits
        # need padding, but it's simplest just to provide it unconditionally.
        in_padding = 1
        in_bytes = samples * template.dig_sample_bits // BYTE_BITS
        self.slots["in"] = accel.IOSlot((accel.Dimension(in_bytes, min_padded_size=in_bytes + in_padding),), np.uint8)
        self.slots["out"] = accel.IOSlot((spectra, accel.Dimension(step, exact=True)), np.float32)
        self.slots["weights"] = accel.IOSlot((step * template.taps,), np.float32)
        self.slots["total_power"] = accel.IOSlot((), np.uint64)
        self.in_offset = 0  # Number of samples to skip from the start of *in
        self.out_offset = 0  # Number of "spectra" to skip from the start of *out.

    def _run(self) -> None:
        if self.spectra == 0:
            return
        step = 2 * self.template.channels
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        in_buffer_samples = in_buffer.shape[0] * BYTE_BITS / self.template.dig_sample_bits
        if self.in_offset + step * (self.spectra + self.template.taps - 1) > in_buffer_samples:
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
                self.buffer("total_power").buffer,
                self.buffer("in").buffer,
                self.buffer("weights").buffer,
                np.int32(out_n),
                np.int32(stepy),
                np.int32(self.in_offset),
                np.int32(self.out_offset * step),  # Must be a multiple of step to make sense.
            ],
            global_size=(groupsx * self.template.wgs, groupsy),
            local_size=(self.template.wgs, 1),
        )
