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
from . import DIG_SAMPLE_BITS_VALID, INPUT_CHUNK_PADDING


class PFBFIRTemplate:
    """Template for the PFB-FIR operation.

    The operation can operate in two different modes. In the first mode
    (intended for a wideband channeliser), the input contains real digitiser
    samples (bit-packed integers). In the second mode (intended for a
    narrowband channeliser), the digitiser samples have already been
    preprocessed and the PFB operates on complex-valued inputs (floating
    point). The mode is selected with the `complex_input` parameter.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    taps
        The number of taps that you want the resulting PFB-FIRs to have.
    channels
        Number of channels into which the input data will be decomposed.
    input_sample_bits
        Bits per each component of input. If `complex_input` is true, the input
        values are floating-point complex numbers and this must equal 32.
        Otherwise, the inputs are packed integers, and the value must be in
        :data:`DIG_SAMPLE_BITS_VALID`.
    unzip_factor
        The output is reordered so that every unzip_factor'ith pair of
        outputs (or single complex output, if `complex_input` is true) is
        placed contiguously.
    complex_input
        Operation mode (see above).
    n_pols
        Number of polarisations to operate over. The polarisations are
        stored contiguously in memory, but have independent offsets.

    Raises
    ------
    ValueError
        If ``taps`` is not positive.
    ValueError
        If ``complex_input`` is true and ``input_sample_bits`` is not 32.
    ValueError
        If ``complex_input`` is false and ``input_sample_bits`` is not in
        :data:`.DIG_SAMPLE_BITS_VALID`.
    ValueError
        If ``channels`` is not an even power of 2.
    ValueError
        If ``channels`` is not a multiple of ``unzip_factor``.
    ValueError
        If ``2*channels`` is not a multiple of the workgroup size (currently
        128).
    """

    def __init__(
        self,
        context: AbstractContext,
        taps: int,
        channels: int,
        input_sample_bits: int,
        unzip_factor: int = 1,
        *,
        complex_input: bool = False,
        n_pols: int,
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be at least 1")
        self.wgs = 128
        self.taps = taps
        self.channels = channels
        self.input_sample_bits = input_sample_bits
        self.unzip_factor = unzip_factor
        self.complex_input = complex_input
        self.n_pols = n_pols
        if complex_input:
            if input_sample_bits != 32:
                raise ValueError("input_sample_bits must be 32 when complex_input is true")
        else:
            if input_sample_bits not in DIG_SAMPLE_BITS_VALID:
                raise ValueError("input_sample_bits must be 2-10, 12 or 16 when complex_input is false")
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
                    "input_sample_bits": input_sample_bits,
                    "unzip_factor": unzip_factor,
                    "complex_input": complex_input,
                    "n_pols": n_pols,
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

    The slots depend on ``template.complex_input``. If it is false, then they are

    **in** : pols × (samples * input_sample_bits // BYTE_BITS), uint8
        Input samples in a big chunk.
    **out** : pols × spectra × 2*channels, float32
        FIR-filtered time data, ready to be processed by the FFT.
    **weights** : 2*channels*taps, float32
        The time-domain transfer function of the FIR filter to be applied.
    **total_power** : pols, uint64
        Sum of squares of input samples. This will not include every input
        sample. Rather, it will contain a specific tap from each PFB window
        (currently, the last tap, but that is an implementation detail).

        This is incremented rather than overwritten. It is the caller's
        responsibility to zero it when desired, or alternatively to track
        values before and after to measure the change.

    Otherwise, they are

    **in** : samples, complex64
        Input samples
    **out** : spectra × channels, complex64
        See above
    **weights** : channels*taps, float32
        See above

    Raises
    ------
    ValueError
        If ``samples`` is not a multiple of 8 and ``complex_input`` is false
    ValueError
        If ``samples`` is too large (more than 2**29)

    Parameters
    ----------
    template
        Template for the PFB-FIR operation.
    command_queue
        The GPU command queue (typically this will be an instance of
        :class:`katsdpsigproc.cuda.CommandQueue` which wraps a CUDA Stream) on
        which actual processing operations are to be scheduled.
    samples
        Number of input samples that will be processed each time the operation is run.
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
        if not template.complex_input and samples % BYTE_BITS != 0:
            raise ValueError(f"samples must be a multiple of {BYTE_BITS}")
        if samples > 2**29:
            # This ensures no overflow in samples_to_bytes in the kernel
            raise ValueError("at most 2**29 samples are supported")
        self.template = template
        self.samples = samples
        self.spectra = spectra  # Can be changed (TODO: documentation)
        if template.complex_input:
            self.slots["in"] = accel.IOSlot((template.n_pols, samples), np.complex64)
            # The output needs to be unpadded so that it will match what the next
            # stage (FFT) expects.
            self.slots["out"] = accel.IOSlot(
                (
                    template.n_pols,
                    accel.Dimension(spectra, exact=True),
                    accel.Dimension(template.channels, exact=True),
                ),
                np.complex64,
            )
            self.slots["weights"] = accel.IOSlot((template.channels * template.taps,), np.float32)
        else:
            step = 2 * template.channels
            # Some load operations can run past the end. Not all input_sample_bits
            # need padding, but it's simplest just to provide it unconditionally.
            # The actual padding needed is only 1 byte, but we use
            # INPUT_CHUNK_PADDING so that the Compute operation ends up with the
            # desired padded size.
            in_padding = INPUT_CHUNK_PADDING
            in_bytes = samples * template.input_sample_bits // BYTE_BITS
            self.slots["in"] = accel.IOSlot(
                (
                    template.n_pols,
                    accel.Dimension(in_bytes, min_padded_size=in_bytes + in_padding),
                ),
                np.uint8,
            )
            # The output needs to be unpadded so that it will match what the next
            # stage (FFT) expects.
            self.slots["out"] = accel.IOSlot(
                (
                    template.n_pols,
                    accel.Dimension(spectra, exact=True),
                    accel.Dimension(step, exact=True),
                ),
                np.float32,
            )
            self.slots["weights"] = accel.IOSlot((step * template.taps,), np.float32)
            self.slots["total_power"] = accel.IOSlot((template.n_pols,), np.uint64)
        self.in_offset = np.zeros(template.n_pols, int)  # Number of samples to skip from the start of *in
        self.out_offset = 0  # Number of "spectra" to skip from the start of *out.

    def _run(self) -> None:
        if self.spectra == 0:
            return
        rps = 2 if self.template.complex_input else 1  # real values per sample
        real_step = 2 * self.template.channels  # step size in real values
        sample_step = real_step // rps  # step size in samples
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        # in_buffer_bytes/in_buffer_samples are per polarisation
        in_buffer_bytes = in_buffer.shape[1] * in_buffer.dtype.itemsize
        in_buffer_samples = in_buffer_bytes * BYTE_BITS // (self.template.input_sample_bits * rps)
        for in_offset in self.in_offset:
            if in_offset + sample_step * (self.spectra + self.template.taps - 1) > in_buffer_samples:
                raise IndexError("Input buffer does not contain sufficient samples")
        if self.out_offset + self.spectra > out_buffer.shape[1]:
            raise IndexError("Output buffer does not contain sufficient spectra")

        # Try to ensure that each workitem has enough work to do to amortise
        # the overhead of loading the initial taps. Each workitem should
        # contribute to work_spectra outputs.
        work_spectra = self.template.taps * 8
        # Number of workgroups along the time axis to match this
        groupsy = accel.divup(self.spectra, work_spectra)
        # Keep a minimum of 128K workitems (across all pols), to avoid starving
        # the GPU for work.
        groupsy = max(groupsy, accel.divup(128 * 1024 // self.template.n_pols, real_step))
        # Re-compute work_spectra to balance the load
        work_spectra = accel.divup(self.spectra, groupsy)
        stepy = work_spectra * real_step
        # Rounding up may have left some workgroups with nothing to do, so recalculate
        # groupsy again.
        groupsy = accel.divup(self.spectra, work_spectra)

        raw_in_offset = (self.in_offset * rps).astype(np.int32)
        out_buffer = self.buffer("out")
        in_buffer = self.buffer("in")
        kernel_args = (
            [
                out_buffer.buffer,
                in_buffer.buffer,
                self.buffer("weights").buffer,
                np.int32(out_buffer.padded_shape[1] * out_buffer.padded_shape[2] * rps),
                np.int32(in_buffer.padded_shape[1] * rps),
                np.int32(real_step * self.spectra),
                np.int32(stepy),
            ]
            + list(raw_in_offset)
            + [
                np.int32(self.out_offset * real_step),
            ]
        )
        if not self.template.complex_input:
            kernel_args.insert(1, self.buffer("total_power").buffer)

        self.command_queue.enqueue_kernel(
            self.template.kernel,
            kernel_args,
            global_size=(real_step, groupsy, self.template.n_pols),
            local_size=(self.template.wgs, 1, 1),
        )
