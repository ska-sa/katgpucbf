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

"""Postproc module.

These classes handle the operation of the GPU in performing the fine-delay,
per-channel gains, requantisation and corner-turn through a mako-templated
kernel.
"""

from importlib import resources

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import N_POLS, utils
from ..curand_helpers import RAND_STATE_DTYPE, RandomStateBuilder
from ..utils import DitherType


class PostprocTemplate:
    """Template for the postproc operation.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    channels
        Number of input channels in each spectrum.
    unzip_factor
        Radix of the final Cooley-Tukey FFT step performed by the kernel.
    complex_pfb
        If true, the PFB is a complex-to-complex transform, and no
        real-to-complex fixup is needed. Additionally, the DC channel is
        considered to be the centre of the band i.e. it is written to the
        middle of the output rather than the start (and similarly, gains for
        it are loaded from the middle of the gain array etc).
    out_bits
        Bits per real/imaginary value. Only 4 or 8 are currently supported.
        When 4, the real part is in the most-significant bits.
    dither
        Type of dithering to apply before quantisation.
    out_channels
        Range of channels to write to the output (defaults to all).
    """

    def __init__(
        self,
        context: AbstractContext,
        channels: int,
        unzip_factor: int = 1,
        *,
        complex_pfb: bool,
        out_bits: int,
        dither: DitherType,
        out_channels: tuple[int, int] | None = None,
    ) -> None:
        assert dither in {DitherType.NONE, DitherType.UNIFORM}
        self.block = 16
        self.vtx = 1
        self.vty = 2
        self.channels = channels
        self.unzip_factor = unzip_factor
        self.out_bits = out_bits
        self.dither = dither
        self.groups_x = accel.divup(channels // unzip_factor // 2 + 1, self.block * self.vtx)
        if channels <= 0 or channels & (channels - 1):
            raise ValueError("channels must be a power of 2")
        if channels % unzip_factor:
            raise ValueError("channels must be a multiple of unzip_factor")
        if unzip_factor not in {1, 2, 4}:
            raise ValueError("unzip_factor must be 1, 2 or 4")
        if out_bits not in {4, 8}:
            raise ValueError("out_bits must be 4 or 8")
        if out_channels is None:
            self.out_channels = (0, channels)
        else:
            if not 0 <= out_channels[0] < out_channels[1] <= channels:
                raise ValueError("out_channels must be a subrange of [0, channels)")
            self.out_channels = out_channels
        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/postproc.mako",
                {
                    "block": self.block,
                    "vtx": self.vtx,
                    "vty": self.vty,
                    "groups_x": self.groups_x,
                    "channels": channels,
                    "out_low": self.out_channels[0],
                    "out_high": self.out_channels[1],
                    "out_bits": self.out_bits,
                    "unzip_factor": unzip_factor,
                    "complex_pfb": complex_pfb,
                    "dither": bool(dither.value),
                },
                extra_dirs=[str(resource_dir), str(resource_dir.parent)],
            )
        self.kernel = program.get_kernel("postproc")

    def instantiate(
        self,
        command_queue: AbstractCommandQueue,
        spectra: int,
        spectra_per_heap: int,
        *,
        seed: int,
        sequence_first: int,
        sequence_step: int = 1,
    ) -> "Postproc":
        """Generate a :class:`Postproc` object based on this template."""
        return Postproc(
            self,
            command_queue,
            spectra,
            spectra_per_heap,
            seed=seed,
            sequence_first=sequence_first,
            sequence_step=sequence_step,
        )


class Postproc(accel.Operation):
    """The fine-delay, requant and corner-turn operations coming after the PFB.

    .. rubric:: Slots

    **in** : N_POLS × spectra × unzip_factor × channels // unzip_factor, complex64
        Input channelised data for the two polarisations. These are formed by
        taking the complex-to-complex Fourier transform of the input
        reinterpreted as a complex input. See :ref:`fgpu-fft` for details.
    **out** : spectra // spectra_per_heap × out_channels × spectra_per_heap × N_POLS
        Output F-engine data, quantised and corner-turned, ready for
        transmission on the network. See :func:`.gaussian_dtype` for the type.
    **saturated** : spectra // spectra_per_heap × N_POLS, uint32
        Number of saturated complex values in **out**.
    **fine_delay** : spectra × N_POLS, float32
        Fine delay in samples (one value per pol).
    **phase** : spectra × N_POLS, float32
        Fixed phase adjustment in radians (one value per pol).
    **gains** : out_channels × N_POLS, complex64
        Per-channel gain (one value per pol).
    **rand_states** : implementation-defined
        Random states. This slot is set up by the constructor and should
        normally not need to be touched. It is only present if dithering
        is enabled.

    Parameters
    ----------
    template: PostprocTemplate
        The template for the post-processing operation.
    command_queue: AbstractCommandQueue
        The GPU command queue (typically this will be a CUDA Stream) on which
        actual processing operations are to be scheduled.
    spectra: int
        Number of spectra on which post-prodessing will be performed.
    spectra_per_heap: int
        Number of spectra to send out per heap.
    seed, sequence_first, sequence_step
        See :class:`.RandomStateBuilder`. These are ignored if the template
        disables dithering.
    """

    def __init__(
        self,
        template: PostprocTemplate,
        command_queue: AbstractCommandQueue,
        spectra: int,
        spectra_per_heap: int,
        *,
        seed: int,
        sequence_first: int,
        sequence_step: int = 1,
    ) -> None:
        super().__init__(command_queue)
        if spectra % spectra_per_heap != 0:
            raise ValueError("spectra must be a multiple of spectra_per_heap")
        heaps = spectra // spectra_per_heap
        block_y = template.block * template.vty
        if spectra_per_heap % block_y != 0:
            raise ValueError(f"spectra_per_heap must be a multiple of {block_y}")
        self.template = template
        self.spectra = spectra
        self.spectra_per_heap = spectra_per_heap
        self._groups_y = spectra_per_heap // block_y
        self._heaps = heaps
        pols = accel.Dimension(N_POLS, exact=True)

        in_shape = (
            accel.Dimension(N_POLS),
            accel.Dimension(spectra),
            accel.Dimension(template.unzip_factor, exact=True),
            accel.Dimension(template.channels // template.unzip_factor, exact=True),
        )
        n_out_channels = template.out_channels[1] - template.out_channels[0]
        out_dtype = utils.gaussian_dtype(template.out_bits)
        self.slots["in"] = accel.IOSlot(in_shape, np.complex64)
        self.slots["out"] = accel.IOSlot((heaps, n_out_channels, spectra_per_heap, pols), out_dtype)
        self.slots["saturated"] = accel.IOSlot((heaps, pols), np.uint32)
        self.slots["fine_delay"] = accel.IOSlot((spectra, pols), np.float32)
        self.slots["phase"] = accel.IOSlot((spectra, pols), np.float32)
        self.slots["gains"] = accel.IOSlot((n_out_channels, pols), np.complex64)
        if template.dither == DitherType.UNIFORM:
            # This could be seen as multi-dimensional, but we flatten it to 1D as an
            # easy way to guarantee that it is not padded.
            rand_states_shape = (template.groups_x * self._groups_y * template.block * template.block,)
            self.slots["rand_states"] = accel.IOSlot(rand_states_shape, RAND_STATE_DTYPE)
            builder = RandomStateBuilder(command_queue.context)
            rand_states = builder.make_states(
                command_queue, rand_states_shape, seed=seed, sequence_first=sequence_first, sequence_step=sequence_step
            )
            self.bind(rand_states=rand_states)

    def _run(self) -> None:
        out = self.buffer("out")
        saturated = self.buffer("saturated")
        in_ = self.buffer("in")
        saturated.zero(self.command_queue)
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out.buffer,
                saturated.buffer,
                in_.buffer,
                self.buffer("fine_delay").buffer,
                self.buffer("phase").buffer,
                self.buffer("gains").buffer,
            ]
            + ([self.buffer("rand_states").buffer] if self.template.dither == DitherType.UNIFORM else [])
            + [
                np.int32(out.padded_shape[1] * out.padded_shape[2]),  # out_stride_z
                np.int32(out.padded_shape[2]),  # out_stride
                np.int32(np.prod(in_.padded_shape[1:])),  # in_stride
                np.int32(self.spectra_per_heap),  # spectra_per_heap
                np.int32(self._heaps),  # heaps
            ],
            global_size=(
                self.template.block * self.template.groups_x,
                self.template.block * self._groups_y,
                1,
            ),
            local_size=(self.template.block, self.template.block, 1),
        )
