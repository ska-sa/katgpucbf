################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Implement the calculations for beamforming."""

from importlib import resources
from typing import Sequence

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import COMPLEX, N_POLS
from ..curand_helpers import RandomStateBuilder


class BeamformTemplate:
    """Template for beamforming.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    beam_pols
        One entry per single-polarisation output beam. Each entry is either
        0 or 1, to indicate which input polarisation to use in the beam.
    n_spectra_per_frame
        Number of samples in time axis for each frame (fine time dimension)
        - see :class:`Beamform`.
    """

    def __init__(self, context: AbstractContext, beam_pols: Sequence[int], n_spectra_per_frame: int) -> None:
        # TODO: tune these.
        self.block_spectra = min(128, n_spectra_per_frame)
        self.beam_pols = beam_pols
        self.n_spectra_per_frame = n_spectra_per_frame
        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/beamform.mako",
                {
                    "block_spectra": self.block_spectra,
                    "beam_pols": self.beam_pols,
                },
                extra_dirs=[str(resource_dir), str(resource_dir.parent)],
            )
        self.kernel = program.get_kernel("beamform")

    def instantiate(
        self,
        command_queue: AbstractCommandQueue,
        n_frames: int,
        n_ants: int,
        n_channels: int,
        seed: int,
        sequence_first: int,
        sequence_step: int = 1,
    ) -> "Beamform":
        """Generate a :class:`Beamform` object based on the template."""
        return Beamform(self, command_queue, n_frames, n_ants, n_channels, seed, sequence_first, sequence_step)


class Beamform(accel.Operation):
    r"""Operation for beamforming.

    For ease-of-use with the data formats used in the rest of katgpucbf,
    time is split into two dimensions: a coarse outer dimension (called
    "frames") and a finer inner dimension ("spectra").

    .. rubric:: Slots

    **in** : n_frames × n_ants × n_channels × n_spectra_per_frame × N_POLS × COMPLEX, int8
        Complex (Gaussian integer) input channelised voltages
    **out** : n_frames × n_beams × n_channels × n_spectra_per_frame × COMPLEX, int8
        Complex (Gaussian integer) output channelised voltages
    **saturated**: n_beams, uint32
        Number of saturated output values, per beam. This value is *incremented*
        by the kernel, so should be explicitly zeroed first if desired.
    **weights** : n_ants × n_beams, complex64
        Complex scale factor to apply to each antenna for each beam
    **delays** : n_ants × n_beams, float32
        Delay used to compute channel-dependent phase rotation. The
        rotation applied is :math:`e^{j\pi cd}` where :math:`c` is
        the channel number and :math:`d` is the delay value. Note
        that this will not apply any rotation to the first channel
        in the data; any such rotation needs to be baked into **weights**.
    **rand_states** : n_frames × n_channels × n_spectra_per_frame, curandStateXORWOW_t (packed)
        Independent random states for generating dither values. This is set
        up by the constructor and should not normally need to be touched.

    Parameters
    ----------
    template
        The template for the operation
    command_queue
        The command queue on which to enqueue the work
    n_frames
        Number of frames (coarse time dimension)
    n_ants
        Number of antennas
    n_channels
        Number of frequency channels
    seed, sequence_first, sequence_step
        See :class:`.RandomStateBuilder`.
    """

    def __init__(
        self,
        template: BeamformTemplate,
        command_queue: AbstractCommandQueue,
        n_frames: int,
        n_ants: int,
        n_channels: int,
        seed: int,
        sequence_first: int,
        sequence_step: int = 1,
    ) -> None:
        super().__init__(command_queue)
        self.template = template
        pol_dim = accel.Dimension(N_POLS, exact=True)
        complex_dim = accel.Dimension(COMPLEX, exact=True)
        n_beams = len(template.beam_pols)
        n_spectra_per_frame = template.n_spectra_per_frame
        builder = RandomStateBuilder(command_queue.context)
        self.slots["in"] = accel.IOSlot(
            (n_frames, n_ants, n_channels, n_spectra_per_frame, pol_dim, complex_dim), np.int8
        )
        self.slots["out"] = accel.IOSlot((n_frames, n_beams, n_channels, n_spectra_per_frame, complex_dim), np.int8)
        self.slots["saturated"] = accel.IOSlot((n_beams,), np.uint32)
        weights_dims = (n_ants, accel.Dimension(n_beams, exact=True))
        self.slots["weights"] = accel.IOSlot(weights_dims, np.complex64)
        self.slots["delays"] = accel.IOSlot(weights_dims, np.float32)
        self.slots["rand_states"] = accel.IOSlot(
            (
                accel.Dimension(n_frames, exact=True),
                accel.Dimension(n_channels, exact=True),
                accel.Dimension(n_spectra_per_frame, exact=True),
            ),
            builder.dtype,
        )
        rand_states = builder.make_states(
            (n_frames, n_channels, n_spectra_per_frame),
            seed=seed,
            sequence_first=sequence_first,
            sequence_step=sequence_step,
        )
        self.bind(rand_states=rand_states)

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                self.buffer("saturated").buffer,
                in_buffer.buffer,
                self.buffer("weights").buffer,
                self.buffer("delays").buffer,
                self.buffer("rand_states").buffer,
                np.int32(out_buffer.padded_shape[3]),
                np.int32(out_buffer.padded_shape[2] * out_buffer.padded_shape[3]),
                np.int32(out_buffer.padded_shape[1] * out_buffer.padded_shape[2] * out_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[2] * in_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[1] * in_buffer.padded_shape[2] * in_buffer.padded_shape[3]),
                np.int32(in_buffer.shape[1]),
                np.int32(in_buffer.shape[3]),
            ],
            global_size=(
                accel.roundup(in_buffer.shape[3], self.template.block_spectra),
                in_buffer.shape[2],
                in_buffer.shape[0],
            ),
            local_size=(self.template.block_spectra, 1, 1),
        )
