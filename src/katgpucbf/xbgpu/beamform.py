################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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


class BeamformTemplate:
    """Template for beamforming.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    pols
        One entry per single-polarisation output beam. Each entry is either
        0 or 1, to indicate which input polarisation to use in the beam.
    """

    def __init__(self, context: AbstractContext, beam_pols: Sequence[int]) -> None:
        # TODO: tune these. And maybe adapt to input shape?
        self.block_spectra = 128
        self.block_channels = 1
        self.beam_pols = beam_pols
        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/beamform.mako",
                {
                    "block_spectra": self.block_spectra,
                    "block_channels": self.block_channels,
                    "beam_pols": self.beam_pols,
                },
                extra_dirs=[str(resource_dir)],
            )
        self.kernel = program.get_kernel("beamform")

    def instantiate(
        self,
        command_queue: AbstractCommandQueue,
        n_frames: int,
        n_ants: int,
        n_channels: int,
        n_spectra_per_frame: int,
    ) -> "Beamform":
        """Generate a :class:`Beamform` object based on the template."""
        return Beamform(self, command_queue, n_frames, n_ants, n_channels, n_spectra_per_frame)


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
    **weights** : n_ants × n_beams, complex64
        Complex scale factor to apply to each antenna for each beam
    **delays** : n_ants × n_beams, float32
        Delay used to compute channel-dependent phase rotation. The
        rotation applied is :math:`e^{\pi cd}` where :math:`c` is
        the channel number and :math:`d` is the delay value. Note
        that this will not apply any rotation to the first channel
        in the data; any such rotation needs to be baked into **weights**.

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
    n_spectra_per_frame
        Number of samples in time axis for each frame (fine time dimension)
    """

    def __init__(
        self,
        template: BeamformTemplate,
        command_queue: AbstractCommandQueue,
        n_frames: int,
        n_ants: int,
        n_channels: int,
        n_spectra_per_frame: int,
    ) -> None:
        super().__init__(command_queue)
        self.template = template
        pol_dim = accel.Dimension(N_POLS, exact=True)
        complex_dim = accel.Dimension(COMPLEX, exact=True)
        n_beams = len(template.beam_pols)
        self.slots["in"] = accel.IOSlot(
            (n_frames, n_ants, n_channels, n_spectra_per_frame, pol_dim, complex_dim), np.int8
        )
        self.slots["out"] = accel.IOSlot((n_frames, n_beams, n_channels, n_spectra_per_frame, complex_dim), np.int8)
        weights_dims = (n_ants, accel.Dimension(n_beams, exact=True))
        self.slots["weights"] = accel.IOSlot(weights_dims, np.complex64)
        self.slots["delays"] = accel.IOSlot(weights_dims, np.float32)

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                in_buffer.buffer,
                self.buffer("weights").buffer,
                self.buffer("delays").buffer,
                np.int32(out_buffer.padded_shape[3]),
                np.int32(out_buffer.padded_shape[2] * out_buffer.padded_shape[3]),
                np.int32(out_buffer.padded_shape[1] * out_buffer.padded_shape[2] * out_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[2] * in_buffer.padded_shape[3]),
                np.int32(in_buffer.padded_shape[1] * in_buffer.padded_shape[2] * in_buffer.padded_shape[3]),
                np.int32(in_buffer.shape[1]),
                np.int32(in_buffer.shape[2]),
                np.int32(in_buffer.shape[3]),
            ],
            global_size=(
                accel.roundup(in_buffer.shape[3], self.template.block_spectra),
                accel.roundup(in_buffer.shape[2], self.template.block_channels),
                in_buffer.shape[0],
            ),
            local_size=(self.template.block_spectra, self.template.block_channels, 1),
        )