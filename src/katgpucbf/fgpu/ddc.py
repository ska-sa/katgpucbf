################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""Digital down-conversion."""

import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import BYTE_BITS
from . import SAMPLE_BITS


class DDCTemplate:
    """Template for digital down-conversion.

    The kernel takes 10-bit integer inputs (real) and produces 32-bit
    floating-point outputs (complex). The user provides a real-valued FIR
    baseband filter and a mixer frequency for translating the signal from
    the desired band to baseband.

    Parameters
    ----------
    context
        The GPU context that we'll operate in
    taps
        Number of taps in the FIR filter
    decimation
        Fraction of samples to retain after filtering

    Raises
    ------
    ValueError
        If `taps` is not a multiple of `decimation`
    """

    def __init__(self, context: AbstractContext, taps: int, decimation: int) -> None:
        if taps <= 0:
            raise ValueError("taps must be positive")
        if decimation <= 0:
            raise ValueError("decimation must be positive")
        if taps % decimation != 0:
            raise ValueError("taps must be a multiple of decimation")
        # TODO: tune the magic numbers and enforce more requirements
        self.wgs = 128
        self._sg_size = 1
        self._coarsen = 1
        self.taps = taps
        self.decimation = decimation
        self._group_out_size = self.wgs // self._sg_size * self._coarsen  # TODO: tune
        program = accel.build(
            context,
            "kernels/ddc_hybrid.mako",
            {
                "wgs": self.wgs,
                "taps": taps,
                "decimation": decimation,
                "group_out_size": self._group_out_size,
                "coarsen": self._coarsen,
                "sg_size": self._sg_size,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("ddc")

    def instantiate(self, command_queue: AbstractCommandQueue, samples: int) -> "DDC":
        """Generate a :class:`DDC` object based on the template."""
        return DDC(self, command_queue, samples)


class DDC(accel.Operation):
    """Operation implementating :class:`DDCTemplate`.

    .. rubric:: Slots

    **in** : samples * SAMPLE_BITS // BYTE_BITS, uint8
        Input digitiser samples in a big chunk.
    **out** : out_samples, complex64
        Filtered and decimated output data
    **weights** : taps, float32
        Baseband filter coefficients. If the filter is asymmetric, the
        coefficients must be reversed: the first element gets
        multiplied by the oldest sample.

    Raises
    ------
    ValueError
        If `samples` is not a multiple of 8, or is less than `template.taps`

    Parameters
    ----------
    template
        Template for the PFB-FIR operation.
    command_queue
        The GPU command queue
    samples
        Number of input samples to store
    """

    def __init__(self, template: DDCTemplate, command_queue: AbstractCommandQueue, samples: int) -> None:
        super().__init__(command_queue)
        if samples % BYTE_BITS != 0:
            raise ValueError(f"samples must be a multiple of {BYTE_BITS}")
        if samples < template.taps:
            raise ValueError("samples must be at least the number of filter taps")
        self.template = template
        self.samples = samples
        self.out_samples = accel.divup(samples - template.taps + 1, template.decimation)
        self.slots["in"] = accel.IOSlot(
            (
                accel.Dimension(
                    samples * SAMPLE_BITS // BYTE_BITS,
                    min_padded_size=samples * SAMPLE_BITS // BYTE_BITS + 65536,
                ),
            ),
            np.uint8,
        )
        self.slots["out"] = accel.IOSlot(
            (accel.Dimension(self.out_samples),),
            np.complex64,
        )
        self.slots["weights"] = accel.IOSlot((template.taps,), np.float32)

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        weights_buffer = self.buffer("weights")
        groups = accel.divup(out_buffer.shape[0], self.template._group_out_size)
        # TODO: set up the offsets and mix frequency
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                in_buffer.buffer,
                weights_buffer.buffer,
                np.int32(0),  # out_offset
                np.int32(0),  # in_offset
                np.int32(out_buffer.shape[0]),  # out_size
                np.int32(in_buffer.shape[0] * BYTE_BITS // SAMPLE_BITS),  # in_size
                np.float32(0),  # mix_scale
                np.float32(0),  # mix_bias
            ],
            global_size=(groups * self.template.wgs, 1, 1),
            local_size=(self.template.wgs, 1, 1),
        )
