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

from typing import TypedDict

import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import BYTE_BITS
from . import SAMPLE_BITS


class _TuningDict(TypedDict):
    wgs: int
    sg_size: int
    coarsen: int
    segment_samples: int


class DDCTemplate:
    """Template for digital down-conversion.

    See :class:`DDC` for a more detailed description of what it does.

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

    def __init__(self, context: AbstractContext, taps: int, decimation: int, tuning: _TuningDict | None = None) -> None:
        if taps <= 0:
            raise ValueError("taps must be positive")
        if decimation <= 0:
            raise ValueError("decimation must be positive")
        if taps % decimation != 0:
            raise ValueError("taps must be a multiple of decimation")
        if tuning is None:
            tuning = self.autotune(context, taps, decimation)
        self.context = context
        self.wgs = tuning["wgs"]
        self._sg_size = tuning["sg_size"]
        self._coarsen = tuning["coarsen"]
        self._segment_samples = tuning["segment_samples"]
        self.taps = taps
        self.decimation = decimation

        self._group_out_size = self.wgs // self._sg_size * self._coarsen
        self._group_in_size = self._group_out_size * decimation
        self._load_size = self._group_in_size + taps - decimation
        self._segments = (self._load_size - 1) // (self._segment_samples * self.wgs) + 1

        # Sanity check the tuning parameters
        if self.wgs % self._sg_size:
            raise ValueError("wgs must be a multiple of sg_size")
        if self.decimation % self._sg_size:
            raise ValueError("decimation must be a multiple of sg_size")
        if self._group_in_size % self._segment_samples:
            raise ValueError("group_in_size must be a multiple of segment_samples (fix sg_size)")
        if self._segment_samples * SAMPLE_BITS % 32:
            raise ValueError("segment_samples * SAMPLE_BITS must be a multiple of 32")

        program = accel.build(
            context,
            "kernels/ddc.mako",
            {
                "wgs": self.wgs,
                "taps": taps,
                "decimation": decimation,
                "coarsen": self._coarsen,
                "sg_size": self._sg_size,
                "sample_bits": SAMPLE_BITS,
                "segment_samples": self._segment_samples,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("ddc")

    def autotune(self, context: AbstractContext, taps: int, decimation: int) -> _TuningDict:
        """Determine tuning parameters.

        .. todo::

           Actually do autotuning instead of using fixed logic.
        """
        wgs = 32
        coarsen = 9
        sg_size = 2
        segment_samples = 16
        while sg_size > 1 and decimation % sg_size != 0:
            sg_size //= 2
        return {"wgs": wgs, "coarsen": coarsen, "sg_size": sg_size, "segment_samples": segment_samples}

    def instantiate(self, command_queue: AbstractCommandQueue, samples: int) -> "DDC":
        """Generate a :class:`DDC` object based on the template."""
        return DDC(self, command_queue, samples)


class DDC(accel.Operation):
    r"""Operation implementating :class:`DDCTemplate`.

    The kernel takes 10-bit integer inputs (real) and produces 32-bit
    floating-point outputs (complex). The user provides a real-valued FIR
    baseband filter and a mixer frequency for translating the signal from
    the desired band to baseband.

    Element j of the output contains the dot product of **weights** with
    elements :math:`dj, d(j+1), \ldots, d(j+taps-1)` of the mixed signal. The
    mixed signal is the product of sample :math:`j` of the input with
    :math:`e^{2\pi i (aj + b)}`, where :math:`a` and :math:`b` are set with
    the :attr:`mix_frequency` and :attr:`mix_phase` properties. Note that
    setting :attr:`mix_frequency` is somewhat expensive as it has to update an
    array on the device.

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
                    alignment=4,
                ),
            ),
            np.uint8,
        )
        self.slots["out"] = accel.IOSlot((self.out_samples,), np.complex64)
        self.slots["weights"] = accel.IOSlot((template.taps,), np.float32)
        self._mix_lookup = accel.DeviceArray(
            template.context, (template._segments, template._segment_samples), np.complex64
        )
        self._mix_lookup_host = self._mix_lookup.empty_like()
        self.mix_frequency = 0.0  # Specify in cycles per sample
        self.mix_phase = 0.0  # Specify in fractions of a cycle (0-1)

    @property
    def mix_frequency(self) -> float:
        """Mixer frequency in cycles per ADC sample."""
        return self._mix_frequency

    @mix_frequency.setter
    def mix_frequency(self, frequency: float) -> None:
        self._mix_frequency = frequency
        major = self.template.wgs * self.template._segment_samples
        sample_indices = (np.arange(self._mix_lookup_host.shape[0]) * major)[:, np.newaxis] + (
            np.arange(self.template._segment_samples)[np.newaxis, :]
        )
        angles = 2 * np.pi * frequency * sample_indices
        self._mix_lookup_host[:] = np.cos(angles) + 1j * np.sin(angles)
        self._mix_lookup.set(self.command_queue, self._mix_lookup_host)

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        weights_buffer = self.buffer("weights")
        groups = accel.divup(out_buffer.shape[0], self.template._group_out_size)
        # TODO: set up the offsets

        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                in_buffer.buffer,
                weights_buffer.buffer,
                np.int32(0),  # out_offset
                np.int32(0),  # in_offset_words
                np.int32(out_buffer.shape[0]),  # out_size
                np.int32(accel.divup(in_buffer.shape[0], 4)),  # in_size_words
                np.float64(self.mix_frequency),  # mix_scale
                np.float64(self.mix_phase),  # mix_bias
                self._mix_lookup.buffer,
            ],
            global_size=(groups * self.template.wgs, 1, 1),
            local_size=(self.template.wgs, 1, 1),
        )
