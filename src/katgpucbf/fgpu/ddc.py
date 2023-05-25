################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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

from importlib import resources
from typing import TypedDict

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import BYTE_BITS, DIG_SAMPLE_BITS
from . import INPUT_CHUNK_PADDING


class _TuningDict(TypedDict):
    wgs: int
    unroll: int


class DDCTemplate:
    """Template for digital down-conversion.

    See :class:`DDC` for a more detailed description of what it does.

    Parameters
    ----------
    context
        The GPU context that we'll operate in
    taps
        Number of taps in the FIR filter
    subsampling
        Fraction of samples to retain after filtering

    Raises
    ------
    ValueError
        If `taps` is not a multiple of `subsampling`
    """

    def __init__(
        self, context: AbstractContext, taps: int, subsampling: int, tuning: _TuningDict | None = None
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be positive")
        if subsampling <= 0:
            raise ValueError("subsampling must be positive")
        if taps % subsampling != 0:
            raise ValueError("taps must be a multiple of subsampling")
        if tuning is None:
            tuning = self.autotune(context, taps, subsampling)
        self.context = context
        self.wgs = tuning["wgs"]
        self.unroll = tuning["unroll"]
        self.taps = taps
        self.subsampling = subsampling
        self.input_sample_bits = DIG_SAMPLE_BITS

        # Sanity check the tuning parameters
        # TODO: re-do for NGC-980
        assert self.subsampling * self.unroll * self.input_sample_bits % 32 == 0

        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/ddc.mako",
                {
                    "wgs": self.wgs,
                    "unroll": self.unroll,
                    "taps": taps,
                    "subsampling": subsampling,
                    "sample_bits": DIG_SAMPLE_BITS,
                },
                extra_dirs=[str(resource_dir)],
            )
        self.kernel = program.get_kernel("ddc")

    def autotune(self, context: AbstractContext, taps: int, subsampling: int) -> _TuningDict:
        """Determine tuning parameters.

        .. todo::

           Actually do autotuning instead of using fixed logic.
        """
        wgs = 32
        unroll = 8
        while unroll * subsampling * DIG_SAMPLE_BITS % 32:
            unroll *= 2
        return {"wgs": wgs, "unroll": unroll}

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

    **in** : samples * DIG_SAMPLE_BITS // BYTE_BITS, uint8
        Input digitiser samples in a big chunk.
    **out** : out_samples, complex64
        Filtered and subsampled output data

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
        self.out_samples = accel.divup(samples - template.taps + 1, template.subsampling)
        # The actual padding requirement is just 4-byte alignment, but using
        # INPUT_CHUNK_PADDING gives consistent padding to wideband pipelines.
        in_bytes = samples * DIG_SAMPLE_BITS // BYTE_BITS
        self.slots["in"] = accel.IOSlot(
            (accel.Dimension(in_bytes, min_padded_size=in_bytes + INPUT_CHUNK_PADDING),),
            np.uint8,
        )
        self.slots["out"] = accel.IOSlot((self.out_samples,), np.complex64)
        self._weights = accel.DeviceArray(template.context, (template.taps,), np.complex64)
        self._weights_host = self._weights.empty_like()
        self._mix_lookup = accel.DeviceArray(template.context, (template.unroll,), np.complex64)
        self._mix_lookup_host = self._mix_lookup.empty_like()
        self._mix_frequency = 0.0  # Specify in cycles per sample
        self.mix_phase = 0.0  # Specify in fractions of a cycle (0-1)

    def configure(self, mix_frequency: float, weights: np.ndarray) -> None:
        """Set the mixer frequency and filter weights."""
        assert weights.shape == self._weights_host.shape
        self._mix_frequency = mix_frequency
        self._weights_host[:] = weights * np.exp(2j * np.pi * mix_frequency * np.arange(len(weights)))
        self._weights.set(self.command_queue, self._weights_host)

        self._mix_lookup_host[:] = np.exp(
            2j * np.pi * mix_frequency * self.template.subsampling * np.arange(self.template.unroll)
        )
        self._mix_lookup.set(self.command_queue, self._mix_lookup_host)

    @property
    def mix_frequency(self) -> float:
        """Mixer frequency in cycles per ADC sample."""
        return self._mix_frequency

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        groups = accel.divup(self.out_samples, self.template.wgs * self.template.unroll)

        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                in_buffer.buffer,
                self._weights.buffer,
                np.int32(out_buffer.shape[0]),  # out_size
                np.int32(accel.divup(in_buffer.shape[0], 4)),  # in_size_words
                np.float64(self.mix_frequency),  # mix_scale
                np.float64(self.mix_phase),  # mix_bias
                self._mix_lookup.buffer,
            ],
            global_size=(groups * self.template.wgs, 1, 1),
            local_size=(self.template.wgs, 1, 1),
        )
