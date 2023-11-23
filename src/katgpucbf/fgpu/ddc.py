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

import math
from importlib import resources
from typing import Callable, TypedDict, cast

import numpy as np
from katsdpsigproc import accel, tune
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import BYTE_BITS, N_POLS
from . import INPUT_CHUNK_PADDING

_SAMPLE_WORD_SIZE = 4


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
    input_sample_bits
        Bits per input sample
    """

    autotune_version = 2

    def __init__(
        self,
        context: AbstractContext,
        taps: int,
        subsampling: int,
        input_sample_bits: int,
        tuning: _TuningDict | None = None,
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be positive")
        if subsampling <= 0:
            raise ValueError("subsampling must be positive")
        if not 1 <= input_sample_bits <= 32:
            raise ValueError("input_sample_bits must be in the range [1, 32]")
        if tuning is None:
            tuning = self.autotune(context, taps, subsampling, input_sample_bits)
        self.context = context
        self.wgs = tuning["wgs"]
        self.unroll = tuning["unroll"]
        self.taps = taps
        self.subsampling = subsampling
        self.input_sample_bits = input_sample_bits

        # Sanity check the tuning parameters
        ua = self.unroll_align(subsampling, input_sample_bits)
        if self.unroll % ua != 0:
            raise ValueError(f"unroll must be a multiple of {ua}")

        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(
                context,
                "kernels/ddc.mako",
                {
                    "wgs": self.wgs,
                    "unroll": self.unroll,
                    "taps": taps,
                    "subsampling": subsampling,
                    "input_sample_bits": input_sample_bits,
                },
                extra_dirs=[str(resource_dir), str(resource_dir.parent)],
            )
        self.kernel = program.get_kernel("ddc")

    @staticmethod
    def unroll_align(subsampling: int, input_sample_bits: int) -> int:
        """Determine the factor that must divide into `unroll`."""
        return 32 // math.gcd(32, subsampling * input_sample_bits)

    @classmethod
    @tune.autotuner(test={"wgs": 32, "unroll": 16})
    def autotune(cls, context: AbstractContext, taps: int, subsampling: int, input_sample_bits: int) -> _TuningDict:
        """Determine tuning parameters."""
        queue = context.create_tuning_command_queue()
        in_samples = 16 * 1024 * 1024
        # Create one just to generate the correct padding for the inputs
        ua = cls.unroll_align(subsampling, input_sample_bits)
        dummy_fn = cls(context, taps, subsampling, input_sample_bits, tuning={"wgs": 32, "unroll": ua}).instantiate(
            queue, in_samples, N_POLS
        )
        dummy_fn.ensure_all_bound()
        in_data = dummy_fn.buffer("in")
        out_data = dummy_fn.buffer("out")
        in_data.zero(queue)

        def generate(wgs: int, unroll: int) -> Callable[[int], float] | None:
            # Making the context current allows `fn._weights` and
            # `fn._weights_host` to be garbage-collected correctly when `fn`
            # goes out of scope.
            with context:
                fn = cls(
                    context, taps, subsampling, input_sample_bits, tuning={"wgs": wgs, "unroll": unroll}
                ).instantiate(queue, in_samples, N_POLS)
                fn.bind(**{"in": in_data, "out": out_data})
                return tune.make_measure(queue, fn)

        return cast(_TuningDict, tune.autotune(generate, wgs=[32, 64], unroll=range(max(8, ua), 17, ua)))

    def instantiate(self, command_queue: AbstractCommandQueue, samples: int, n_pols: int) -> "DDC":
        """Generate a :class:`DDC` object based on the template."""
        return DDC(self, command_queue, samples, n_pols)


class DDC(accel.Operation):
    r"""Operation implementating :class:`DDCTemplate`.

    The kernel takes 10-bit integer inputs (real) and produces 32-bit
    floating-point outputs (complex). The user provides a real-valued FIR
    baseband filter and a mixer frequency for translating the signal from
    the desired band to baseband.

    Element j of the output contains the dot product of **weights** with
    elements :math:`dj, d(j+1), \ldots, d(j+taps-1)` of the mixed signal. The
    mixed signal is the product of sample :math:`j` of the input with
    :math:`e^{2\pi i (aj + b)}`, where :math:`a` is the `mix_frequency`
    argument to :meth:`configure` and :math:`b` is the (settable)
    :attr:`mix_phase` property.

    .. rubric:: Slots

    **in** : samples * input_sample_bits // BYTE_BITS, uint8
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
        Number of input samples to store, per polarisation
    n_pols
        Number of polarisations
    """

    def __init__(self, template: DDCTemplate, command_queue: AbstractCommandQueue, samples: int, n_pols: int) -> None:
        super().__init__(command_queue)
        if samples % BYTE_BITS != 0:
            raise ValueError(f"samples must be a multiple of {BYTE_BITS}")
        if samples < template.taps:
            raise ValueError("samples must be at least the number of filter taps")
        self.template = template
        self.samples = samples
        self.out_samples = accel.divup(samples - template.taps + 1, template.subsampling)
        # The actual padding requirement is just sample_word alignment, but
        # using INPUT_CHUNK_PADDING gives consistent padding to wideband
        # pipelines.
        in_bytes = samples * template.input_sample_bits // BYTE_BITS
        self.slots["in"] = accel.IOSlot(
            (
                n_pols,
                accel.Dimension(
                    in_bytes,
                    min_padded_size=in_bytes + INPUT_CHUNK_PADDING,
                    alignment=_SAMPLE_WORD_SIZE,
                ),
            ),
            np.uint8,
        )
        self.slots["out"] = accel.IOSlot((n_pols, self.out_samples), np.complex64)
        self._weights = accel.DeviceArray(template.context, (template.taps,), np.complex64)
        self._weights_host = self._weights.empty_like()
        self._mix_scale = 0  # Specify in cycles per output sample, times 2**32
        self.mix_phase = 0.0  # Specify in fractions of a cycle (0-1)

    def configure(self, mix_frequency: float, weights: np.ndarray) -> None:
        """Set the mixer frequency and filter weights.

        This is a somewhat expensive operation, as it computes lookup tables
        and transfers them to the device synchronously. It is only intended
        to be used at startup rather than continuously.

        .. note::

           The provided `mix_frequency` is quantised. The actual mixer
           frequency can be retrieved from the :attr:`mix_frequency`
           property.
        """
        assert weights.shape == self._weights_host.shape
        # Quantise the mixer frequency so that cycles per *output* sample are
        # represented in fixed point with 32 fractional bits.
        self._mix_scale = round(mix_frequency * self.template.subsampling * 2**32)
        self._weights_host[:] = weights * np.exp(2j * np.pi * self.mix_frequency * np.arange(len(weights)))
        self._weights.set(self.command_queue, self._weights_host)

    @property
    def mix_frequency(self) -> float:
        """Mixer frequency in cycles per ADC sample."""
        return self._mix_scale / self.template.subsampling / 2**32

    def _run(self) -> None:
        in_buffer = self.buffer("in")
        out_buffer = self.buffer("out")
        groups = accel.divup(self.out_samples, self.template.wgs * self.template.unroll)

        mix_scale = self._mix_scale % 2**32
        mix_bias = round(self.mix_phase * 2**32) % 2**32
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out_buffer.buffer,
                in_buffer.buffer,
                self._weights.buffer,
                np.uint32(out_buffer.padded_shape[1]),  # out_stride
                np.uint32(in_buffer.padded_shape[1] // _SAMPLE_WORD_SIZE),  # in_stride in sample_words
                np.uint32(out_buffer.shape[1]),  # out_size
                np.uint32(accel.divup(in_buffer.shape[1], _SAMPLE_WORD_SIZE)),  # in_size_words
                np.uint32(mix_scale),
                np.uint32(mix_bias),
            ],
            global_size=(groups * self.template.wgs, in_buffer.shape[0], 1),
            local_size=(self.template.wgs, 1, 1),
        )
