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

"""The :class:`Compute` class specifies the sequence of operations on the GPU.

Allocations of memory for input, intermediate and output are also handled here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from katsdpsigproc import accel, fft
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import N_POLS
from . import ddc, pfb, postproc


@dataclass
class NarrowbandConfig:
    """Configuration for a narrowband stream."""

    #: Factor by which bandwidth is reduced
    decimation: int
    #: Mixer frequency, in cycles per ADC sample
    mix_frequency: float
    #: Downconversion filter weights (float)
    weights: np.ndarray


class ComputeTemplate:
    """Template for the channelisation operation sequence.

    The reason for doing things this way can be read in the relevant
    `katsdpsigproc docs`_.

    .. _katsdpsigproc docs: https://katsdpsigproc.readthedocs.io/en/latest/user/operations.html#operation-templates

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    taps
        The number of taps that you want the resulting PFB-FIRs to have.
    channels
        Number of output channels into which the input data will be decomposed.
    dig_sample_bits
        Number of bits per digitiser sample.
    out_bits
        Number of bits per output real component.
    narrowband
        Configuration for narrowband operation. If ``None``, wideband is assumed.
    """

    def __init__(
        self,
        context: AbstractContext,
        taps: int,
        channels: int,
        dig_sample_bits: int,
        out_bits: int,
        narrowband: NarrowbandConfig | None,
    ) -> None:
        self.context = context
        self.taps = taps
        self.channels = channels
        self.narrowband = narrowband
        self.unzip_factor = 4 if channels >= 8 else 1
        if narrowband is None:
            self.internal_channels = channels
            self.postproc = postproc.PostprocTemplate(
                context, channels, self.unzip_factor, complex_pfb=False, out_bits=out_bits
            )
            self.pfb_fir = pfb.PFBFIRTemplate(
                context, taps, channels, dig_sample_bits, self.unzip_factor, n_pols=N_POLS
            )
            self.ddc: ddc.DDCTemplate | None = None
        else:
            self.internal_channels = 2 * channels
            self.postproc = postproc.PostprocTemplate(
                context,
                self.internal_channels,
                self.unzip_factor,
                complex_pfb=True,
                out_bits=out_bits,
                out_channels=(channels // 2, 3 * channels // 2),
            )
            self.pfb_fir = pfb.PFBFIRTemplate(
                context, taps, self.internal_channels, 32, self.unzip_factor, complex_input=True, n_pols=N_POLS
            )
            self.ddc = ddc.DDCTemplate(context, len(narrowband.weights), narrowband.decimation, dig_sample_bits)

    def instantiate(
        self,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
        spectra_per_batch: int,
    ) -> "Compute":  # We have to put the return type in quotes because we haven't declared the `Compute` class yet.
        """Generate a :class:`Compute` object based on the template."""
        return Compute(self, command_queue, samples, spectra, spectra_per_batch)


class Compute(accel.OperationSequence):
    """The DSP processing pipeline handling F-engine operation.

    The actual running of this operation isn't done through the :meth:`_run`
    method or by calling it directly, if you're familiar with the usual method
    of `composing operations`_. Fgpu's compute is streaming rather than
    batched, i.e. we have to coordinate the receiving of new data and the
    transmission of processed data along with the actual processing operation.

    Currently, no internal checks for consistency of the parameters are
    performed. The following constraints are assumed, Bad Things(TM) may happen
    if they aren't followed:

    - spectra_per_batch <= spectra - i.e. a chunk of data must be enough to send out at
      least one batch.
    - spectra % spectra_per_batch == 0
    - samples >= output.window (see :class:`.fgpu.output.Output`). An input chunk requires
      at least enough samples to output a single spectrum.
    - samples % 8 == 0

    .. _composing operations: https://katsdpsigproc.readthedocs.io/en/latest/user/operations.html#composing-operations

    Parameters
    ----------
    template
        Template for the channelisation operation sequence.
    command_queue
        The GPU command queue (typically this will be a CUDA Stream) on which
        actual processing operations are to be scheduled.
    samples
        Number of samples in each input chunk (per polarisation), including
        padding samples.
    spectra
        Number of spectra in each output chunk.
    spectra_per_batch
        Number of spectra to send in each output batch.
    """

    def __init__(
        self,
        template: ComputeTemplate,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
        spectra_per_batch: int,
    ) -> None:
        self.template = template
        self.samples = samples
        self.spectra = spectra
        self.spectra_per_batch = spectra_per_batch

        operations: list[tuple[str, accel.Operation]] = []
        # DDC, PFB-FIR and FFT each happen for each polarisation.
        if template.ddc is None:
            # Wideband
            self.ddc: ddc.DDC | None = None
        else:
            # Narrowband
            assert template.narrowband is not None
            if samples % template.ddc.subsampling != 0:
                raise ValueError(f"samples ({samples}) must be a multiple of subsampling ({template.ddc.subsampling})")
            self.ddc = template.ddc.instantiate(command_queue, samples, N_POLS)
            self.ddc.configure(template.narrowband.mix_frequency, template.narrowband.weights)
            operations.append(("ddc", self.ddc))
            samples = self.ddc.out_samples  # Number of samples available to remainder of pipeline
        self.pfb_fir = template.pfb_fir.instantiate(command_queue, samples, spectra)
        fft_shape = (N_POLS, spectra, template.unzip_factor, template.internal_channels // template.unzip_factor)
        fft_template = fft.FftTemplate(
            template.context,
            1,
            fft_shape,
            np.complex64,
            np.complex64,
            fft_shape,
            fft_shape,
        )
        self.fft = fft_template.instantiate(command_queue, fft.FftMode.FORWARD)
        self.postproc = template.postproc.instantiate(command_queue, spectra, spectra_per_batch)

        operations.append(("pfb_fir", self.pfb_fir))
        operations.append(("fft", self.fft))
        operations.append(("postproc", self.postproc))

        compounds = {
            "fft_work": ["fft:work_area"],
            "fft_out": ["fft:dest", "postproc:in"],
            "weights": ["pfb_fir:weights"],
            "out": ["postproc:out"],
            "saturated": ["postproc:saturated"],
            "fine_delay": ["postproc:fine_delay"],
            "phase": ["postproc:phase"],
            "gains": ["postproc:gains"],
        }
        aliases = {}
        if template.ddc is None:
            compounds["in"] = ["pfb_fir:in"]
            compounds["dig_total_power"] = ["pfb_fir:total_power"]
        else:
            compounds["in"] = ["ddc:in"]
            compounds["subsampled"] = ["ddc:out", "pfb_fir:in"]
        # pfb_fir:out is an array of real values (in wideband) while
        # fft:src reinterprets it as an array of complex values. We thus
        # have to make them aliases to view the memory as different
        # types.
        #
        # Since the dimensions aren't linked, it is important that both
        # Operations specify the dimensions as unpadded.
        aliases["fft_in"] = ["pfb_fir:out", "fft:src"]
        super().__init__(command_queue, operations, compounds, aliases)

    def run_ddc(self, samples: accel.DeviceArray, first_sample: int) -> None:
        """Run the narrowband DDC kernel on the received samples.

        Parameters
        ----------
        samples
            A device array containing the samples.
        first_sample
            Timestamp (in samples) of the initial sample. This is used to
            correctly phase the mixer.
        """
        assert self.ddc is not None
        self.bind(**{"in": samples})
        self.ensure_bound("subsampled")
        # Compute the fractional part of first_sample * mix_frequency.
        # Using Fraction avoids the serious rounding errors that would
        # occur using floating point.
        phase = Fraction(self.ddc.mix_frequency) * first_sample
        phase -= round(phase)
        self.ddc.mix_phase = float(phase)
        self.ddc()

    def _run_frontend_common(
        self,
        in_offsets: Sequence[int],
        out_offset: int,
        spectra: int,
    ) -> None:
        """Do common parts of :meth:`run_wideband_frontend` and :meth:`run_narrowband_frontend`."""
        self.ensure_bound("weights")
        self.ensure_bound("fft_in")
        self.pfb_fir.in_offset[:] = in_offsets
        self.pfb_fir.out_offset = out_offset
        self.pfb_fir.spectra = spectra
        self.pfb_fir()

    def run_wideband_frontend(
        self,
        samples: accel.DeviceArray,
        dig_total_power: accel.DeviceArray,
        in_offsets: Sequence[int],
        out_offset: int,
        spectra: int,
    ) -> None:
        """Run the PFB-FIR on the received samples, for a wideband pipeline.

        Parameters
        ----------
        samples
            A device arrays containing the samples
        dig_total_power
            A device array holding digitiser total power for each pol. This
            is not zeroed.
        in_offsets
            Index of first sample in input array to process (one for each pol).
        out_offset
            Index of first sample in output array to write.
        spectra
            How many spectra worth of samples to push through the PFB-FIR.
        """
        assert self.ddc is None
        if samples.shape[0] != N_POLS:
            raise ValueError(f"samples must have size {N_POLS} on the first dimension")
        if len(in_offsets) != N_POLS:
            raise ValueError(f"in_offsets must contain {N_POLS} elements")
        self.bind(**{"in": samples, "dig_total_power": dig_total_power})
        self._run_frontend_common(in_offsets, out_offset, spectra)

    def run_narrowband_frontend(
        self,
        in_offsets: Sequence[int],
        out_offset: int,
        spectra: int,
    ) -> None:
        """Run the PFB-FIR on the received samples, for a narrowband pipeline.

        Parameters
        ----------
        in_offsets
            Index of first sample in input array to process (one for each pol).
        out_offset
            Index of first sample in output array to write.
        spectra
            How many spectra worth of samples to push through the PFB-FIR.
        """
        assert self.ddc is not None
        self.ensure_bound("subsampled")
        self._run_frontend_common(in_offsets, out_offset, spectra)

    def run_backend(self, out: accel.DeviceArray, saturated: accel.DeviceArray) -> None:
        """Run the FFT and postproc on the data which has been PFB-FIRed.

        Postproc incorporates fine-delay, requantisation and corner-turning.

        Parameters
        ----------
        out
            Destination for the processed data.
        """
        self.bind(out=out, saturated=saturated)
        # Note: we only actually need to bind the slots specific to the
        # backend, but there are quite a few to keep track of, and by the
        # time the backend is run the frontend slots should all be bound
        # anyway.
        self.ensure_all_bound()
        self.fft()
        self.postproc()
