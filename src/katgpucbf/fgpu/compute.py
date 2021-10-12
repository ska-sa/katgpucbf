################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

from typing import List, Sequence, Tuple

from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from .. import N_POLS
from . import SAMPLE_BITS, pfb, postproc


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
    """

    def __init__(self, context: AbstractContext, taps: int) -> None:
        self.context = context
        self.taps = taps
        self.pfb_fir = pfb.PFBFIRTemplate(context, taps)
        self.postproc = postproc.PostprocTemplate(context)

    def instantiate(
        self, command_queue: AbstractCommandQueue, samples: int, spectra: int, spectra_per_heap: int, channels: int
    ) -> "Compute":  # We have to put the return type in quotes because we haven't declared the `Compute` class yet.
        """Generate a :class:`Compute` object based on the template."""
        return Compute(self, command_queue, samples, spectra, spectra_per_heap, channels)


class Compute(accel.OperationSequence):
    """The DSP processing pipeline handling F-engine operation.

    The actual running of this operation isn't done through the :meth:`_run`
    method or by calling it directly, if you're familiar with the usual method
    of `composing operations`_. Katfgpu's compute is streaming rather than
    batched, i.e. we have to coordinate the receiving of new data and the
    transmission of processed data along with the actual processing operation.

    Currently, no internal checks for consistency of the parameters are
    performed. The following constraints are assumed, Bad Things(TM) may happen
    if they aren't followed:

    - spectra_per_heap <= spectra - i.e. a chunk of data must be enough to send out at
      least one heap.
    - spectra % spectra_per_heap == 0
    - samples >= spectra*channels*2 + taps*channels*2 - 1. The factor of 2 is
      because the PFB input is real, 2*channels samples are needed for each
      output spectrum. The "extra samples" are to ensure continuity in the PFB-
      FIR from one chunk to the next.
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
        Number of samples that will be processed each time the operation is run.
    spectra
        Number of spectra that we will get from each chunk of samples.
    spectra_per_heap
        Number of spectra to send out per heap.
    channels
        Number of channels into which the input data will be decomposed.
    """

    def __init__(
        self,
        template: ComputeTemplate,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
        spectra_per_heap: int,
        channels: int,
    ) -> None:
        self.sample_bits = SAMPLE_BITS
        self.template = template
        self.channels = channels
        self.samples = samples
        self.spectra = spectra
        self.spectra_per_heap = spectra_per_heap

        # PFB-FIR and FFT each happen for each polarisation.
        self.pfb_fir = [
            template.pfb_fir.instantiate(command_queue, samples, spectra, channels) for pol in range(N_POLS)
        ]
        self.fft = [pfb.FFT(command_queue, spectra, channels) for pol in range(N_POLS)]

        # Postproc is single though because it involves the corner turn which
        # combines the two pols.
        self.postproc = template.postproc.instantiate(command_queue, spectra, spectra_per_heap, channels)

        operations: List[Tuple[str, accel.Operation]] = []
        for pol in range(N_POLS):
            operations.append((f"pfb_fir{pol}", self.pfb_fir[pol]))
        for pol in range(N_POLS):
            operations.append((f"fft{pol}", self.fft[pol]))
        operations.append(("postproc", self.postproc))

        compounds = {
            # fft0:work and fft1:work are just scratchpad memory. Since the FFTs
            # are run sequentially they won't interfere with each other, fft0 is
            # finished by the time fft1 starts.
            "fft_work": [f"fft{pol}:work" for pol in range(N_POLS)],
            # We expect the weights on the PFB-FIR taps to be the same for both
            # pols so they can share memory.
            "weights": [f"pfb_fir{pol}:weights" for pol in range(N_POLS)],
            "out": ["postproc:out"],
            "fine_delay": ["postproc:fine_delay"],
            "phase": ["postproc:phase"],
        }
        for pol in range(N_POLS):
            compounds[f"in{pol}"] = [f"pfb_fir{pol}:in"]
            compounds[f"fft_in{pol}"] = [f"pfb_fir{pol}:out", f"fft{pol}:in"]
            compounds[f"fft_out{pol}"] = [f"fft{pol}:out", f"postproc:in{pol}"]
        super().__init__(command_queue, operations, compounds)

    def run_frontend(self, samples: Sequence[accel.DeviceArray], in_offset: int, out_offset: int, spectra: int) -> None:
        """Run the PFB-FIR on the received samples.

        Coarse delay also seems to be involved.

        Parameters
        ----------
        samples
            A pair of device arrays containing the samples, one for each pol.
        in_offset
            TODO: Figure out what this is. Something to do with coarse delay I think.
        out_offset
            TODO: Figure out what this is. Need to refer to the actual pfb_fir kernel.
        spectra
            How many spectra worth of samples to push through the PFB-FIR.
        """
        if len(samples) != N_POLS:
            raise ValueError(f"samples must contain {N_POLS} elements")
        for pol in range(N_POLS):
            self.bind(**{f"in{pol}": samples[pol]})
        # TODO: only bind relevant slots for frontend
        self.ensure_all_bound()
        for pol in range(N_POLS):
            # TODO: could run these in parallel, but that would require two
            # command queues.
            self.pfb_fir[pol].in_offset = in_offset
            self.pfb_fir[pol].out_offset = out_offset
            self.pfb_fir[pol].spectra = spectra
            self.pfb_fir[pol]()

    def run_backend(self, out: accel.DeviceArray) -> None:
        """Run the FFT and postproc on the data which has been PFB-FIRed.

        Postproc incorporates fine-delay, requantisation and corner-turning.

        Parameters
        ----------
        out
            Destination for the processed data.
        """
        self.bind(out=out)
        # TODO: only bind relevant slots for backend
        self.ensure_all_bound()
        for fft in self.fft:
            fft()
        self.postproc()

    @property
    def quant_gain(self) -> float:  # noqa: D401
        """Scaling factor used for requantisation."""
        return self.postproc.quant_gain

    @quant_gain.setter
    def quant_gain(self, value: float) -> None:
        self.postproc.quant_gain = value
