"""Combines all the device operations"""

from typing import List, Tuple, Sequence

from katsdpsigproc import accel

from . import pfb, postproc
from .types import AbstractContext, AbstractCommandQueue


class ComputeTemplate:
    def __init__(self, context: AbstractContext, taps: int) -> None:
        self.context = context
        self.taps = taps
        self.pfb_fir = pfb.PFBFIRTemplate(context, taps)
        self.postproc = postproc.PostprocTemplate(context)

    def instantiate(self, command_queue: AbstractCommandQueue,
                    samples: int, spectra: int, acc_len: int, channels: int) -> 'Compute':
        return Compute(self, command_queue, samples, spectra, acc_len, channels)


class Compute(accel.OperationSequence):
    def __init__(self, template: ComputeTemplate,
                 command_queue: AbstractCommandQueue,
                 samples: int, spectra: int, acc_len: int, channels: int) -> None:
        self.pols = 2
        self.sample_bits = 10
        self.template = template
        self.channels = channels
        self.samples = samples
        self.spectra = spectra
        self.acc_len = acc_len
        self.pfb_fir = [
            template.pfb_fir.instantiate(command_queue, samples, spectra, channels)
            for pol in range(self.pols)
        ]
        self.fft = [
            pfb.FFT(command_queue, spectra, channels)
            for pol in range(self.pols)
        ]
        self.postproc = template.postproc.instantiate(command_queue, spectra, acc_len, channels)

        operations: List[Tuple[str, accel.Operation]] = []
        for pol in range(self.pols):
            operations.append((f'pfb_fir{pol}', self.pfb_fir[pol]))
        for pol in range(self.pols):
            operations.append((f'fft{pol}', self.fft[pol]))
        operations.append(('postproc', self.postproc))

        compounds = {
            'fft_work': [f'fft{pol}:work' for pol in range(self.pols)],
            'weights': [f'pfb_fir{pol}:weights' for pol in range(self.pols)],
            'out': ['postproc:out'],
            'fine_delay': ['postproc:fine_delay']
        }
        for pol in range(self.pols):
            compounds[f'in{pol}'] = [f'pfb_fir{pol}:in']
            compounds[f'fft_in{pol}'] = [f'pfb_fir{pol}:out', f'fft{pol}:in']
            compounds[f'fft_out{pol}'] = [f'fft{pol}:out', f'postproc:in{pol}']
        super().__init__(command_queue, operations, compounds)

    def run_frontend(self, samples: Sequence[accel.DeviceArray],
                     in_offset: int, out_offset: int, spectra: int) -> None:
        if len(samples) != self.pols:
            raise ValueError(f'samples must contain {self.pols} elements')
        for pol in range(self.pols):
            self.bind(**{f'in{pol}': samples[pol]})
        # TODO: only bind relevant slots for frontend
        self.ensure_all_bound()
        for pol in range(self.pols):
            # TODO: could run these in parallel, but that would require two
            # command queues.
            self.pfb_fir[pol].in_offset = in_offset
            self.pfb_fir[pol].out_offset = out_offset
            self.pfb_fir[pol].spectra = spectra
            self.pfb_fir[pol]()

    def run_backend(self, out: accel.DeviceArray) -> None:
        self.bind(out=out)
        # TODO: only bind relevant slots for backend
        self.ensure_all_bound()
        for fft in self.fft:
            fft()
        self.postproc()

    @property
    def quant_scale(self) -> float:
        return self.postproc.quant_scale

    @quant_scale.setter
    def quant_scale(self, value: float) -> None:
        self.postproc.quant_scale = value
