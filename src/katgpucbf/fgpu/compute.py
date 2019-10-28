"""Combines all the device operations"""

from katsdpsigproc import accel

from . import pfb, postproc

class ComputeTemplate:
    def __init__(self, context, taps):
        self.context = context
        self.taps = taps
        self.pfb_fir = pfb.PFBFIRTemplate(context, taps)
        self.postproc = postproc.PostprocTemplate(context)

    def instantiate(self, command_queue, samples, spectra, acc_len, channels):
        return Compute(self, command_queue, samples, spectra, acc_len, channels)


class Compute(accel.OperationSequence):
    def __init__(self, template, command_queue, samples, spectra, acc_len, channels):
        pols = 2
        self.template = template
        self.channels = channels
        self.samples = samples
        self.spectra = spectra
        self.acc_len = acc_len
        self.pfb_fir = [
            template.pfb_fir.instantiate(command_queue, samples, spectra, channels)
            for pol in range(pols)
        ]
        self.fft = [
            pfb.FFT(command_queue, spectra, channels)
            for pol in range(pols)
        ]
        self.postproc = template.postproc.instantiate(command_queue, spectra, acc_len, channels)

        operations = []
        for pol in range(pols):
            operations.append((f'pfb_fir{pol}', self.pfb_fir[pol]))
        for pol in range(pols):
            operations.append((f'fft{pol}', self.fft[pol]))
        operations.append(('postproc', self.postproc))

        compounds = {
            'weights': [f'pfb_fir{pol}:work' for pol in range(pols)],
            'fft_work': [f'fft{pol}:work' for pol in range(pols)],
            'out': ['postproc:out'],
            'fine_delay': ['postproc:fine_delay']
        }
        for pol in range(pols):
            compounds[f'in{pol}'] = [f'pfb_fir{pol}:in']
            compounds[f'fft_in{pol}'] = [f'pfb_fir{pol}:out', f'fft{pol}:in']
            compounds[f'fft_out{pol}'] = [f'fft{pol}:out', 'postproc:in{pol}']
        super().__init__(command_queue, operations, compounds)

    def run_frontend(self, samples, in_offset, out_offset, spectra):
        pols = len(self.pfb_fir)
        if len(samples) != pols:
            raise ValueError(f'samples must contain {pols} elements')
        for pol in range(pols):
            self.bind(**{f'in{pol}': samples[pol])
            # TODO: could run these in parallel, but that would require two
            # command queues.
            self.pfb_fir[pol].in_offset = in_offset
            self.pfb_fir[pol].out_offset = out_offset
            self.pfb_fir[pol].spectra = spectra
            self.pfb_fir[pol]()

    def run_backend(self, out):
        self.bind(out=out)
        for fft in self.fft:
            fft()
        self.postproc()
