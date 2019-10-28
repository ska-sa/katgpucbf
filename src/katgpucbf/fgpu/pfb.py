import threading

import pkg_resources
import numpy as np
import skcuda.fft
import skcuda.cufft
from katsdpsigproc import accel


class PFBFIRTemplate:
    def __init__(self, context, taps):
        if taps <= 0:
            raise ValueError('taps must be at least 1')
        self.wgs = 128
        self.taps = taps
        program = accel.build(
            context, 'kernels/pfb_fir.mako',
            {'wgs': self.wgs, 'taps': self.taps},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('pfb_fir')

    def instantiate(self, command_queue, n, step):
        return PFBFIR(self, command_queue, n, step)


class PFBFIR(accel.Operation):
    def __init__(self, template, command_queue, n, step):
        super().__init__(command_queue)
        # n is number of *output* samples
        if step % template.wgs != 0:
            raise ValueError(f'step must be a multiple of {template.wgs}')
        if n % step != 0:
            raise ValueError(f'n must be a multiple of step')
        self.template = template
        self.n = n
        self.step = step
        self.slots['in'] = accel.IOSlot((n + step * (template.taps - 1),), np.int16)
        self.slots['out'] = accel.IOSlot(
            (n // step, accel.Dimension(step, exact=True)),
            np.float32)
        self.slots['weights'] = accel.IOSlot((template.taps,), np.float32)

    def _run(self):
        # Aim for 256K workitems i.e. step * (n / stepy) == 256K
        stepy = accel.roundup(accel.divup(self.step * self.n, 256 * 1024), self.step)
        groupsx = self.step // self.template.wgs
        groupsy = accel.divup(self.n, stepy)
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                self.buffer('out').buffer,
                self.buffer('in').buffer,
                self.buffer('weights').buffer,
                np.int32(self.n), np.int32(self.step), np.int32(stepy)
            ],
            global_size=(groupsx * self.template.wgs, groupsy),
            local_size=(self.template.wgs, 1)
        )


class FFT(accel.Operation):
    def __init__(self, command_queue, spectra, channels):
        super().__init__(command_queue)
        self.spectra = spectra
        self.channels = channels
        with command_queue.context:
            self.plan = skcuda.fft.Plan(
                2 * channels, np.float32, np.complex64, spectra,
                stream=command_queue._pycuda_stream,
                inembed=np.array([2 * channels], np.int32),
                idist=2 * channels,
                onembed=np.array([channels + 1], np.int32),
                odist=channels + 1,
                auto_allocate=False)
            work_size = skcuda.cufft.cufftGetSize(self.plan.handle)
        self.slots['in'] = accel.IOSlot((spectra, accel.Dimension(2 * channels, exact=True)),
                                        np.float32)
        self.slots['out'] = accel.IOSlot((spectra, accel.Dimension(channels + 1, exact=True)),
                                         np.complex64)
        self.slots['work'] = accel.IOSlot((work_size,), np.uint8)

    def _run(self):
        with self.command_queue.context:
            self.plan.set_work_area(self.buffer('work').buffer)
            skcuda.fft.fft(self.buffer('in').buffer,
                           self.buffer('out').buffer,
                           self.plan)
