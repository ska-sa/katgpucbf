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

    def instantiate(self, command_queue, samples, spectra, channels):
        return PFBFIR(self, command_queue, samples, spectra, channels)


class PFBFIR(accel.Operation):
    def __init__(self, template, command_queue, samples, spectra, channels):
        super().__init__(command_queue)
        if samples % 8 != 0:
            raise ValueError('samples must be a multiple of 8')
        if (2 * channels) % template.wgs != 0:
            raise ValueError(f'2*channels must be a multiple of {template.wgs}')
        self.template = template
        self.samples = samples
        self.spectra = spectra        # Can be changed (TODO: documentation)
        self.channels = channels
        self.slots['in'] = accel.IOSlot((samples * 10 // 8,), np.uint8)
        self.slots['out'] = accel.IOSlot(
            (spectra, accel.Dimension(2 * channels, exact=True)),
            np.float32)
        self.slots['weights'] = accel.IOSlot((template.taps,), np.float32)
        self.in_offset = 0            # TODO: docs
        self.out_offset = 0           # TODO: docs

    def _run(self):
        if self.spectra == 0:
            return
        step = 2 * self.channels
        in_buffer = self.buffer('in')
        out_buffer = self.buffer('out')
        if self.in_offset + step * (self.spectra + self.template.taps - 1) > in_buffer.shape[0]:
            raise IndexError('Input buffer does not contain sufficient samples')
        if self.out_offset + self.spectra > out_buffer.shape[0]:
            raise IndexError('Output buffer does not contain sufficient spectra')
        # Aim for 256K workitems i.e. step * (out_n / stepy) == 256K
        out_n = step * self.spectra
        stepy = accel.roundup(accel.divup(step * out_n, 256 * 1024), step)
        groupsx = step // self.template.wgs
        groupsy = accel.divup(out_n, stepy)
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                self.buffer('out').buffer,
                self.buffer('in').buffer,
                self.buffer('weights').buffer,
                np.int32(out_n), np.int32(step), np.int32(stepy),
                np.int32(self.in_offset), np.int32(self.out_offset * step)
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
