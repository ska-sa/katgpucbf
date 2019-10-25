import pkg_resources
import numpy as np
from katsdpsigproc import accel


class PostprocTemplate:
    def __init__(self, context):
        self.block = 32
        self.vtx = 1
        self.vty = 1
        program = accel.build(
            context, 'kernels/postproc.mako',
            {'block': self.block, 'vtx': self.vtx, 'vty': self.vty},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('postproc')

    def instantiate(self, command_queue, channels, spectra, acc_len):
        return Postproc(self, command_queue, channels, spectra, acc_len)


class Postproc(accel.Operation):
    def __init__(self, template, command_queue, channels, spectra, acc_len):
        super().__init__(command_queue)
        if spectra % acc_len != 0:
            raise ValueError('spectra must be a multiple of acc_len')
        block_x = template.block * template.vtx
        block_y = template.block * template.vty
        if channels % block_x != 0:
            raise ValueError(f'channels must be a multiple of {block_x}')
        if acc_len % block_y != 0:
            raise ValueError(f'acc_len must be a multiple of {block_y}')
        self.template = template
        self.channels = channels
        self.spectra = spectra
        self.acc_len = acc_len
        _2 = accel.Dimension(2, exact=True)
        in_shape = (accel.Dimension(spectra), accel.Dimension(channels + 1))
        self.slots['in0'] = accel.IOSlot(in_shape, np.complex64)
        self.slots['in1'] = accel.IOSlot(in_shape, np.complex64)
        self.slots['out'] = accel.IOSlot((spectra // acc_len, channels, acc_len, _2, _2), np.int8)
        self.slots['fine_delay'] = accel.IOSlot((spectra,), np.float32)
        self.quant_scale = 1.0

    def _run(self):
        block_x = self.template.block * self.template.vtx
        block_y = self.template.block * self.template.vty
        groups_x = self.channels // block_x
        groups_y = self.acc_len // block_y
        groups_z = self.spectra // self.acc_len
        out = self.buffer('out')
        in0 = self.buffer('in0')
        in1 = self.buffer('in1')
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out.buffer,
                in0.buffer,
                in1.buffer,
                self.buffer('fine_delay').buffer,
                np.int32(out.padded_shape[1] * out.padded_shape[2]),
                np.int32(out.padded_shape[2]),
                np.int32(in0.padded_shape[1]),
                np.int32(self.acc_len),
                np.float32(1 / self.channels),
                np.float32(self.quant_scale)
            ],
            global_size=(self.template.block * groups_x,
                         self.template.block * groups_y,
                         groups_z),
            local_size=(self.template.block, self.template.block, 1)
        )
