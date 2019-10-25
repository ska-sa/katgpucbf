import pkg_resources
import numpy as np
from katsdpsigproc import accel


class Decode10BitTemplate:
    def __init__(self, context):
        self.wgs = 128
        program = accel.build(
            context, 'kernels/decode.mako',
            {'wgs': self.wgs},
            extra_dirs=[pkg_resources.resource_filename(__name__, '')])
        self.kernel = program.get_kernel('decode_10bit')

    def instantiate(self, command_queue, samples):
        return Decode10Bit(self, command_queue, samples)


class Decode10Bit(accel.Operation):
    def __init__(self, template, command_queue, samples):
        if samples % 8 != 0:
            raise ValueError('samples must be a multiple of 8')
        super().__init__(command_queue)
        padded_samples = accel.roundup(samples, 16 * template.wgs)
        in_dim = accel.Dimension(samples * 10 // 8, min_padded_size=padded_samples * 10 // 8)
        out_dim = accel.Dimension(samples, template.wgs)
        self.template = template
        self.samples = samples
        self.slots['in'] = accel.IOSlot((in_dim,), np.uint8)
        self.slots['out'] = accel.IOSlot((out_dim,), np.int16)

    def _run(self):
        in_buf = self.buffer('in')
        out_buf = self.buffer('out')
        work_items = accel.divup(self.samples, 16)
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                self.buffer('out').buffer,
                self.buffer('in').buffer
            ],
            global_size=(accel.roundup(work_items, self.template.wgs),),
            local_size=(self.template.wgs,)
        )
