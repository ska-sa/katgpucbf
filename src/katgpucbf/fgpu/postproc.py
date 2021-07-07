"""Postproc module.

These classes handle the operation of the GPU in performing the fine-delay,
requantisation and corner-turn through a mako-templated kernel.

.. todo::

  Phase is currently added from the bottom of the band. Needs to be added from
  the centre. I haven't implemented that properly yet because I haven't thought
  of an elegant and efficient way to do it (i.e. expensive calculations on the
  host vs lots of wasted calculations by each thread on the device). This note
  is here so that I don't forget about this.
"""


import numpy as np
import pkg_resources
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext


class PostprocTemplate:
    """Template for the postproc operation.

    Parameters
    ----------
    context: AbstractContext
        The GPU context that we'll operate in.
    """

    def __init__(self, context: AbstractContext) -> None:
        self.block = 32
        self.vtx = 1
        self.vty = 1
        program = accel.build(
            context,
            "kernels/postproc.mako",
            {"block": self.block, "vtx": self.vtx, "vty": self.vty},
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        self.kernel = program.get_kernel("postproc")

    def instantiate(self, command_queue: AbstractCommandQueue, spectra: int, acc_len: int, channels: int) -> "Postproc":
        """Generate a :class:`Postproc` object based on this template."""
        return Postproc(self, command_queue, spectra, acc_len, channels)


class Postproc(accel.Operation):
    """The fine-delay, requant and corner-turn operations coming after the PFB.

    .. rubric:: Slots

    **in0** : spectra × channels+1, complex64
        Input channelised data, pol0.
    **in1** : spectra × channels+1, complex64
        Input channelised data, pol1.
    **out** : (spectra // acc_len, channels, acc_len, 2, 2), int8
        Output F-engine data, quantised and corner-turned, ready for
        transmission on the network.

    The inputs need to have dimension `channels+1` because cuFFT calculates
    N/2+1 output channels, i.e. the Nyquist frequency is included.  The kernel
    just loads N/2 (channels) values to work on, ignoring the Nyquist (+1).

    Parameters
    ----------
    template: PostprocTemplate
        The template for the post-processing operation.
    command_queue: AbstractCommandQueue
        The GPU command queue (typically this will be a CUDA Stream) on which
        actual processing operations are to be scheduled.
    spectra: int
        Number of spectra on which post-prodessing will be performed.
    acc_len: int
        Number of spectra to send out per heap.
    channels: int
        Number of channels in each spectrum.
    """

    def __init__(
        self, template: PostprocTemplate, command_queue: AbstractCommandQueue, spectra: int, acc_len: int, channels: int
    ) -> None:
        super().__init__(command_queue)
        if spectra % acc_len != 0:
            raise ValueError("spectra must be a multiple of acc_len")
        block_x = template.block * template.vtx
        block_y = template.block * template.vty
        if channels % block_x != 0:
            raise ValueError(f"channels must be a multiple of {block_x}")
        if acc_len % block_y != 0:
            raise ValueError(f"acc_len must be a multiple of {block_y}")
        self.template = template
        self.channels = channels
        self.spectra = spectra
        self.acc_len = acc_len
        _2 = accel.Dimension(2, exact=True)

        in_shape = (accel.Dimension(spectra), accel.Dimension(channels + 1))
        self.slots["in0"] = accel.IOSlot(in_shape, np.complex64)
        self.slots["in1"] = accel.IOSlot(in_shape, np.complex64)

        # TODO: this needs to be more explicit. 2 pols, and complexity.
        self.slots["out"] = accel.IOSlot((spectra // acc_len, channels, acc_len, _2, _2), np.int8)

        self.slots["fine_delay"] = accel.IOSlot((spectra,), np.float32)
        self.slots["phase"] = accel.IOSlot((spectra,), np.float32)
        self.quant_scale = 1.0

    def _run(self) -> None:
        block_x = self.template.block * self.template.vtx
        block_y = self.template.block * self.template.vty
        groups_x = self.channels // block_x
        groups_y = self.acc_len // block_y
        groups_z = self.spectra // self.acc_len
        out = self.buffer("out")
        in0 = self.buffer("in0")
        in1 = self.buffer("in1")
        self.command_queue.enqueue_kernel(
            self.template.kernel,
            [
                out.buffer,
                in0.buffer,
                in1.buffer,
                self.buffer("fine_delay").buffer,
                self.buffer("phase").buffer,
                np.int32(out.padded_shape[1] * out.padded_shape[2]),  # out_stride_z
                np.int32(out.padded_shape[2]),  # out_stride
                np.int32(in0.padded_shape[1]),  # in_stride
                np.int32(self.acc_len),  # acc_len
                np.float32(-1 / self.channels),  # delay_scale
                np.float32(self.quant_scale),  # quant_scale
            ],
            global_size=(self.template.block * groups_x, self.template.block * groups_y, groups_z),
            local_size=(self.template.block, self.template.block, 1),
        )
