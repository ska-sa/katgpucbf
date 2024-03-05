################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Helpers to initialise random state with curand."""

import importlib.resources

import numpy as np
from katsdpsigproc.accel import DeviceArray, roundup
from katsdpsigproc.cuda import Context


class RandomStateHelper:
    """Helper to initialise random state with curand."""

    def __init__(self, context: Context) -> None:
        source = (importlib.resources.files(__package__) / "kernels" / "curand_helpers.cu").read_text()
        program = context.compile(source)
        size_align_kernel = program.get_kernel("sizeof_alignof_curandStateXORWOW_t")
        size_align_device = DeviceArray(context, (2,), np.int32)
        command_queue = context.create_command_queue()
        command_queue.enqueue_kernel(size_align_kernel, [size_align_device.buffer], (1,), (1,))
        size_align = size_align_device.get(command_queue)

        self.size = int(size_align[0])
        self.align = int(size_align[1])
        self.dtype = np.dtype(
            {"names": ["_align"], "formats": [np.dtype(f"u{self.align}")], "itemsize": self.size}, align=True
        )
        assert self.dtype.itemsize == self.size
        assert self.dtype.alignment == self.align
        self.command_queue = command_queue
        self.init_kernel = program.get_kernel("init_curandStateXORWOW_t")

    def make_states(
        self, shape: tuple[int, ...], seed: int, sequence_first: int, sequence_step: int = 1, offset: int = 0
    ) -> DeviceArray:
        """Create a multi-dimensional array of random states.

        This method is not particularly efficient. It's intended to be used
        just during startup, after which the random states will be persisted in
        global memory and reused.
        """
        states = DeviceArray(self.command_queue.context, shape, self.dtype)
        n = int(np.prod(shape))
        wgs = 256
        self.command_queue.enqueue_kernel(
            self.init_kernel,
            [
                states.buffer,
                np.uint64(seed),
                np.uint64(sequence_first),
                np.uint64(sequence_step),
                np.uint64(offset),
                np.uint32(n),
            ],
            global_size=(roundup(n, wgs),),
            local_size=(wgs,),
        )
        self.command_queue.finish()
        return states
