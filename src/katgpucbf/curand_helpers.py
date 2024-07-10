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

"""Helpers to initialise random state with curand.

See :ref:`dithering` for an explanation of why we introduce a separate
:c:struct:`randState_t` structure. Code using this module should **not**
generate Gaussian distributions without understanding the implications.
"""

from importlib import resources

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

#: sizeof(randState_t)
RAND_STATE_SIZE = 24
#: alignof(randState_t)
RAND_STATE_ALIGNMENT = 8
#: opaque dtype corresponding to randState_t (only size and alignment matter)
RAND_STATE_DTYPE = np.dtype(
    {"names": ["_align"], "formats": [np.dtype(f"u{RAND_STATE_ALIGNMENT}")], "itemsize": RAND_STATE_SIZE}, align=True
)
assert RAND_STATE_DTYPE.itemsize == RAND_STATE_SIZE
assert RAND_STATE_DTYPE.alignment == RAND_STATE_ALIGNMENT


class RandomStateBuilder:
    """Build array of initialised random states for curand."""

    def __init__(self, context: AbstractContext) -> None:
        with resources.as_file(resources.files(__package__)) as resource_dir:
            program = accel.build(context, "kernels/curand_init.mako", extra_dirs=[str(resource_dir)])
        self._init_kernel = program.get_kernel("rand_state_init")

    def make_states(
        self,
        command_queue: AbstractCommandQueue,
        shape: tuple[int, ...],
        seed: int,
        sequence_first: int,
        sequence_step: int = 1,
        offset: int = 0,
    ) -> accel.DeviceArray:
        """Create a multi-dimensional array of random states.

        This method is not particularly efficient. It's intended to be used
        just during startup, after which the random states will be persisted in
        global memory and reused.

        The initialisation process is enqueued to `command_queue`.
        """
        states = accel.DeviceArray(command_queue.context, shape, RAND_STATE_DTYPE)
        n = int(np.prod(shape))
        wgs = 256
        command_queue.enqueue_kernel(
            self._init_kernel,
            [
                states.buffer,
                np.uint64(seed),
                np.uint64(sequence_first),
                np.uint64(sequence_step),
                np.uint64(offset),
                np.uint32(n),
            ],
            global_size=(accel.roundup(n, wgs),),
            local_size=(wgs,),
        )
        return states
