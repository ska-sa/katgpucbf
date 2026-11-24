################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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

"""A collection of utilities for benchmarking/profiling the kernels in katgpucbf."""

from collections.abc import Sequence
from typing import overload

import numpy as np
from katsdpsigproc.abc import AbstractCommandQueue
from katsdpsigproc.accel import DeviceArray, Operation


def _duplicate(array: DeviceArray, command_queue: AbstractCommandQueue) -> DeviceArray:
    new_array = DeviceArray(array.context, array.shape, array.dtype, array.padded_shape)
    array.copy_region(command_queue, new_array, (), ())
    return new_array


@overload
def replicate_slots(
    operation: Operation, slots: Sequence[str] | None = None, *, count: int
) -> list[dict[str, DeviceArray]]: ...


@overload
def replicate_slots(
    operation: Operation, slots: Sequence[str] | None = None, *, size: int
) -> list[dict[str, DeviceArray]]: ...


@overload
def replicate_slots(
    operation: Operation, slots: Sequence[str] | None = None, *, cache_factor: float = ...
) -> list[dict[str, DeviceArray]]: ...


def replicate_slots(
    operation: Operation,
    slots: Sequence[str] | None = None,
    **kwargs,
) -> list[dict[str, DeviceArray]]:
    """Generate additional copies of bound slots.

    This is intended to be used to increase the memory footprint when
    benchmarking repeated application of a kernel. Doing this naïvely
    can lead to all the memory for the kernel being held in a cache.
    This function will return a list of groups of buffers to bind to the
    slots in round-robin fashion.

    The operation must already have buffers bound to the relevant slots.
    These buffers will be replicated some number of times. The original
    buffers will be included in the list returned.

    There are several mutually-exclusive ways to specify the number of repeats
    to make:

    1. Pass `count` as a keyword argument. This directly specifies the length
       of the return value.

    2. Pass `size` as a keyword argument. This specifies a minimum total size
       for the returned buffers (the actual size may be larger due to
       rounding).

    3. Pass `cache_factor` as a keyword argument. The total size of the
       returned buffers will be at least `cache_factor` times the size of the
       cache. For CUDA this is specifically the L2 cache. For OpenCL it is
       whatever ``CL_DEVICE_GLOBAL_MEM_CACHE_SIZE`` returns, which might be
       the total size of L1 caches.

    4. Pass nothing, which is equivalent to passing ``cache_factor=2``.

    The copying of data is performed asynchronously using the command queue
    associated with `operation` and not flushed.
    """
    item_size = 0
    if slots is None:
        slots = list(operation.slots)
    for slot in slots:
        buffer = operation.buffer(slot)
        if buffer is None:
            raise ValueError(f"slot {slot} is not populated")
        item_size += int(np.prod(buffer.padded_shape)) * buffer.dtype.itemsize
    queue = operation.command_queue
    ctx = queue.context

    if len(kwargs) > 1:
        raise TypeError("too many kwargs (pass only one of count and cache_factor)")

    if "count" in kwargs:
        count = kwargs["count"]
        if count < 1:
            raise ValueError("count must be at least 1")
    else:
        if "size" in kwargs:
            target_size = kwargs["size"]
        else:
            target_size = int(ctx.device.cache_size * kwargs.get("cache_factor", 2))
        count = max(1, (target_size + item_size - 1) // item_size)

    ret = [{slot: operation.buffer(slot) for slot in slots}]
    for _ in range(count - 1):
        ret.append({slot: _duplicate(ret[0][slot], queue) for slot in slots})
    return ret
