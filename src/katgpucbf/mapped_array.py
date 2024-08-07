################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Implement :class:`MappedArray`."""

from dataclasses import dataclass
from typing import Self

import katsdpsigproc.cuda
import numpy as np
import vkgdr
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext, AbstractDevice


def make_vkgdr(device: AbstractDevice) -> vkgdr.Vkgdr:
    """Create a :class:`vkgdr.Vkgdr` for a given device.

    The device must be a CUDA device. The returned handle uses coherent
    mappings, which are suitable for use with :class:`MappedArray`.
    """
    if not isinstance(device, katsdpsigproc.cuda.Device):
        raise TypeError("device is not a CUDA device")
    # This is hacky: pycuda.driver.Device doesn't have an API to get the
    # underlying CUdevice, but its __hash__ implementation returns it.
    raw_device = hash(device._pycuda_device)
    # We could quite easily make do with non-coherent mappings and
    # explicit flushing, but since NVIDIA currently only provides
    # host-coherent memory, this is a simpler option.
    return vkgdr.Vkgdr(raw_device, vkgdr.OpenFlags.REQUIRE_COHERENT_BIT)


@dataclass
class MappedArray:
    """An array visible in the address space of both the host and the device.

    The host view can be updated using normal numpy code, and the changes will
    be visible to subsequently enqueued kernels on the device.
    """

    host: np.ndarray
    device: accel.DeviceArray

    @classmethod
    def from_slot(cls, vkgdr_handle: vkgdr.Vkgdr, context: AbstractContext, slot: accel.IOSlotBase) -> Self:
        """Allocate a :class:`MappedArray` to match a slot.

        Parameters
        ----------
        vkgdr_handle
            Handle for allocating memory from vkgdr. It must be created from the same device
            as `context`.
        context
            CUDA context in which the device view will be used.
        slot
            Slot from which the dtype, shape and padded shape will be
            extracted. The parameter is annotated as
            :class:`~katsdpsigproc.accel.IOSlotBase` for convenience, but it
            must actually be an instance of :class:`~katsdpsigproc.accel.IOSlot`.
        """
        assert isinstance(slot, accel.IOSlot)
        padded_shape = slot.required_padded_shape()
        n_bytes = int(np.prod(padded_shape)) * slot.dtype.itemsize
        with context:
            handle = vkgdr.pycuda.Memory(vkgdr_handle, n_bytes)
        # Slice out the shape from the padded shape
        index = tuple(slice(0, x) for x in slot.shape)
        host = np.asarray(handle).view(slot.dtype).reshape(padded_shape)[index]
        device = accel.DeviceArray(context, slot.shape, slot.dtype, padded_shape, raw=handle)
        return cls(host=host, device=device)
