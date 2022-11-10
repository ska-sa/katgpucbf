################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""A collection of utility functions for working with katsdpsigproc."""

from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext


def device_allocate_slot(context: AbstractContext, slot: accel.IOSlotBase) -> accel.DeviceArray:
    """Allocate a :class:`~katsdpsigproc.accel.DeviceArray` to match a given slot.

    The `slot` argument is annotated as :class:`.IOSlotBase` to reduce the
    boilerplate needed to call this helper, but it must be a real
    :class:`.IOSlot`.
    """
    assert isinstance(slot, accel.IOSlot)
    return accel.DeviceArray(context, slot.shape, slot.dtype, slot.required_padded_shape())


def host_allocate_slot(context: AbstractContext, slot: accel.IOSlotBase) -> accel.HostArray:
    """Allocate a :class:`~katsdpsigproc.accel.HostArray` to match a given slot.

    The `slot` argument is annotated as :class:`.IOSlotBase` to reduce the
    boilerplate needed to call this helper, but it must be a real
    :class:`.IOSlot`.
    """
    assert isinstance(slot, accel.IOSlot)
    return accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(), context=context)
