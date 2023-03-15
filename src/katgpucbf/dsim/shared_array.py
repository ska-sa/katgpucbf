################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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

"""Shared memory arrays."""

import mmap
import multiprocessing.connection
import multiprocessing.reduction
import os
from typing import Callable

import numpy as np
from numpy.typing import DTypeLike


class SharedArray:
    """An array that can be passed to another process.

    Unlike :mod:`multiprocessing.shared_memory`, the shared memory used for
    this is not backed by a file, and so is guaranteed to be cleaned up when
    the processes involved die off, without the need for a manager process.

    This is UNIX (probably Linux) specific.

    Do not construct directly. Instead, either use :meth:`create` to allocate a
    new array, or :meth:`multiprocessing.connection.Connection.recv` to
    construct a new reference to an existing array in another process.
    """

    @staticmethod
    def _byte_size(shape: tuple[int, ...], dtype: DTypeLike) -> int:
        return int(np.product(shape)) * np.dtype(dtype).itemsize

    def __init__(self, fd: int, shape: tuple[int, ...], dtype: DTypeLike) -> None:
        size = self._byte_size(shape, dtype)
        self._mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
        self._fd = fd
        self.buffer = np.ndarray(shape, dtype, buffer=self._mapping)  # type: ignore

    def close(self) -> None:
        """Close the reference shared array and release the mapping.

        Accessing the array after this will most likely crash. It is safe to
        call twice.
        """
        if self._mapping.closed:
            return  # It's already closed
        self._mapping.close()
        os.close(self._fd)

    @classmethod
    def create(cls, name: str, shape: tuple[int, ...], dtype: DTypeLike) -> "SharedArray":
        """Create a new array from scratch.

        Parameters
        ----------
        name
            An arbitrary name to associate with the array. See
            :func:`os.memfd_create`.
        shape
            Shape of the array. To simplify this function, it requires a tuple
            (a scalar cannot be used).
        dtype
            The type of the array.
        """
        fd = os.memfd_create(name)
        try:
            # Resize to the appropriate size.
            os.ftruncate(fd, cls._byte_size(shape, dtype))
            array = cls(fd, shape, dtype)
        except Exception:
            os.close(fd)  # Clean up the file descriptor if there is a failure
            raise
        array.buffer.fill(0)  # Ensure memory is actually allocated
        return array

    def __del__(self) -> None:
        self.close()


# Register with multiprocessing so that a SharedArray can be sent through a
# pipe by sending the file descriptor and metadata and constructing a new
# memory mapping on the other side.


def _reduce(a: SharedArray) -> tuple[Callable, tuple]:
    return _rebuild, (multiprocessing.reduction.DupFd(a._fd), a.buffer.shape, a.buffer.dtype)


def _rebuild(dupfd, shape: tuple[int, ...], dtype: DTypeLike) -> SharedArray:
    return SharedArray(dupfd.detach(), shape, dtype)


multiprocessing.reduction.register(SharedArray, _reduce)
