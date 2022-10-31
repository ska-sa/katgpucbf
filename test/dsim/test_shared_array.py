################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`.shared_array`."""

import multiprocessing
from collections.abc import Generator

import numpy as np
import pytest

from katgpucbf.dsim.shared_array import SharedArray

# Process isn't an attribute of BaseContext but of each subclass
_MPContext = (
    multiprocessing.context.ForkContext
    | multiprocessing.context.ForkServerContext
    | multiprocessing.context.SpawnContext
)


@pytest.fixture
def array() -> Generator[SharedArray, None, None]:  # noqa: D401
    """A pre-created shared array."""
    array = SharedArray.create("test_shared_array", (10000,), np.int32)
    yield array
    array.close()


@pytest.fixture(params=["fork", "forkserver", "spawn"])
def mp_context(request) -> multiprocessing.context.BaseContext:  # noqa: D401
    """Multiprocessing context (launch method)."""
    return multiprocessing.get_context(request.param)


def _child(conn: multiprocessing.connection.Connection) -> None:
    """Run the child process of one of the tests.

    It
    - Receives an array.
    - Replies with its sum.
    - Receives an arbitrary value.
    - Replies with the sum again.
    """
    array = conn.recv()
    conn.send(int(np.sum(array.buffer)))
    # Wait until the parent tells us it has updated the array
    conn.recv()
    conn.send(int(np.sum(array.buffer)))


class TestSharedArray:
    """Test :class:`.SharedArray`."""

    def test_create(self, array: SharedArray) -> None:
        """Test that an array created in this process has the right properties."""
        assert array.buffer.shape == (10000,)
        assert array.buffer.dtype == np.int32

    def test_share(self, array: SharedArray, mp_context: _MPContext) -> None:
        """Test sharing an array between processes."""
        conn, child_conn = mp_context.Pipe()
        proc = mp_context.Process(target=_child, args=(child_conn,))
        proc.start()
        # Close our copy of the child connection
        child_conn.close()
        # Put some values into the array and share it
        array.buffer[:] = np.arange(len(array.buffer))
        conn.send(array)
        # Read back what the child thinks is the sum
        total1 = conn.recv()
        assert total1 == np.sum(array.buffer)
        # Change the array, to make sure the child sees the updates
        array.buffer *= 3
        conn.send(None)  # Arbitrary value to wake it up
        total2 = conn.recv()
        assert total2 == np.sum(array.buffer)
        assert total2 != total1
        proc.join()

    def test_double_close(self, array: SharedArray) -> None:
        """Test that closing twice does not explode."""
        array.close()
        array.close()
