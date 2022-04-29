################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""Fixtures for use in xbgpu unit tests."""

from typing import Optional, Tuple

import pytest
import spead2


@pytest.fixture
def mock_recv_stream(mocker) -> spead2.InprocQueue:
    """Mock out :func:`katgpucbf.recv.add_reader` to use in-process queues.

    Returns
    -------
    queue
        An in-process queue to use for sending to the X-engine under test.
    """
    queue = spead2.InprocQueue()

    def add_reader(
        stream: spead2.recv.ChunkRingStream,
        *,
        src: Tuple[str, int],
        interface: Optional[str],
        ibv: bool,
        comp_vector: int,
        buffer: int,
    ) -> None:
        """Mock implementation of :func:`katgpucbf.recv.add_reader`."""
        stream.add_inproc_reader(queue)

    mocker.patch("katgpucbf.recv.add_reader", autospec=True, side_effect=add_reader)
    return queue
