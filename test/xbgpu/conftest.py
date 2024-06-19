################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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

import pytest
import spead2


@pytest.fixture
def n_recv_streams() -> int:  # noqa: D401
    """Number of source streams for an xbgpu instance."""
    return 1


@pytest.fixture
def mock_recv_streams(mocker, n_recv_streams: int) -> list[spead2.InprocQueue]:
    """Mock out :func:`katgpucbf.recv.add_reader` to use in-process queues.

    Returns
    -------
    queues
        A list of in-process queue to use for sending data. The number of queues
        in the list is determined by ``n_recv_streams``.
    """
    queues = [spead2.InprocQueue() for _ in range(n_recv_streams)]
    queue_iter = iter(queues)  # Each call to add_reader gets the next queue

    def add_reader(
        stream: spead2.recv.ChunkRingStream,
        *,
        src: str | list[tuple[str, int]],
        interface: str | None,
        ibv: bool,
        comp_vector: int,
        buffer: int,
    ) -> None:
        """Mock implementation of :func:`katgpucbf.recv.add_reader`."""
        queue = next(queue_iter)
        stream.add_inproc_reader(queue)

    mocker.patch("katgpucbf.recv.add_reader", autospec=True, side_effect=add_reader)
    return queues
