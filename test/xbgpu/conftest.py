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

"""Fixtures for use in xbgpu unit tests."""

import pytest
import spead2


@pytest.fixture
def n_src_streams() -> int:  # noqa: D401
    """Number of source streams for an xbgpu instance."""
    return 1


@pytest.fixture
def mock_send_stream(mocker) -> spead2.InprocQueue:
    """Mock out creation of the send stream.

    Calls to :class:`spead2.send.asyncio.UdpStream` are replaced by an
    in-process stream. Returns an inproc queue to receive the output from that
    stream.
    """
    queue = spead2.InprocQueue()

    def constructor(thread_pool, endpoints, config, *args, **kwargs):
        return spead2.send.asyncio.InprocStream(thread_pool, [queue], config)

    mocker.patch("spead2.send.asyncio.UdpStream", autospec=True, side_effect=constructor)
    return queue
