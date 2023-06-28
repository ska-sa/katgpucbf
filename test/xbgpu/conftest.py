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

from ipaddress import IPv4Address, IPv4Network

import pytest
import spead2.send.asyncio


@pytest.fixture
def n_src_streams() -> int:  # noqa: D401
    """Number of source streams for an xbgpu instance."""
    return 1


@pytest.fixture
def mock_send_stream_network() -> IPv4Network:
    """Network mask to filter the queues returned by :func:`mock_send_stream`.

    Test classes can override this to select only a subset.
    """
    return IPv4Network("0.0.0.0/0")


@pytest.fixture
def mock_send_stream(mocker, mock_send_stream_network: IPv4Network) -> list[spead2.InprocQueue]:
    """Mock out creation of the send stream.

    Each time a :class:`spead2.send.asyncio.UdpStream` is created, it instead
    creates an in-process stream and appends an equivalent inproc queue to
    the list returned by the fixture.

    The queues returned can be filtered by IP address by overriding the
    :func:`mock_send_stream_network` fixture.
    """
    queues: list[spead2.InprocQueue] = []

    def constructor(thread_pool, endpoints, config, *args, **kwargs):
        stream_queues = [spead2.InprocQueue() for _ in endpoints]
        queues.extend(
            queue
            for queue, endpoint in zip(stream_queues, endpoints)
            if IPv4Address(endpoint[0]) in mock_send_stream_network
        )
        return spead2.send.asyncio.InprocStream(thread_pool, stream_queues, config)

    mocker.patch("spead2.send.asyncio.UdpStream", autospec=True, side_effect=constructor)
    return queues
