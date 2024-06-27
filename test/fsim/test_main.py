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

"""Test :mod:`katgpucbf.fsim.main`."""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import spead2.recv.asyncio

from katgpucbf.fsim.main import DTYPE, Sender, parse_args


@pytest.fixture
def args() -> argparse.Namespace:
    """Create command-line arguments."""
    return parse_args(
        [
            "--channels=32768",
            "--channels-per-substream=512",
            "--array-size=8",
            "--jones-per-batch=1048576",
            "--adc-sample-rate=1e12",  # Much faster than realistic, to make the test finish quickly
            "--interface=lo",
            "239.1.1.1:7148",
        ]
    )


@pytest.fixture
def mock_queue(monkeypatch: pytest.MonkeyPatch) -> spead2.InprocQueue:
    """Mock out the Sender's stream, and return the underlying :class:`spead2.InprocQueue`."""
    queue = spead2.InprocQueue()

    def fake_stream(
        thread_pool: spead2.ThreadPool,
        endpoints: list[tuple[str, int]],
        config: spead2.send.StreamConfig,
        ttl: int,
        interface_address: str,
    ) -> "spead2.send.asyncio.AsyncStream":
        return spead2.send.asyncio.InprocStream(thread_pool, [queue], config)

    monkeypatch.setattr("spead2.send.asyncio.UdpStream", fake_stream)
    return queue


@pytest.fixture
def mock_stream(mock_queue: spead2.InprocQueue) -> Generator[spead2.recv.asyncio.Stream, None, None]:
    """Get a receive stream corresponding to the mocked send stream."""
    config = spead2.recv.StreamConfig(max_heaps=64)  # More than needed
    stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(), config)
    stream.add_inproc_reader(mock_queue)
    yield stream
    stream.stop()


@pytest.fixture
def sender(args: argparse.Namespace, mock_queue) -> Sender:
    """Get an instance of :class:`katgpucbf.fsim.main.Sender`."""
    return Sender(args, 0)


@pytest.fixture
async def sender_run(sender: Sender) -> AsyncGenerator[asyncio.Task, None]:
    """Run the `sender` fixture and also send descriptors, and cancel it at the end of the test."""
    await sender.stream.async_send_heap(sender.descriptor_heap, rate=0.0)
    task = asyncio.create_task(sender.run(0, False))
    yield task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_sender(
    args: argparse.Namespace, sender_run: asyncio.Task, mock_stream: spead2.recv.asyncio.Stream
) -> None:
    """Test the shape and metadata of sent data.

    This does not check the payload values, since they're largely irrelevant to
    how fsim is used.
    """
    batches = 20
    ig = spead2.ItemGroup()
    for i in range(batches):
        seen = set()  # Antennas seen this batch
        for _ in range(args.array_size):
            while not (updated := ig.update(await mock_stream.get())):  # Loop until non-descriptor
                pass
            assert set(updated.keys()) == {"timestamp", "feng_id", "frequency", "feng_raw"}
            assert updated["timestamp"].value == 2**21 * i
            assert updated["frequency"].value == 0
            seen.add(updated["feng_id"].value)
            assert updated["feng_raw"].shape == (512, 32, 2, 2)
            assert updated["feng_raw"].dtype == DTYPE
        assert sorted(seen) == list(range(args.array_size))
