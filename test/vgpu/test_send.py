################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.vgpu.send`."""

import asyncio
import errno
import logging
import socket
import struct
from collections.abc import Buffer
from dataclasses import dataclass
from typing import override
from unittest import mock

import async_solipsism
import numpy as np
import pytest
import pytest_mock

from katgpucbf.vgpu.main import VTP_DEFAULT_PORT
from katgpucbf.vgpu.send import RateLimiter, VDIFFrame, VDIFSender


@pytest.fixture
def event_loop_policy() -> async_solipsism.EventLoopPolicy:
    """Use async_solipsism event loop."""
    return async_solipsism.EventLoopPolicy()


@dataclass(frozen=True)
class DummyItem:
    """Item type for :class:`DummyRateLimiter`."""

    size: int  #: Size of the item for the rate-limiting algorithm
    sleep: float = 0.0  #: Time that processing the item will sleep


class DummyRateLimiter(RateLimiter[DummyItem]):
    """Process items and store the times they were processed."""

    def __init__(self, rate: float, burst_rate: float, capacity: int) -> None:
        super().__init__(rate, burst_rate, capacity)
        self.times: list[float] = []

    @override
    def item_size(self, item: DummyItem) -> int:
        return item.size

    @override
    async def _process_item(self, item: DummyItem) -> None:
        self.times.append(asyncio.get_running_loop().time())
        await asyncio.sleep(item.sleep)


class TestRateLimiter:
    """Test :class:`.RateLimiter`."""

    async def test(self) -> None:
        """Test :class:`.RateLimiter`."""
        # Use a TaskGroup so that we can schedule items to be sent at
        # precisely-controlled times, regardless of any sleeping that
        # the tasks do.
        limiter = DummyRateLimiter(10, 20, 2)
        async with asyncio.TaskGroup() as tg:
            # Two back-to-back
            tg.create_task(limiter.send(DummyItem(size=1)))
            tg.create_task(limiter.send(DummyItem(size=1)))
            # Long gap so that we reset the reference point
            await asyncio.sleep(0.5)
            # More back-to-back, for which processing causes delays.
            for _ in range(3):
                tg.create_task(limiter.send(DummyItem(size=4, sleep=0.5)))
            # Some more back-to-back for which we will be catching up
            for _ in range(5):
                tg.create_task(limiter.send(DummyItem(size=2)))
        await limiter.stop()
        assert limiter.times == pytest.approx(
            [
                0.0,
                0.1,
                0.5,
                1.0,
                1.5,
                2.0,
                2.1,
                2.2,
                2.3,
                2.5,  # All caught up now, revert to standard rate
            ]
        )

    async def test_flush(self) -> None:
        """Test :meth:`.RateLimiter.flush`."""
        # We use a large capacity so that limiter.send will always be instant
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        limiter = DummyRateLimiter(10, 20, 100)
        for _ in range(5):
            await limiter.send(DummyItem(size=1))
        # Start a flush, and asynchronously also add more data
        async with asyncio.TaskGroup() as tg:
            flush_task = tg.create_task(limiter.flush())
            await asyncio.sleep(0.1)  # Give flush_task time to start
            for _ in range(5):
                await limiter.send(DummyItem(size=1))
            await flush_task
            assert len(limiter.times) == 5
            assert loop.time() - start_time == pytest.approx(0.4)
        await limiter.stop()
        assert len(limiter.times) == 10


class TestVDIFSender:
    """Test :class:.`VDIFSender`."""

    BUFFER = 1048576
    TTL = 4

    def _make_sender(self, dsts: list[tuple[str, int]]) -> VDIFSender:
        return VDIFSender(
            dsts=dsts,
            rate=0.0,  # Rate limiting is tested in TestRateLimiter, so disable it here
            burst_rate=0.0,
            capacity=2,
            ttl=self.TTL,
            buffer_size=self.BUFFER,
            interfaces=["127.0.0.1", "127.0.0.2"],
        )

    # This has to be async because the constructor for RateLimiter expects an event loop
    @pytest.fixture
    async def sender(self) -> VDIFSender:
        """A sender with mocked sockets."""

        def make_socket(*args, **kwargs):
            sock = mock.MagicMock()
            # We need getsockopt(..., SO_SNDBUF) to return the requested buffer size
            sock.getsockopt.return_value = self.BUFFER
            return sock

        with mock.patch("socket.socket", autospec=True, side_effect=make_socket):
            return self._make_sender([(f"239.102.0.{i}", VTP_DEFAULT_PORT) for i in range(4)])

    async def test_settings(self, sender: VDIFSender) -> None:
        """Test that the sockets are correctly initialised."""
        assert len(sender._socks) == 4
        # The first four bytes in each case are unused; the second four encode IP addresses
        # 127.0.0.1 and 127.0.0.2 (in big endian).
        multicast_if = [b"\x00\x00\x00\x00\x7f\x00\x00\x01", b"\x00\x00\x00\x00\x7f\x00\x00\x02"]
        for i, sock in enumerate(sender._socks):
            sock.setsockopt.assert_any_call(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.TTL)  # type: ignore
            sock.setsockopt.assert_any_call(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, multicast_if[i % 2])  # type: ignore
            sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_SNDBUF, self.BUFFER)  # type: ignore
            sock.connect.assert_called_with((f"239.102.0.{i}", VTP_DEFAULT_PORT))  # type: ignore

    async def test_non_multicast_destination(self) -> None:
        """Test the exception raised if a destination is not a multicast address."""
        with pytest.raises(ValueError, match=r"Destination address 1\.2\.3\.4 is not a multicast address"):
            self._make_sender([("1.2.3.4", VTP_DEFAULT_PORT)])

    async def test_sndbuf_failed(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test the logging if setting the socket buffer size fails."""

        def setsockopt_side_effect(level: int, optname: int, value: int | Buffer, /) -> None:
            if level == socket.SOL_SOCKET and optname == socket.SO_SNDBUF:
                raise OSError(errno.EPERM, "Operation not permitted")

        def make_socket(*args, **kwargs):
            sock = mock.MagicMock()
            sock.setsockopt.side_effect = setsockopt_side_effect
            sock.getsockopt.return_value = 1024
            return sock

        with caplog.at_level(logging.WARNING, "katgpucbf.vgpu.send"):
            with mock.patch("socket.socket", autospec=True, side_effect=make_socket):
                self._make_sender(dsts=[("239.102.0.0", VTP_DEFAULT_PORT)])
        assert f"Failed to set socket buffer size to {self.BUFFER}: [Errno 1] Operation not permitted" in caplog.text
        assert f"Requested socket buffer size {self.BUFFER} but actual size is 1024" in caplog.text

    async def test_send(self, sender: VDIFSender) -> None:
        """Test that data is sent to the correct sockets."""
        n_framesets = 5
        n_threads = len(sender._socks)
        framesets = [
            [
                VDIFFrame(
                    np.array(bytearray(f"header{i}.{j}".encode())),
                    np.array(bytearray(f"payload{i}.{j}".encode())),
                )
                for j in range(n_threads)
            ]
            for i in range(n_framesets)
        ]
        for frameset in framesets:
            await sender.send(frameset)
        await asyncio.sleep(1)  # Allow all the background processing to complete
        packets = []
        for sock in sender._socks:
            for call in sock.sendmsg.call_args_list:  # type: ignore
                packets.append(b"".join(call.args[0]))
        # This sorts by the sequence number. The sequence number is
        # little-endian, so this only works because they're all less than
        # 256.
        packets.sort()

        expected_packets = []
        seq = 0
        for i in range(n_framesets):
            for j in range(n_threads):
                expected_packets.append(struct.pack("<Q", seq) + b"header%d.%dpayload%d.%d" % (i, j, i, j))
                seq += 1
        assert packets == expected_packets

    async def _test_resend_prologue(self, sender: VDIFSender, mocker: pytest_mock.MockerFixture) -> None:
        """Perform common prefix to :meth:`test_resend` and :meth:`test_resend_fail`.

        This sets up the packet to send and arranges for the first attempt to fail
        with :exc:`BlockingIOError`. It also arranges for the write callback to
        be triggered on the next event loop iteration.
        """
        # This is tricky to test because the retry mechanism interacts with the
        # event loop, and we don't have a real socket (and async-solipsism's
        # fake sockets don't support UDP yet). So this is behaviour-driven
        # testing.
        loop = asyncio.get_running_loop()
        mocker.patch.object(loop, "add_writer")
        mocker.patch.object(loop, "remove_writer")
        sender._socks[0].sendmsg.side_effect = BlockingIOError  # type: ignore
        frame = VDIFFrame(np.array(bytearray(b"hello")), np.array(bytearray(b"world")))
        await sender.send([frame])  # Will return immediately, since there is space in the queue
        await asyncio.sleep(1)  # Allow async stuff to happen
        sender._socks[0].sendmsg.assert_called_once()  # type: ignore
        loop.add_writer.assert_called_once()  # type: ignore
        loop.remove_writer.assert_not_called()  # type: ignore

        # Set up the next attempt.
        write_callback = loop.add_writer.call_args_list[0].args[1]  # type: ignore
        write_callback_args = loop.add_writer.call_args_list[0].args[2:]  # type: ignore
        loop.call_soon(write_callback, *write_callback_args)

    async def test_resend(self, sender: VDIFSender, mocker: pytest_mock.MockerFixture) -> None:
        """Test that data is re-sent if the socket buffer is full."""
        await self._test_resend_prologue(sender, mocker)
        packet = b""

        # Allow the next attempt to succeed. The sender clears the list of
        # buffers immediately after sending it (in-place), so to check the
        # actual packet content we need to intercept the call rather than rely
        # on the mock.
        def sendmsg(buffers, *args) -> None:
            nonlocal packet
            packet = b"".join(bytearray(buffer) for buffer in buffers)

        sender._socks[0].sendmsg.reset_mock()  # type: ignore
        sender._socks[0].sendmsg.side_effect = sendmsg  # type: ignore
        await asyncio.sleep(1)  # Allow async stuff to happen
        asyncio.get_running_loop().remove_writer.assert_called()  # type: ignore
        # Check that the retry call used the right values
        sender._socks[0].sendmsg.assert_called_once_with(mock.ANY, [], socket.MSG_DONTWAIT)  # type: ignore
        assert packet == b"\0" * 8 + b"helloworld"

    async def test_resend_fail(
        self, sender: VDIFSender, mocker: pytest_mock.MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test error reporting when an attempt to resend a packet fails permanently."""
        await self._test_resend_prologue(sender, mocker)

        sender._socks[0].sendmsg.reset_mock()  # type: ignore
        sender._socks[0].sendmsg.side_effect = OSError(1, "Permission denied")  # type: ignore
        with caplog.at_level(logging.WARNING, "katgpucbf.vgpu.send"):
            await asyncio.sleep(1)
        asyncio.get_running_loop().remove_writer.assert_called()  # type: ignore
        assert "Exception in processing rate-limited item" in caplog.text
        assert "[Errno 1] Permission denied" in caplog.text
