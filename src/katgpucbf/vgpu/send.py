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

"""Send VDIF frames at a steady rate."""

import asyncio
import ipaddress
import logging
import socket
import struct
from abc import ABC, abstractmethod
from collections.abc import Buffer
from typing import override

from katcbf_vlbi_resample.vdif_writer import VDIFFrame

logger = logging.getLogger(__name__)


def _set_result(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


class RateLimiter[T](ABC):
    """Process items at a limited rate.

    This is an abstract base class which requires implementation to actually
    process the items.

    There must be a running event loop when this class is instantiated, and
    it must be the same event loop running when the asynchronous methods are
    called.

    Parameters
    ----------
    rate
        Normal pace for acceptance, in units per second, where the units are
        determined by :meth:`item_size`. It can also be zero to disable this
        pacing.
    burst_rate
        Pace at which to catch up if falling behind (for example, because a
        sleep took too long, or a garbage collection pause). It can also be
        zero to disable this pacing. Note that if the producer doesn't
        produce items fast enough, this is *not* used to make up for time that
        the queue is empty.
    capacity
        Maximum number of items in the queue (0 for unlimited).
    """

    def __init__(self, rate: float, burst_rate: float, capacity: int) -> None:
        self.rate = rate
        self.burst_rate = burst_rate
        # Loop time at which next item could be sent based on `rate`
        self._next = 0.0
        # Loop time at which next item could be sent based on `burst_rate`
        self._next_burst = 0.0
        # Seconds per unit for `rate`
        self._per_unit = 1 / rate if rate else 0.0
        # Seconds per unit for `burst_rate`
        self._per_unit_burst = 1 / burst_rate if burst_rate else 0.0
        self._queue: asyncio.Queue[T] = asyncio.Queue(capacity)
        self._run_task: asyncio.Task | None = None

    @abstractmethod
    def item_size(self, item: T) -> int:
        """Get the number of units in an item."""

    @abstractmethod
    async def _process_item(self, item: T) -> None:
        """Implement processing of an item.

        This method does not handle the rate limiting.
        """

    async def _run(self) -> None:
        """Process queue items.

        This is scheduled as an asyncio task, only when the queue is non-empty.
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                now = loop.time()
                target = max(self._next, self._next_burst)
                # Don't try to sleep for short times. We tend to oversleep and
                # then are unable to catch up.
                if target - now > 1e-3:
                    future = loop.create_future()
                    loop.call_at(target, _set_result, future)
                    await future
                else:
                    await asyncio.sleep(0)  # Give other asyncio tasks a chance to run
                now = loop.time()
                size = self.item_size(item)
                self._next += self._per_unit * size
                self._next_burst = max(self._next_burst, max(target, now)) + self._per_unit_burst * size
                try:
                    await self._process_item(item)
                except Exception:
                    logger.exception("Exception in processing rate-limited item")

        finally:
            self._run_task = None

    async def send(self, item: T) -> None:
        """Add an item to the queue.

        Note that this will return as soon as the item is admitted to the
        queue, rather than when it is processed. The item should thus not be
        modified after submitting it.
        """
        await self._queue.put(item)
        if not self._queue.empty() and self._run_task is None:
            now = asyncio.get_running_loop().time()
            self._next = max(self._next, now)
            self._next_burst = max(self._next_burst, now)
            self._run_task = asyncio.create_task(self._run(), name="RateLimiter")

    async def join(self) -> None:
        """Wait until all queued items have been processed."""
        while self._run_task is not None:
            await self._run_task


class VDIFSender(RateLimiter[list[VDIFFrame]]):
    """Send VDIF frames at a limited rate to a set of multicast addresses.

    The units for `rate` and `burst_rate` are samples per second.
    """

    def __init__(
        self,
        dsts: list[tuple[str, int]],
        rate: float,
        burst_rate: float,
        capacity: int,
        *,
        ttl: int,
        buffer_size: int,
        interfaces: list[str],
    ) -> None:
        super().__init__(rate, burst_rate, capacity)
        # Create a socket per destination, distributing them
        # over the interfaces.
        if_addrs = [ipaddress.IPv4Address(address) for address in interfaces]
        self._socks = []
        self._sequence = 0
        for i, dst in enumerate(dsts):
            if not ipaddress.IPv4Address(dst[0]).is_multicast:
                raise ValueError(f"Destination address {dst[0]} is not a multicast address")
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            if_addr = if_addrs[i % len(if_addrs)]
            # struct ip_mreq contains an address and an interface address;
            # IP_MULTICAST_IF only uses the latter.
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, b"\0\0\0\0" + if_addr.packed)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
            except OSError as exc:
                logger.warning("Failed to set socket buffer size to %d: %s", buffer_size, exc)
            actual_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            if actual_buffer < buffer_size:
                logger.warning("Requested socket buffer size %d but actual size is %d", buffer_size, actual_buffer)
            sock.connect(dst)
            self._socks.append(sock)
        self._next_sock = 0

    @override
    def item_size(self, item: list[VDIFFrame]) -> int:
        # 2 bits per sample, so 4 samples per byte
        return item[0].payload.nbytes * 4

    @staticmethod
    def _try_send(sock: socket.socket, buffers: list[Buffer]) -> bool:
        try:
            sock.sendmsg(buffers, [], socket.MSG_DONTWAIT)
            return True
        except (BlockingIOError, InterruptedError):
            return False

    @staticmethod
    def _write_callback(sock: socket.socket, buffers: list[Buffer], future: asyncio.Future) -> None:
        try:
            if VDIFSender._try_send(sock, buffers):
                _set_result(future)
                # The finally in _process_item will also do this, but doing it
                # now ensures that we don't get called back again before that
                # happens.
                asyncio.get_running_loop().remove_writer(sock.fileno())
                buffers.clear()  # Makes doubly sure we can't send twice
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    async def _send_frame(self, frame: VDIFFrame) -> None:
        buffers: list[Buffer] = [struct.pack("<Q", self._sequence), frame.header, frame.payload]
        sock = self._socks[self._next_sock]
        self._next_sock = (self._next_sock + 1) % len(self._socks)
        self._sequence += 1
        # Try to send immediately
        if not self._try_send(sock, buffers):
            # Fall back to doing it asynchronously via a callback when
            # the socket is ready for writing.
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            try:
                loop.add_writer(sock.fileno(), self._write_callback, sock, buffers, future)
                await future
            finally:
                loop.remove_writer(sock.fileno())

    @override
    async def _process_item(self, item: list[VDIFFrame]) -> None:
        for frame in item:
            await self._send_frame(frame)
