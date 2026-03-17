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
import functools
import ipaddress
import logging
import socket
import struct
from abc import ABC, abstractmethod
from collections.abc import Buffer
from typing import Self, overload, override

from katcbf_vlbi_resample.vdif_writer import VDIFFrame

logger = logging.getLogger(__name__)


@functools.total_ordering
class _PreciseTimeBase:
    """Common base class for :class:`PreciseTime` and :class:`PreciseTimeDelta`."""

    def __init__(self, t: float) -> None:
        self._ticks = round(t * 1e15)

    @classmethod
    def _from_ticks(cls, ticks: int) -> Self:
        t = cls(0.0)
        t._ticks = ticks
        return t

    def __float__(self) -> float:
        return self._ticks * 1e-15

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        # mypy doesn't recognise the comparison on type()
        return self._ticks == other._ticks  # type: ignore[attr-defined]

    def __lt__(self, other: Self) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._ticks < other._ticks

    def __hash__(self) -> int:
        return hash(self._ticks)


class PreciseTime(_PreciseTimeBase):
    """Time with femtosecond (1e-15 second) precision.

    The time is internally represented as a integer number of femtoseconds
    since an arbitrary epoch. When cast to float, it returns the time as
    floating-point seconds, losing precision.

    This represents a point in time. Use :class:`PreciseTimeDelta` for a
    time difference.
    """

    @overload
    def __sub__(self, other: "PreciseTime") -> "PreciseTimeDelta": ...
    @overload
    def __sub__(self, other: "PreciseTimeDelta") -> Self: ...

    def __sub__(self, other):
        if isinstance(other, PreciseTime):
            return PreciseTimeDelta._from_ticks(self._ticks - other._ticks)
        elif isinstance(other, PreciseTimeDelta):
            return self._from_ticks(self._ticks - other._ticks)
        else:
            return NotImplemented

    def __add__(self, other: "PreciseTimeDelta") -> Self:
        if not isinstance(other, PreciseTimeDelta):
            return NotImplemented
        return self._from_ticks(self._ticks + other._ticks)


class PreciseTimeDelta(_PreciseTimeBase):
    """Time difference with femtosecond (1e-15 second) precision.

    The delta is internally represented as a integer number of femtoseconds.
    When cast to float, it returns the value as floating-point seconds, losing
    precision.

    This represents a change in time. Use :class:`PreciseTime` for an absolute
    time.
    """

    def __add__(self, other: "PreciseTimeDelta") -> Self:
        if not isinstance(other, PreciseTimeDelta):
            return NotImplemented
        return self._from_ticks(self._ticks + other._ticks)

    def __sub__(self, other: "PreciseTimeDelta") -> Self:
        if not isinstance(other, PreciseTimeDelta):
            return NotImplemented
        return self._from_ticks(self._ticks - other._ticks)

    def __mul__(self, other: float) -> Self:
        if isinstance(other, int):
            return self._from_ticks(self._ticks * other)
        elif isinstance(other, float):
            # This loses precision, but `other` is imprecise anyway
            return type(self)(float(self) * other)
        else:
            return NotImplemented


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
        zero to disable this pacing.
    """

    def __init__(self, rate: float, burst_rate: float) -> None:
        self.rate = rate
        self.burst_rate = burst_rate
        self._loop = asyncio.get_running_loop()
        self._next: PreciseTime | None = None
        self._next_burst: PreciseTime | None = None
        self._per_unit = PreciseTimeDelta(1 / rate if rate else 0.0)
        self._per_unit_burst = PreciseTimeDelta(1 / burst_rate if burst_rate else 0.0)
        self._lock = asyncio.Lock()

    @abstractmethod
    def item_size(self, item: T) -> int:
        """Get the number of units in an item."""

    @abstractmethod
    async def _process_item(self, item: T) -> None:
        """Implement processing of an item.

        This method does not handle the rate limiting.
        """

    async def send(self, item: T) -> None:
        """Wait for the rate limiter, then process an item."""
        async with self._lock:
            now = PreciseTime(self._loop.time())
            if self._next is None:
                self._next = now
            if self._next_burst is None:
                self._next_burst = now
            target = max(self._next, self._next_burst)
            if now < target:
                future = self._loop.create_future()
                self._loop.call_at(float(target), _set_result, future)
                await future
            else:
                await asyncio.sleep(0)  # Give other asyncio tasks a chance to run
            now = PreciseTime(self._loop.time())
            size = self.item_size(item)
            self._next += self._per_unit * size
            self._next_burst = max(self._next_burst, now) + self._per_unit_burst * size
            await self._process_item(item)


class VDIFSender(RateLimiter[list[VDIFFrame]]):
    """Send VDIF frames at a limited rate to a set of multicast addresses.

    The units for `rate` and `burst_rate` are samples per second.
    """

    def __init__(
        self,
        dsts: list[tuple[str, int]],
        rate: float,
        burst_rate: float,
        *,
        ttl: int,
        buffer: int,
        interfaces: list[str],
    ) -> None:
        super().__init__(rate, burst_rate)
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
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer)
            except OSError as exc:
                logger.warning("Failed to set socket buffer size to %d: %s", buffer, exc)
            actual_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            if actual_buffer < buffer:
                logger.warning("Requested socket buffer size %d but actual size is %d", buffer, actual_buffer)
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
        buffers = [struct.pack(">Q", self._sequence), frame.header, frame.payload]
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
