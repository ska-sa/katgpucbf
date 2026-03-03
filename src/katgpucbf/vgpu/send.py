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
from abc import ABC, abstractmethod
from typing import Self, overload


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
        determined by :meth:`item_size`.
    burst_rate
        Pace at which to catch up if falling behind (for example, because a
        sleep took too long, or a garbage collection pause).
    """

    def __init__(self, rate: float, burst_rate: float) -> None:
        self.rate = rate
        self.burst_rate = burst_rate
        self._loop = asyncio.get_running_loop()
        self._next = PreciseTime(self._loop.time())
        self._next_burst = self._next
        self._per_unit = PreciseTimeDelta(1 / rate)
        self._per_unit_burst = PreciseTimeDelta(1 / burst_rate)
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
            target = max(self._next, self._next_burst)
            if now < target:
                future = self._loop.create_future()
                self._loop.call_at(float(target), _set_result, future)
                await future
                now = PreciseTime(self._loop.time())
            size = self.item_size(item)
            self._next += self._per_unit * size
            self._next_burst = max(self._next_burst, now) + self._per_unit_burst * size
            await self._process_item(item)
