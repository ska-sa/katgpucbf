"""
Class for wrapping SPEAD2 ringbuffers.

Defines two classes that enable the SPEAD2 C++ ringbuffer to operate in Python:
1. RingbufferProtocol - This object seems to define some functions that already exist in the SPEAD2 ringbuffer class. It
   does not add any functionality to this ringbuffer. As far as I can tell it is just there as a placeholder for a
   pybind11 wrapped SPEAD2 C++ ringbuffer object.
2. AsyncRingbuffer - This takes a ringbuffer object and gives it asyncio functionality. This is the main object that
   will be used when receiving data from a SPEAD2 object. For example if the receiver ringbuffer object is wrapped in an
   AsyncRingbuffer object called asyncRingbuffer then this can be iterated through asynchronously as follows:
   ---
   async for chunk in asyncRingbuffer:
       print(chunk)
   ---

This file has been copied directly from the katfgpu documentation. The functionality is not fully understood.

TODO: Look at moving this class to a different repo to avoid repitition between here and katxgpu.
"""

import asyncio
from typing import Optional, AsyncIterator, Generic, TypeVar

from typing_extensions import Protocol

from katxgpu._katxgpu import Empty, Stopped
from katxgpu.monitor import Monitor


_T = TypeVar("_T")


class RingbufferProtocol(Protocol[_T]):
    """Protocol for Python wrappers of spead2's C++ ringbuffer class."""

    @property
    def data_fd(self) -> int:
        """TODO: Create docstring."""
        ...

    def try_push(self, item: _T) -> None:
        """TODO: Create docstring."""
        ...

    def try_pop(self) -> _T:
        """TODO: Create docstring."""
        ...

    def pop(self) -> _T:
        """TODO: Create docstring."""
        ...


class AsyncRingbuffer(Generic[_T]):
    """Wraps a C++-provided ringbuffer class to provide asyncio.

    It is **not** safe to use more than one wrapper of the same base within the
    same event loop.
    """

    def __init__(self, base: RingbufferProtocol[_T], monitor: Monitor, name: str, task_name: str) -> None:
        """TODO: Create docstring."""
        self._base = base
        self._waiter = None  # type: Optional[asyncio.Future[_T]]
        self._monitor = monitor
        self._name = name
        self._task_name = task_name

    @property
    def base(self) -> RingbufferProtocol[_T]:
        """Return the wrapped C++ ringbuffer class."""
        return self._base

    async def async_pop(self) -> _T:
        """TODO: Create docstring."""
        if self._waiter is not None:
            raise RuntimeError("Cannot have more than one waiter on a Ringbuffer")
        loop = asyncio.get_event_loop()
        future = loop.create_future()  # type: asyncio.Future[_T]
        self._waiter = future
        loop.add_reader(self._base.data_fd, self._ready_callback)
        try:
            with self._monitor.with_state(self._task_name, "wait ringbuffer"):
                item = await future
            self._monitor.event_qsize_delta(self._name, -1)
            return item
        finally:
            self._waiter = None
            loop.remove_reader(self._base.data_fd)

    def try_push(self, item: _T) -> None:
        """TODO: Create docstring."""
        self._base.try_push(item)

    def _ready_callback(self) -> None:
        """TODO: Create docstring."""
        if self._waiter is None or self._waiter.done():
            return
        try:
            self._waiter.set_result(self._base.try_pop())
        except Empty:
            # Spurious wakeup, no action required
            pass
        except Exception as exc:
            self._waiter.set_exception(exc)

    def __aiter__(self) -> AsyncIterator[_T]:
        """TODO: Create docstring."""
        return self

    async def __anext__(self) -> _T:
        """TODO: Create docstring."""
        try:
            return await self.async_pop()
        except Stopped:
            raise StopAsyncIteration from None
