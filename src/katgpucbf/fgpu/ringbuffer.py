import asyncio
from typing import Optional, AsyncIterator, Generic, TypeVar

from typing_extensions import Protocol

from . import Empty, Stopped


_T = TypeVar('_T')


class RingbufferProtocol(Protocol[_T]):
    """Protocol for Python wrappers of spead2's C++ ringbuffer class."""
    @property
    def data_fd(self) -> int:
        ...

    def try_push(self, item: _T) -> None:
        ...

    def try_pop(self) -> _T:
        ...

    def pop(self) -> _T:
        ...


class AsyncRingbuffer(Generic[_T]):
    """Wraps a C++-provided ringbuffer class to provide asyncio.

    It is **not** safe to use more than one wrapper of the same base within the
    same event loop.
    """

    def __init__(self, base: RingbufferProtocol[_T]) -> None:
        self._base = base
        self._waiter = None     # type: Optional[asyncio.Future[_T]]

    @property
    def base(self) -> RingbufferProtocol[_T]:
        """The wrapped C++ ringbuffer class"""
        return self._base

    async def async_pop(self) -> _T:
        if self._waiter is not None:
            raise RuntimeError('Cannot have more than one waiter on a Ringbuffer')
        loop = asyncio.get_event_loop()
        future = loop.create_future()    # type: asyncio.Future[_T]
        self._waiter = future
        loop.add_reader(self._base.data_fd, self._ready_callback)
        try:
            return await future
        finally:
            self._waiter = None
            loop.remove_reader(self._base.data_fd)

    def try_push(self, item: _T) -> None:
        self._base.try_push(item)

    def _ready_callback(self) -> None:
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
        return self

    async def __anext__(self) -> _T:
        try:
            return await self.async_pop()
        except Stopped:
            raise StopAsyncIteration from None
