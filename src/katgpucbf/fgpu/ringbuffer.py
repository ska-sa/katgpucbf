# noqa D100
import asyncio
from typing import AsyncIterator, Generic, Optional, Protocol, TypeVar

from ..monitor import Monitor
from . import Empty, Stopped

_T = TypeVar("_T")


class RingbufferProtocol(Protocol[_T]):
    """Protocol for Python wrappers of spead2's C++ ringbuffer class."""

    @property
    def data_fd(self) -> int:  # noqa D102
        ...

    def try_push(self, item: _T) -> None:  # noqa D102
        ...

    def try_pop(self) -> _T:  # noqa D102
        ...

    def pop(self) -> _T:  # noqa D102
        ...


class AsyncRingbuffer(Generic[_T]):
    """Wraps a C++-provided ringbuffer class to provide asyncio.

    It is **not** safe to use more than one wrapper of the same base within the
    same event loop.

    More information can be obtained by looking at
    :class:`katgpucbf.fgpu.recv.Ringbuffer` or
    :class:`katgpucbf.fgpu.send.Ringbuffer`.

    Parameters
    ----------
    base
        base c++ ringbuffer object
    monitor
        `Monitor` to use for performance monitoring.
    name
        Name of the ringbuffer, used for reporting.
    task_name
        Name of the task (e.g. send or receive) using the ringbuffer, used for
        reporting.
    """

    def __init__(self, base: RingbufferProtocol[_T], monitor: Monitor, name: str, task_name: str) -> None:
        self._base = base
        self._waiter = None  # type: Optional[asyncio.Future[_T]]
        self._monitor = monitor
        self._name = name
        self._task_name = task_name

    @property
    def base(self) -> RingbufferProtocol[_T]:  # noqa D401
        """The wrapped C++ ringbuffer class."""
        return self._base

    async def async_pop(self) -> _T:
        """Pop from the ringbuffer asynchronously.

        If there is an item on the ringbuffer, retrieve it, otherwise await
        until there is an item available to pop.
        """
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

    def try_push(self, item: _T) -> None:  # noqa D102
        self._base.try_push(item)

    def _ready_callback(self) -> None:  # noqa D102
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
