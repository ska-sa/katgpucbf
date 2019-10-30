import asyncio
from typing import Optional, AsyncIterator

from ._katfgpu.send import Sender, Chunk, Ringbuffer   # noqa: F401
from . import Empty, Stopped


# TODO: unify with code in recv.py
class AsyncRingbuffer:
    def __init__(self, base: Ringbuffer) -> None:
        self._waiter: Optional[asyncio.Future] = None
        self._base = base

    async def async_pop(self) -> Chunk:
        if self._waiter is not None:
            raise RuntimeError('Cannot have more than one waiter on a Ringbuffer')
        loop = asyncio.get_event_loop()
        future = self._waiter = loop.create_future()
        loop.add_reader(self._base.data_fd, self._ready_callback)
        try:
            return await future
        finally:
            self._waiter = None
            loop.remove_reader(self._base.data_fd)

    def try_push(self, chunk: Chunk) -> None:
        self._base.try_push(chunk)

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

    def __aiter__(self) -> AsyncIterator[Chunk]:
        return self

    async def __anext__(self) -> Chunk:
        try:
            return await self.async_pop()
        except Stopped:
            raise StopAsyncIteration from None
