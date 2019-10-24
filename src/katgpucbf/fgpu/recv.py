import asyncio
from typing import AsyncIterator

from . import _katfgpu
from ._katfgpu.recv import Receiver, Chunk
from ._katfgpu.recv import Ringbuffer as _Ringbuffer


class Ringbuffer(_Ringbuffer):
    def __init__(self, cap: int) -> None:
        super().__init__(cap)
        self._waiter: Optional[asyncio.Future] = None

    async def async_pop(self) -> Chunk:
        if self._waiter is not None:
            raise RuntimeError('Cannot have more than one waiter on a Ringbuffer')
        loop = asyncio.get_event_loop()
        future = self._waiter = loop.create_future()
        loop.add_reader(self.data_fd, self._ready_callback)
        try:
            return await future
        finally:
            self._waiter = None
            loop.remove_reader(self.data_fd)

    def _ready_callback(self) -> None:
        if self._waiter is None or self._waiter.done():
            return
        try:
            chunk = self.try_pop()
            self._waiter.set_result(chunk)
        except _katfgpu.Empty:
            # Spurious wakeup, no action required
            pass
        except Exception as exc:
            self._waiter.set_exception(exc)

    async def __aiter__(self) -> AsyncIterator[Chunk]:
        return self

    async def __anext__(self) -> Chunk:
        try:
            return await self.async_pop()
        except _katfgpu.Stopped:
            raise AsyncStopIteration
