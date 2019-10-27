import asyncio
from typing import List, AsyncIterator, AsyncGenerator, Optional

from . import _katfgpu
from ._katfgpu.recv import Stream, Chunk
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
            raise StopAsyncIteration from None


async def chunk_sets(streams: List[Stream]) -> AsyncGenerator[List[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.
    """
    n_pol = len(streams)
    buf = [None] * n_pol
    ring = streams[0].ringbuffer
    lost = 0
    try:
        async for chunk in ring:
            total = len(chunk.present)
            good = sum(chunk.present)
            lost += total - good
            print('Received chunk: timestamp={chunk.timestamp} pol={chunk.pol} ({good}/{total}, lost {lost})'.format(
                chunk=chunk, good=good, total=total, lost=lost))
            if buf[chunk.pol] is not None:
                # Chunk was passed by without getting used. Return to the pool.
                streams[chunk.pol].add_chunk(buf[chunk.pol])
                buf[chunk.pol] = None
            buf[chunk.pol] = chunk
            if all(c is not None and c.timestamp == chunk.timestamp for c in buf):
                # We have a matched set, yield it
                yield buf
                buf = [None] * n_pol
    finally:
        for chunk in buf:
            if chunk is not None:
                streams[chunk.pol].add_chunk(chunk)
