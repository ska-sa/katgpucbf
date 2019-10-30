import asyncio
from typing import List, AsyncIterator, AsyncGenerator, Optional, cast

from . import Empty, Stopped
from .ringbuffer import AsyncRingbuffer
from ._katfgpu.recv import Stream, Chunk, Ringbuffer


async def chunk_sets(streams: List[Stream]) -> AsyncGenerator[List[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.
    """
    n_pol = len(streams)
    buf: List[Optional[Chunk]] = [None] * n_pol
    ring = AsyncRingbuffer(streams[0].ringbuffer)
    lost = 0
    try:
        async for chunk in ring:
            total = len(chunk.present)
            good = sum(chunk.present)
            lost += total - good
            print(f'Received chunk: timestamp={chunk.timestamp} '
                  f'pol={chunk.pol} ({good}/{total}, lost {lost})')
            old = buf[chunk.pol]
            if old is not None:
                # Chunk was passed by without getting used. Return to the pool.
                streams[chunk.pol].add_chunk(old)
                buf[chunk.pol] = None
            buf[chunk.pol] = chunk
            if all(c is not None and c.timestamp == chunk.timestamp for c in buf):
                # We have a matched set, yield it. mypy isn't smart enough to
                # see that the list can't have Nones in it at this point.
                yield buf      # type: ignore
                buf = [None] * n_pol
    finally:
        for c in buf:
            if c is not None:
                streams[chunk.pol].add_chunk(c)
