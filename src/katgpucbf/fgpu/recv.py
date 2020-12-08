import logging
from typing import List, Optional, AsyncGenerator

import katsdpsigproc.accel

from .monitor import Monitor
from .ringbuffer import AsyncRingbuffer
from ._katfgpu.recv import Stream, Chunk, Ringbuffer


logger = logging.getLogger(__name__)


async def chunk_sets(streams: List[Stream],
                     monitor: Monitor) -> AsyncGenerator[List[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.
    """
    n_pol = len(streams)
    buf: List[Optional[Chunk]] = [None] * n_pol
    ring = AsyncRingbuffer(streams[0].ringbuffer, monitor, 'recv_ringbuffer', 'run_receive')
    lost = 0
    try:
        async for chunk in ring:
            total = len(chunk.present)
            good = sum(chunk.present)
            lost += total - good
            if good < total:
                logger.warning('Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)',
                               chunk.timestamp, chunk.pol, good, total, lost)
            old = buf[chunk.pol]
            if old is not None:
                logger.warning('Chunk not matched: timestamp=%#x pol=%d',
                               old.timestamp, old.pol)
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
                streams[c.pol].add_chunk(c)


__all__ = ['chunk_sets', 'Stream', 'Chunk', 'Ringbuffer']
