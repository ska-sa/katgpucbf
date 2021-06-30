"""Recv module."""

import logging
from typing import List, Optional, AsyncGenerator
from aiokatcp import Sensor

from .monitor import Monitor
from .ringbuffer import AsyncRingbuffer
from ._katfgpu.recv import Stream, Chunk, Ringbuffer


logger = logging.getLogger(__name__)


async def chunk_sets(
    streams: List[Stream], monitor: Monitor, dropped_pkt_sensor: Sensor = None
) -> AsyncGenerator[List[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    This code receives chunks of data from the C++-domain Ringbuffer, matches
    them by timestamp, and ``yield`` to the caller.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.

    Parameters
    ----------
    streams
        A list of stream objects - there should be only two of them, because
        each represents a polarisation.
    monitor
        Used for performance monitoring of the ringbuffer.
    """
    n_pol = len(streams)
    buf: List[Optional[Chunk]] = [None] * n_pol  # Working buffer to match up pairs of chunks from both pols.
    ring = AsyncRingbuffer(streams[0].ringbuffer, monitor, "recv_ringbuffer", "run_receive")
    lost = 0  # TODO this is probably something that can be reported as a katcp sensor.

    # `try`/`finally` block acting as a quick-and-dirty context manager,
    # to ensure that we clean up nicely after ourselves if we are stopped.
    try:
        async for chunk in ring:
            # Inspect the chunk we have just received.
            total = len(chunk.present)
            good = sum(chunk.present)
            lost += total - good
            if good < total:
                if dropped_pkt_sensor:
                    dropped_pkt_sensor.set_value(lost)
                logger.warning(
                    "Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)",
                    chunk.timestamp,
                    chunk.pol,
                    good,
                    total,
                    lost,
                )

            # Check whether we have a chunk already for this pol.
            old = buf[chunk.pol]
            if old is not None:
                logger.warning("Chunk not matched: timestamp=%#x pol=%d", old.timestamp, old.pol)
                # Chunk was passed by without getting used. Return to the pool.
                streams[chunk.pol].add_chunk(old)
                buf[chunk.pol] = None

            # Stick the chunk in the buffer.
            buf[chunk.pol] = chunk

            # If we have both chunks and they match up, then we can yield.
            if all(c is not None and c.timestamp == chunk.timestamp for c in buf):
                # mypy isn't smart enough to see that the list can't have Nones
                # in it at this point.
                yield buf  # type: ignore
                # Empty the buffer again for next use.
                buf = [None] * n_pol
    finally:
        for c in buf:
            if c is not None:
                streams[c.pol].add_chunk(c)


__all__ = ["chunk_sets", "Stream", "Chunk", "Ringbuffer"]
