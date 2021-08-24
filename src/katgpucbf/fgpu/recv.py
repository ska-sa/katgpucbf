"""Recv module."""

import ctypes
import logging
import time
from typing import AsyncGenerator, Final, List, Optional, cast

import numba
import numpy as np
import scipy
import spead2.recv.asyncio
from aiokatcp import Sensor, SensorSet
from numba import types
from numpy.typing import NDArray
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from .monitor import Monitor

logger = logging.getLogger(__name__)
TIMESTAMP_ID = 0x1600


class Chunk(spead2.recv.Chunk):
    # Refine the type used in the base class
    present: NDArray[np.uint8]
    data: NDArray[np.uint8]
    # New fields
    timestamp: int

    def __init__(self, *args, device: object = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.timestamp = 0  # Actual value filled in when chunk received


chunk_layout_dtype: Final[np.dtype] = np.dtype(
    [
        ("heap_samples", np.int32),
        ("heap_bytes", np.int32),
        ("chunk_heaps", np.int64),
        ("chunk_samples", np.int64),
        ("timestamp_mask", np.uint64),
    ]
)


# numba.types doesn't have a size_t, so assume it is the same as uintptr_t
@numba.cfunc(
    types.void(types.CPointer(chunk_place_data), types.uintp, types.voidptr),
    nopython=True,
)
def chunk_place(data_ptr, data_size, layout_ptr):
    data = numba.carray(data_ptr, 1)
    layout = numba.carray(layout_ptr, 1, dtype=chunk_layout_dtype)
    items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
    timestamp = items[0]
    payload_size = items[1]
    if payload_size != layout[0].heap_bytes or timestamp < 0:
        # It's something unexpected - maybe it has descriptors or a stream
        # control item. Ignore it.
        return
    timestamp &= layout[0].timestamp_mask
    if timestamp % layout[0].heap_samples != 0:
        # TODO: log/count. The timestamp is broken.
        return
    data[0].chunk_id = timestamp // layout[0].chunk_samples
    data[0].heap_index = timestamp // layout[0].heap_samples % layout[0].chunk_heaps
    data[0].heap_offset = data[0].heap_index * layout[0].heap_bytes


def add_chunk(stream: spead2.recv.ChunkRingStream, chunk: Chunk):
    """Return a chunk to the free ring."""
    # TODO: this functionality may move into spead2
    chunk.present.fill(0)
    try:
        stream.free_ringbuffer.put_nowait(chunk)
    except spead2.Stopped:
        # We're shutting down; just drop the chunk
        pass


async def chunk_sets(
    streams: List[spead2.recv.ChunkRingStream],
    layout: np.generic,
    monitor: Monitor,
    sensors: Optional[SensorSet] = None,
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
    sensors
        Sensors through which networking statistics will be reported (if provided).
    """
    n_pol = len(streams)
    # Working buffer to match up pairs of chunks from both pols.
    buf: List[Optional[Chunk]] = [None] * n_pol
    # TODO: bring back ringbuffer monitoring capabilities
    ring = cast(spead2.recv.asyncio.ChunkRingbuffer, streams[0].data_ringbuffer)
    lost = 0
    first_timestamp = -1  # Updated to the actual first timestamp on the first chunk
    # Convert from numpy types to plain Python ints
    heap_samples = int(layout["heap_samples"])
    heap_bytes = int(layout["heap_bytes"])
    chunk_heaps = int(layout["chunk_heaps"])
    chunk_samples = int(layout["chunk_samples"])

    # `try`/`finally` block acting as a quick-and-dirty context manager,
    # to ensure that we clean up nicely after ourselves if we are stopped.
    try:
        async for chunk in ring:
            assert isinstance(chunk, Chunk)
            # Inspect the chunk we have just received.
            chunk.timestamp = chunk.chunk_id * chunk_samples
            pol = chunk.stream_id
            if first_timestamp == -1:
                # TODO: use chunk.present to determine the actual first timestamp
                first_timestamp = chunk.timestamp
            good = np.sum(chunk.present)
            lost += chunk_heaps - good
            if good < chunk_heaps:
                logger.debug(
                    "Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)",
                    chunk.timestamp,
                    pol,
                    good,
                    chunk_heaps,
                    lost,
                )

            # Check whether we have a chunk already for this pol.
            old = buf[pol]
            if old is not None:
                logger.warning("Chunk not matched: timestamp=%#x pol=%d", old.chunk_id * chunk_samples, pol)
                # Chunk was passed by without getting used. Return to the pool.
                add_chunk(streams[pol], old)
                buf[pol] = None

            # Stick the chunk in the buffer.
            buf[pol] = chunk

            # If we have both chunks and they match up, then we can yield.
            if all(c is not None and c.chunk_id == chunk.chunk_id for c in buf):
                if sensors:
                    # Get explicit timestamp to ensure that all the updated sensors
                    # have the same timestamp.
                    sensor_timestamp = time.time()

                    def increment(sensor: Sensor, incr: int):
                        sensor.set_value(sensor.value + incr, timestamp=sensor_timestamp)

                    # mypy isn't smart enough to see that the list can't have Nones
                    # in it at this point. The cast is to force numpy ints to
                    # Python ints.
                    buf_good = sum(int(np.sum(c.present)) for c in buf)  # type: ignore
                    increment(sensors["input-heaps-total"], buf_good)
                    increment(sensors["input-chunks-total"], n_pol)
                    increment(sensors["input-bytes-total"], buf_good * heap_bytes)
                    # Determine how many heaps we expected to have seen by
                    # now, and subtract from it the number actually seen to
                    # determine the number missing. This accounts for both
                    # heaps lost within chunks and lost chunks.
                    received_heaps = sensors["input-heaps-total"].value
                    expected_heaps = (chunk.timestamp + chunk_samples) * n_pol // heap_samples
                    missing = expected_heaps - received_heaps
                    if missing > sensors["input-missing-heaps-total"].value:
                        sensors["input-missing-heaps-total"].set_value(missing, timestamp=sensor_timestamp)

                # mypy isn't smart enough to see that the list can't have Nones
                # in it at this point.
                yield buf  # type: ignore
                # Empty the buffer again for next use.
                buf = [None] * n_pol
    finally:
        for c in buf:
            if c is not None:
                add_chunk(streams[c.stream_id], c)


def make_chunk_layout(
    sample_bits: int,
    packet_samples: int,
    chunk_samples: int,
    mask_timestamp: bool,
) -> np.generic:
    layout = np.zeros((), chunk_layout_dtype)[()]
    layout["heap_samples"] = packet_samples
    layout["heap_bytes"] = packet_samples * sample_bits // 8
    layout["chunk_heaps"] = chunk_samples // packet_samples
    layout["chunk_samples"] = chunk_samples
    layout["timestamp_mask"] = ~np.uint64(packet_samples - 1 if mask_timestamp else 0)
    return layout


def make_stream(
    pol: int,
    layout: np.generic,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    affinity: int,
    use_gdrcopy: bool,
    monitor: Monitor,
) -> spead2.recv.ChunkRingStream:
    stream_config = spead2.recv.StreamConfig(
        max_heaps=1,
        memcpy=spead2.MEMCPY_NONTEMPORAL if use_gdrcopy else spead2.MEMCPY_STD,
        stream_id=pol,
    )
    place = scipy.LowLevelCallable(
        chunk_place.ctypes,
        signature="void (void *, size_t, void *)",
        user_data=np.array(layout).ctypes.data_as(ctypes.c_void_p),
    )
    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=[TIMESTAMP_ID, spead2.HEAP_LENGTH_ID], max_chunks=2, place=place
    )
    # Ringbuffer size is largely arbitrary: just needs to be big enough to
    # never fill up.
    free_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(128)
    return spead2.recv.ChunkRingStream(
        spead2.ThreadPool(1, [] if affinity < 0 else [affinity]),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )
    # TODO: hook up Monitor somehow


# TODO: update
__all__ = ["Chunk", "chunk_sets", "make_chunk_layout", "make_stream"]
