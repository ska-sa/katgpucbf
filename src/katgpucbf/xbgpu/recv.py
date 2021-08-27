"""SPEAD receiver utilities."""

import numba
import numpy as np
import scipy
import spead2.recv.asyncio
import spead2.send
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101


class Chunk(spead2.recv.Chunk):
    """Extend :class:`spead2.recv.Chunk` to track timestamp."""

    # Refine base class types
    data: np.ndarray
    present: np.ndarray
    # New attributes
    timestamp: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = -1  # Computed from chunk_id when we receive the chunk


def make_stream(
    n_ants: int,
    n_channels_per_stream: int,
    n_samples_per_channel: int,
    n_pols: int,
    sample_bits: int,
    timestamp_step: int,
    heaps_per_fengine_per_chunk: int,
    max_active_chunks: int,
    ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    thread_affinity: int,
):
    """Create a SPEAD receiver stream.

    Parameters
    ----------
    n_ants
        The number of antennas that data will be received from
    n_channels
        The total number of frequency channels out of the F-Engine.
    n_channels_per_stream
        The number of frequency channels contained in the stream.
    n_samples_per_channel
        The number of time samples received per frequency channel.
    n_pols
        The number of pols per antenna. Expected to always be 2 at the moment.
    sample_bits
        The number of bits per sample. Only 8 bits is supported at the moment.
    timestamp_step
        Each heap contains a timestamp. The timestamp between consecutive heaps
        changes depending on the FFT size and the number of time samples per
        channel. This parameter defines the difference in timestamp values
        between consecutive heaps. This parameter can be calculated from the
        array configuration parameters for power-of-two array sizes, but is
        configurable to allow for greater flexibility during testing.
    heaps_per_fengine_per_chunk
        Each chunk out of the SPEAD2 receiver will contain multiple heaps from
        each antenna. This parameter specifies the number of heaps per antenna
        that each chunk will contain.
    max_active_chunks
        Maximum number of chunks under construction.
    ringbuffer
        All completed heaps will be queued on this ringbuffer object.
    thread_affinity
        CPU Thread that this receiver will use for processing.
    """
    heap_bytes = n_channels_per_stream * n_pols * 2 * sample_bits // 8  # * 2 because samples are complex

    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
    def chunk_place_impl(data_ptr, data_size):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
        timestamp = items[0]
        fengine = items[1]
        payload_size = items[2]
        if payload_size != heap_bytes or timestamp < 0 or fengine < 0:
            # It's something unexpected - possibly descriptors. Ignore it.
            return
        if timestamp % timestamp_step != 0:
            # Invalid timestamp. TODO: log/count it somehow
            return
        if fengine >= n_ants:
            # Invalid F-engine ID. TODO: log/count it somehow
            return
        # Compute position of this heap on the time axis, starting from timestamp 0
        heap_time_abs = timestamp // timestamp_step
        data[0].chunk_id = heap_time_abs // heaps_per_fengine_per_chunk
        # Position of this heap on the time axis, from the start of the chunk
        heap_time = heap_time_abs % heaps_per_fengine_per_chunk
        data[0].heap_index = heap_time * n_ants + fengine
        data[0].heap_offset = data[0].heap_index * heap_bytes

    # max_heaps is set quite high because timing jitter/bursting means there could
    # be multiple heaps from one F-engine during the time it takes another to
    # transmit.
    stream_config = spead2.recv.StreamConfig(
        max_heaps=n_ants * (2 + spead2.send.StreamConfig.DEFAULT_BURST_SIZE // heap_bytes),
    )
    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=[TIMESTAMP_ID, FENGINE_ID, spead2.HEAP_LENGTH_ID],
        max_chunks=max_active_chunks,
        place=scipy.LowLevelCallable(chunk_place_impl.ctypes, signature="void (void *, size_t)"),
    )
    free_ringbuffer = spead2.recv.ChunkRingbuffer(ringbuffer.maxsize)
    return spead2.recv.ChunkRingStream(
        spead2.ThreadPool(1, [] if thread_affinity < 0 else [thread_affinity]),
        stream_config,
        chunk_stream_config,
        ringbuffer,
        free_ringbuffer,
    )
