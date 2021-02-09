"""TODO: Add a comment."""

# import katxgpu
import katxgpu._katxgpu.recv as recv
import katxgpu.monitor
import katxgpu.ringbuffer
import spead2
import spead2.send

import logging
import asyncio
import numpy as np
import katsdpsigproc.accel as accel

logger = logging.getLogger(__name__)

# TODO: Check data format is correct
# TODO: Add katxgpu receiver to this

# 1. Define Constants
# SPEAD IDs
TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x4300

# Configuration parameters
n_ants = 16
n_channels_total = 32768
n_channels_per_stream = 128
n_samples_per_channel = 256
n_pols = 2
sample_bits = 8
heaps_per_fengine_per_chunk = 10
complexity = 2

timestamp_step = (
    n_channels_total * 2 * n_samples_per_channel
)  # Multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.

n_heaps_in_flight_per_antenna = 10

# 2. Set up receiver
max_packet_size = (
    n_samples_per_channel * n_pols * complexity * sample_bits // 8 + 96
)  # Header is 12 fields of 8 bytes each: So 96 bytes of header
thread_pool = spead2.ThreadPool()
streamSource = spead2.send.BytesStream(
    thread_pool,
    spead2.send.StreamConfig(max_packet_size=max_packet_size, max_heaps=n_ants * heaps_per_fengine_per_chunk),
)
del thread_pool

# 3. Define Heap Format
shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
ig = spead2.send.ItemGroup(flavour=spead2.Flavour(4, 64, 48, 0))
item_timestamp = ig.add_item(TIMESTAMP_ID, "timestamp", "timestamp description", shape=[], format=[("u", 48)])
item_fengine = ig.add_item(FENGINE_ID, "fengine id", "F-Engine heap is received from", shape=[], format=[("u", 48)])
item_channel = ig.add_item(
    CHANNEL_OFFSET, "Channel offset", "Value of first channel in collections stored here", shape=[], format=[("u", 48)]
)
item_data = ig.add_item(DATA_ID, "FENG_RAW", "Raw Channelised data", shape=shape, dtype=np.int8)
# Adding padding
item_padding = []
for i in range(3):
    item_padding.append(
        ig.add_item(
            CHANNEL_OFFSET + 1 + i,
            f"Padding {i}",
            "Padding field {i} to align header to 256-bit boundary.",
            shape=[],
            format=[("u", 48)],
        )
    )
heap = (
    ig.get_heap()
)  # Throwaway heap - need to get this as it contains a bunch of descriptor information that we dont want for the purposes of this test.


# 4. Create and array of heaps to send

sample_value = 1


def createHeaps(timestamp: int):
    """
    Generate a list of heaps to send in an interleaved manner.

    The list is of HeapReference objects which point to the heaps as this is what the send_heaps() function requires.
    """
    global sample_value
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        item_timestamp.value = timestamp
        item_fengine.value = ant_index
        item_channel.value = n_channels_per_stream * 4  # Arbitrary multiple for now
        item_data.value = np.full(shape, sample_value, np.int8)
        for item in item_padding:
            item.value = 0
        heap = ig.get_heap()
        heap.repeat_pointers = True
        sample_value += 1

        # NOTE: The substream_index is set to zero as the SPEAD BytesStream transport has not had the concept of substreams introduced. It has not been updated along with the rest of the transports. As such the unit test cannot yet test that packet interleaving works correctly. I am not sure if this feature is planning to be added. If it is, then set `substream_index=ant_index`. If this starts becoming an issue, then we will need to lok at using the inproc transport. The inproc transport would be much better, but requires porting a bunch of things from SPEAD2 python to katxgpu python. So it will require much more work.
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=0))
    return heaps


# 5. Transmit initial few heaps
for i in range(n_heaps_in_flight_per_antenna):
    heaps = createHeaps(timestamp_step * i)
    streamSource.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

# 6. Create all receiver data

# 6.1 Create monitor for file
use_file_monitor = False
if use_file_monitor:
    monitor = katxgpu.monitor.FileMonitor("temp_file.log")
else:
    monitor = katxgpu.monitor.NullMonitor()

# 6.2 Create ringbuffer
ringbuffer_capacity = 8
ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)

# 6.3 Create Receiver
thread_affinity = 2
receiverStream = recv.Stream(
    n_ants,
    n_channels_per_stream,
    n_samples_per_channel,
    n_pols,
    sample_bits,
    timestamp_step,
    heaps_per_fengine_per_chunk,
    ringbuffer,
    thread_affinity,
    use_gdrcopy=False,
    monitor=monitor,
)

# 6.4 Add free chunks to SPEAD2 receiver
context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
src_chunks_per_stream = 8
monitor.event_qsize("free_chunks", 0, src_chunks_per_stream)
for i in range(src_chunks_per_stream):
    buf = accel.HostArray((receiverStream.chunk_bytes,), np.uint8, context=context)
    chunk = recv.Chunk(buf)
    receiverStream.add_chunk(chunk)

receiverStream.add_buffer_reader(streamSource.getvalue())
# receiverStream.add_udp_ibv_reader([("239.10.10.10", 7149)], "10.100.44.1", 10000000, 0)

asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
    receiverStream.ringbuffer, monitor, "recv_ringbuffer", "get_chunks"
)


async def get_chunks():
    """TODO: Create docstring."""
    print("Main asyncio loop now running.")
    i = 0
    dropped = 0
    received = 0
    async for chunk in asyncRingbuffer:
        received += len(chunk.present)
        dropped += len(chunk.present) - sum(chunk.present)
        print(
            f"Chunk: {i:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}. {len(chunk.base)}"
        )
        receiverStream.add_chunk(chunk)
        i += 1


loop = asyncio.get_event_loop()
loop.run_until_complete(get_chunks())
loop.close()
