"""TODO: Write a description."""

import spead2
import spead2.send
import spead2.recv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

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

n_heaps_in_flight_per_antenna = 1

# 2. Set up receiver and transmitter - link together via queues for an "in process" transport. One queue is required per
# substream.
queues = []
for ant_index in range(n_ants):
    queues.append(spead2.InprocQueue())

max_packet_size = (
    n_samples_per_channel * n_pols * complexity * sample_bits // 8 + 96
)  # Header is 12 fields of 8 bytes each: So 96 bytes of header
thread_pool = spead2.ThreadPool()
streamSource = spead2.send.InprocStream(
    thread_pool,
    queues,
    spead2.send.StreamConfig(max_packet_size=max_packet_size, max_heaps=n_ants * heaps_per_fengine_per_chunk),
)
del thread_pool

receiver_buffer_size_bytes = 838860800

thread_pool = spead2.ThreadPool()
streamDest = spead2.recv.Stream(
    thread_pool,
    spead2.recv.StreamConfig(
        memory_allocator=spead2.MemoryPool(receiver_buffer_size_bytes, receiver_buffer_size_bytes, 12, 8)
    ),
)
del thread_pool

for ant_index in range(n_ants):
    streamDest.add_inproc_reader(queues[ant_index])

# 3. Define Heap Format
shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
ig = spead2.send.ItemGroup(flavour=spead2.Flavour(4, 64, 48, 0))
item_timestamp = ig.add_item(TIMESTAMP_ID, "timestamp", "timestamp description", shape=[], format=[("u", 48)])
item_fengine = ig.add_item(FENGINE_ID, "fengine id", "F-Engine heap is received from", shape=[], format=[("u", 48)])
item_channel = ig.add_item(
    CHANNEL_OFFSET, "Channel offset", "Value of first channel in collections stored here", shape=[], format=[("u", 48)]
)
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

item_data = ig.add_item(DATA_ID, "FENG_RAW", "Raw Channelised data", shape=shape, dtype=np.int8)


# 4. Create and array of heaps to send
def createHeaps(timestamp: int):
    """
    Generate a list of heaps to send in an interleaved manner.

    The list is of HeapReference objects which point to the heaps as this is what the send_heaps() function requires.
    """
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        item_timestamp.value = timestamp
        item_fengine.value = ant_index
        item_channel.value = n_channels_per_stream * 4  # Arbitrary multiple for now
        item_data.value = np.zeros(shape, np.int8)
        for item in item_padding:
            item.value = 0
        heap = ig.get_heap()
        heap.repeat_pointers = True
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=ant_index))
    return heaps


# 5. Transmit initial few heaps
for i in range(n_heaps_in_flight_per_antenna):
    heaps = createHeaps(timestamp_step * i)
    streamSource.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

# 6. Loop through all data

print("Starting Receiver")

timestamp_index = n_heaps_in_flight_per_antenna
i = 0
for heap in streamDest:

    print("Got heap", heap.cnt)
    items = ig.update(heap)
    for item in items.values():
        if item.shape == ():
            print(f"\t {item.name}, {hex(item.value)}")
        else:
            print(f"\t {item.name}, {item.shape}")

    if (i - 1) % n_ants == 0:
        heaps = createHeaps(timestamp_step * timestamp_index)
        streamSource.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        timestamp_index += 1

    i += 1

    if i > 50000:
        streamSource.send_heap(ig.get_end())

print("Done")
