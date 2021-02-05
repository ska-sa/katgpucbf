"""TODO: Write a description."""

import spead2
import spead2.send
import spead2.recv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# TODO: Send multiple heaps in one go
# TODO: Check data format is correct
# TODO: Add katxgpu receiver to this
# TODO: Add in

# Configuration parameters
n_ants = 64
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


# SPEAD IDs
TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x4300

queue = spead2.InprocQueue()

thread_pool = spead2.ThreadPool()
streamSource = spead2.send.InprocStream(thread_pool, [queue], spead2.send.StreamConfig())  # Packet size
del thread_pool

shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
ig = spead2.send.ItemGroup(flavour=spead2.Flavour(4, 64, 48, 0))
item_timestamp = ig.add_item(TIMESTAMP_ID, "timestamp", "timestamp description", shape=[], format=[("u", 48)])
item_timestamp.value = timestamp_step
item_fengine = ig.add_item(FENGINE_ID, "fengine id", "F-Engine heap is received from", shape=[], format=[("u", 48)])
item_fengine.value = 32
item_channel = ig.add_item(
    CHANNEL_OFFSET, "Channel offset", "Value of first channel in collections stored here", shape=[], format=[("u", 48)]
)
item_channel.value = n_channels_per_stream * 4  # Arbitrary multiple for now

# Adding padding
for i in range(3):
    item_padding = ig.add_item(
        CHANNEL_OFFSET + 1 + i,
        f"Padding {i}",
        "Padding field {i} to align header to 256-bit boundary.",
        shape=[],
        format=[("u", 48)],
    )
    item_padding.value = 0  # Arbitrary multiple for now

item_data = ig.add_item(DATA_ID, "FENG_RAW", "Raw Channelised data", shape=shape, dtype=np.int8)
item_data.value = np.zeros(shape, np.int8)
heap = ig.get_heap()
heap.repeat_pointers = True
print(heap.repeat_pointers)
streamSource.send_heaps([spead2.send.HeapReference(heap, cnt=-1, substream_index=0)], spead2.send.GroupMode.ROUND_ROBIN)

thread_pool = spead2.ThreadPool()
streamDest = spead2.recv.Stream(
    thread_pool, spead2.recv.StreamConfig(memory_allocator=spead2.MemoryPool(16384, 26214400, 12, 8))
)
del thread_pool

streamDest.add_inproc_reader(queue)
i = 0
for heap in streamDest:
    i += 1
    print("Got heap", heap.cnt)
    items = ig.update(heap)
    for item in items.values():
        if item.shape == ():
            print(f"\t {item.name}, {hex(item.value)}")
        else:
            print(f"\t {item.name}, {item.shape}")
    streamSource.send_heap(ig.get_heap())
    if i > 50000:
        streamSource.send_heap(ig.get_end())

print("Done")
