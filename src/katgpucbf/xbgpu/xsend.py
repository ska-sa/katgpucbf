"""TODO: Write this."""


# 2. Create sourceStream object - transforms "transmitted" heaps into a byte array to simulate received data.

import spead2
import spead2.send.asyncio
import katsdpsigproc.accel as accel
import numpy as np
import time
import asyncio


default_spead_flavour = {"version": 4, "item_pointer_bits": 64, "heap_address_bits": 48, "bug_compat": 0}

# 3.1 SPEAD IDs
TIMESTAMP_ID = 0x1600
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x1800

max_payload_size = 2048
max_packet_size = max_payload_size + 64

complexity = 2
n_ants = 64
n_channels_total = 32768
n_channels_per_stream = 128
n_samples_per_channel = 256
n_pols = 2
n_baselines = (n_pols * n_ants + 1) * (n_ants * n_pols) // 2
sample_bits = 8

heap_size_bytes = n_channels_per_stream * n_baselines * complexity * sample_bits // 8

endpoint = ("239.10.10.10", 7149)

context = accel.create_some_context(device_filter=lambda x: x.is_cuda)

n_send_chunks = 1
bufs = []

for i in range(n_send_chunks):
    # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
    buf = accel.HostArray((heap_size_bytes,), np.uint8, context=context)
    # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
    bufs.append(buf)

print(bufs)

thread_pool = spead2.ThreadPool()
sourceStream = spead2.send.asyncio.UdpIbvStream(
    thread_pool,
    spead2.send.StreamConfig(
        max_packet_size=max_packet_size, max_heaps=10, rate_method=spead2.send.RateMethod.AUTO, rate=10e6
    ),
    spead2.send.UdpIbvConfig(
        endpoints=[endpoint], interface_address="10.100.44.1", ttl=4, comp_vector=2, memory_regions=bufs
    ),
)
del thread_pool  # This line is copied from the SPEAD2 examples.
sourceStream.repeat_pointers = True

ig = spead2.send.ItemGroup(flavour=spead2.Flavour(**default_spead_flavour))
heap_shape = (n_channels_per_stream, n_baselines, complexity)
ig.add_item(
    CHANNEL_OFFSET,
    "channel offset",
    "Value of first channel in collections stored here",
    shape=[],
    format=[("u", default_spead_flavour["heap_address_bits"])],
)
ig.add_item(
    TIMESTAMP_ID,
    "timestamp",
    "timestamp description",
    shape=[],
    format=[("u", default_spead_flavour["heap_address_bits"])],
)
ig.add_item(DATA_ID, "xeng_raw", "Integrated baseline correlation products", shape=(heap_size_bytes,), dtype=np.int8)

# 3.2 Throw away first heap - need to get this as it contains a bunch of descriptor information that we dont want
# for the purposes of this test.
# ig.get_heap()

for i in range(n_send_chunks):
    ig["timestamp"].value = i * 0x1000
    ig["channel offset"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
    ig["xeng_raw"].value = bufs[i]
    print("Sending", time.time())
    futures = [sourceStream.async_send_heap(ig.get_heap())]
    asyncio.get_event_loop().run_until_complete(asyncio.wait(futures))
    print("Sent   ", time.time())
    print()
