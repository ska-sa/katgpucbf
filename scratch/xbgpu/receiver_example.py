"""TODO: Add a comment."""

# import katxgpu
import katxgpu._katxgpu.recv as recv
import katxgpu.monitor
import katxgpu.ringbuffer

import logging
import asyncio
import numpy as np
import katsdpsigproc.accel as accel

logger = logging.getLogger(__name__)

# Create monitor for file
use_file_monitor = True
if use_file_monitor:
    monitor = katxgpu.monitor.FileMonitor("temp_file.log")
else:
    monitor = katxgpu.monitor.NullMonitor()

# Create ringbuffer
ringbuffer_capacity = 8
ringbuffer = recv.Ringbuffer(ringbuffer_capacity)
monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)

# Create SPEAD2 receiver


packet_size_bytes = 1024
thread_affinity = 2

n_ants = 64
n_channels_total = 32768
n_channels_per_stream = 128
n_samples_per_channel = 256
n_pols = 2
sample_bits = 8
heaps_per_fengine_per_chunk = 10

# Multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.
timestamp_step = n_channels_total * 2 * n_samples_per_channel

print("Timestamp step: ", hex(timestamp_step))

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

# Add free chunks to SPEAD2 receiver
context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
src_chunks_per_stream = 8
monitor.event_qsize("free_chunks", 0, src_chunks_per_stream)
for i in range(src_chunks_per_stream):
    buf = accel.HostArray((receiverStream.chunk_bytes,), np.uint8, context=context)
    chunk = recv.Chunk(buf)
    receiverStream.add_chunk(chunk)

receiverStream.add_udp_ibv_reader([("239.10.10.10", 7149)], "10.100.44.1", 10000000, 0)

asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(ringbuffer, monitor, "recv_ringbuffer", "get_chunks")


async def get_chunks():
    """TODO: Create docstring."""
    print("Starting Main Loop")
    i = 0
    dropped = 0
    received = 0
    async for chunk in asyncRingbuffer:
        received += len(chunk.present)
        dropped += len(chunk.present) - sum(chunk.present)
        print(
            "Chunk:",
            i,
            "Received:",
            sum(chunk.present),
            "of",
            len(chunk.present),
            "expected heaps. All time dropped/received heaps:",
            dropped,
            "/",
            received,
        )
        receiverStream.add_chunk(chunk)
        i += 1


print("Here")

loop = asyncio.get_event_loop()
loop.run_until_complete(get_chunks())
loop.close()
