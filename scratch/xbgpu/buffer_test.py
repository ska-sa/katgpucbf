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

context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
buf = accel.HostArray((1024 * 1024 * 1024,), np.uint8, context=context)

buf[2] = 15

print(buf)

chunk = recv.Chunk(buf)

print(buf)

print(chunk.timestamp)
print(chunk.present)

print(recv.Ringbuffer.__doc__)

use_file_monitor = False

# Create monitor for file
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

receiverStream = recv.Stream(
    2,
    8,
    packet_size_bytes,
    packet_size_bytes * 8192,
    ringbuffer,
    thread_affinity,
    mask_timestamp=True,
    use_gdrcopy=False,
    monitor=monitor,
)

# Add free chunks to SPEAD2 receiver
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
    async for chunk in asyncRingbuffer:
        print("Here", chunk)
        receiverStream.add_chunk(chunk)


print("Here")

loop = asyncio.get_event_loop()
loop.run_until_complete(get_chunks())
loop.close()
