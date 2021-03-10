"""Katxgpu receiver example script.

This script demonstrates how to configure a katxgpu receiver object to receive data from an fsim or MeerKAT
channelised voltage stream (or more simply an X-Engine input stream).

It also shows how a chunks can be received asynchronously and shows how to pass a used chunk back to the katxgpu
receiver.
"""
# 1. Imports
# 1.1 Local imports
import katxgpu._katxgpu.recv as recv
import katxgpu.monitor
import katxgpu.ringbuffer

# 1.1 External imports
import argparse
import logging
import asyncio
import numpy as np
import katsdpsigproc.accel as accel

logger = logging.getLogger(__name__)

# 2. Relevant variables

# 2.1 Parsing command line arguments

parser = argparse.ArgumentParser(description="Simple example demonstrating how to use katxgpu receiver software.")
parser.add_argument("--mcast_src_ip", default="239.10.10.10", help="IP address of multicast stream to subscribe to.")
parser.add_argument("--mcast_src_port", default="7149", type=int, help="Port of multicast stream to subscribe to.")
parser.add_argument(
    "--src_interface", default="10.100.44.1", help="IP Address of interface that will receive the data."
)
args = parser.parse_args()
print(args)
src_multicast_ip = args.mcast_src_ip
src_multicast_port = args.mcast_src_port
src_interface_ip = args.src_interface

print(src_multicast_ip, src_multicast_port, src_interface_ip)


# 2.2 Hard coded variables declaration
thread_affinity = 2
n_ants = 64
n_channels_total = 32768
n_channels_per_stream = 128
n_samples_per_channel = 256
n_pols = 2
sample_bits = 8
heaps_per_fengine_per_chunk = 10

# This step represents the difference in timestamp between two consecutive heaps received from the same F-Engine. We
# multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.
# While we can workout the timestamp_step from other parameters that configure the receiver, we pass it as a seperate
# argument to the reciever for cases where the n_samples_per_channel changes across streams (likely for non-power-of-
# two array sizes).
timestamp_step = n_channels_total * 2 * n_samples_per_channel

print(f"Timestamp step: {hex(timestamp_step)}")

# 3. Create a monitor to monitor resource use on the reciever. If use_file_monitor is set to true, this monitor will
# write to file for later examination (this likely has a performance hit, but this has not been quantified yet). If this
# is set to false, then a null moitor is used bypassing error reporting. Examine the temp_file.log file that is produced
# when this is set to true to see the format of the log file.
use_file_monitor = False
if use_file_monitor:
    monitor: katxgpu.monitor.Monitor = katxgpu.monitor.FileMonitor("temp_file.log")
else:
    monitor = katxgpu.monitor.NullMonitor()

# 4. Create ringbuffer - All chunks that the receiver assembles are placed on this ringbuffer.
ringbuffer_capacity = 8
ringbuffer = recv.Ringbuffer(ringbuffer_capacity)

# 4.1 Sets the initial value that the monitor uses to track the ringbuffer capacity.
monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)


# 5. Create receiver object
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

# 6. Create empty chunks and give them to the receiver. The receiver will then place received packets in these given
# chunks.

# 6.1 We need am accel context as the buffers from this context can be transferred to the GPU.
context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
src_chunks_per_stream = 8
monitor.event_qsize("free_chunks", 0, src_chunks_per_stream)
for i in range(src_chunks_per_stream):
    # 6.1.1 Create a buffer from this accel context. The size of the buffer is equal to the chunk size.
    buf = accel.HostArray((receiverStream.chunk_bytes,), np.uint8, context=context)
    # 6.2 Create a chunk - the buffer object is given to this chunk. This is where sample data in a chunk is stored.
    chunk = recv.Chunk(buf)
    # 6.3 Give the chunk to the receiver - once this is done we no longer need to track the chunk object.
    receiverStream.add_chunk(chunk)

# 7. Add a "transport" to the reciever. The add_udp_ibv_reader() transport tells the receiver to listen on a specific
# ethernet interface using the ibverbs acceleration tools. Once this transport is added, the receiver stream will start
# receiving any relevant packets off of the network.

receiverStream.add_udp_ibv_reader([(src_multicast_ip, src_multicast_port)], src_interface_ip, 10000000, 0)

# 8. Receive chunks asyncronously in python from the receiver.

# 8.1 Wrap the receiver ringbuffer in an AsyncRIngbuffer object so that chunks can be passed to python in an asyncio
# loop.
asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
    receiverStream.ringbuffer, monitor, "recv_ringbuffer", "get_chunks"
)


async def get_chunks():
    """Receive and process completed chunks from the katxgpu receiver."""
    print("Main asyncio loop now running.")
    i = 0
    dropped = 0
    received = 0
    # 8.2 Run async for loop to wait for completed chunks from the receiver using the asyncRingbuffer object.
    async for chunk in asyncRingbuffer:
        received += len(chunk.present)
        dropped += len(chunk.present) - sum(chunk.present)
        print(
            f"Chunk: {i:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}. {len(chunk.base)}"
        )
        # 8.3 Once we are done with the chunk, give it back to the receiver so that the receiver has access to more
        # chunks without having to allocate more memory.
        receiverStream.add_chunk(chunk)
        i += 1


# 9. Run get_chunks() function in an asyncio event loop.
loop = asyncio.get_event_loop()
loop.run_until_complete(get_chunks())
loop.close()

# 10. Cleanup - Sometimes closing everything is a bit of a pain. I have not quite got it right yet, so watch out for
# crashes when
