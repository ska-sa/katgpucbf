"""Katxgpu X-Engine send example script.

This script demonstrates how to configure a katxgpu.xsend.XEngineSPEADIbvSend object to transmit a "Baseline
Correlation Products Hardware Heaps" stream (or more simply an X-Engine output stream) onto the network
channelised voltage stream

It also shows how a chunk can be received asynchronously and how to pass a used chunk back to the katxgpu receiver.
"""
# 1. Imports
# 1.1 Local imports
import katxgpu.xsend

# 1.2 External imports
import argparse
import asyncio
import numpy as np
import katsdpsigproc.accel as accel

# 2. Relevant variables
# 2.1 Parsing command line arguments
parser = argparse.ArgumentParser(description="Simple example demonstrating how to use katxgpu receiver software.")
parser.add_argument("--mcast_dest_ip", default="239.10.10.11", help="IP address of multicast stream to transmit on.")
parser.add_argument("--mcast_dest_port", default="7149", type=int, help="Port of multicast stream to transmit on.")
parser.add_argument("--interface", default="10.100.44.1", help="IP Address of interface that will receive the data.")
args = parser.parse_args()
dest_multicast_ip = args.mcast_dest_ip
dest_multicast_port = args.mcast_dest_port
interface_ip = args.interface

print(f"Transmitting to {dest_multicast_ip}:{dest_multicast_port} on the {interface_ip} interface.")

# 2.2 Adjustable parameters - The description of these parameters can be found in the documentation for the
# katxgpu.xsend.XEngineSPEADIbvSend object.
thread_affinity = 3
n_ants = 64
n_channels_total = 32768
n_channels_per_stream = n_channels_total // n_ants // 4
n_pols = 2
dump_rate_s = 0.4

# 3. Create cuda context - all buffers created in the XEngineSPEADIbvSend object are created from this context.
context = accel.create_some_context(device_filter=lambda x: x.is_cuda)

# 4. Create object to transmit X-Engine output heaps. This constructor creates its own internal collection of buffers
# that are registered in such a way as to allow zero copy sends.
transmitStream = katxgpu.xsend.XEngineSPEADIbvSend(
    n_ants=n_ants,
    n_channels_per_stream=n_channels_per_stream,
    n_pols=n_pols,
    dump_rate_s=dump_rate_s,
    channel_offset=n_channels_per_stream * 4,  # Arbitrary for now
    context=context,
    endpoint=(dest_multicast_ip, dest_multicast_port),
    interface_address=interface_ip,
    thread_affinity=thread_affinity,
)


# 5. This is the main processing function. It repeatedly sends new heaps out onto the network.
async def send_process():
    """
    Continously sends X-Engine output heaps onto the network.

    This function retrieves available buffers from the transmitStream, populates them and then tells the transmit stream
    to send the filled buffer onto the network.

    It is very important that only the buffers returned from the transmitStream.get_free_heap() function are transmitted
    using the transmitStream.send_heap() function as these buffers have been registered as memory regions that ibverbs
    will be able to use to zero copy send data onto the network.
    """
    num_sent = 0
    while 1:
        # 5.1 Get a free buffer to store the next heap.
        buffer_wrapper = await transmitStream.get_free_heap()

        # 5.2 Populate the buffer with dummy data - notice how we copy new values into the buffer, we dont overwrite
        # the buffer. Attempts to overwrite the buffer will throw an error. This is intended behavour as the memory
        # regions in the buffer have been configured for zero-copy sends.
        buffer_wrapper.buffer[:] = np.full(
            buffer_wrapper.buffer.shape, num_sent, np.uint16
        )  # [:] forces a copy, not an overwrite
        num_sent += 1

        # 5.3 Give the buffer back to the transmitStream to transmit out onto the network.
        transmitStream.send_heap(num_sent * 0x1000, buffer_wrapper)
        print(f"Sent heap {num_sent-1}. Values: [{buffer_wrapper.buffer[0]}...{buffer_wrapper.buffer[0]}]")


# 6. This is a function that wraps the send_process() function. See desription in function docstring.
async def run() -> None:
    """
    Kicks off the send process by calling the send_process() function asynchronously.

    This function needs to be called asynchronously and in turn calls the send_process() function asynchronously.

    This function is not necessary in this example as the send_process function can be launched directly in the __main__
    section but by copying the task1 = loop.create_task(send_process()) and await task1 pattern, this function can
    launch multiple tasks on the same IO loop.

    I left it in here as a reminder as it will be used in the main X-Engine program and getting multiple coroutines to
    run in parallel was a bit challenging.

    The method of creating tasks in this function is the only way to do it in Python 3.6. From Python 3.7 onwards, the
    loop object is not longer necessary as the mechanism to launch tasks has been simplified.
    """
    loop = asyncio.get_event_loop()
    task1 = loop.create_task(send_process())
    await task1


# 7. Call the run() function in an asynio loop.
if __name__ == "__main__":
    # These two lines can be replaced with asyncio.run(run()) in python 3.7 and later
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
