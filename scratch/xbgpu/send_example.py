"""TODO: Write this."""
import katxgpu.xsend

import argparse
import asyncio
import numpy as np
import katsdpsigproc.accel as accel


parser = argparse.ArgumentParser(description="Simple example demonstrating how to use katxgpu receiver software.")
parser.add_argument("--mcast_dest_ip", default="239.10.10.11", help="IP address of multicast stream to transmit on.")
parser.add_argument("--mcast_dest_port", default="7149", type=int, help="Port of multicast stream to transmit on.")
parser.add_argument("--interface", default="10.100.44.1", help="IP Address of interface that will receive the data.")
args = parser.parse_args()
dest_multicast_ip = args.mcast_dest_ip
dest_multicast_port = args.mcast_dest_port
interface_ip = args.interface

print(f"Transmitting to {dest_multicast_ip}:{dest_multicast_port} on the {interface_ip} interface.")

context = accel.create_some_context(device_filter=lambda x: x.is_cuda)

x = katxgpu.xsend.XEngineSPEADIbvSend(
    n_ants=64,
    n_channels_per_stream=128,
    n_pols=2,
    dump_rate_s=0.4,
    channel_offset=128 * 4,
    context=context,
    endpoint=(dest_multicast_ip, dest_multicast_port),
    interface_address=interface_ip,
    thread_affinity=3,
)


async def send_process():
    """TODO: Write this docstring."""
    num_sent = 0
    while 1:
        buffer_wrapper = await x.get_free_heap()
        buffer_wrapper.buffer = np.full(buffer_wrapper.buffer.shape, num_sent, np.uint16)
        num_sent += 1
        x.send_heap(num_sent * 0x1000, buffer_wrapper)
        print(f"Sent heap {num_sent-1}. Values: [{buffer_wrapper.buffer[0]}...{buffer_wrapper.buffer[0]}]")


async def run() -> None:
    """TODO: Write this docstring."""
    # This ayncio stuff is for python3.6, will need to be upgraded for Python3.7 and above
    loop = asyncio.get_event_loop()
    task1 = loop.create_task(send_process())
    await task1


if __name__ == "__main__":
    # This ayncio stuff is for python3.6, will need to be upgraded for Python3.7 and above
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
