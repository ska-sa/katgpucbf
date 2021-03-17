"""TODO: Write this."""
import katxgpu.xsend
import asyncio
import numpy as np


x = katxgpu.xsend.XEngineSPEADIbvSend(
    n_ants=64,
    n_channels_per_stream=128,
    n_pols=2,
    dump_rate_s=0.4,
    channel_offset=128 * 4,
    endpoint=("239.10.10.11", 7149),
    interface_address="10.100.44.1",
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
