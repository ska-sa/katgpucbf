"""TODO: Write this."""

import katxgpu.xsend
import asyncio

x = katxgpu.xsend.XEngineSPEADSend()

x.send_heap(0x1, x._free_heaps_queue.get())
x.send_heap(0x2, x._free_heaps_queue.get())
x.send_heap(0x3, x._free_heaps_queue.get())
x.send_heap(0x4, x._free_heaps_queue.get())
x.send_heap(0x5, x._free_heaps_queue.get())


async def send_process():
    """TODO: Write this docstring."""
    print("Sending.")
    num_sent = 0
    while 1:
        num_sent += 1
        await x.get_free_heap()
        print(num_sent)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_process())
    loop.close()
