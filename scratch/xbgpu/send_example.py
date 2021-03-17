"""TODO: Write this."""

import katxgpu.xsend
import asyncio
import numpy as np


x = katxgpu.xsend.XEngineSPEADSend()


async def send_process():
    """TODO: Write this docstring."""
    num_sent = 0
    while 1:
        buffer_wrapper = await x.get_free_heap()
        buffer_wrapper.buffer = np.full(buffer_wrapper.buffer.shape, num_sent, np.uint16)
        num_sent += 1
        x.send_heap(num_sent * 0x1000, buffer_wrapper)
        print(num_sent, buffer_wrapper.buffer[0:10])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_process())
    loop.close()
