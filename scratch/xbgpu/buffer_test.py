"""TODO: Add a comment."""

# import katxgpu
import katxgpu._katxgpu.recv as recv

import numpy as np
import katsdpsigproc.accel as accel

context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
buf = accel.HostArray((1024 * 1024 * 1024,), np.uint8, context=context)

buf[2] = 15

print(buf)

chunk = recv.Chunk(buf)

print(buf)
