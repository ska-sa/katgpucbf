#!/usr/bin/env python3

################################################################################
# Copyright (c) 2019-2023 National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import argparse
import mmap
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver
from pycuda.gpuarray import GPUArray

parser = argparse.ArgumentParser()
parser.add_argument("--size", "-s", type=int, default=1024 * 1024 * 1024)
parser.add_argument("--repeat", "-r", type=int, default=20)
parser.add_argument("--forever", action="store_true")
parser.add_argument("--mem", default="pagelocked", choices=("pagelocked", "wc", "huge"))
parser.add_argument("--fill", type=int, default=1)
parser.add_argument("direction", choices=("htod", "dtoh", "dtod", "peer"))
args = parser.parse_args()

size = args.size
rep = args.repeat
device = GPUArray((size,), np.uint8)
if args.direction in ["htod", "dtoh"]:
    if args.mem == "pagelocked":
        host = pycuda.driver.pagelocked_empty((size,), np.uint8)
    elif args.mem == "wc":
        host = pycuda.driver.pagelocked_empty((size,), np.uint8, mem_flags=pycuda.driver.host_alloc_flags.WRITECOMBINED)
    elif args.mem == "huge":
        mapping = mmap.mmap(-1, size, flags=mmap.MAP_SHARED | 0x40000)  # MAP_HUGETLB
        array = np.frombuffer(mapping, np.uint8)
        host = pycuda.driver.register_host_memory(array)
        assert len(host) == size
    else:
        parser.error(f"Unknown memory type {args.mem}")
    host.fill(args.fill)
    device.set(host)
elif args.direction == "dtod":
    device2 = GPUArray((size,), np.uint8)
    device[:] = device2
else:
    peer_dev = pycuda.driver.Device(1)
    peer_ctx = peer_dev.make_context()  # Becomes current
    device2 = GPUArray((size,), np.uint8)  # Allocated from peer_ctx
    pycuda.driver.Context.pop()  # Default context current again
    pycuda.autoinit.context.enable_peer_access(peer_ctx)
try:
    while True:
        start = time.time()
        for _ in range(rep):
            if args.direction == "htod":
                device.set(host)
            elif args.direction == "dtoh":
                device.get(host)
            else:
                device[:] = device2
                pycuda.driver.Context.synchronize()
        stop = time.time()
        rate = size * rep / (stop - start)
        print(size * rep, "bytes ", rate / 1e9, "GB/s")
        if not args.forever:
            break
finally:
    # Free device2 from the right context
    if args.direction == "peer":
        peer_ctx.push()
        device2.gpudata.free()
        pycuda.driver.Context.pop()
