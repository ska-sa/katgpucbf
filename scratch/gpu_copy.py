#!/usr/bin/env python3

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
parser.add_argument("direction", choices=("htod", "dtoh", "dtod"))
args = parser.parse_args()

size = args.size
rep = args.repeat
device = GPUArray((size,), np.uint8)
if args.direction != "dtod":
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
    host.fill(1)
    device.set(host)
else:
    device2 = GPUArray((size,), np.uint8)
    device[:] = device2
while True:
    start = time.time()
    for _ in range(rep):
        if args.direction == "htod":
            device.set(host)
        elif args.direction == "dtoh":
            device.get(host)
        else:
            device[:] = device2
    stop = time.time()
    rate = size * rep / (stop - start)
    print(size * rep, "bytes ", rate / 1e9, "GB/s")
    if not args.forever:
        break
