#!/usr/bin/env python3

import sys
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver
from pycuda.gpuarray import GPUArray

size = 1024 * 1024 * 1024
host = pycuda.driver.pagelocked_empty((size,), np.uint8)
host.fill(1)
device = GPUArray((size,), np.uint8)
rep = 10
while True:
    start = time.time()
    for i in range(rep):
        if sys.argv[1] == 'htod':
            device.set(host)
        else:
            device.get(host)
    stop = time.time()
    rate = size * rep / (stop - start)
    print(rate / 1e9)
