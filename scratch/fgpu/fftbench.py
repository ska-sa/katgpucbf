#!/usr/bin/env python

import argparse
import ctypes
import math

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import skcuda.cufft as cufft
import skcuda.fft as fft
from skcuda.cufft import _libcufft

_libcufft.cufftGetSize.restype = int
_libcufft.cufftGetSize.argtypes = [ctypes.c_int, ctypes.c_void_p]


def cufftGetSize(handle):  # noqa: N802
    work_size = ctypes.c_size_t()
    status = _libcufft.cufftGetSize(handle, ctypes.byref(work_size))
    cufft.cufftCheckStatus(status)
    return work_size.value


def time_gpu(func, passes):
    func()  # Warmup
    cuda.Context.synchronize()
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for _ in range(args.passes):
        func()
    end.record()
    end.synchronize()
    return (end.time_since(start)) / args.passes * 1e-3  # Convert ms to s


def benchmark_real(args):
    shape = (args.batch, 2 * args.channels)
    fshape = (args.batch, args.channels + 1)

    plan = fft.Plan(
        shape[1],
        np.float32,
        np.complex64,
        args.batch,
        inembed=np.array([shape[1]], np.int32),
        idist=shape[1],
        onembed=np.array([fshape[1]], np.int32),
        odist=fshape[1],
    )

    x = np.random.standard_normal(shape).astype(np.float32)
    x_gpu = gpuarray.to_gpu(x)
    X_gpu = gpuarray.empty(fshape, dtype=np.complex64)  # noqa: N806

    flops = 5 * shape[0] * shape[1] * int(math.ceil(math.log(shape[1], 2)))
    bw = x.nbytes + X_gpu.nbytes
    fmt = "{0}: Time: {1} Speed: {2} GFlop/s BW: {3} GiB/s Scratch: {4} MiB"

    time = time_gpu(lambda: fft.fft(x_gpu, X_gpu, plan), args.passes)
    print(fmt.format("R2C", time, flops / time / 1e9, bw / time / 1e9, cufftGetSize(plan.handle) / 1024 ** 2))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--channels", type=int, default=32768)
parser.add_argument("-b", "--batch", type=int, default=1024)
parser.add_argument("-p", "--passes", type=int, default=100)
parser.add_argument("-i", "--in-place", action="store_true")
args = parser.parse_args()

benchmark_real(args)
