#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Measure the accuracy of CUDA's sincos implementations."""

import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver
from pycuda.compiler import SourceModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=float, default=1.0, help="Maximum value to test, in units of pi")
    parser.add_argument("--func", choices=["__sincosf", "sincosf"], default="__sincosf")
    args = parser.parse_args()

    source = SourceModule(
        f"""
    #include <math.h>

    __global__ void sincos_kernel(float2 *out, const float *in)
    {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        {args.func}(in[idx], &out[idx].x, &out[idx].y);
    }}
    """
    )
    kernel = source.get_function("sincos_kernel")

    info = np.finfo(np.float32)
    block = 128
    n = 2**info.nmant
    sc = np.zeros((n, 2), np.float32)

    max_test = np.pi * args.max
    # Iterate through the exponent portion of the float32. For each value,
    # we populate angle with all possible mantissa bits. We exclude the
    # largest value since that is used to encode infinity and NaNs.
    max_sin_err = 0.0
    max_cos_err = 0.0
    max_tot_err = 0.0
    for raw_exp in range(0, 2**info.nexp - 1):
        angle = np.arange(raw_exp << info.nmant, (raw_exp + 1) << info.nmant, dtype=np.uint32).view(np.float32)
        if angle[0] > max_test:
            break
        cut = np.searchsorted(angle, max_test, side="right")

        kernel(pycuda.driver.Out(sc), pycuda.driver.In(angle), block=(block, 1, 1), grid=(n // block, 1, 1))
        # Clip to max_test if needed
        sin_err = np.abs(sc[:cut, 0].astype(np.float64) - np.sin(angle[:cut].astype(np.float64)))
        cos_err = np.abs(sc[:cut, 1].astype(np.float64) - np.cos(angle[:cut].astype(np.float64)))
        tot_err = np.hypot(sin_err, cos_err)
        max_sin_err = max(max_sin_err, np.max(sin_err))
        max_cos_err = max(max_cos_err, np.max(cos_err))
        max_tot_err = max(max_tot_err, np.max(tot_err))

    print(f"Max sin err: {max_sin_err}  (2**{np.log2(max_sin_err)})")
    print(f"Max cos err: {max_cos_err}  (2**{np.log2(max_cos_err)})")
    print(f"Max tot err: {max_tot_err}  (2**{np.log2(max_tot_err)})")


if __name__ == "__main__":
    main()
