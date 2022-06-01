#!/usr/bin/env python3

################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

import numpy as np
from katsdpsigproc import accel, fft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--mode", choices=["r2c", "c2c"], default="r2c")
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "r2c" and args.size % 2:
        parser.error("size must be even for r2c transformations")

    context = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    with context:
        dtype_src = np.float32 if args.mode == "r2c" else np.complex64
        dtype_dest = np.complex64
        shape = (args.batches, args.size)
        shape_dest = (args.batches, args.size if args.mode == "c2c" else args.size // 2 + 1)
        template = fft.FftTemplate(context, 1, shape, dtype_src, dtype_dest, shape, shape_dest)
        command_queue = context.create_tuning_command_queue()
        fn = template.instantiate(command_queue, fft.FftMode.FORWARD)
        # Fill everything with zeros just ensure performance isn't affected
        # by stray NaNs and the like.
        fn.ensure_all_bound()
        fn.buffer("src").zero(command_queue)
        fn()  # Warmup pass
        start = command_queue.enqueue_marker()
        for _ in range(args.passes):
            fn()
        stop = command_queue.enqueue_marker()
        command_queue.finish()
        average = stop.time_since(start) / args.passes
        print(f"{average * 1000:.3f} ms")


if __name__ == "__main__":
    main()
