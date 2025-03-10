#!/usr/bin/env python3

################################################################################
# Copyright (c) 2022-2023, 2025, National Research Foundation (SARAO)
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
from fractions import Fraction

import numpy as np
from katsdpsigproc import accel

from katgpucbf import DIG_SAMPLE_BITS, N_POLS
from katgpucbf.fgpu.ddc import DDCTemplate
from katgpucbf.fgpu.main import DEFAULT_DDC_TAPS_RATIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int)  # Default is computed
    parser.add_argument("--subsampling", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--input-sample-bits", type=int, default=DIG_SAMPLE_BITS)
    parser.add_argument("--pols", type=int, default=N_POLS)
    parser.add_argument("--passes", type=int, default=1000)
    args = parser.parse_args()
    if args.taps is None:
        args.taps = args.subsampling * DEFAULT_DDC_TAPS_RATIO

    context = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    with context:
        command_queue = context.create_tuning_command_queue()
        template = DDCTemplate(
            context, taps=args.taps, subsampling=args.subsampling, input_sample_bits=args.input_sample_bits
        )
        fn = template.instantiate(command_queue, samples=args.samples, n_pols=args.pols)
        fn.ensure_all_bound()
        fn.buffer("in").zero(command_queue)
        fn.configure(Fraction(0.25), np.ones(args.taps, np.float32))
        fn()  # Do a warmup pass
        start = command_queue.enqueue_marker()
        for _ in range(args.passes):
            fn()
        stop = command_queue.enqueue_marker()
        command_queue.finish()
        average = stop.time_since(start) / args.passes
        print(f"{average * 1000000:.3f} us")


if __name__ == "__main__":
    main()
