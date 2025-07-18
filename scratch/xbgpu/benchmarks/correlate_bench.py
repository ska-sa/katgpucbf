#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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

import katsdpsigproc.accel
import numpy as np

from katgpucbf import DEFAULT_JONES_PER_BATCH
from katgpucbf.xbgpu.correlation import CorrelationTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-size", type=int, default=80, help="Antennas in the array [%(default)s]")
    parser.add_argument(
        "--channels", type=int, default=1024, help="Total number of channels in the stream [%(default)s]"
    )
    parser.add_argument(
        "--channels-per-substream", type=int, default=16, help="Channels processed by one engine [%(default)s]"
    )
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        help="Number of antenna-channelised-voltage Jones vectors in each F-engine batch [%(default)s]",
    )
    parser.add_argument("--heaps-per-fengine-per-chunk", type=int, default=32, help="Frames per chunk [%(default)s]")
    parser.add_argument("--passes", type=int, default=10000, help="Number of times to repeat the test [%(default)s]")
    args = parser.parse_args()
    if args.jones_per_batch % args.channels != 0:
        parser.error("--jones-per-batch must be a multiple of --channels")
    spectra_per_heap = args.jones_per_batch // args.channels

    ctx = katsdpsigproc.accel.create_some_context()
    command_queue = ctx.create_command_queue()
    template = CorrelationTemplate(ctx, args.array_size, args.channels_per_substream, spectra_per_heap, 8)
    fn = template.instantiate(command_queue, args.heaps_per_fengine_per_chunk)

    fn.ensure_all_bound()
    buf = fn.buffer("in_samples")
    h_data = buf.empty_like()
    assert h_data.dtype == np.int8
    rng = np.random.default_rng(seed=1)
    h_data[:] = rng.integers(-128, 127, size=h_data.shape, dtype=h_data.dtype)
    fn.buffer("in_samples").set(command_queue, h_data)
    fn.zero_visibilities()

    fn()  # Warmup pass
    command_queue.finish()

    start = command_queue.enqueue_marker()
    for _ in range(args.passes):
        fn()
    stop = command_queue.enqueue_marker()
    elapsed = stop.time_since(start)
    voltages = (
        args.array_size
        * args.channels_per_substream
        * spectra_per_heap
        * args.heaps_per_fengine_per_chunk
        * args.passes
    )
    rate = voltages / elapsed
    print(f"{rate * 1e-6:.3f} M input dual-pol samples/second")


if __name__ == "__main__":
    main()
