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

import argparse

import katsdpsigproc.accel

from katgpucbf.curand_helpers import RandomStateBuilder
from katgpucbf.xbgpu.beamform import BeamformTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-size", type=int, default=80, help="Antennas in the array [%(default)s]")
    parser.add_argument(
        "--channels-per-substream", type=int, default=16, help="Channels processed by one engine [%(default)s]"
    )
    parser.add_argument("--spectra-per-batch", type=int, default=256, help="Spectra in each batch [%(default)s]")
    parser.add_argument("--heaps-per-fengine-per-chunk", type=int, default=32, help="Batches per chunk [%(default)s]")
    parser.add_argument("--beams", type=int, default=4, help="Number of dual-pol beams [%(default)s]")
    parser.add_argument("--passes", type=int, default=10000, help="Number of times to repeat the test [%(default)s]")
    args = parser.parse_args()

    ctx = katsdpsigproc.accel.create_some_context()
    command_queue = ctx.create_command_queue()
    template = BeamformTemplate(ctx, [0, 1] * args.beams, n_spectra_per_batch=args.spectra_per_heap)
    fn = template.instantiate(
        command_queue,
        n_batches=args.heaps_per_fengine_per_chunk,
        n_ants=args.array_size,
        n_channels_per_substream=args.channels_per_substream,
        seed=1,
        sequence_first=0,
    )

    builder = RandomStateBuilder(ctx)
    slot = fn.slots["rand_states"]
    fn.bind(rand_states=builder.make_states(slot.shape, seed=1234567, sequence_first=0))

    fn.ensure_all_bound()
    # Set non-trivial weights so the whole thing isn't just zero
    h_weights = fn.buffer("weights").empty_like()
    h_weights.fill(1)
    fn.buffer("weights").set(command_queue, h_weights)
    fn.buffer("delays").zero(command_queue)
    fn.buffer("in").zero(command_queue)

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
        * args.spectra_per_heap
        * args.heaps_per_fengine_per_chunk
        * args.passes
    )
    rate = voltages / elapsed
    print(f"{rate * 1e-6:.3f} M input dual-pol samples/second")


if __name__ == "__main__":
    main()
