#!/usr/bin/env python3

import argparse

import katsdpsigproc.accel

from katgpucbf.xbgpu.beamform import BeamformTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-size", type=int, default=80, help="Antennas in the array [%(default)s]")
    parser.add_argument(
        "--channels-per-substream", type=int, default=16, help="Channels processed by one engine [%(default)s]"
    )
    parser.add_argument("--spectra-per-heap", type=int, default=256, help="Spectra in each frame [%(default)s]")
    parser.add_argument("--heaps-per-fengine-per-chunk", type=int, default=5, help="Frames per chunk [%(default)s]")
    parser.add_argument("--beams", type=int, default=4, help="Number of dual-pol beams [%(default)s]")
    parser.add_argument("--passes", type=int, default=10000, help="Number of times to repeat the test [%(default)s]")
    args = parser.parse_args()

    ctx = katsdpsigproc.accel.create_some_context()
    command_queue = ctx.create_command_queue()
    template = BeamformTemplate(ctx, [0, 1] * args.beams)
    fn = template.instantiate(
        command_queue,
        n_frames=args.heaps_per_fengine_per_chunk,
        n_ants=args.array_size,
        n_channels=args.channels_per_substream,
        n_spectra_per_frame=args.spectra_per_heap,
    )
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