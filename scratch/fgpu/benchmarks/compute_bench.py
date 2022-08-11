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

from katsdpsigproc import accel

from katgpucbf.fgpu.compute import ComputeTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--channels", type=int, default=32768)
    parser.add_argument("--spectra-per-heap", type=int, default=256)
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()

    context = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    with context:
        template = ComputeTemplate(context, args.taps, args.channels)
        command_queue = context.create_tuning_command_queue()
        extra_samples = (args.taps - 1) * args.channels * 2
        spectra = args.samples // (args.channels * 2)
        fn = template.instantiate(
            command_queue,
            samples=args.samples + extra_samples,
            spectra=spectra,
            spectra_per_heap=args.spectra_per_heap,
        )
        # Fill everything with zeros just ensure performance isn't affected
        # by stray NaNs and the like.
        fn.ensure_all_bound()
        for name in ["weights", "fine_delay", "phase", "gains"]:
            fn.buffer(name).zero(command_queue)

        def run():
            fn.run_frontend([fn.buffer("in0"), fn.buffer("in1")], [0, 0], 0, spectra)
            fn.run_backend(fn.buffer("out"))

        run()  # Warmup pass
        start = command_queue.enqueue_marker()
        for _ in range(args.passes):
            run()
        stop = command_queue.enqueue_marker()
        command_queue.finish()
        average = stop.time_since(start) / args.passes
        print(f"{average * 1000:.3f} ms")


if __name__ == "__main__":
    main()
