#!/usr/bin/env python3

################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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
from katsdpsigproc import accel

from katgpucbf import DIG_SAMPLE_BITS, N_POLS
from katgpucbf.fgpu.compute import ComputeTemplate, NarrowbandConfig
from katgpucbf.fgpu.engine import generate_ddc_weights, generate_pfb_weights


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int, default=16)
    parser.add_argument("--src-chunk-samples", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--dst-chunk-jones", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--channels", type=int, default=32768)
    parser.add_argument("--spectra-per-heap", type=int, default=256)
    parser.add_argument("--dig-sample-bits", type=int, default=DIG_SAMPLE_BITS)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--ddc-taps", type=int, default=128)
    parser.add_argument("--narrowband", action="store_true")
    parser.add_argument("--narrowband-decimation", type=int, default=8)
    parser.add_argument("--kernel", choices=["all", "ddc", "pfb_fir", "fft", "postproc"], default="all")
    parser.add_argument(
        "--sem",
        type=int,
        metavar="N",
        help="Measure after every N iterations to estimate SEM (standard error in the mean)",
    )
    args = parser.parse_args()
    if args.sem is not None:
        if args.sem <= 0:
            parser.error("--sem must be positive")
        if args.passes % args.sem != 0:
            parser.error("--sem must divide into --passes")

    rng = np.random.default_rng(seed=1)
    context = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    with context:
        if args.narrowband:
            narrowband_config = NarrowbandConfig(
                decimation=args.narrowband_decimation,
                taps=args.ddc_taps,
                mix_frequency=0.25,
                weights=generate_ddc_weights(args.ddc_taps, args.narrowband_decimation, 0.005),
            )
            spectra_samples = 2 * args.channels * args.narrowband_decimation
            window = args.taps * spectra_samples + args.ddc_taps - args.narrowband_decimation
        else:
            narrowband_config = None
            spectra_samples = 2 * args.channels
            window = args.taps * spectra_samples
        template = ComputeTemplate(
            context, args.taps, args.channels, args.dig_sample_bits, narrowband=narrowband_config
        )
        command_queue = context.create_tuning_command_queue()
        out_spectra = accel.roundup(args.dst_chunk_jones // args.channels, args.spectra_per_heap)
        frontend_spectra = min(args.src_chunk_samples // spectra_samples, out_spectra)
        extra_samples = window - spectra_samples
        fn = template.instantiate(
            command_queue,
            samples=args.src_chunk_samples + extra_samples,
            spectra=out_spectra,
            spectra_per_heap=args.spectra_per_heap,
        )
        fn.ensure_all_bound()

        h_weights = fn.buffer("weights").empty_like()
        h_weights[:] = generate_pfb_weights(2 * args.channels, args.taps, 1.0)
        fn.buffer("weights").set(command_queue, h_weights)

        h_gains = fn.buffer("gains").empty_like()
        h_gains[:] = 1
        fn.buffer("gains").set(command_queue, h_gains)

        for i in range(N_POLS):
            buf = fn.buffer(f"in{i}")
            h_data = buf.empty_like()
            assert h_data.dtype == np.uint8
            h_data[:] = rng.integers(0, 256, size=h_data.shape, dtype=np.uint8)
            buf.set(command_queue, h_data)

        # Fill these with zeros just to ensure performance isn't affected
        # by stray NaNs and the like.
        for name in ["fine_delay", "phase"]:
            fn.buffer(name).zero(command_queue)

        def run_ddc():
            fn.ddc[0]()

        def run_pfb_fir():
            fn.pfb_fir[0].spectra = frontend_spectra
            fn.pfb_fir[0]()

        def run_fft():
            fn.fft[0]()

        def run_postproc():
            fn.postproc()

        def run_all():
            if args.narrowband:
                fn.run_ddc([fn.buffer("in0"), fn.buffer("in1")], 0)
                fn.run_narrowband_frontend([0, 0], 0, frontend_spectra)
            else:
                fn.run_wideband_frontend(
                    [fn.buffer("in0"), fn.buffer("in1")],
                    [fn.buffer("dig_total_power0"), fn.buffer("dig_total_power1")],
                    [0, 0],
                    0,
                    frontend_spectra,
                )
            fn.run_backend(fn.buffer("out"), fn.buffer("saturated"))

        run = locals()[f"run_{args.kernel}"]
        run()  # Warmup pass

        markers = [command_queue.enqueue_marker()]
        for i in range(args.passes):
            run()
            if args.sem is not None and (i + 1) % args.sem == 0:
                markers.append(command_queue.enqueue_marker())
        if args.sem is None:
            markers.append(command_queue.enqueue_marker())
        command_queue.finish()
        average = markers[-1].time_since(markers[0]) / args.passes
        if args.sem is not None:
            times = []
            for i in range(1, len(markers)):
                times.append(markers[i].time_since(markers[i - 1]))
            # Standard error in the mean of elements of times
            measure_sem = np.std(times, ddof=1) / np.sqrt(len(times))
            # Standard error in `average`
            sem = measure_sem / args.sem
            print(f"{average * 1000:.5f} ± {sem * 1000:.5f} ms")
        else:
            print(f"{average * 1000:.5f} ms")


if __name__ == "__main__":
    main()
