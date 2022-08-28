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
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractContext, AbstractEvent

from katgpucbf import N_POLS
from katgpucbf.fgpu.compute import ComputeTemplate
from katgpucbf.fgpu.engine import generate_weights


@dataclass
class Item:
    """Combines an on-device array with events corresponding to usage of that array."""

    data: accel.DeviceArray
    events: List[AbstractEvent] = field(default_factory=list)


def _allocate_item(context: AbstractContext, slot: accel.IOSlot) -> Item:
    """Allocate an :class:`Item` with the shape and dtype to match a slot."""
    data = accel.DeviceArray(context, slot.shape, slot.dtype, slot.required_padded_shape())
    slot.bind(data)
    return Item(data, [])


def _host_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.HostArray:
    """Allocate a :class:`~.HostArray` with the shape and dtype to match a slot."""
    return accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(), context=context)


def _get_slot(fn: accel.Operation, name: str) -> accel.IOSlot:
    """Get a slot from an operation, asserting that it is an IOSlot."""
    slot = fn.slots[name]
    assert isinstance(slot, accel.IOSlot)
    return slot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--channels", type=int, default=32768)
    parser.add_argument("--spectra-per-heap", type=int, default=256)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--transfers", action="store_true", help="Include host-device memory transfers")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=1)
    context = accel.create_some_context(device_filter=lambda device: device.is_cuda)
    n_in = 2 * N_POLS
    n_out = 2
    # These "queues" are always kept at the same capacity; they're just used to
    # round-robin over multiple items.
    in_queue: Deque[Item] = deque()
    out_queue: Deque[Item] = deque()
    with context:
        template = ComputeTemplate(context, args.taps, args.channels)
        compute_queue = context.create_tuning_command_queue()
        upload_queue = context.create_command_queue()
        download_queue = context.create_command_queue()
        extra_samples = (args.taps - 1) * args.channels * 2
        spectra = args.samples // (args.channels * 2)
        fn = template.instantiate(
            compute_queue,
            samples=args.samples + extra_samples,
            spectra=spectra,
            spectra_per_heap=args.spectra_per_heap,
        )

        h_in = []  # Host-side random arrays of input
        for i in range(N_POLS):
            h_in.append(_host_allocate_slot(context, _get_slot(fn, f"in{i}")))
            assert h_in[-1].dtype == np.uint8
            h_in[-1][:] = rng.integers(0, 256, size=h_in[-1].shape, dtype=np.uint8)
        h_out = _host_allocate_slot(context, _get_slot(fn, "out"))

        for i in range(n_in):
            pol = i % N_POLS
            in_queue.append(_allocate_item(context, _get_slot(fn, f"in{pol}")))
            in_queue[-1].data.set(upload_queue, h_in[pol])
        for _ in range(n_out):
            out_queue.append(_allocate_item(context, _get_slot(fn, "out")))
        fn.ensure_all_bound()

        # Fill in some somewhat realistic weights and gains
        h_weights = fn.buffer("weights").empty_like()
        h_weights[:] = generate_weights(args.channels, args.taps)
        fn.buffer("weights").set(upload_queue, h_weights)
        h_gains = fn.buffer("gains").empty_like()
        h_gains[:] = 1
        fn.buffer("gains").set(upload_queue, h_gains)

        # Fill these with zeros just to ensure performance isn't affected
        # by stray NaNs and the like.
        for name in ["fine_delay", "phase"]:
            fn.buffer(name).zero(upload_queue)

        def run():
            in_items = []
            if args.transfers:
                for i in range(N_POLS):
                    in_item = in_queue.popleft()
                    upload_queue.enqueue_wait_for_events(in_item.events)
                    fn.bind(**{f"in{i}": in_item.data})
                    in_item.data.set_async(upload_queue, h_in[i])
                    in_items.append(in_item)
                # Ensure that uploads are complete before we start computation
                compute_queue.enqueue_wait_for_events([upload_queue.enqueue_marker()])
                out_item = out_queue.popleft()
                # Ensure the previous read of the out_item has completed before
                # we overwrite it.
                compute_queue.enqueue_wait_for_events(out_item.events)
                fn.bind(out=out_item.data)
            # Do the actual calculations
            fn.run_frontend([fn.buffer("in0"), fn.buffer("in1")], [0, 0], 0, spectra)
            fn.run_backend(fn.buffer("out"))
            if args.transfers:
                # Ensure computation has completed before we start download.
                download_queue.enqueue_wait_for_events([compute_queue.enqueue_marker()])
                fn.buffer("out").get_async(download_queue, h_out)
                # Mark the output as in use until download completes
                out_item.events = [download_queue.enqueue_marker()]
                out_queue.append(out_item)
                # Mark the inputs as in use until the computation completes
                compute_marker = compute_queue.enqueue_marker()
                for in_item in in_items:
                    in_item.events = [compute_marker]
                    in_queue.append(in_item)

        run()  # Warmup pass
        start = compute_queue.enqueue_marker()
        for _ in range(args.passes):
            run()
        stop = compute_queue.enqueue_marker()
        upload_queue.finish()
        compute_queue.finish()
        download_queue.finish()
        # Note: this excludes time for the first upload and the last
        # download, but over a large number of passes it doesn't really
        # matter.
        average = stop.time_since(start) / args.passes
        print(f"{average * 1000:.3f} ms")


if __name__ == "__main__":
    main()
