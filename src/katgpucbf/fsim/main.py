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

"""
Simulate channelised data from the MeerKAT F-Engines destined for one or more XB-Engines.

Refer to :ref:`feng-packet-sim` for more information.
"""

import argparse
import asyncio
import functools
import itertools
import os
import time
from collections.abc import Sequence

import katsdpservices
import numpy as np
import prometheus_async
import spead2.send.asyncio
from katsdptelstate.endpoint import endpoint_list_parser
from numba import njit
from prometheus_client import Counter, Gauge

from .. import (
    BYTE_BITS,
    COMPLEX,
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_PACKET_PAYLOAD_BYTES,
    DEFAULT_TTL,
    N_POLS,
    SPEAD_DESCRIPTOR_INTERVAL_S,
    spead,
)
from ..fgpu.send import PREAMBLE_SIZE, make_descriptor_heap, make_item_group
from ..send import DescriptorSender
from ..utils import TimeConverter, add_gc_stats, comma_split
from . import METRIC_NAMESPACE

DTYPE = np.dtype(np.int8)
#: Number of heaps in time to keep in flight
QUEUE_DEPTH = 8
output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter("output_bytes", "number of payload bytes transmitted", namespace=METRIC_NAMESPACE)
time_error_gauge = Gauge(
    "time_error_s", "elapsed time minus expected elapsed time", ["stream"], namespace=METRIC_NAMESPACE
)


def parse_args(arglist: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="fsim")

    parser.add_argument(
        "--adc-sample-rate", type=float, default=1712e6, help="Digitiser sampling rate (Hz) [%(default)s]"
    )
    parser.add_argument("--interface", default="lo", help="Network interface on which to send packets [%(default)s]")
    parser.add_argument(
        "--array-size", type=int, default=80, help="Number of antennas in the simulated array [%(default)s]"
    )
    parser.add_argument(
        "--channels", type=int, default=32768, help="Total number of channels in the simulated stream [%(default)s]"
    )
    parser.add_argument(
        "--channels-per-substream", type=int, default=512, help="Number of channels sent by this fsim [%(default)s]"
    )
    parser.add_argument(
        "--samples-between-spectra", type=int, help="Number of digitiser samples between spectra [2*channels]"
    )
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        help="Number of antenna-channelised-voltage Jones vectors in each F-engine batch [%(default)s]",
    )
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="IP TTL for multicast [%(default)s]")
    parser.add_argument("--ibv", action="store_true", help="Use ibverbs for acceleration")
    parser.add_argument(
        "--send-packet-payload",
        type=int,
        default=DEFAULT_PACKET_PAYLOAD_BYTES,
        metavar="BYTES",
        help="Size for output packets (voltage payload only) [%(default)s]",
    )
    parser.add_argument(
        "--affinity", type=comma_split(int), default=[-1], help="Core affinity for the sending thread [not bound]"
    )
    parser.add_argument(
        "--main-affinity", type=int, default=-1, help="Core affinity for the main Python thread [not bound]"
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        help="Network port on which to serve Prometheus metrics [none]",
    )
    parser.add_argument("--run-once", action="store_true", help="Transmit a single collection of heaps before exiting")
    parser.add_argument(
        "dest",
        type=endpoint_list_parser(spead.DEFAULT_PORT),
        metavar="X.X.X.X[+N]:PORT",
        help="Destination addresses (one per polarisation)",
    )
    args = parser.parse_args(arglist)

    if args.samples_between_spectra is None:
        args.samples_between_spectra = 2 * args.channels
    if args.jones_per_batch % args.channels != 0:
        parser.error("--jones-per-batch must be divisible by --channels")

    return args


@njit
def make_heap_payload(
    out: np.ndarray,
    heap_index: int,
    feng_id: int,
    n_ants: int,
) -> None:
    """Create the simulated payload data for a heap.

    A pattern is chosen that will hopefully be easy to verify at the receiver
    graphically. On each F-Engine, the signal amplitude will increase linearly
    over time for each channel. Each channel will have a different starting
    amplitude but the rate of increase will be the same for all channels.

    Each F-Engine will have the same same signal amplitude for the same
    timestamp, but the signal phase will be different. The signal phase remains
    constant across all channels in a single F-Engine. By examining the signal
    phase it can be verified that correct feng_id is attached to the correct
    data.

    These samples need to be stored as 8 bit samples. As such, the amplitude is
    wrapped each time it reaches 127. 127 is used as the amplitude when
    multiplied by the phase can reach -127. The full range of values is
    covered.

    This current format is not fixed and it is likely that it will be adjusted
    to be suited for different verification needs.

    Parameters
    ----------
    out
        Output array, with shape (n_channels_per_substream, n_spectra_per_heap, N_POLS, COMPLEX)
    heap_index
        Heap index on time axis
    feng_id
        Heap index on antenna axis
    n_ants
        Number of antennas in the array
    """
    n_channels_per_substream = out.shape[0]
    n_spectra_per_heap = out.shape[1]
    initial_offset = heap_index * n_spectra_per_heap
    sample_angle = 2.0 * np.pi / (n_ants * N_POLS) * (feng_id * N_POLS + np.arange(2))
    for c in range(n_channels_per_substream):
        for t in range(n_spectra_per_heap):
            sample_amplitude = (initial_offset + c * 10 + t) % 127
            for p in range(N_POLS):
                out[c][t][p][0] = sample_amplitude * np.cos(sample_angle[p])
                out[c][t][p][1] = sample_amplitude * np.sin(sample_angle[p])


def make_heap(
    timestamp: np.ndarray, feng_id: int, channel_offset: int, payload: np.ndarray
) -> spead2.send.HeapReference:
    """Create a heap to transmit.

    The `timestamp` must be a zero-dimensional array of dtype ``>u8``. It will
    be transmitted by reference, so it can be updated in place to change the
    stored timestamp.
    """
    # The heap setup should be equivalent to katgpucbf.fgpu.send.Batch
    item_group = make_item_group(shape=payload.shape, dtype=payload.dtype)
    item_group[spead.TIMESTAMP_ID].value = timestamp
    item_group[spead.FENG_ID_ID].value = feng_id
    item_group[spead.FREQUENCY_ID].value = channel_offset
    item_group[spead.FENG_RAW_ID].value = payload
    heap = item_group.get_heap(descriptors="none", data="all")
    heap.repeat_pointers = True
    return spead2.send.HeapReference(heap)


def make_stream(args: argparse.Namespace, idx: int, data: np.ndarray) -> "spead2.send.asyncio.AsyncStream":
    """Create a SPEAD stream for a single destination.

    Parameters
    ----------
    args
        Command-line arguments
    idx
        Index into the destinations to use
    data
        All payload data for this destination
    """
    overhead = 1 + PREAMBLE_SIZE / args.send_packet_payload
    # Data rate for the entire array, excluding packet overhead
    full_rate = args.adc_sample_rate * N_POLS * DTYPE.itemsize * args.array_size
    rate = full_rate * args.channels_per_substream / args.channels * overhead
    print(f"Rate for {args.dest[idx]}: {rate * 8e-9:.3f} Gbps", flush=True)
    config = spead2.send.StreamConfig(
        max_packet_size=args.send_packet_payload + PREAMBLE_SIZE,
        rate=rate,
        max_heaps=QUEUE_DEPTH * args.array_size + 1,  # + 1 for descriptor heaps
    )
    interface_address = katsdpservices.get_interface_address(args.interface)
    affinity = args.affinity[idx % len(args.affinity)]
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    endpoints = [(args.dest[idx].host, args.dest[idx].port)]
    if args.ibv:
        udp_ibv_config = spead2.send.UdpIbvConfig(
            endpoints=endpoints,
            interface_address=interface_address,
            ttl=args.ttl,
            memory_regions=[data],
        )
        return spead2.send.asyncio.UdpIbvStream(thread_pool, config, udp_ibv_config)
    else:
        return spead2.send.asyncio.UdpStream(thread_pool, endpoints, config, args.ttl, interface_address)


class Sender:
    """Manage sending data to a single XB-engine."""

    def __init__(self, args: argparse.Namespace, idx: int) -> None:
        n_spectra_per_heap = args.jones_per_batch // args.channels
        self.data = np.empty(
            (
                QUEUE_DEPTH,
                args.array_size,
                args.channels_per_substream,
                n_spectra_per_heap,
                N_POLS,
                COMPLEX,
            ),
            DTYPE,
        )
        self.timestamp_step = args.samples_between_spectra * n_spectra_per_heap
        self.timestamps = np.empty(QUEUE_DEPTH, spead.IMMEDIATE_DTYPE)
        self.batches: list[spead2.send.HeapReferenceList] = []
        for i in range(QUEUE_DEPTH):
            batch_heaps = []
            for j in range(args.array_size):
                payload = self.data[i, j]
                make_heap_payload(payload, i, j, args.array_size)
                # The ... makes numpy return a 0d array instead of a scalar
                batch_heaps.append(
                    make_heap(
                        self.timestamps[i, ...], j, channel_offset=idx * args.channels_per_substream, payload=payload
                    )
                )
            self.batches.append(spead2.send.HeapReferenceList(batch_heaps))
        self.stream = make_stream(args, idx, self.data)
        self.time_error_gauge = time_error_gauge.labels(str(idx))
        self.batch_heaps = args.array_size
        self.batch_bytes = self.data[0].nbytes
        # Actual sync time will be filled in by run().
        self.time_converter = TimeConverter(0.0, args.adc_sample_rate)
        self.descriptor_heap = make_descriptor_heap(
            channels_per_substream=args.channels_per_substream,
            spectra_per_heap=n_spectra_per_heap,
            sample_bits=DTYPE.itemsize * BYTE_BITS,
        )

    def _update_metrics(self, end_timestamp: int, future: asyncio.Future) -> None:
        end_time = self.time_converter.adc_to_unix(end_timestamp)
        self.time_error_gauge.set(time.time() - end_time)
        output_heaps_counter.inc(self.batch_heaps)
        output_bytes_counter.inc(self.batch_bytes)

    async def run(self, sync_time: float, run_once: bool) -> None:
        """Send heaps until cancelled."""
        self.time_converter.sync_time = sync_time
        futures: list[asyncio.Future[int]] = [asyncio.get_running_loop().create_future() for _ in range(QUEUE_DEPTH)]
        for i in range(QUEUE_DEPTH):
            futures[i].set_result(0)  # Make the future ready
        timestamp = 0
        for idx in itertools.cycle(range(QUEUE_DEPTH)):
            await futures[idx]  # Wait for previous transmission of this batch
            self.timestamps[idx] = timestamp
            futures[idx] = self.stream.async_send_heaps(self.batches[idx], spead2.send.GroupMode.ROUND_ROBIN)
            timestamp += self.timestamp_step
            if run_once:
                break
            futures[idx].add_done_callback(functools.partial(self._update_metrics, timestamp))
        await asyncio.gather(*futures)


async def _async_main(tg: asyncio.TaskGroup) -> None:
    """Real implementation of :func:`async_main`.

    This is split into a separate function to avoid having to indent the whole
    thing inside the `tg` context manager.
    """
    args = parse_args()
    add_gc_stats()

    if args.prometheus_port is not None:
        await prometheus_async.aio.web.start_http_server(port=args.prometheus_port)

    senders = [Sender(args, i) for i in range(len(args.dest))]
    descriptor_senders = [
        DescriptorSender(sender.stream, sender.descriptor_heap, SPEAD_DESCRIPTOR_INTERVAL_S) for sender in senders
    ]
    for descriptor_sender in descriptor_senders:
        tg.create_task(descriptor_sender.run())

    if args.main_affinity >= 0:
        os.sched_setaffinity(0, [args.main_affinity])
    sync_time = time.time()
    async with asyncio.TaskGroup() as sender_tg:
        for sender in senders:
            sender_tg.create_task(sender.run(sync_time, args.run_once))
    for descriptor_sender in descriptor_senders:
        descriptor_sender.halt()


async def async_main() -> None:
    """Run main program."""
    async with asyncio.TaskGroup() as tg:
        await _async_main(tg)


def main() -> None:
    """Run main program."""
    katsdpservices.setup_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
