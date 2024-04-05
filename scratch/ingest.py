#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021-2023, National Research Foundation (SARAO)
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
import ast
import asyncio
import logging
import sys
from typing import List, Sequence, Tuple

import aiokatcp
import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

CPLX = 2
CORR2_SENSOR_REMAP = {
    "antenna-channelised-voltage.adc-sample-rate": "adc-sample-rate",
}
# According to the ICD.
TIMESTAMP = 0x1600
FREQUENCY = 0x4103


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        return endpoint_parser(None)(str(await client.sensor_value(f"{product_name}.katcp-address", aiokatcp.Address)))


async def get_subordinate_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for an array's subordinate controller from the CMC."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        reply, informs = await client.request("subordinate-list")
        for inform in informs:
            if aiokatcp.decode(str, inform.arguments[0]) == product_name:
                ports = aiokatcp.decode(str, inform.arguments[1]).split(",")
                return Endpoint(mc_endpoint.host, int(ports[0]))


def make_chunk_place(
    *,
    n_bls: int,
    n_chans_per_substream: int,
    n_bits_per_sample: int,
    n_spectra_per_acc: int,
    n_samples_between_spectra: int,
):
    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * CPLX * n_bits_per_sample // 8  # noqa: N806
    timestamp_step = n_samples_between_spectra * n_spectra_per_acc

    # Heap placement function. Gets compiled so that spead2's C code can call it.
    # A chunk consists of all baselines and channels for a single point in time.
    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
    def chunk_place(data_ptr, data_size):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
        channel_offset = items[0]
        timestamp = items[1]
        payload_size = items[2]
        # If the payload size doesn't match, discard the heap (could be descriptors etc).
        if payload_size == HEAP_PAYLOAD_SIZE:
            data[0].chunk_id = timestamp // timestamp_step
            data[0].heap_index = channel_offset // n_chans_per_substream
            data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE

    return chunk_place


def get_bls_subset(baselines: Sequence[str], bls_ordering: Sequence[Tuple[str, str]]) -> List[int]:
    """Parse the command-line --baseline arguments."""
    if baselines:
        bls_subset = []
        for bls in baselines:
            bls_tuple = tuple(bls.split(","))  # Turn "m000h,m000v" into ("m000h", "m000v")
            try:
                idx = bls_ordering.index(bls_tuple)
            except ValueError:
                print(f"Baseline {bls} not found", file=sys.stderr)
                sys.exit(1)
            bls_subset.append(idx)
    else:
        bls_subset = list(range(len(bls_ordering)))
    return bls_subset


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        type=get_interface_address,
        required=True,
        help="Name of network interface.",
    )
    parser.add_argument(
        "--ibv",
        action="store_true",
        help="Use ibverbs",
    )
    parser.add_argument(
        "--mc-address",
        type=endpoint_parser(5001),
        default="lab5.sdp.kat.ac.za:5001",  # Naturally this applies only to our lab...
        metavar="HOST:PORT",
        help="Master controller to query for details about the product. [%(default)s]",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show dumps as they are received",
    )
    parser.add_argument(
        "--corr2",
        action="store_true",
        help="Target a correlator based on corr2 rather than katgpucbf",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="INPUT,INPUT",
        help="Restrict output to a specific baseline (repeat for multiple baselines)",
    )
    parser.add_argument(
        "--phase",
        action="store_true",
        help="Plot phase instead of magnitude",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns to plot",
    )
    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()
    if args.interactive:
        matplotlib.use("WebAgg")
        # plt.show will run a Tornado event loop, which in turn runs an asyncio
        # event loop. So we can't directly run an event loop ourselves, but
        # must instead schedule async_main onto the one run by matplotlib.
        asyncio.get_event_loop().create_task(async_main(args))
        plt.show()
    else:
        asyncio.run(async_main(args))


async def async_main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    if args.corr2:
        host, port = await get_subordinate_endpoint(args.mc_address, args.product_name)
    else:
        host, port = await get_product_controller_endpoint(args.mc_address, args.product_name)
    client = await aiokatcp.Client.connect(host, port)

    async with client:

        async def sensor_val(name, sensor_type=None):
            if args.corr2:
                name = CORR2_SENSOR_REMAP.get(name, name.replace(".", "-"))
            return await client.sensor_value(name, sensor_type)

        # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
        # But we can cast Endpoints to tuples, which it does know.
        multicast_endpoints = [
            tuple(endpoint)
            for endpoint in endpoint_list_parser(7148)(
                await sensor_val("baseline-correlation-products.destination", str)
            )
        ]

        # We need these parameters for various useful reasons.
        n_bls = await sensor_val("baseline-correlation-products.n-bls", int)
        n_chans = await sensor_val("baseline-correlation-products.n-chans", int)
        n_chans_per_substream = await sensor_val("baseline-correlation-products.n-chans-per-substream", int)
        n_bits_per_sample = await sensor_val("baseline-correlation-products.xeng-out-bits-per-sample", int)
        n_spectra_per_acc = await sensor_val("baseline-correlation-products.n-accs", int)
        n_samples_between_spectra = await sensor_val("antenna-channelised-voltage.n-samples-between-spectra", int)
        adc_sample_rate = await sensor_val("antenna-channelised-voltage.adc-sample-rate", float)
        int_time = await sensor_val("baseline-correlation-products.int-time", float)

        # I quite like this trick. It gives us a list of tuples.
        bls_ordering = ast.literal_eval(await sensor_val("baseline-correlation-products.bls-ordering", str))

    bls_subset = get_bls_subset(args.baseline, bls_ordering)

    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806

    stream_config = spead2.recv.StreamConfig(max_heaps=HEAPS_PER_CHUNK * 3)

    # These are the SPEAD items that we will need for placing the individual
    # heaps within the chunk.
    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]
    # Assuming X-engines are at most 1 second out of sync, with one extra chunk for luck.
    # May need to revisit that assumption for much larger array sizes.
    max_chunks = round(1 // int_time) + 1
    chunk_place = make_chunk_place(
        n_bls=n_bls,
        n_chans_per_substream=n_chans_per_substream,
        n_spectra_per_acc=n_spectra_per_acc,
        n_samples_between_spectra=n_samples_between_spectra,
        n_bits_per_sample=n_bits_per_sample,
    )
    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=items,
        max_chunks=max_chunks,
        place=scipy.LowLevelCallable(chunk_place.ctypes, signature="void (void *, size_t)"),
    )
    free_ringbuffer = spead2.recv.ChunkRingbuffer(max_chunks)
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(max_chunks)
    stream = spead2.recv.ChunkRingStream(
        spead2.ThreadPool(),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )

    for _ in range(max_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty((n_chans, n_bls, CPLX), dtype=getattr(np, f"int{n_bits_per_sample}")),
        )
        stream.add_free_chunk(chunk)

    config = spead2.recv.UdpIbvConfig(
        endpoints=multicast_endpoints, interface_address=args.interface, buffer_size=int(16e6), comp_vector=-1
    )
    if args.ibv:
        stream.add_udp_ibv_reader(config)
    else:
        for ep in multicast_endpoints:
            stream.add_udp_reader(*ep, interface_address=args.interface)

    # Preparation for the plot. Doing it outside the for-loop, no need to redo it lots.
    frequency_axis = np.fft.rfftfreq(2 * n_chans, d=1 / adc_sample_rate)[:-1]  # -1 because it goes all the way to n/2

    columns = args.columns
    rows = (len(bls_subset) + columns - 1) // columns
    fig, axs = plt.subplots(
        rows,
        columns,
        sharex=True,
        squeeze=False,
        figsize=(12, 8) if args.interactive else (24, 16),
        constrained_layout=True,
    )
    axs = axs.ravel()  # Now that the plot is several columns wide, this comes as a 2-d array. Which is a pain.
    axs = axs[: len(bls_subset)]  # If there are gaps at the end, ignore them
    lines = []
    for i, idx in enumerate(bls_subset):
        lines.append(axs[i].plot(frequency_axis, [0.0] * len(frequency_axis))[0])
        axs[i].set_ylabel(bls_ordering[idx])
        axs[i].xaxis.set_major_formatter(lambda x, _: x / 1e6)  # Display frequency in MHz.
        axs[i].set_xlabel("Frequency [MHz]")
    if args.interactive:
        fig.canvas.mpl_connect("close_event", lambda event: stream.stop())
        plt.ion()
        plt.show()
    async for chunk in stream.data_ringbuffer:
        received_heaps = int(np.sum(chunk.present))
        if received_heaps == HEAPS_PER_CHUNK:
            # We have a full chunk.
            for i, idx in enumerate(bls_subset):
                # We're just plotting the magnitude for now. Phase is easy enough,
                # and is left as an exercise to the reader.
                data = chunk.data[:, idx, 0] + 1j * chunk.data[:, idx, 0]
                if args.phase:
                    data = np.angle(data)
                else:
                    data = np.abs(data)
                lines[i].set_ydata(data)
                axs[i].relim()
                axs[i].autoscale_view()
            plt.title(f"Chunk {chunk.chunk_id}")
            if not args.interactive:
                plt.savefig(f"{chunk.chunk_id}.png")
                logger.info("Wrote chunk %d", chunk.chunk_id)
        else:
            logger.warning(
                "Chunk %d missing %d/%d heaps! (This is expected for the first few.)",
                chunk.chunk_id,
                HEAPS_PER_CHUNK - received_heaps,
                HEAPS_PER_CHUNK,
            )
        stream.add_free_chunk(chunk)


if __name__ == "__main__":
    main()
