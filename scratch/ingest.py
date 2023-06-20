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


async def get_sensor_val(client: aiokatcp.Client, sensor_name: str) -> int | float | str:
    """Get the value of a katcp sensor.

    If the sensor value can't be cast as an int or a float (in that order), the
    value will get returned as a string. This simple implementation ignores the
    actual type advertised by the server.
    """
    _reply, informs = await client.request("sensor-value", sensor_name)

    expected_types = [int, float, str]
    for t in expected_types:
        try:
            return aiokatcp.decode(t, informs[0].arguments[4])
        except ValueError:
            continue


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        return endpoint_parser(None)(await get_sensor_val(client, f"{product_name}.katcp-address"))


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
        help="Master controller to query for details about the product. [%(default)s]",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show dumps as they are received",
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
    host, port = await get_product_controller_endpoint(args.mc_address, args.product_name)
    client = await aiokatcp.Client.connect(host, port)

    async with client:
        # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
        # But we can cast Endpoints to tuples, which it does know.
        multicast_endpoints = [
            tuple(endpoint)
            for endpoint in endpoint_list_parser(7148)(
                await get_sensor_val(client, "baseline-correlation-products.destination")
            )
        ]

        # We need these parameters for various useful reasons.
        n_bls = await get_sensor_val(client, "baseline-correlation-products.n-bls")
        n_chans = await get_sensor_val(client, "baseline-correlation-products.n-chans")
        n_chans_per_substream = await get_sensor_val(client, "baseline-correlation-products.n-chans-per-substream")
        n_bits_per_sample = await get_sensor_val(client, "baseline-correlation-products.xeng-out-bits-per-sample")
        n_spectra_per_acc = await get_sensor_val(client, "baseline-correlation-products.n-accs")
        adc_sample_rate = await get_sensor_val(client, "antenna-channelised-voltage.adc-sample-rate")
        int_time = await get_sensor_val(client, "baseline-correlation-products.int-time")

        # The only reason for getting this info is to annotate the plot we make at the end.
        # I quite like this trick. It gives us a list of tuples.
        bls_ordering = ast.literal_eval(await get_sensor_val(client, "baseline-correlation-products.bls-ordering"))

    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * CPLX * n_bits_per_sample // 8  # noqa: N806
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806

    # According to the ICD.
    TIMESTAMP = 0x1600  # noqa: N806
    FREQUENCY = 0x4103  # noqa: N806

    # These are the spead items that we will need for placing the individual
    # heaps within the chunk.
    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]
    timestamp_step = 2 * n_chans * n_spectra_per_acc

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

    stream_config = spead2.recv.StreamConfig(max_heaps=HEAPS_PER_CHUNK * 3)

    # Assuming X-engines are at most 1 second out of sync, with one extra chunk for luck.
    # May need to revisit that assumption for much larger array sizes.
    max_chunks = round(1 // int_time) + 1
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

    fig, axs = plt.subplots(
        n_bls // 4, 4, sharex=True, figsize=(12, 8) if args.interactive else (24, 16), constrained_layout=True
    )
    axs = axs.ravel()  # Now that the plot is four columns wide, this comes as a 2-d array. Which is a pain.
    lines = []
    for i in range(len(axs)):
        lines.append(axs[i].plot(frequency_axis, [0.0] * len(frequency_axis))[0])
        axs[i].set_ylabel(bls_ordering[i])
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
            for i in range(n_bls):
                # We're just plotting the magnitude for now. Phase is easy enough,
                # and is left as an exercise to the reader.
                lines[i].set_ydata(np.abs(chunk.data[:, i, 0] + 1j * chunk.data[:, i, 0]))
                axs[i].relim()
                axs[i].autoscale_view()
            plt.title(f"Chunk {chunk.chunk_id}")
            if not args.interactive:
                plt.savefig(f"{chunk.chunk_id}.png")
                logger.info("Wrote chunk %d", chunk.chunk_id)
        else:
            logger.warning("Chunk %d missing heaps! (This is expected for the first few.)", chunk.chunk_id)
        stream.add_free_chunk(chunk)


if __name__ == "__main__":
    main()
