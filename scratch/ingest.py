#!/usr/bin/env python3

import argparse
import ast
import asyncio
from math import frexp
from typing import Optional, Sequence

import aiokatcp
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from aiokatcp.core import Timestamp
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

complexity = 2


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str):
    """Get the katcp address for a named product controller from the master."""
    mc_host, mc_port = mc_endpoint
    client = await aiokatcp.Client.connect(mc_host, mc_port)
    _reply, informs = await client.request("sensor-value", f"{product_name}.katcp-address")
    pc_host, pc_port = endpoint_parser(7148)(informs[0].arguments[4].decode("ascii"))
    return pc_host, pc_port


async def async_main(host: str, port: int):
    client = await aiokatcp.Client.connect(host, port)

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-destination")
    multicast_endpoints = endpoint_list_parser(7148)(informs[0].arguments[4].decode("ascii"))

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-n-bls")
    n_bls = int(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-n-chans")
    n_chans = int(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-n-chans-per-substream")
    n_chans_per_substream = int(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-xeng-out-bits-per-sample")
    n_bits_per_sample = int(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-n-xengs")
    n_xengs = int(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "antenna_channelised_voltage-adc-sample-rate")
    adc_sample_rate = float(informs[0].arguments[4])

    _reply, informs = await client.request("sensor-value", "baseline_correlation_products-bls-ordering")
    bls_ordering = ast.literal_eval(informs[0].arguments[4].decode())

    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * complexity * n_bits_per_sample // 8

    print(f"heap pl size: {HEAP_PAYLOAD_SIZE}")

    TIMESTAMP = 0x1600
    FREQUENCY = 0x4103

    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]

    max_chunks = 5
    timestamp_step = (
        2 * n_chans * (spectra_per_heap := 256) * (heaps_per_accum := 1)
    )  # values from the low-speed correlator I created
    print(f"timestamp_step {timestamp_step}")

    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
    def chunk_place(data_ptr, data_size):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
        channel_offset = items[0]
        timestamp = items[1]
        payload_size = items[2]
        # If the payload size doesn't match, discard the heap (could be descriptors etc).
        if payload_size == HEAP_PAYLOAD_SIZE:  # This isn't working. Somehow payload size is wrong.
            data[0].chunk_id = timestamp // timestamp_step
            data[0].heap_index = channel_offset // n_chans_per_substream
            data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE

    stream_config = spead2.recv.StreamConfig(
        max_heaps=n_xengs * 10,
        allow_out_of_order=True,
    )  # just an arbitrary guess for now
    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=items,
        max_chunks=max_chunks,
        place=scipy.LowLevelCallable(chunk_place.ctypes, signature="void (void *, size_t)"),
    )
    free_ringbuffer = spead2.recv.ChunkRingbuffer(max_chunks)
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(max_chunks)
    stream = spead2.recv.ChunkRingStream(
        spead2.ThreadPool(1, []),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )

    HEAPS_PER_CHUNK = n_xengs  # for now
    CHUNK_PAYLOAD_SIZE = HEAPS_PER_CHUNK * HEAP_PAYLOAD_SIZE

    for _ in range(max_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8), data=np.empty(CHUNK_PAYLOAD_SIZE, np.uint8)
        )
        stream.add_free_chunk(chunk)

    mc_ep = [(str(host), int(port)) for host, port in multicast_endpoints]
    print(mc_ep)
    config = spead2.recv.UdpIbvConfig(
        endpoints=mc_ep, interface_address=args.interface, buffer_size=1000000, comp_vector=-1
    )
    stream.add_udp_ibv_reader(config)

    async_ringbuffer = stream.data_ringbuffer

    expected_heaps_total = 0
    dropped_heaps_total = 0

    frequency_axis = np.fft.rfftfreq(2 * n_chans, d=1 / adc_sample_rate)[:-1]  # Because it goes up to n/2 as well

    async for chunk in async_ringbuffer:
        expected_heaps = len(chunk.present)
        received_heaps = int(np.sum(chunk.present))
        dropped_heaps = expected_heaps - received_heaps
        expected_heaps_total += expected_heaps
        dropped_heaps_total += dropped_heaps

        if dropped_heaps == 0:
            # We have a full chunk. Discard the first few, they're likely to be messy.
            data = chunk.data.view(dtype=np.int32).reshape(n_chans, n_bls, 2)

            fig = plt.figure(figsize=(8, 24))
            # fig.suptitle(f"Baselines for frame with timestamp {chunk.chunk_id * timestamp_step}")
            for i in range(n_bls):

                ax = plt.subplot(n_bls, 1, i + 1)
                plt.plot(frequency_axis, np.abs(data[:, i, 0] + 1j * data[:, i, 0]))
                plt.setp(ax.get_xticklabels(), visible=(True if i == n_bls - 1 else False))
                ax.set_ylabel(bls_ordering[i])

            ax.xaxis.set_major_formatter(lambda x, y: x / 1e6)
            ax.set_xlabel("Frequency [MHz]")

            plt.savefig(f"{chunk.chunk_id}.png")

        stream.add_free_chunk(chunk)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        type=str,
        required=True,
        help="IP address of ibverbs interface",
    )
    parser.add_argument(
        "--mc-address",
        type=endpoint_parser(5001),
        default="lab5.sdp.kat.ac.za:5001",
        help="Master controller to query for details about the product. [%(default)s]",
    )
    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()

    host, port = asyncio.run(get_product_controller_endpoint(args.mc_address, args.product_name))
    asyncio.run(async_main(host, int(port)))
