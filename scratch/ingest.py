#!/usr/bin/env python3

import argparse
import ast
import asyncio
from typing import Union

import aiokatcp
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

complexity = 2


async def get_katcp_sensor_value(client: aiokatcp.Client, sensor_name: str) -> Union[int, float]:
    """Get the relevant value from a katcp sensor.

    If the sensor isn't either an int or a float, the value will get returned
    as a string.
    """
    _reply, informs = await client.request("sensor-value", sensor_name)

    expected_types = [int, float, str]
    for T in expected_types:
        try:
            return aiokatcp.decode(T, informs[0].arguments[4])
        except ValueError:
            continue


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    return endpoint_parser(None)(await get_katcp_sensor_value(client, f"{product_name}.katcp-address"))


async def async_main(host: str, port: int):
    """TODO: This functionality should be wrapped up in a class really."""
    client = await aiokatcp.Client.connect(host, port)

    # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
    # But we can cast Endpoints to tuples, which it does know.
    multicast_endpoints = [
        tuple(endpoint)
        for endpoint in endpoint_list_parser(7148)(
            await get_katcp_sensor_value(client, "baseline_correlation_products-destination")
        )
    ]

    # We need these parameters for various useful reasons.
    n_bls = await get_katcp_sensor_value(client, "baseline_correlation_products-n-bls")
    n_chans = await get_katcp_sensor_value(client, "baseline_correlation_products-n-chans")
    n_chans_per_substream = await get_katcp_sensor_value(client, "baseline_correlation_products-n-chans-per-substream")
    n_bits_per_sample = await get_katcp_sensor_value(client, "baseline_correlation_products-xeng-out-bits-per-sample")
    n_spectra_per_acc = await get_katcp_sensor_value(client, "baseline_correlation_products-n-accs")
    n_xengs = await get_katcp_sensor_value(client, "baseline_correlation_products-n-xengs")
    adc_sample_rate = await get_katcp_sensor_value(client, "antenna_channelised_voltage-adc-sample-rate")

    # The only reason for getting this info is to annotate the plot we make at the end.
    bls_ordering = ast.literal_eval(await get_katcp_sensor_value(client, "baseline_correlation_products-bls-ordering"))
    # I quite like this trick. It gives us a list of tuples.

    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * complexity * n_bits_per_sample // 8

    # According to the ICD.
    TIMESTAMP = 0x1600
    FREQUENCY = 0x4103

    # These are the spead items that we will need for placing the individual
    # heaps within the chunk.
    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]
    timestamp_step = 2 * n_chans * n_spectra_per_acc

    # Heap placement function. Gets translated from Python to C so that spead2
    # can use it.
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

    max_chunks = 5  # Just a guess. No logic to this just yet.
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

    # For the time being we'll make a chunk look like what katsdpingest calls
    # a "frame".
    HEAPS_PER_CHUNK = n_xengs
    CHUNK_PAYLOAD_SIZE = HEAPS_PER_CHUNK * HEAP_PAYLOAD_SIZE

    for _ in range(max_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8), data=np.empty(CHUNK_PAYLOAD_SIZE, np.uint8)
        )
        stream.add_free_chunk(chunk)

    config = spead2.recv.UdpIbvConfig(
        endpoints=multicast_endpoints, interface_address=args.interface, buffer_size=1000000, comp_vector=-1
    )
    stream.add_udp_ibv_reader(config)

    # Preparation for the plot. Doing it outside the for-loop, no need to redo it lots.
    frequency_axis = np.fft.rfftfreq(2 * n_chans, d=1 / adc_sample_rate)[:-1]  # -1 because it goes all the way to n/2

    async for chunk in stream.data_ringbuffer:
        received_heaps = int(np.sum(chunk.present))
        if HEAPS_PER_CHUNK - received_heaps == 0:
            # We have a full chunk.
            data = chunk.data.view(dtype=np.int32).reshape(n_chans, n_bls, 2)

            plt.figure(figsize=(8, 24))
            for i in range(n_bls):
                ax = plt.subplot(n_bls, 1, i + 1)
                # We're just plotting the magnitude for now. Phase is easy enough,
                # and is left as an exercise to the reader.
                plt.plot(frequency_axis, np.abs(data[:, i, 0] + 1j * data[:, i, 0]))
                # This just makes the x-ticks only on the bottom graph, it makes
                # the plot somewhat less cluttered.
                plt.setp(ax.get_xticklabels(), visible=(True if i == n_bls - 1 else False))
                ax.set_ylabel(bls_ordering[i])

            ax.xaxis.set_major_formatter(lambda x, _: x / 1e6)  # Numbers are a bit large, let's display in MHz instead.
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
        default="lab5.sdp.kat.ac.za:5001",  # Naturally this applies only to our lab...
        help="Master controller to query for details about the product. [%(default)s]",
    )
    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()

    host, port = asyncio.run(get_product_controller_endpoint(args.mc_address, args.product_name))
    asyncio.run(async_main(host, int(port)))
