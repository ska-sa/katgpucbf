#!/usr/bin/env python3
import argparse
import asyncio
import logging
from typing import List, Union

import aiokatcp
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


async def print_all_sensors(client: aiokatcp.Client):
    _reply, informs = await client.request("sensor-value")
    for inform in informs:
        print(inform)


async def get_sensor_val(client: aiokatcp.Client, sensor_name: str) -> Union[int, float, str]:
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


async def get_dsim_endpoint(pc_client: Endpoint) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    return endpoint_parser(None)(await get_sensor_val(pc_client, "sim.m800.1712000000.0.port"))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        type=get_interface_address,
        required=True,
        help="Name of network  to use for ingest.",
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
    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()
    asyncio.run(async_main(args))


async def async_main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    host, port = await get_product_controller_endpoint(args.mc_address, args.product_name)
    logger.info("Connecting to product controller on %s:%d", host, port)
    pc_client = await aiokatcp.Client.connect(host, port)
    logger.info("Successfully connected to product controller.")

    async with pc_client:
        dsim_host, dsim_port = await get_dsim_endpoint(pc_client)
        logger.info("Connecting to dsim 0 on %s:%d", dsim_host, dsim_port)
        dsim_client = await aiokatcp.Client.connect(dsim_host, dsim_port)
        logger.info("Successfully connected to dsim.")

        # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
        # But we can cast Endpoints to tuples, which it does know.
        multicast_endpoints = [
            tuple(endpoint)
            for endpoint in endpoint_list_parser(7148)(
                await get_sensor_val(pc_client, "baseline_correlation_products-destination")
            )
        ]

        # We need these parameters for various useful reasons.
        n_bls = await get_sensor_val(pc_client, "baseline_correlation_products-n-bls")
        n_chans = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans")
        n_chans_per_substream = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans-per-substream")
        n_bits_per_sample = await get_sensor_val(pc_client, "baseline_correlation_products-xeng-out-bits-per-sample")
        n_spectra_per_acc = await get_sensor_val(pc_client, "baseline_correlation_products-n-accs")
        bandwidth = await get_sensor_val(pc_client, "antenna_channelised_voltage-bandwidth")
        channel_width = bandwidth / n_chans
        int_time = await get_sensor_val(pc_client, "baseline_correlation_products-int-time")

    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * CPLX * n_bits_per_sample // 8  # noqa: N806
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806

    # According to the ICD.
    TIMESTAMP = 0x1600  # noqa: N806
    FREQUENCY = 0x4103  # noqa: N806

    # These are the spead items that we will need for placing the individual
    # heaps within the chunk.
    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]
    timestamp_step = 2 * n_chans * n_spectra_per_acc  # True only for wideband.

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

    stream_config = spead2.recv.StreamConfig(substreams=HEAPS_PER_CHUNK)

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
        chunk.chunk_id

    if args.ibv:
        config = spead2.recv.UdpIbvConfig(
            endpoints=multicast_endpoints, interface_address=args.interface, buffer_size=int(16e6), comp_vector=-1
        )
        stream.add_udp_ibv_reader(config)
    else:
        for ep in multicast_endpoints:
            stream.add_udp_reader(*ep, interface_address=args.interface)

    # Channelisation test.
    channel = 1234  # picked fairly arbitrarily.
    channel_centre_freq = channel * channel_width  # TODO: check this.
    logger.info("Checking channel %d at frequency %f", channel, channel_centre_freq)
    logger.info("We expect the channel width to be %f", channel_width)

    chans_on_either_side = 2
    points_per_channel = 50
    num_points_on_either_side = round((chans_on_either_side + 0.5) * points_per_channel - 1)

    # TODO: This formula may be wrong. It gets me what I want for the time being,
    # but if the code gets reused in a more formal setting, it'll want checking.
    frequencies_to_check = np.linspace(
        channel_centre_freq - (chans_on_either_side + 0.5) * channel_width,
        channel_centre_freq + (chans_on_either_side + 0.5) * channel_width,
        2 * num_points_on_either_side + 1,
        endpoint=True,
    )
    frequency_response: List[float] = []
    to_the_left: List[float] = []
    to_the_right: List[float] = []
    n_freqs = len(frequencies_to_check)

    for n, freq in enumerate(frequencies_to_check):
        logger.info("Setting dsim cw freq to %f", freq)
        reply, _informs = await dsim_client.request("signals", f"common=cw(0.15,{freq})+wgn(0.01);common;common;")
        expected_timestamp = int(reply[0])

        async for chunk in stream.data_ringbuffer:
            recvd_timestamp = chunk.chunk_id * timestamp_step
            if not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
                stream.add_free_chunk(chunk)
            elif recvd_timestamp <= expected_timestamp + timestamp_step:  # give ourselves a bit of buffer for luck
                logger.info("Skipping chunk with timestamp %d", recvd_timestamp)
                stream.add_free_chunk(chunk)
            else:
                logger.info("Received a chunk with %f Hz, recording response! (%d/%d)", freq, n, n_freqs)
                frequency_response.append(chunk.data[channel, 0, 0])
                to_the_left.append(chunk.data[channel - 1, 0, 0])
                to_the_right.append(chunk.data[channel + 1, 0, 0])
                stream.add_free_chunk(chunk)
                break

    frequency_response = np.maximum(frequency_response, 1e-6)
    to_the_left = np.maximum(to_the_left, 1e-6)
    to_the_right = np.maximum(to_the_right, 1e-6)
    xaxis = (frequencies_to_check - channel_centre_freq) / channel_width
    plt.plot(xaxis, 10 * np.log10(frequency_response))
    plt.plot(xaxis, 10 * np.log10(to_the_left))
    plt.plot(xaxis, 10 * np.log10(to_the_right))
    plt.xlabel(f"Channel relative to {channel}")
    plt.ylabel("Channel response [dB]")
    plt.title("Channelisation Test result")
    plt.savefig("channelisation.png")


if __name__ == "__main__":
    main()
