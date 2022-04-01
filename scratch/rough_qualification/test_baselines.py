#!/usr/bin/env python3
import argparse
import ast
import asyncio
import logging
import time
from collections import namedtuple
from typing import Tuple, Union

import aiokatcp
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

logger = logging.getLogger(__name__)

Baseline = namedtuple("Baseline", ["ant0", "pol0", "ant1", "pol1"])


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


async def get_dsim_endpoint(pc_client: aiokatcp.Client) -> Endpoint:
    """Get the katcp address for a dsim on a product controller (with hardcoded name)."""
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
    host, port = await get_product_controller_endpoint(args.mc_address, args.product_name)
    logger.info("Connecting to product controller on %s:%d", host, port)
    pc_client = await aiokatcp.Client.connect(host, port)
    logger.info("Successfully connected to product controller.")

    # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
    # But we can cast Endpoints to tuples, which it does know.
    multicast_endpoints = [
        tuple(endpoint)
        for endpoint in endpoint_list_parser(7148)(
            await get_sensor_val(pc_client, "baseline_correlation_products-destination")
        )
    ]

    # We need these parameters for various useful reasons.
    n_ants = await get_sensor_val(pc_client, "antenna_channelised_voltage-n-fengs")
    n_bls = await get_sensor_val(pc_client, "baseline_correlation_products-n-bls")
    n_chans = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans")
    n_chans_per_substream = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans-per-substream")
    n_bits_per_sample = await get_sensor_val(pc_client, "baseline_correlation_products-xeng-out-bits-per-sample")
    n_spectra_per_acc = await get_sensor_val(pc_client, "baseline_correlation_products-n-accs")
    bandwidth = await get_sensor_val(pc_client, "antenna_channelised_voltage-bandwidth")
    int_time = await get_sensor_val(pc_client, "baseline_correlation_products-int-time")
    sync_time = await get_sensor_val(pc_client, "antenna_channelised_voltage-sync-time")
    timestamp_scale_factor = await get_sensor_val(pc_client, "antenna_channelised_voltage-scale-factor-timestamp")

    bls_ordering = ast.literal_eval(await get_sensor_val(pc_client, "baseline_correlation_products-bls-ordering"))

    # Get dsim ready.
    channel_width = bandwidth / n_chans
    dsim_host, dsim_port = await get_dsim_endpoint(pc_client)
    channel = 1234  # picked fairly arbitrarily. We just need to see the tone.
    await setup_dsim(dsim_host, dsim_port, channel, channel_width)

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

    setup_stream(args, multicast_endpoints, n_bls, n_chans, n_bits_per_sample, HEAPS_PER_CHUNK, max_chunks, stream)

    # Baseline test.
    # Let's have some functions to help us.
    async def zero_all_gains():
        for ant in range(n_ants):
            for pol in ["v", "h"]:
                logger.debug(f"Setting gain to zero on m{800 + ant}{pol}")
                await pc_client.request("gain", "antenna_channelised_voltage", f"m{800 + ant}{pol}", "0")

    async def unzero_a_baseline(baseline_tuple: Tuple[str]):
        logger.debug(f"Unzeroing gain on {baseline_tuple}")
        for ant in baseline_tuple:
            await pc_client.request("gain", "antenna_channelised_voltage", ant, "1e-4")

    for bl_idx, bl in enumerate(bls_ordering):
        current_bl = Baseline(int(bl[0][3]), bl[0][4], int(bl[1][3]), bl[1][4])
        logger.info("Checking baseline %r (%d)", bl, bl_idx)
        await zero_all_gains()
        await unzero_a_baseline(bl)
        expected_timestamp = (time.time() + 1 - sync_time) * timestamp_scale_factor
        # Note that we are making an assumption that nothing is straying too far
        # from wall time here. I don't have a way other than adjusting the dsim
        # signal of ensuring that we get going after a specific timestamp in the
        # DSP pipeline itself.

        async for chunk in stream.data_ringbuffer:
            recvd_timestamp = chunk.chunk_id * timestamp_step
            if not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
                stream.add_free_chunk(chunk)

            elif recvd_timestamp <= expected_timestamp:
                logger.debug("Skipping chunk with timestamp %d", recvd_timestamp)
                stream.add_free_chunk(chunk)

            else:
                loud_bls = np.nonzero(chunk.data[channel, :, 0])[0]
                logger.info("%d bls had signal in them: %r", len(loud_bls), loud_bls)
                assert bl_idx in loud_bls  # Best to check the expected baselin is actually in the list.
                for loud_bl in loud_bls:
                    check_signal_expected_in_bl(bl_idx, bl, current_bl, loud_bl, bls_ordering)
                stream.add_free_chunk(chunk)
                break


def setup_stream(args, multicast_endpoints, n_bls, n_chans, n_bits_per_sample, heaps_per_chunk, max_chunks, stream):
    """Set up the spead2 stream needed for ingest."""
    for _ in range(max_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(heaps_per_chunk, np.uint8),
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


async def setup_dsim(dsim_host, dsim_port, channel, channel_width):
    logger.info("Connecting to dsim 0 on %s:%d", dsim_host, dsim_port)
    dsim_client = await aiokatcp.Client.connect(dsim_host, dsim_port)
    logger.info("Successfully connected to dsim.")
    channel_centre_freq = channel * channel_width  # TODO: check this.
    # Set the dsim with a tone.
    async with dsim_client:
        await dsim_client.request("signals", f"common=cw(0.15,{channel_centre_freq})+wgn(0.01);common;common;")


def check_signal_expected_in_bl(bl_idx, bl, current_bl, loud_bl, bls_ordering):
    def get_bl_idx(ant0: int, pol0: str, ant1: int, pol1: str) -> int:
        return bls_ordering.index((f"m{800 + ant0}{pol0}", f"m{800 + ant1}{pol1}"))

    if loud_bl == bl_idx:
        logger.info("Signal confirmed in bl %d for %r where expected", loud_bl, bl)
    elif loud_bl == get_bl_idx(current_bl.ant0, current_bl.pol0, current_bl.ant0, current_bl.pol0):
        logger.debug("Signal in %r - fine - it's ant0's autocorrelation.", loud_bl)
    elif loud_bl == get_bl_idx(current_bl.ant1, current_bl.pol1, current_bl.ant1, current_bl.pol1):
        logger.debug("Signal in %r - fine - it's ant1's autocorrelation.", loud_bl)
    elif loud_bl == get_bl_idx(current_bl.ant1, current_bl.pol1, current_bl.ant0, current_bl.pol0):
        logger.debug(
            "Signal in %r - fine - it's the negative of what we expect.",
            loud_bl,
        )
    else:
        logger.error("Signal in %d but it wasn't expected there!", loud_bl)


if __name__ == "__main__":
    main()
