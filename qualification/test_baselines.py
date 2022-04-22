#!/usr/bin/env python3

"""Baseline verification tests."""

import ast
import logging
import time
from typing import Tuple

import numpy as np
import spead2
import spead2.recv
import spead2.recv.asyncio

from . import CorrelatorRemoteControl, get_sensor_val

CPLX = 2

logger = logging.getLogger(__name__)


async def test_baselines(
    n_antennas: int, n_channels: int, correlator: CorrelatorRemoteControl, receive_stream: spead2.recv.ChunkRingStream
):
    """Test that the baseline ordering indicated in the sensor matches the output data."""
    pc_client = correlator.product_controller_client

    # Get some necessary sensor values from the correlator.
    bls_ordering = ast.literal_eval(await get_sensor_val(pc_client, "baseline_correlation_products-bls-ordering"))
    sync_time = await get_sensor_val(pc_client, "antenna_channelised_voltage-sync-time")
    timestamp_scale_factor = await get_sensor_val(pc_client, "antenna_channelised_voltage-scale-factor-timestamp")
    n_samples_between_spectra = await get_sensor_val(pc_client, "antenna_channelised_voltage-n-samples-between-spectra")
    n_spectra_per_acc = await get_sensor_val(pc_client, "baseline_correlation_products-n-accs")
    bandwidth = await get_sensor_val(pc_client, "antenna_channelised_voltage-bandwidth")

    timestamp_step = n_samples_between_spectra * n_spectra_per_acc

    # Get dsim ready with a tone in a known channel that we can check for on the output.
    channel = n_channels // 3  # picked fairly arbitrarily. We just need to know where to set and look for the tone.
    channel_width = bandwidth / n_channels
    channel_centre_freq = channel * channel_width
    await correlator.dsim_client.request("signals", f"common=cw(0.15,{channel_centre_freq})+wgn(0.01);common;common;")

    # Some helper functions:
    async def zero_all_gains():
        for ant in range(n_antennas):
            for pol in ["v", "h"]:
                logger.debug(f"Setting gain to zero on m{800 + ant}{pol}")
                await pc_client.request("gain", "antenna_channelised_voltage", f"m{800 + ant}{pol}", "0")

    async def unzero_a_baseline(baseline_tuple: Tuple[str]):
        logger.debug(f"Unzeroing gain on {baseline_tuple}")
        for ant in baseline_tuple:
            # This was done prior to NGC-535, so the gain used here will need
            # to be tweaked if the test is repeated later. 1 may be fine, but
            # it'll need to be tested.
            await pc_client.request("gain", "antenna_channelised_voltage", ant, "1")

    for bl in bls_ordering:
        logger.info("Checking baseline %r", bl)
        await zero_all_gains()
        await unzero_a_baseline(bl)
        expected_timestamp = (time.time() + 1 - sync_time) * timestamp_scale_factor
        # Note that we are making an assumption that nothing is straying too far
        # from wall time here. I don't have a way other than adjusting the dsim
        # signal of ensuring that we get going after a specific timestamp in the
        # DSP pipeline itself. See NGC-549

        async for chunk in receive_stream.data_ringbuffer:
            recvd_timestamp = chunk.chunk_id * timestamp_step
            if not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
                receive_stream.add_free_chunk(chunk)

            elif recvd_timestamp <= expected_timestamp:
                logger.debug("Skipping chunk with timestamp %d", recvd_timestamp)
                receive_stream.add_free_chunk(chunk)

            else:
                loud_bls = np.nonzero(chunk.data[channel, :, 0])[0]
                logger.info("%d bls had signal in them: %r", len(loud_bls), loud_bls)
                assert bls_ordering.index(bl) in loud_bls  # Check that the expected baseline is actually in the list.
                for loud_bl in loud_bls:
                    assert is_signal_expected_in_baseline(bl, bls_ordering[loud_bl])
                receive_stream.add_free_chunk(chunk)
                break


def is_signal_expected_in_baseline(expected_bl: Tuple[str, str], loud_bl: Tuple[str, str]) -> bool:
    """Check whether signal is expected in the loud baseline, given which one had a test signal injected.

    It isn't possible in the general case to get signal in only a single
    baseline. There will be auto-correlations, and the conjugate correlations
    which will show signal as well.

    Parameters
    ----------
    expected_bl
        A tuple of the form ("m801h", "m802v") indicating which baseline we are
        checking.
    loud_bl
        A baseline where signal has been detected.

    Returns
    -------
    bool
        Indication of whether signal is expected, i.e. whether the test can pass.
    """
    if loud_bl == expected_bl:
        logger.info("Signal confirmed in bl %r where expected", expected_bl)
        return True
    elif loud_bl == (expected_bl[0], expected_bl[0]):
        logger.debug("Signal in %r - fine - it's ant0's autocorrelation.", loud_bl)
        return True
    elif loud_bl == (expected_bl[1], expected_bl[1]):
        logger.debug("Signal in %r - fine - it's ant1's autocorrelation.", loud_bl)
        return True
    elif loud_bl == (expected_bl[1], expected_bl[0]):
        logger.debug("Signal in %r - fine - it's the conjugate of what we expect.", loud_bl)
        return True
    else:
        logger.error("Signal injected into bl %r wasn't expected to show up in %r!", expected_bl, loud_bl)
        return False
