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
from .reporter import Reporter

CPLX = 2

logger = logging.getLogger(__name__)


async def test_baseline_correlation_products(
    n_antennas: int,
    n_channels: int,
    correlator: CorrelatorRemoteControl,
    receive_stream: spead2.recv.ChunkRingStream,
    pdf_report: Reporter,
    expect,
):
    """Test that the baseline ordering indicated in the sensor matches the output data.

    Requirements verified:

    CBF-REQ-0087
        The CBF shall, on request via the CAM interface, compute all cross-
        correlation and auto-correlation products for all configured baselines
        in each defined sub-array.

    CBF-REQ-0104
        The CBF, when requested to produce the Baseline Correlation Products
        data product, shall transfer the appropriate data continuously to the
        subscribed user(s) via the interface as specified in the appropriate
        ICD.


    Verification method:

    Verification by means of test. Verify by testing all correlation product
    combinations. Use gain correction after channelisation to turn the input
    signal on or off. Iterate through all combinations and verify that the
    expected doutput appears in the correct baseline product.

    """
    pdf_report.step(
        "Connect to correlator's product controller to retrieve " "configuration for the running correlator."
    )
    pc_client = correlator.product_controller_client

    # Get some necessary sensor values from the correlator.
    bls_ordering = ast.literal_eval(await get_sensor_val(pc_client, "baseline_correlation_products-bls-ordering"))
    sync_time = await get_sensor_val(pc_client, "antenna_channelised_voltage-sync-time")
    timestamp_scale_factor = await get_sensor_val(pc_client, "antenna_channelised_voltage-scale-factor-timestamp")
    n_samples_between_spectra = await get_sensor_val(pc_client, "antenna_channelised_voltage-n-samples-between-spectra")
    n_spectra_per_acc = await get_sensor_val(pc_client, "baseline_correlation_products-n-accs")
    bandwidth = await get_sensor_val(pc_client, "antenna_channelised_voltage-bandwidth")

    timestamp_step = n_samples_between_spectra * n_spectra_per_acc

    pdf_report.step("Select and configure the D-sim with a strong tone.")
    # Get dsim ready with a tone in a known channel that we can check for on the output.
    channel = n_channels // 3  # picked fairly arbitrarily. We just need to know where to set and look for the tone.
    channel_width = bandwidth / n_channels
    channel_centre_freq = channel * channel_width
    pdf_report.detail(f"Tone frequency of {channel_centre_freq} Hz selected, in the centre of channel {channel}.")

    await correlator.dsim_client.request("signals", f"common=cw(0.15,{channel_centre_freq})+wgn(0.01);common;common;")
    pdf_report.detail(
        "Set D-sim with {channel_centre_freq} Hz tone, amplitude=0.15, with wgn amplitude 0.01 on both pols."
    )

    # Some helper functions:
    async def zero_all_gains():
        pdf_report.detail("Setting all F-engine gains to zero.")
        for ant in range(n_antennas):
            for pol in ["v", "h"]:
                # This may be useful debug info but we don't need it in the PDF report.
                logger.debug(f"Setting gain to zero on m{800 + ant}{pol}")
                await pc_client.request("gain", "antenna_channelised_voltage", f"m{800 + ant}{pol}", "0")

    async def unzero_a_baseline(baseline_tuple: Tuple[str]):
        pdf_report.detail(f"Unzeroing gain on {baseline_tuple}")
        for ant in baseline_tuple:
            await pc_client.request("gain", "antenna_channelised_voltage", ant, "1")

    for idx, bl in enumerate(bls_ordering):
        pdf_report.step(f"Check baseline {bl} ({idx+1}/{len(bls_ordering)})")
        await zero_all_gains()
        await unzero_a_baseline(bl)
        expected_timestamp = (time.time() + 1 - sync_time) * timestamp_scale_factor
        # Note that we are making an assumption that nothing is straying too far
        # from wall time here. I don't have a way other than adjusting the dsim
        # signal of ensuring that we get going after a specific timestamp in the
        # DSP pipeline itself. See NGC-549
        data_ringbuffer = receive_stream.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        async for chunk in data_ringbuffer:
            # These asserts aren't particularly important, but they keep mypy happy.
            assert isinstance(chunk.present, np.ndarray)
            assert isinstance(chunk.data, np.ndarray)
            recvd_timestamp = chunk.chunk_id * timestamp_step
            if not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
                receive_stream.add_free_chunk(chunk)

            elif recvd_timestamp <= expected_timestamp:
                logger.debug("Skipping chunk with timestamp %d", recvd_timestamp)
                receive_stream.add_free_chunk(chunk)

            else:
                loud_bls = np.nonzero(chunk.data[channel, :, 0])[0]
                pdf_report.detail(
                    f"{len(loud_bls)} baseline{'s' if len(loud_bls) != 1 else ''} "
                    f"had signal in {'them' if len(loud_bls) != 1 else 'it'}: {loud_bls}"
                )
                expect(
                    bls_ordering.index(bl) in loud_bls,
                    f"{bl} ({bls_ordering.index(bl)}) doesn't show up in the list ({loud_bls})!",
                )
                for loud_bl in loud_bls:
                    expect(
                        is_signal_expected_in_baseline(bl, bls_ordering[loud_bl], pdf_report),
                        "Signal found in unexpected baseline.",
                    )
                receive_stream.add_free_chunk(chunk)
                break


def is_signal_expected_in_baseline(
    expected_bl: Tuple[str, str], loud_bl: Tuple[str, str], pdf_report: Reporter
) -> bool:
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
        pdf_report.detail(f"Signal confirmed in bl {expected_bl} where expected")
        return True
    elif loud_bl == (expected_bl[0], expected_bl[0]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's ant0's autocorrelation.")
        return True
    elif loud_bl == (expected_bl[1], expected_bl[1]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's ant1's autocorrelation.")
        return True
    elif loud_bl == (expected_bl[1], expected_bl[0]):
        pdf_report.detail(f"Signal in {loud_bl} is ok, it's the conjugate of what we expect.")
        return True
    else:
        pdf_report.detail(f"Signal injected into bl {expected_bl} wasn't expected to show up in {loud_bl}!")
        return False
