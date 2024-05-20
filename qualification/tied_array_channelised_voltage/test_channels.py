################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Test that each input channel appears in the correct place in the output."""

import random

import numpy as np
from pytest_check import check

from .. import CBFRemoteControl, TiedArrayChannelisedVoltageReceiver
from ..reporter import Reporter


async def test_channels(
    cbf: CBFRemoteControl,
    receive_tied_array_channelised_voltage: TiedArrayChannelisedVoltageReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that each input channel appears in the correct place in the output.

    Verification method
    -------------------
    Verification by means of test. The dsim is configured with a number of
    tones in randomly-selected channels. The output is then checked to ensure
    that the tones appear in the same channels.
    """
    receiver = receive_tied_array_channelised_voltage
    client = cbf.product_controller_client

    pdf_report.step("Select channels")
    rng = random.Random(1)
    channels = sorted(rng.sample(range(receiver.n_chans), 30))
    pdf_report.detail(f"Using channels {channels}")

    pdf_report.step("Configure dsim")
    amplitude = 0.05
    freqs = [receiver.channel_frequency(c) for c in channels]
    signal = " + ".join(f"cw({amplitude}, {freq})" for freq in freqs)
    await cbf.dsim_clients[0].request("signals", f"common = {signal}; common; common;")

    pdf_report.step("Set F-engine gain")
    gain = receiver.compute_tone_gain(amplitude, 100)
    await client.request("gain-all", "antenna-channelised-voltage", gain)
    pdf_report.detail(f"Set gain to {gain} for all inputs")

    pdf_report.step("Set XB-engine quantisation gain")
    gain = 1.0 / receiver.n_ants  # Average rather than summing antennas
    for beam in receiver.stream_names:
        await client.request("beam-quant-gains", beam, gain)
    pdf_report.detail(f"Set quantisation gain to {gain} for all beams")

    pdf_report.step("Check results")
    _, data = await receiver.next_complete_chunk()
    for i, beam in enumerate(receiver.stream_names):
        beam_data = data[i]
        # Sum over polarisation and real/imag, but not channel
        power = np.sum(np.square(beam_data.astype(np.float64)), axis=(1, 2))
        # Detect tones. Assume anything over 50% of max is a tone.
        max_power = np.max(power)
        tones = list(np.where(power > 0.5 * max_power)[0])
        pdf_report.detail(f"Tones in {beam}: {tones}")
        with check:
            assert tones == channels
