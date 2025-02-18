################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

"""Tests for the F-engines.

These use the X-engine output, but only as a way to obtain information about
what is coming out of the F-engines rather than to test the X-engines.
"""

import asyncio

import numpy as np
from numpy.typing import ArrayLike

from katgpucbf import N_POLS

from ..cbf import CBFRemoteControl
from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import Reporter


async def sample_tone_response_hdr(
    cbf: CBFRemoteControl,
    receiver: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
    amplitude: float,
    rel_freqs: ArrayLike,
    iterations: int = 3,
    gain_step: float = 100.0,
) -> np.ndarray:
    """Compute spectra with high dynamic range.

    Compute high dynamic range spectra per requested tone. One spectrum is
    computed per tone. The HDR data is computed by iterating and applying various
    gain values adjusted per iteration by the gain step.

    Parameters
    ----------
    cbf
        Container for CBF control using katcp.
    receiver
        Correlation receiver stream.
    pdf_report
        Pytest report logger.
    iterations
        Number of iterations to loop through when adjusting gain and computing HDR data.
    gain_step
        Step size for gain adjustment per iteration. This is a scaling factor.
    amplitude
        Required amplitude for generated CW tone. This is also used in computing initial gain.
    rel_freqs
        List of channels (can be fractional units of channels) for which CWs will be generated.

    Returns
    -------
    hdr_data
        HDR data per spectrum.
    """
    # Determine the ideal F-engine output level at the peak. Maximum target_voltage is 127, but some headroom is good.
    gain = receiver.compute_tone_gain(amplitude=amplitude, target_voltage=110)

    rel_freqs = np.asarray(rel_freqs)

    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.
    for i in range(iterations):
        pdf_report.detail(f"Set gain to {gain}.")
        await cbf.product_controller_client.request("gain-all", "antenna-channelised-voltage", gain)

        data = await sample_tone_response(rel_freqs, amplitude, receiver)

        # Store gain adjusted data for SFDR measurement.
        # Use current gain to capture spikes, then adjust gain.
        if i == 0:
            peak_data = np.max(data)
            hdr_data = data
        else:
            # Compute HDR data
            power_scale = gain_step ** (i * 2)
            hdr_data = np.where(hdr_data >= peak_data / power_scale, hdr_data, data / power_scale)
        gain *= gain_step
    return hdr_data


async def sample_tone_response(
    rel_freqs: ArrayLike,
    amplitude: ArrayLike,
    receiver: BaselineCorrelationProductsReceiver,
) -> np.ndarray:
    """Measure power response to tones.

    The input arrays are broadcast with each other. The result has the same
    dimensions plus an axis for frequency in the response.

    Parameters
    ----------
    rel_freqs
        Frequencies to measure, in units of channels (integers give the centre
        of frequency bins).
    amplitude
        Amplitude of the tones, on a scale of 0 to 1.
    receiver
        Receiver for obtaining the output data.
    """
    # Identify baselines using the two pols from each dsim.
    # The fixtures set up a one-to-one relationship between dsims and antennas.
    assert len(receiver.input_labels) == N_POLS * len(receiver.cbf.dsim_clients)
    corrs = []
    for i in range(0, len(receiver.input_labels), 2):
        corrs.append(receiver.bls_ordering.index((receiver.input_labels[i], receiver.input_labels[i + 1])))

    freqs = receiver.channel_frequency(np.asarray(rel_freqs))
    amplitude = np.asarray(amplitude)
    out_shape = np.broadcast_shapes(freqs.shape, amplitude.shape) + (receiver.n_chans,)
    out = np.empty(out_shape, np.float64)
    # Each element is an (out_index, signal_spec) pair. When it fills up to the
    # number of antennas available, `flush` is called.
    tasks: list[tuple[tuple[int, ...], str]] = []

    async def flush() -> None:
        """Execute all the work in `tasks`."""
        async with asyncio.TaskGroup() as tg:
            for i in range(len(tasks)):
                signal = tasks[i][1] * N_POLS
                tg.create_task(receiver.cbf.dsim_clients[i].request("signals", signal))
        _, data = await receiver.next_complete_chunk()
        for task, bl_idx in zip(tasks, corrs):
            # In the absence of noise this should be purely real, but
            # due to quantization noise it is complex. Take the absolute
            # value.
            np.hypot(data[:, bl_idx, 0], data[:, bl_idx, 1], out=out[task[0]])

    with np.nditer([freqs, amplitude], flags=["multi_index"]) as it:
        for f, a in it:
            tasks.append((it.multi_index, f"cw({a}, {f});"))
            if len(tasks) == len(receiver.cbf.dsim_clients):
                await flush()
                tasks = []
    if tasks:
        await flush()
    return out
