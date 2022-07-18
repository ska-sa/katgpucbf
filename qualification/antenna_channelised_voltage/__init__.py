################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from katgpucbf import DIG_SAMPLE_BITS, N_POLS

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter


async def sample_tone_response_hdr(
    correlator: CorrelatorRemoteControl,
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
    gain values adjuested per iteration by the gain step.

    Parameters
    ----------
    correlator
        Container for correlator control using katcp.
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
        List of channels (can fractional units of channels) which CW will be generated.

    Returns
    ----------
    hdr_data
        HDR data per spectrum.
    """
    # Determine the ideal F-engine output level at the peak. Maximum target_voltage is 127, but some headroom is good.
    gain = compute_tone_gain(receiver=receiver, amplitude=amplitude, target_voltage=110)

    rel_freqs = np.asarray(rel_freqs)

    # Get a high dynamic range result (hdr_data) by using several gain settings
    # and using the high-gain results to more accurately measure the samples
    # whose power is low enough not to saturate.
    for i in range(iterations):
        pdf_report.detail(f"Set gain to {gain}.")
        await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

        data = await sample_tone_response(rel_freqs, amplitude, receiver)
        data = data.astype(np.float64)

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


def compute_tone_gain(
    receiver: BaselineCorrelationProductsReceiver,
    amplitude: float,
    target_voltage: int,
) -> float:
    """Compute F-Engine gain.

    Compute gain to be applied to the F-Engine to maximise output dynamic range
    when the input is a tone (for example, for use with :func:``sample_tone_response``).
    The F-Engine output is 8-bit signed (max 127).

    Parameters
    ----------
    correlator
        Connection to the correlator.
    amplitude
        Amplitude of the tones, on a scale of 0 to 1.
    target_voltage
        Desired magnitude of F-engine output values. The calculation uses
        an approximation, so the actual value may be slightly higher than
        the target. The target may also be reduced if necessary to avoid
        saturating the X-engine output.
    """
    # We need to avoid saturating the signed 32-bit X-engine accumulation as
    # well (2e9 is comfortably less than 2^31).
    # The PFB is scaled for fixed incoherent gain, but we need to be concerned
    # about coherent gain to avoid overflowing the F-engine output. Coherent gain
    # scales approximately with np.sqrt(correlator.n_chans / 2).
    target_voltage = min(target_voltage, np.sqrt(2e9 / receiver.n_spectra_per_acc))
    dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
    return target_voltage / (amplitude * dig_max * np.sqrt(receiver.n_chans / 2))


async def sample_tone_response(
    rel_freqs: ArrayLike,
    amplitude: ArrayLike,
    receiver: BaselineCorrelationProductsReceiver,
) -> np.ndarray:
    """Measure power response to tones.

    The input arrays are broadcast with each other. The rest has the same
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
    # Identify auto-correlation baselines corresponding to the dsim outputs.
    # The fixtures set up a one-to-one relationship between dsims and antennas.
    # katsdpcontroller ensures that the dsim is started with V pol first and
    # the fixtures put V pol first in the axis ordering, so this is fairly
    # straightforward.
    assert len(receiver.input_labels) == N_POLS * len(receiver.correlator.dsim_clients)
    autos = []
    for inp in receiver.input_labels:
        autos.append(receiver.bls_ordering.index((inp, inp)))

    channel_width = receiver.bandwidth / receiver.n_chans
    freqs = np.asarray(rel_freqs) * channel_width  # Baseband, which is what dsims work on
    amplitude = np.asarray(amplitude)
    out_shape = np.broadcast_shapes(freqs.shape, amplitude.shape) + (receiver.n_chans,)
    out = np.empty(out_shape, np.int32)
    # Each element is an (out_index, signal_spec) pair. When it fills up to the
    # number of inputs available, `flush` is called.
    tasks: List[Tuple[Tuple[int, ...], str]] = []

    async def flush() -> None:
        """Execute all the work in `tasks`."""
        # Each dsim has two inputs. Simplify handling an odd number of tasks
        # (in the last batch) by just duplicating the last task. We'll end up
        # writing to the output twice, but it's not a big performance issue.
        while len(tasks) % N_POLS:
            tasks.append(tasks[-1])
        requests = []
        n_dsims = len(tasks) // N_POLS
        for i in range(n_dsims):
            signal = ""
            for j in range(N_POLS):
                signal += tasks[i * N_POLS + j][1]
            requests.append(asyncio.create_task(receiver.correlator.dsim_clients[i].request("signals", signal)))
        await asyncio.gather(*requests)
        _, data = await receiver.next_complete_chunk()
        for task, bl_idx in zip(tasks, autos):
            out[task[0]] = data[:, bl_idx, 0]  # Only keep real part

    with np.nditer([freqs, amplitude], flags=["multi_index"]) as it:
        for f, a in it:
            tasks.append((it.multi_index, f"cw({a}, {f});"))
            if len(tasks) == receiver.n_inputs:
                await flush()
                tasks = []
    if tasks:
        await flush()
    return out
