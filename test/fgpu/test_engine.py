################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

"""Unit tests for Engine functions."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from ipaddress import IPv4Network

import aiokatcp
import numpy as np
import pytest
import scipy.signal
import spead2.send
from katsdpsigproc.accel import roundup
from numpy.typing import ArrayLike

from katgpucbf import COMPLEX, DIG_SAMPLE_BITS, N_POLS
from katgpucbf.fgpu import METRIC_NAMESPACE
from katgpucbf.fgpu.delay import wrap_angle
from katgpucbf.fgpu.engine import Engine, InQueueItem, Pipeline, generate_ddc_weights, generate_pfb_weights
from katgpucbf.fgpu.main import parse_narrowband, parse_wideband
from katgpucbf.fgpu.output import NarrowbandOutput, Output
from katgpucbf.utils import TimeConverter

from .. import PromDiff, packbits
from .test_recv import gen_heaps

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.cuda_only]
# Command-line arguments
SYNC_TIME = 1632561921
CHANNELS = 1024
JONES_PER_BATCH = 262144
# Lower than the default to make tests quicker
# TODO: use a number that's not a multiple of the number of channels,
# once _send_data can handle partial chunks.
CHUNK_SAMPLES = 524288
CHUNK_JONES = 131072
MAX_DELAY_DIFF = 16384  # Needs to be lowered because CHUNK_SAMPLES is lowered
PACKET_SAMPLES = 4096
TAPS = 16
FENG_ID = 42
ADC_SAMPLE_RATE = 1712e6
DSTS = 16

WIDEBAND_ARGS = f"name=test_wideband,dst=239.10.11.0+{DSTS - 1}:7149,taps={TAPS}"
# Centre frequency is not a multiple of the channel width, but it does ensure
# that the two copies of the same data in test_missing are separated by a
# whole number of cycles.
NARROWBAND_ARGS = (
    f"name=test_narrowband,dst=239.10.12.0+{DSTS - 1}:7149,taps={TAPS},decimation=8,centre_frequency=408173015.5944824"
)


@pytest.fixture
def channels() -> int:
    return CHANNELS


@pytest.fixture
def jones_per_batch(channels: int, request: pytest.FixtureRequest) -> int:
    if marker := request.node.get_closest_marker("spectra_per_heap"):
        return marker.args[0] * channels
    else:
        return JONES_PER_BATCH


@dataclass
class CW:
    r"""Specification of a cosine wave.

    The value at sample :math:`t` is :math:`(A\cos(\pi f(t - d) + p)`, where
    :math:`A`, :math:`f`, :math:`d` and :math:`p` are `magnitude`,
    `frac_channel`, `delay` and `phase` respectively. Note that having both
    `delay` and `phase` is redundant (they achieve equivalent effects), but
    convenient for different tests.

    Parameters
    ----------
    frac_channel
        Frequency, as a fraction of the overall digitised bandwidth
        (e.g., 0.5 means the wideband centre frequency)
    magnitude
        Voltage magnitude
    phase
        Phase to add to the signal, in radians
    delay
        An amount by which to delay the signal, in samples
    """

    frac_channel: float
    magnitude: float = 1.0
    phase: float = 0.0
    delay: float = 0.0

    def __call__(self, t: ArrayLike) -> np.ndarray:
        """Evaluate the cosine wave at given points in time (in units of samples)."""
        t = np.asarray(t)
        return self.magnitude * np.cos(np.pi * self.frac_channel * (t - self.delay) + self.phase)


def frac_channel(output: Output, channel: float) -> float:
    """Convert a channel number to a `frac_channel` parameter for :class:`CW`."""
    if isinstance(output, NarrowbandOutput):
        # Convert centre frequency to a frac_channel
        offset = output.centre_frequency / (ADC_SAMPLE_RATE / 2)
        return (channel - output.channels // 2) / (output.channels * output.decimation) + offset
    else:
        return channel / output.channels


def assert_angles_allclose(a, b, **kwargs) -> None:
    """Assert that two angles (or arrays of angles) are equal to within a tolerance."""
    a = np.asarray(a)
    b = np.asarray(b)
    np.testing.assert_allclose(wrap_angle(a - b), 0.0, **kwargs)


class TestEngine:
    r"""Grouping of unit tests for :class:`.Engine`\'s various functionality."""

    @pytest.fixture
    def wideband_args(self, channels: int, jones_per_batch: int) -> str:
        """Arguments to pass to the command-line parser for the wideband output."""
        return f"{WIDEBAND_ARGS},channels={channels},jones_per_batch={jones_per_batch}"

    @pytest.fixture
    def narrowband_args(self, channels: int, jones_per_batch: int) -> str:
        """Arguments to pass to the command-line parser for the narrowband output."""
        return f"{NARROWBAND_ARGS},channels={channels},jones_per_batch={jones_per_batch}"

    @pytest.fixture(params=["wideband", "narrowband"])
    def output(self, wideband_args: str, narrowband_args: str, request: pytest.FixtureRequest) -> Output:
        """The output to run tests against."""
        if request.param == "wideband":
            return parse_wideband(wideband_args)
        else:
            return parse_narrowband(narrowband_args)

    @pytest.fixture
    def mock_send_stream_network(self, output: Output) -> IPv4Network:
        """Filter the output queues to just those corresponding to `output`.

        This overrides the default implementation in conftest.py.
        """
        return IPv4Network((output.dst[0].host, 24))

    @pytest.fixture
    def coherent_scale(self, output: Output) -> np.ndarray:
        """Gain for tones.

        Expected frequency-domain magnitude for a tone with time-domain
        magnitude 1 when the eq gain is 1. The array is 1D, indexed by
        channel.
        """
        pfb = generate_pfb_weights(output.spectra_samples // output.subsampling, output.taps, output.w_cutoff)
        gain = np.repeat(np.sum(pfb), output.channels)
        if isinstance(output, NarrowbandOutput):
            ddc = generate_ddc_weights(output.ddc_taps, output.subsampling, output.weight_pass)
            response = np.fft.fftshift(scipy.signal.freqz(ddc, worN=output.spectra_samples, whole=True)[1])
            # Discard higher frequencies
            response = response[
                (output.spectra_samples - output.channels) // 2 : (output.spectra_samples + output.channels) // 2
            ]
            gain *= np.abs(response)
        # Factor of 2 is because the power is split between the positive and
        # negative frequencies, and only the positive frequency is returned.
        return gain / 2

    @pytest.fixture
    def default_gain(self, coherent_scale: np.ndarray) -> np.float32:
        """Default value passed to ``?gain`` command."""
        # Centre chain gets defined power. In narrowband, other channels will
        # have less power.
        return np.float32(1 / coherent_scale[len(coherent_scale) // 2])

    @pytest.fixture
    def extra_delay_samples(self, output: Output) -> float:
        """Extra samples by which to delay tones for narrowband tests.

        The narrowband pipeline uses a low-pass filter, which affects phase
        because the output timestamp corresponds to the first sample rather
        than the centre of mass of the filter. To compensate for this:

        - tones must be delayed by this much; and
        - tones phases must be increased by :meth:`extra_phase` (because the
          mixer is *not* shifted)
        """
        if isinstance(output, NarrowbandOutput):
            return (output.ddc_taps - 1) / 2
        else:
            return 0.0

    @pytest.fixture
    def extra_phase(self, output: Output, extra_delay_samples: float) -> float:
        """Extra phase due to narrowband low-pass filter.

        See :meth:`extra_delay_samples` for details. Note that this is valid
        only for baseband tones. If tones are generated in sky frequencies and
        shifted, a further correction is needed.
        """
        if isinstance(output, NarrowbandOutput):
            return wrap_angle(2 * np.pi * (output.centre_frequency / ADC_SAMPLE_RATE) * extra_delay_samples)
        else:
            return 0.0

    @pytest.fixture
    def engine_arglist(self, wideband_args: str, narrowband_args: str, default_gain: np.float32) -> list[str]:
        return [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            "--src-interface=lo",
            "--dst-interface=lo",
            f"--sync-time={SYNC_TIME}",
            f"--src-chunk-samples={CHUNK_SAMPLES}",
            f"--dst-chunk-jones={CHUNK_JONES}",
            f"--max-delay-diff={MAX_DELAY_DIFF}",
            f"--src-packet-samples={PACKET_SAMPLES}",
            f"--feng-id={FENG_ID}",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--gain={default_gain}",
            "--send-rate-factor=0",  # Infinitely fast
            f"--wideband={wideband_args}",
            f"--narrowband={narrowband_args}",
            "239.10.10.0+15:7149",  # src
        ]

    def test_engine_required_arguments(self, engine_server: Engine) -> None:
        """Test proper setting of required arguments.

        .. note::

          This doesn't test if the functionality described by these is in any
          way correct, just whether or not the member variables are being
          correctly populated.
        """
        assert engine_server._port == 0
        assert engine_server._src_interface == ["127.0.0.1"]
        # TODO: `dst_interface` goes to the _sender member, which doesn't have anything we can query.
        assert engine_server._pipelines[0].output.channels == CHANNELS
        assert engine_server.time_converter.sync_time == SYNC_TIME
        assert engine_server._srcs == [(f"239.10.10.{i}", 7149) for i in range(16)]
        # TODO: same problem for `dst` itself.

    def _make_digitiser(self, queue: spead2.InprocQueue) -> "spead2.send.asyncio.AsyncStream":
        """Create send stream for a fake digitiser."""
        config = spead2.send.StreamConfig(max_packet_size=9000)  # Just needs to be bigger than the heaps
        return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), [queue], config)

    def _make_tone(self, timestamps: np.ndarray, tone: CW, pol: int) -> np.ndarray:
        """Synthesize digitiser data containing a tone.

        Only one polarisation (`pol`) contains the tone; the other is all zeros.

        The result includes random dithering, but with a fixed seed, so it will
        be the same for all calls with the same number of samples.

        Parameters
        ----------
        timestamps
            Timestamps for the samples to generate
        tone
            The cosine wave to synthesize
        pol
            The polarisation containing the tone
        """
        rng = np.random.default_rng(1)
        data = tone(timestamps)
        # Dither the signal to reduce quantisation artifacts, then quantise
        data += rng.random(size=data.shape)
        data = np.floor(data).astype(np.int16)
        # Fill in zeros for the other pol
        out = np.zeros((N_POLS, data.size), data.dtype)
        out[pol] = data
        return out

    async def _send_data(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine: Engine,
        output: Output,
        dig_data: np.ndarray,
        *,
        first_timestamp: int = 0,
        expected_first_timestamp: int | None = None,
        src_present: np.ndarray | None = None,
        dst_present: int | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Send a contiguous stream of data to the engine and retrieve results.

        `dig_data` must contain integer values rather than packed 10-bit samples.

        Parameters
        ----------
        mock_recv_stream, mock_send_stream, engine, output
            Fixtures
        dig_data
            2xN array of samples (not yet packed), which must currently be a
            whole number of chunks
        first_timestamp
            Timestamp to send with the first sample
        expected_first_timestamp
            Timestamp expected for the first output heap; if none is provided
            the first timestamp in the data is not checked.
        src_present
            If present, a bitmask per pol and input heap indicating which heaps
            will be sent.
        dst_present
            A bitmask per output batch indicating which batches should be
            present. As a shortcut, specifying an integer indicates the number
            of expected output batches, which must all be present; and specifying
            None indicates that this integer should be calculated from the
            input data length, assuming default state for the engine (in
            particular, it will not be correct if there are non-zero delays).

            Missing batches still take space in the output but are zeroed out.

        Returns
        -------
        data
            Array of shape channels × times × pols
        timestamps
            Labels for the time axis of `data`
        """
        # Reshape into heap-size pieces (now has indices pol, heap, offset)
        src_layout = engine.src_layout
        channels = output.channels
        spectra_per_heap = output.spectra_per_heap
        n_samples = dig_data.shape[1]
        assert dig_data.shape[0] == N_POLS
        assert n_samples % src_layout.chunk_samples == 0, "samples must be a whole number of chunks"
        saturation_value = 2 ** (src_layout.sample_bits - 1) - 1
        saturated = np.abs(dig_data) >= saturation_value
        saturated = np.sum(saturated.reshape(N_POLS, -1, src_layout.heap_samples), axis=-1, dtype=np.uint16)
        dig_data = packbits(dig_data, src_layout.sample_bits)
        dig_stream = self._make_digitiser(mock_recv_stream)
        heap_gen = gen_heaps(
            src_layout,
            dig_data,
            first_timestamp,
            present=src_present,
            saturated=saturated,
        )

        for dig_heap in heap_gen:
            await dig_stream.async_send_heap(dig_heap)
        mock_recv_stream.stop()

        n_out_streams = len(mock_send_stream)
        assert n_out_streams == 16, "Number of output streams does not match command line"
        out_config = spead2.recv.StreamConfig()
        out_tp = spead2.ThreadPool()

        timestamp_step = spectra_per_heap * output.spectra_samples
        if dst_present is None:
            expected_spectra = (n_samples - output.window) // output.spectra_samples
            dst_present_mask = np.ones(expected_spectra // spectra_per_heap, dtype=bool)
        elif isinstance(dst_present, int):
            dst_present_mask = np.ones(dst_present, dtype=bool)
        else:
            dst_present_mask = dst_present
        assert np.sum(dst_present_mask) > 0

        data = np.zeros((channels, len(dst_present_mask) * spectra_per_heap, N_POLS, COMPLEX), np.int8)
        channels_per_substream = channels // n_out_streams
        for i, queue in enumerate(mock_send_stream):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(queue)
            ig = spead2.ItemGroup()
            expected_timestamp = expected_first_timestamp

            # First heap should be the descriptor heap
            descriptor_heap = await stream.get()
            items = ig.update(descriptor_heap)
            assert items == {}, "This heap contains data, not just descriptors"

            # Now, for the actual processing
            for j, present in enumerate(dst_present_mask):
                if present:
                    heap = await stream.get()
                    while (updated_items := set(ig.update(heap))) == set():
                        logger.warning("Test has gone on long enough that we've gotten another descriptor.")
                        heap = await stream.get()
                    assert updated_items == {"timestamp", "feng_id", "frequency", "feng_raw"}
                    assert ig["feng_id"].value == FENG_ID
                    if expected_timestamp is not None:
                        assert ig["timestamp"].value == expected_timestamp
                    else:
                        expected_timestamp = expected_first_timestamp = ig["timestamp"].value
                    assert ig["frequency"].value == i * channels_per_substream
                    assert ig["feng_raw"].shape == (channels_per_substream, spectra_per_heap, N_POLS, COMPLEX)
                    data[
                        i * channels_per_substream : (i + 1) * channels_per_substream,
                        j * spectra_per_heap : (j + 1) * spectra_per_heap,
                    ] = ig["feng_raw"].value
                if expected_timestamp is not None:
                    expected_timestamp += timestamp_step
            # Check that we didn't get more heaps we weren't expecting
            with pytest.raises(spead2.Stopped):
                await stream.get()

        # Convert to complex for analysis
        data_cplx = data[..., 0] + 1j * data[..., 1]
        assert expected_first_timestamp is not None
        timestamps = np.arange(data_cplx.shape[1], dtype=np.int64) * output.spectra_samples + expected_first_timestamp
        return data_cplx, timestamps

    # One delay value is tested with vkgdr, another with smaller output chunks
    @pytest.mark.parametrize(
        "delay_samples",
        [
            (0.0, 0.0),
            (8192.0, 234.5),
            (-42.0, 58.0),
            (42.4, 24.2),
            pytest.param((42.7, 24.9), marks=[pytest.mark.cmdline_args("--dst-chunk-jones=65536")]),
            pytest.param((42.8, 24.5), marks=[pytest.mark.use_vkgdr]),
        ],
    )
    async def test_channel_centre_tones(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        coherent_scale: np.ndarray,
        default_gain: np.float32,
        extra_delay_samples: float,
        extra_phase: float,
        delay_samples: tuple[float, float],
    ) -> None:
        """Put in tones at channel centre frequencies, with delays and gains, and check the result."""
        # extra_phase is for baseband, but we're simulating the second Nyquist zone
        extra_phase += np.pi * extra_delay_samples
        # Delay the tone by a negative amount, then compensate with a positive
        # delay (delay_samples and delay_s are correction terms).
        # The tones are placed in the second Nyquist zone (the "1 +" in
        # frac_channel) then down-converted to baseband, simulating what
        # happens in MeerKAT L-band.
        tone_channels = [64, 271]
        tone_phases = [0.0, 1.23]
        tones = [
            CW(
                frac_channel=1 + frac_channel(output, tone_channels[0]),
                magnitude=110.0,
                phase=tone_phases[0] + extra_phase,
                delay=-delay_samples[0] + extra_delay_samples,
            ),
            CW(
                frac_channel=1 + frac_channel(output, tone_channels[1]),
                magnitude=80.0,
                phase=tone_phases[1] + extra_phase,
                delay=-delay_samples[1] + extra_delay_samples,
            ),
        ]
        delay_s = np.array(delay_samples) / ADC_SAMPLE_RATE
        sky_centre_frequency = 0.75 * ADC_SAMPLE_RATE
        if isinstance(output, NarrowbandOutput):
            sky_centre_frequency += output.centre_frequency - 0.25 * ADC_SAMPLE_RATE
        # Compute phase correction to compensate for the down-conversion.
        # (delay_s is negated here because in the original it is the signal
        # delay rather than the correction).
        # Based on katpoint.delay.DelayCorrection.corrections
        phase = -2.0 * np.pi * sky_centre_frequency * -delay_s
        phase_correction = -phase
        coeffs = [f"{d},0.0:{p},0.0" for d, p in zip(delay_s, phase_correction)]
        await engine_client.request("delays", output.name, SYNC_TIME, *coeffs)

        # Use constant-magnitude gains to avoid throwing off the magnitudes
        rng = np.random.default_rng(123)
        gain_phase = rng.uniform(0, 2 * np.pi, (CHANNELS, N_POLS))
        gains = default_gain * np.exp(1j * gain_phase).astype(np.complex64)
        for pol in range(N_POLS):
            await engine_client.request("gain", output.name, pol, *(str(gain) for gain in gains[:, pol]))

        # Don't send the first chunk, to avoid complications with the step
        # change in the delay at SYNC_TIME.
        src_layout = engine_server.src_layout
        heap_samples = output.spectra_samples * output.spectra_per_heap
        first_timestamp = roundup(src_layout.chunk_samples, heap_samples)
        n_samples = 20 * src_layout.chunk_samples
        tone_timestamps = np.arange(n_samples) + first_timestamp
        dig_data = self._make_tone(tone_timestamps, tones[0], 0) + self._make_tone(tone_timestamps, tones[1], 1)
        dig_data[:, 1::2] *= -1  # Down-convert to baseband

        expected_first_timestamp = first_timestamp
        # The data should have as many samples as the input, minus a reduction
        # from windowing, rounded down to a full heap.
        expected_spectra = (n_samples + round(min(delay_samples)) - output.window) // output.spectra_samples
        if max(delay_samples) > 0:
            # The first output heap would require data from before the first
            # timestamp, so it does not get produced
            expected_first_timestamp += heap_samples
            expected_spectra -= output.spectra_per_heap
        out_data, _ = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
            first_timestamp=first_timestamp,
            expected_first_timestamp=expected_first_timestamp,
            dst_present=expected_spectra // output.spectra_per_heap,
        )

        # Check for the tones
        for pol in range(2):
            tone_data = out_data[tone_channels[pol], :, pol]
            expected_mag = tones[pol].magnitude * coherent_scale[tone_channels[pol]] * default_gain
            assert 50 <= expected_mag < 127, "Magnitude is outside of good range for testing"
            np.testing.assert_equal(np.abs(tone_data), pytest.approx(expected_mag, abs=3))
            # The frequency (relative to the centre frequency) corresponds to
            # an integer number of cycles per spectrum, so the phase will be
            # consistent across spectra.
            # The accuracy is limited by the quantisation.
            expected_phase = tone_phases[pol] + gain_phase[tone_channels[pol], pol]
            assert_angles_allclose(np.angle(tone_data), expected_phase, atol=0.01)
            # Suppress the tone and check that everything is now zero (the
            # spectral leakage should be below the quantisation threshold).
            tone_data.fill(0)
            np.testing.assert_equal(out_data[..., pol], 0)

    async def test_spectral_leakage(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        coherent_scale: np.ndarray,
        default_gain: np.float32,
    ) -> None:
        """Test leakage from tones that are not in the frequency centre."""
        # Rather than parametrize the test (which would be slow), send in
        # lots of different tones at different times. Each tone is maintained
        # for at least a full window, and we just discard the outputs
        # corresponding to times that mix the tones. The tones are all placed
        # in the centre channel, but linearly spaced over the frequencies in
        # that channel's frequency bin.
        n_tones = 128
        # We want to start each tone on a spectrum boundary
        step = roundup(output.window, output.spectra_samples)
        tones = [
            CW(frac_channel=frac_channel(output, CHANNELS // 2 - 0.5 + (i + 0.5) / n_tones), magnitude=500)
            for i in range(n_tones)
        ]
        dig_data = np.concatenate([self._make_tone(np.arange(step), tone, 0) for tone in tones], axis=1)
        # Add some extra data to align to an input heap, and to fill out the
        # last output chunk.
        output_chunk_samples = engine_server.chunk_jones * 2 * output.decimation
        padded_size = roundup(dig_data.shape[1] + output_chunk_samples, engine_server.src_layout.chunk_samples)
        n_pad = padded_size - dig_data.shape[1]
        padding = np.zeros((2, n_pad), dig_data.dtype)
        dig_data = np.concatenate([dig_data, padding], axis=1)

        # Crank up the gain so that leakage is measurable
        gain = 100 * default_gain
        for pol in range(N_POLS):
            await engine_client.request("gain", output.name, pol, gain)
        # CBF-REQ-0126: The CBF shall perform channelisation such that the 53 dB
        # attenuation is ≤ 2x (twice) the pass band width.
        #
        # The division by 20 (not 10) is because we're dealing with voltage,
        # not power.
        tol = 10 ** (-53 / 20) * (tones[0].magnitude * coherent_scale[CHANNELS // 2]) * gain

        out_data, _ = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        for i in range(n_tones):
            # Get the data for the PFB window that holds the tone
            data = out_data[:, i * (step // output.spectra_samples), 0]
            # Check that the tone was in the right place (it should saturate)
            assert np.abs(data[CHANNELS // 2]) >= 127
            # Blank out the channel that is expected to have the tone, and
            # the nearer adjacent one (with is within the 2x tolerance).
            data[CHANNELS // 2] = 0
            if i < n_tones // 2:
                data[CHANNELS // 2 - 1] = 0
            else:
                data[CHANNELS // 2 + 1] = 0
            np.testing.assert_equal(data, pytest.approx(0, abs=tol))

    # Just test 2 values for dig_sample_bits and dst_sample_bits; it gets
    # expensive otherwise. Also just do it for one test, as a sanity check.
    @pytest.mark.parametrize(
        "dig_sample_bits,dst_sample_bits",
        [
            pytest.param(DIG_SAMPLE_BITS, 4, marks=pytest.mark.cmdline_args("--dst-sample-bits=4")),
            pytest.param(12, 8, marks=pytest.mark.cmdline_args("--dig-sample-bits=12", "--dst-sample-bits=8")),
        ],
    )
    async def test_delay_phase_rate(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        extra_delay_samples: float,
        extra_phase: float,
        dig_sample_bits: int,
        dst_sample_bits: int,
    ) -> None:
        """Test that delay rate and phase rate setting works."""
        # One tone at centre frequency to test the absolute phase, and one at another
        # frequency to test the slope across the band.
        tone_channels = [CHANNELS // 2, CHANNELS - 123]
        tones = [
            CW(
                frac_channel=frac_channel(output, channel),
                magnitude=round(0.4 * 2**dst_sample_bits),
                delay=extra_delay_samples,
                phase=extra_phase,
            )
            for channel in tone_channels
        ]
        src_layout = engine_server.src_layout
        n_samples = 32 * src_layout.chunk_samples

        # Should be high enough to cause multiple coarse delay changes per chunk
        delay_rate = np.array([1e-5, 1.2e-5])
        # Should wrap multiple times over the test
        phase_rate_per_sample = np.array([30, 32.5]) / n_samples
        phase_rate = phase_rate_per_sample * ADC_SAMPLE_RATE
        coeffs = [f"0.0,{dr}:0.0,{pr}" for dr, pr in zip(delay_rate, phase_rate)]
        await engine_client.request("delays", output.name, SYNC_TIME, *coeffs)

        first_timestamp = roundup(100 * src_layout.chunk_samples, output.spectra_samples * output.spectra_per_heap)
        end_delay = round(min(delay_rate) * n_samples)
        expected_spectra = (n_samples + end_delay - output.window) // output.spectra_samples
        tone_timestamps = np.arange(n_samples) + first_timestamp
        dig_data = np.sum([self._make_tone(tone_timestamps, tone, 0) for tone in tones], axis=0)
        dig_data[1] = dig_data[0]  # Copy data from pol 0 to pol 1
        out_data, timestamps = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
            first_timestamp=first_timestamp,
            # The first output heap would require data from before first_timestamp, so
            # is omitted.
            expected_first_timestamp=first_timestamp + output.spectra_samples * output.spectra_per_heap,
            dst_present=expected_spectra // output.spectra_per_heap - 1,
        )
        # Add a polarisation dimension to timestamps to simplify some
        # broadcasting computations below.
        atol = 2 * 0.5**dst_sample_bits
        timestamps = timestamps[:, np.newaxis]
        expected_phase = phase_rate_per_sample * timestamps
        assert_angles_allclose(np.angle(out_data[tone_channels[0]]), expected_phase, atol=atol)

        # Adjust expected phase from the centre frequency to the other channel
        bw = ADC_SAMPLE_RATE / 2 / output.decimation
        channel_bw = bw / CHANNELS
        expected_phase -= (
            2 * np.pi * (tone_channels[1] - tone_channels[0]) * channel_bw * delay_rate * (timestamps / ADC_SAMPLE_RATE)
        )
        assert_angles_allclose(np.angle(out_data[tone_channels[1]]), expected_phase, atol=atol)

    def _watch_sensors(self, sensors: Sequence[aiokatcp.Sensor]) -> dict[str, list]:
        """Set up observers on sensors that record updates.

        The updates are returned in a dictionary whose key is the sensor name and
        whose value is a list of sensor values.
        """
        sensor_updates_dict: dict[str, list[aiokatcp.Reading]] = {sensor.name: [] for sensor in sensors}

        def sensor_observer(
            sensor: aiokatcp.Sensor, sensor_reading: aiokatcp.Reading, *, updates_list: list[aiokatcp.Reading]
        ) -> None:
            """Populate a list to compare at the end of this unit-test."""
            updates_list.append(sensor_reading)

        for sensor in sensors:
            sensor.attach(partial(sensor_observer, updates_list=sensor_updates_dict[sensor.name]))
        return sensor_updates_dict

    async def test_delay_changes(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        extra_delay_samples: float,
        extra_phase: float,
        output: Output,
    ) -> None:
        """Test loading several future delay models."""
        # Set up infrastructure for testing delay sensor updates
        delay_sensors = [engine_server.sensors[f"{output.name}.input{pol}.delay"] for pol in range(N_POLS)]
        sensor_updates_dict = self._watch_sensors(delay_sensors)

        # To keep things simple, we'll just use phase, not delay.
        tone_channel = CHANNELS // 2
        tone = CW(
            frac_channel=frac_channel(output, tone_channel), magnitude=110, delay=extra_delay_samples, phase=extra_phase
        )
        src_layout = engine_server.src_layout
        n_samples = 10 * src_layout.chunk_samples
        dig_data = self._make_tone(np.arange(n_samples), tone, 0)

        # Load some delay models for the future (the last one beyond the end of the data)
        update_times = [0, 123456, 400000, 1234567, 1234567890]  # in samples
        update_phases = [1.0, 0.2, -0.2, -2.0, 0.0]
        for time, phase in zip(update_times, update_phases):
            coeffs = f"0.0,0.0:{phase},0.0"
            await engine_client.request("delays", output.name, SYNC_TIME + time / ADC_SAMPLE_RATE, coeffs, coeffs)

        out_data, timestamps = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        out_data = out_data[tone_channel, :, 0]  # Only pol 0, centre channel matters

        for i in range(len(update_times) - 1):
            # Check which timestamps this delay model applies to
            valid = (update_times[i] <= timestamps) & (timestamps < update_times[i + 1])
            assert np.any(valid)
            phases = np.angle(out_data[valid])
            assert_angles_allclose(phases, update_phases[i], atol=0.01)

        for delay_sensor in delay_sensors:
            for expected_time, expected_phase, sensor_update in zip(
                update_times, update_phases, sensor_updates_dict[delay_sensor.name]
            ):
                # (timestamp, delay, delay_rate, phase, phase_rate)
                sensor_values = sensor_update.value[1:-1].split(",")
                sensor_values = [float(field.strip()) for field in sensor_values]
                # NOTE: This tolerance is in place as the ADC timestamp gets
                # converted to a UNIX time and back again, losing some precision
                # during the conversion process.
                np.testing.assert_allclose(int(sensor_values[0]), expected_time, atol=200)
                # NOTE: Using the default relative tolerance of 1e-07
                np.testing.assert_allclose(sensor_values[3], expected_phase)

    @pytest.mark.parametrize("delay_samples", [0.0, 8192.0, 234.5, 42.8])
    async def test_delay_slope(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        channels: int,
        delay_samples: float,
    ) -> None:
        """Test the slope and intercept of the phase response to delay.

        This serves mostly as a sanity check on the other tests of delay
        tracking, which use complicated formulae to determine the expected
        absolute phase of the response. Here we delay one polarisation
        relative to the other, and check the difference in phase.
        """
        delay_s = delay_samples / ADC_SAMPLE_RATE
        coeffs = ["0.0,0.0:0.0,0.0", f"{delay_s},0.0:0.0,0.0"]
        await engine_client.request("delays", output.name, SYNC_TIME, *coeffs)

        src_layout = engine_server.src_layout
        # Don't send the first chunk, to avoid complications with the step
        # change in the delay at SYNC_TIME.
        heap_samples = output.spectra_samples * output.spectra_per_heap
        first_timestamp = roundup(src_layout.chunk_samples, heap_samples)
        n_samples = 20 * src_layout.chunk_samples
        tone_timestamps = np.arange(n_samples) + first_timestamp

        rng = np.random.default_rng(123)
        n_tones = 10
        tone_channels = rng.integers(0, channels, size=n_tones)
        tone_channels[0] = channels // 2  # Ensure we test the intercept exactly
        tone_phases = rng.uniform(0, 2 * np.pi, size=n_tones)
        tones = [
            CW(frac_channel=frac_channel(output, channel), magnitude=60, phase=phase)
            for channel, phase in zip(tone_channels, tone_phases)
        ]
        dig_data = np.sum([self._make_tone(tone_timestamps, tone, 0) for tone in tones], axis=0)
        dig_data[1] = dig_data[0]  # Copy data from pol 0 to pol 1
        # Ensure we haven't saturated
        assert np.max(np.abs(dig_data)) < 2 ** (DIG_SAMPLE_BITS - 1) - 1

        expected_first_timestamp = first_timestamp
        # The data should have as many samples as the input, minus a reduction
        # from windowing, rounded down to a full heap.
        expected_spectra = (n_samples - output.window) // output.spectra_samples
        if delay_samples > 0:
            # The first output heap would require data from before the first
            # timestamp, so it does not get produced
            expected_first_timestamp += heap_samples
            expected_spectra -= output.spectra_per_heap
        out_data, _ = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
            first_timestamp=first_timestamp,
            expected_first_timestamp=expected_first_timestamp,
            dst_present=expected_spectra // output.spectra_per_heap,
        )

        # Ensure we haven't saturated
        assert np.max(np.abs(out_data.real)) < 127
        assert np.max(np.abs(out_data.imag)) < 127
        orig_phase = np.angle(out_data[tone_channels, :, 0])
        delayed_phase = np.angle(out_data[tone_channels, :, 1])
        channel_bw = ADC_SAMPLE_RATE / 2 / output.decimation / channels
        phase_ramp = -2 * np.pi * delay_s * channel_bw * (tone_channels - channels // 2)
        phase_ramp = phase_ramp[:, np.newaxis]
        # There is quite a lot of quantisation noise, so we need a large tolerance
        assert_angles_allclose(orig_phase + phase_ramp, delayed_phase, atol=3e-2)

    # Test with spectra_samples less than, equal to and greater than src-packet-samples
    @pytest.mark.parametrize("channels", [64, 2048, 8192])
    # Use small jones-per-batch to get finer-grained testing of which spectra
    # were ditched. Fewer would be better, but there are internal alignment
    # requirements. --src-chunk-samples needs to be increased (from
    # CHUNK_SAMPLES) to ensure narrowband windows fit.
    @pytest.mark.spectra_per_heap(32)
    @pytest.mark.cmdline_args("--src-chunk-samples=4194304")
    async def test_missing_heaps(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        channels: int,
    ) -> None:
        """Test that the right output heaps are omitted when input heaps are missing.

        The test sends the same set of data twice, with gaps only in the first half.
        It then checks that the heaps successfully received in the first half match
        the heaps in the second half.
        """
        spectra_per_heap = output.spectra_per_heap
        chunk_samples = engine_server.src_layout.chunk_samples
        n_samples = 16 * chunk_samples
        # Half-open ranges of input heaps that are missing
        missing_ranges = [
            (8, 10),
            (15, 16),
            (117, 133),
            (6 * chunk_samples // PACKET_SAMPLES, 7 * chunk_samples // PACKET_SAMPLES),
        ]
        rng = np.random.default_rng(seed=1)
        dig_data = np.tile(rng.integers(-255, 255, size=(2, n_samples // 2), dtype=np.int16), 2)
        src_present = np.ones((2, n_samples // PACKET_SAMPLES), bool)
        for a, b in missing_ranges:
            assert b < src_present.shape[1]
            src_present[:, a:b] = False
        # The data should have as many samples as the input, minus a reduction
        # from windowing, rounded down to a full heap.
        total_spectra = (n_samples - output.window) // output.spectra_samples
        total_heaps = total_spectra // spectra_per_heap
        dst_present = np.ones(total_heaps, bool)
        # Compute which output heaps should be missing. first_* and last_* are
        # both inclusive (b is exclusive)
        for a, b in missing_ranges:
            first_sample = a * PACKET_SAMPLES
            last_sample = b * PACKET_SAMPLES - 1  # -1 to make it inclusive
            assert last_sample < n_samples // 2  # Make sure gaps are restricted to first half
            first_spectrum = max(0, (first_sample - output.window + 1) // output.spectra_samples)
            last_spectrum = last_sample // output.spectra_samples
            first_heap = first_spectrum // spectra_per_heap
            last_heap = last_spectrum // spectra_per_heap
            dst_present[first_heap : last_heap + 1] = False

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            out_data, timestamps = await self._send_data(
                mock_recv_stream,
                mock_send_stream,
                engine_server,
                output,
                dig_data,
                expected_first_timestamp=0,
                src_present=src_present,
                dst_present=dst_present,
            )
        # Position in dst_present corresponding to the second half of dig_data.
        middle = (n_samples // 2) // (output.spectra_samples * spectra_per_heap)
        for i, p in enumerate(dst_present):
            if p and i + middle < len(dst_present):
                x = out_data[:, i * spectra_per_heap : (i + 1) * spectra_per_heap]
                y = out_data[:, (i + middle) * spectra_per_heap : (i + middle + 1) * spectra_per_heap]
                # For narrowband they're only guaranteed to be equal because
                # the time difference is a multiple of the mixer wavelength.
                np.testing.assert_equal(x, y)

        for pol in range(N_POLS):
            input_missing_heaps = np.sum(~src_present[pol])
            assert prom_diff.get_sample_diff("input_missing_heaps_total", {"pol": str(pol)}) == input_missing_heaps
        n_substreams = len(mock_send_stream)
        output_heaps = np.sum(dst_present) * n_substreams
        assert prom_diff.get_sample_diff("output_heaps_total", {"stream": output.name}) == output_heaps
        batch_samples = channels * spectra_per_heap * N_POLS
        batch_size = batch_samples * COMPLEX * np.dtype(np.int8).itemsize
        assert (
            prom_diff.get_sample_diff("output_bytes_total", {"stream": output.name}) == np.sum(dst_present) * batch_size
        )
        assert (
            prom_diff.get_sample_diff("output_samples_total", {"stream": output.name})
            == np.sum(dst_present) * batch_samples
        )
        assert (
            prom_diff.get_sample_diff("output_skipped_heaps_total", {"stream": output.name})
            == np.sum(~dst_present) * n_substreams
        )

    async def test_dig_clip_cnt_sensors(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
    ) -> None:
        """Test that the ``dig-clip-cnt`` sensors are set correctly."""
        sensors = [engine_server.sensors[f"input{pol}.dig-clip-cnt"] for pol in range(N_POLS)]
        sensor_update_dict = self._watch_sensors(sensors)
        n_samples = 9 * CHUNK_SAMPLES
        dig_data = np.zeros((2, n_samples), np.int16)
        saturation_value = 2 ** (engine_server.src_layout.sample_bits - 1) - 1
        dig_data[0, 10000:15000] = saturation_value
        dig_data[1, 2 * CHUNK_SAMPLES + 50000 : 2 * CHUNK_SAMPLES + 60000] = -saturation_value
        await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        time_converter = TimeConverter(SYNC_TIME, ADC_SAMPLE_RATE)
        expected_timestamps = [time_converter.adc_to_unix(t * CHUNK_SAMPLES) for t in range(1, 10)]
        assert sensor_update_dict[sensors[0].name] == [
            aiokatcp.Reading(t, aiokatcp.Sensor.Status.NOMINAL, 5000) for t in expected_timestamps
        ]
        assert sensor_update_dict[sensors[1].name] == [
            aiokatcp.Reading(t, aiokatcp.Sensor.Status.NOMINAL, v)
            for t, v in zip(expected_timestamps, [0, 0] + [10000] * 7)
        ]

    # It's easier to use a constant voltage. Also need to check the case were
    # the input power is zero.
    @pytest.mark.parametrize(
        "input_voltage,output_power_dbfs", [(0, np.finfo(np.float64).min), (64, pytest.approx(-15.0345186))]
    )
    async def test_dig_rms_dbfs_sensors(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        input_voltage: int,
        output_power_dbfs: float,
    ) -> None:
        """Test that the ``dig-rms-dbfs`` sensors are set correctly."""
        sensors = [engine_server.sensors[f"input{pol}.dig-rms-dbfs"] for pol in range(N_POLS)]
        sensor_update_dict = self._watch_sensors(sensors)
        n_samples = 10 * CHUNK_SAMPLES
        dig_data = np.full((2, n_samples), input_voltage, np.int16)

        await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        time_converter = TimeConverter(SYNC_TIME, ADC_SAMPLE_RATE)
        expected_timestamps = [time_converter.adc_to_unix(t * CHUNK_SAMPLES) for t in range(1, 10)]
        for pol in range(N_POLS):
            assert sensor_update_dict[sensors[pol].name] == [
                aiokatcp.Reading(t, aiokatcp.Sensor.Status.WARN, output_power_dbfs) for t in expected_timestamps
            ]

    @pytest.mark.parametrize("tone_pol", [0, 1])
    async def test_output_clip_count(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        default_gain: np.float32,
        tone_pol: int,
    ) -> None:
        """Test that ``output_clipped_samples`` metric and ``feng-clip-count`` sensor increase when a channel clips."""
        tone = CW(frac_channel=frac_channel(output, 271), magnitude=110.0)
        for pol in range(N_POLS):
            # Set gain high enough to make the tone saturate
            await engine_client.request("gain", output.name, pol, default_gain * 2)

        src_layout = engine_server.src_layout
        n_samples = 20 * src_layout.chunk_samples
        dig_data = self._make_tone(np.arange(n_samples), tone, tone_pol)
        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            _, timestamps = await self._send_data(
                mock_recv_stream,
                mock_send_stream,
                engine_server,
                output,
                dig_data,
            )

        assert prom_diff.get_sample_diff(
            "output_clipped_samples_total", {"stream": output.name, "pol": f"{tone_pol}"}
        ) == len(timestamps)
        assert (
            prom_diff.get_sample_diff("output_clipped_samples_total", {"stream": output.name, "pol": f"{1 - tone_pol}"})
            == 0
        )

        # Compute the expected timestamp. The timestamp is associated with the
        # output chunk, so we need to round up to output chunk size.
        last_timestamp = roundup(timestamps[-1] + 1, engine_server.chunk_jones * output.decimation * 2)

        sensor = engine_server.sensors[f"{output.name}.input{tone_pol}.feng-clip-cnt"]
        assert sensor.reading == aiokatcp.Reading(
            SYNC_TIME + last_timestamp / ADC_SAMPLE_RATE, aiokatcp.Sensor.Status.NOMINAL, len(timestamps)
        )
        sensor = engine_server.sensors[f"{output.name}.input{1 - tone_pol}.feng-clip-cnt"]
        assert sensor.reading == aiokatcp.Reading(
            SYNC_TIME + last_timestamp / ADC_SAMPLE_RATE, aiokatcp.Sensor.Status.NOMINAL, 0
        )

    def _patch_fill_in(self, monkeypatch, engine_client: aiokatcp.Client, output: Output, *request) -> list[int]:
        """Patch :meth:`~.Pipeline._fill_in` to make a request partway through the stream.

        The returned list will be populated with the value of the
        ``steady-state-timestamp`` sensor immediately after executing the
        request.
        """
        counter = 0
        timestamp = []

        async def fill_in(self) -> InQueueItem | None:
            if self._in_item is None and self.output.name == output.name:
                nonlocal counter
                counter += 1
                if counter == 12:
                    await engine_client.request(*request)
                    timestamp.append(await engine_client.sensor_value("steady-state-timestamp", int))
            return await orig_fill_in(self)

        orig_fill_in = Pipeline._fill_in
        monkeypatch.setattr(Pipeline, "_fill_in", fill_in)
        return timestamp

    async def test_steady_state_gain(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        monkeypatch,
    ) -> None:
        """Test that the ``steady-state-timestamp`` is updated correctly after ``?gain``."""
        n_samples = max(16 * CHUNK_SAMPLES, output.spectra_samples * output.spectra_per_heap * 3)
        rng = np.random.default_rng(1)
        dig_data = rng.integers(-255, 255, size=(2, n_samples), dtype=np.int16)

        timestamp_list = self._patch_fill_in(monkeypatch, engine_client, output, "gain-all", output.name, 0)
        out_data, timestamps = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )

        steady_state_timestamp = timestamp_list[0]
        # The steady state timestamp must fall somewhere in the middle of the
        # data for the test to be meaningful.
        assert timestamps[1] <= steady_state_timestamp <= timestamps[-2]
        # After the steady state timestamp, all the data much be zero.
        after = timestamps >= steady_state_timestamp
        assert np.all(out_data[:, after, :] == 0)
        # The effect may take effect earlier than the indicated timestamp.
        # Check that it doesn't affect the first timestamp, which would suggest
        # we've messed up the test setup.
        assert not np.all(out_data[:, 0, :] == 0)

    async def test_steady_state_delay(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        monkeypatch,
    ) -> None:
        """Test that the ``steady-state-timestamp`` is updated correctly after ``?delays``."""
        n_samples = max(16 * CHUNK_SAMPLES, output.spectra_samples * output.spectra_per_heap * 3)
        tone = CW(frac_channel=frac_channel(output, CHANNELS // 2), magnitude=100)
        dig_data = self._make_tone(np.arange(n_samples), tone, 0)

        timestamp_list = self._patch_fill_in(
            monkeypatch, engine_client, output, "delays", output.name, SYNC_TIME, "0,0:3,0", "0,0:3,0"
        )
        out_data, timestamps = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )

        steady_state_timestamp = timestamp_list[0]
        # The steady state timestamp must fall somewhere in the middle of the
        # data for the test to be meaningful.
        assert timestamps[1] <= steady_state_timestamp <= timestamps[-2]
        # After the steady state timestamp, all the data have the phase applied.
        after = timestamps >= steady_state_timestamp
        assert_angles_allclose(np.angle(out_data[CHANNELS // 2, after, 0]), 3.0, atol=0.1)
        # The effect may take effect earlier than the indicated timestamp.
        # Check that it doesn't affect the first timestamp, which would suggest
        # we've messed up the test setup.
        assert np.angle(out_data[CHANNELS // 2, 0, 0]) == pytest.approx(0, abs=0.1)
