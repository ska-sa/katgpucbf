################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
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
from katgpucbf.fgpu.output import NarrowbandOutput, NarrowbandOutputNoDiscard, Output
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
CHUNK_SAMPLES = 1048576
CHUNK_JONES = 262144
MAX_DELAY_DIFF = 16384  # Needs to be lowered because CHUNK_SAMPLES is lowered
PACKET_SAMPLES = 4096
TAPS = 16
FENG_ID = 42
ADC_SAMPLE_RATE = 1712e6
DSTS = 16
TIME_CONVERTER = TimeConverter(SYNC_TIME, ADC_SAMPLE_RATE)

WIDEBAND_ARGS = f"name=test_wideband,dst=239.10.11.0+{DSTS - 1}:7149,taps={TAPS}"
# Centre frequency is not a multiple of the channel width, but it does ensure
# that the two copies of the same data in test_missing are separated by a
# whole number of cycles.
NARROWBAND_ARGS = f"name=test_narrowband,dst=239.10.12.0+{DSTS - 1}:7149,taps={TAPS},centre_frequency=408173015.5944824"


@pytest.fixture
def channels() -> int:
    """Number of channels."""
    return CHANNELS


@pytest.fixture
def jones_per_batch(channels: int, request: pytest.FixtureRequest) -> int:
    """Number of Jones vectors per batch.

    This defaults to :const:`JONES_PER_BATCH`, but can be overridden by
    ``pytest.mark.spectra_per_heap(spectra_per_heap)``.
    """
    if marker := request.node.get_closest_marker("spectra_per_heap"):
        return marker.args[0] * channels
    else:
        return JONES_PER_BATCH


@pytest.fixture
def dig_rms_dbfs_window_chunks() -> int:
    """Number of chunks per window for ``dig-rms-dbfs`` sensors."""
    return 2


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
    r"""Test :class:`.Engine`."""

    @pytest.fixture(params=["wideband", "narrowband_discard", "narrowband_no_discard"])
    def output_type(self, request: pytest.FixtureRequest) -> str:
        """Which output to test.

        In all cases there are wideband and narrowband streams, but only one
        of them is tested for each parametrisation.
        """
        return request.param

    @pytest.fixture
    def wideband_args(self, channels: int, jones_per_batch: int) -> str:
        """Arguments to pass to the command-line parser for the wideband output."""
        return f"{WIDEBAND_ARGS},channels={channels},jones_per_batch={jones_per_batch}"

    @pytest.fixture(params=[4, 8, 16])
    def decimation(self, request: pytest.FixtureRequest) -> int:
        """Narrowband decimation factor."""
        return request.param

    @pytest.fixture
    def narrowband_args(self, channels: int, jones_per_batch: int, decimation: int, output_type: str) -> str:
        """Arguments to pass to the command-line parser for the narrowband output."""
        args = f"{NARROWBAND_ARGS},channels={channels},jones_per_batch={jones_per_batch},decimation={decimation}"
        if output_type == "narrowband_no_discard":
            bandwidth = 0.5 * ADC_SAMPLE_RATE / decimation
            args += f",pass_bandwidth={0.6 * bandwidth}"
        return args

    @pytest.fixture
    def output(
        self,
        output_type: str,
        wideband_args: str,
        narrowband_args: str,
        decimation: int,
        request: pytest.FixtureRequest,
    ) -> Output:
        """The output to run tests against."""
        if output_type == "wideband":
            return parse_wideband(wideband_args)
        else:
            return parse_narrowband(narrowband_args)

    @pytest.fixture
    def dig_rms_dbfs_window_samples(self) -> list[int]:
        """The window size for dig-rms-dbfs sensors, in input samples.

        The return value is a list, which will be populated with a single
        element during engine startup by the
        :meth:`mock_dig_rms_dbfs_window_samples` fixture. This roundabout
        mechanism is needed because the actual `chunk_jones` value for the
        engine is computed when it starts up, but we need the mock in place
        before that.
        """
        return []

    @pytest.fixture(autouse=True)
    def mock_dig_rms_dbfs_window_samples(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dig_rms_dbfs_window_samples: list[int],
        dig_rms_dbfs_window_chunks: int,
        output: Output,
    ) -> None:
        """Mock :meth:`.Pipeline._dig_rms_dbfs_window_samples`.

        This overrides the calculation to use
        :func:`dig_rms_dbfs_window_chunks`, and also populates
        :meth:`dig_rms_dbfs_window_samples` with the computed value.

        This is marked autouse to ensure it will be run before the
        engine_server fixture.
        """

        def _dig_rms_dbfs_window_samples(self: Pipeline) -> int:
            chunk_samples = self.spectra * self.output.spectra_samples
            window_samples = dig_rms_dbfs_window_chunks * chunk_samples
            dig_rms_dbfs_window_samples.append(window_samples)
            return window_samples

        monkeypatch.setattr("katgpucbf.fgpu.engine.Pipeline._dig_rms_dbfs_window_samples", _dig_rms_dbfs_window_samples)

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
        pfb = generate_pfb_weights(
            output.spectra_samples // output.subsampling, output.taps, output.w_cutoff, output.window_function
        )
        gain = np.repeat(np.sum(pfb), output.channels)
        if isinstance(output, NarrowbandOutput):
            ddc = generate_ddc_weights(output, ADC_SAMPLE_RATE)
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
        # Centre channel gets defined power. In narrowband, other channels will
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
        """Command-line arguments to pass to the engine."""
        return [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            "--recv-interface=lo",
            "--send-interface=lo",
            f"--sync-time={SYNC_TIME}",
            f"--recv-chunk-samples={CHUNK_SAMPLES}",
            f"--send-chunk-jones={CHUNK_JONES}",
            f"--max-delay-diff={MAX_DELAY_DIFF}",
            f"--recv-packet-samples={PACKET_SAMPLES}",
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
        assert engine_server._recv_interface == ["127.0.0.1"]
        # TODO: `send_interface` goes to the _sender member, which doesn't have anything we can query.
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
        recv_present: np.ndarray | None = None,
        send_present: int | np.ndarray | None = None,
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
            Timestamp expected for the first output batch; if none is provided
            the first timestamp in the data is not checked.
        recv_present
            If present, a bitmask per pol and input heap indicating which heaps
            will be sent.
        send_present
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
        recv_layout = engine.recv_layout
        channels = output.channels
        spectra_per_heap = output.spectra_per_heap
        n_samples = dig_data.shape[1]
        assert dig_data.shape[0] == N_POLS
        assert n_samples % recv_layout.chunk_samples == 0, "samples must be a whole number of chunks"
        saturation_value = 2 ** (recv_layout.sample_bits - 1) - 1
        saturated = np.abs(dig_data) >= saturation_value
        saturated = np.sum(saturated.reshape(N_POLS, -1, recv_layout.heap_samples), axis=-1, dtype=np.uint16)
        dig_data = packbits(dig_data, recv_layout.sample_bits)
        dig_stream = self._make_digitiser(mock_recv_stream)
        heap_gen = gen_heaps(
            recv_layout,
            dig_data,
            first_timestamp,
            present=recv_present,
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
        if send_present is None:
            expected_spectra = (n_samples - output.window) // output.spectra_samples
            send_present_mask = np.ones(expected_spectra // spectra_per_heap, dtype=bool)
        elif isinstance(send_present, int):
            send_present_mask = np.ones(send_present, dtype=bool)
        else:
            send_present_mask = send_present
        assert np.sum(send_present_mask) > 0

        data = np.zeros((channels, len(send_present_mask) * spectra_per_heap, N_POLS, COMPLEX), np.int8)
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
            for j, present in enumerate(send_present_mask):
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
            pytest.param((42.7, 24.9), marks=[pytest.mark.cmdline_args("--send-chunk-jones=65536")]),
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
        tone_channels = [192, 271]
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
        recv_layout = engine_server.recv_layout
        heap_samples = output.spectra_samples * output.spectra_per_heap
        first_timestamp = roundup(recv_layout.chunk_samples, heap_samples)
        n_samples = 20 * recv_layout.chunk_samples
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
            send_present=expected_spectra // output.spectra_per_heap,
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
            assert_angles_allclose(np.angle(tone_data), expected_phase, atol=0.02)
            # Suppress the tone and check that everything is now almost zero.
            # The spectral leakage should be less than 1, but dithering can cause
            # some values to be 1 rather than 0, and if that happens in both real
            # and imaginary then the error will be sqrt(2).
            tone_data.fill(0)
            np.testing.assert_allclose(out_data[..., pol], 0, atol=1.5)

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
        padded_size = roundup(dig_data.shape[1] + output_chunk_samples, engine_server.recv_layout.chunk_samples)
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

    # Just test 2 values for dig_sample_bits and send_sample_bits; it gets
    # expensive otherwise. Also just do it for one test, as a sanity check.
    @pytest.mark.parametrize(
        "dig_sample_bits,send_sample_bits",
        [
            pytest.param(
                DIG_SAMPLE_BITS, 4, marks=[pytest.mark.cmdline_args("--send-sample-bits=4"), pytest.mark.slow]
            ),
            pytest.param(12, 8, marks=pytest.mark.cmdline_args("--dig-sample-bits=12", "--send-sample-bits=8")),
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
        send_sample_bits: int,
    ) -> None:
        """Test that delay rate and phase rate setting works."""
        # One tone at centre frequency to test the absolute phase, and one at another
        # frequency to test the slope across the band.
        tone_channels = [CHANNELS // 2, CHANNELS // 2 + 123]
        tones = [
            CW(
                frac_channel=frac_channel(output, channel),
                magnitude=round(0.4 * 2**send_sample_bits),
                delay=extra_delay_samples,
                phase=extra_phase,
            )
            for channel in tone_channels
        ]
        recv_layout = engine_server.recv_layout
        n_samples = 32 * recv_layout.chunk_samples

        # Should be high enough to cause multiple coarse delay changes per chunk
        delay_rate = np.array([1e-5, 1.2e-5])
        # Should wrap multiple times over the test
        phase_rate_per_sample = np.array([30, 32.5]) / n_samples
        phase_rate = phase_rate_per_sample * ADC_SAMPLE_RATE
        coeffs = [f"0.0,{dr}:0.0,{pr}" for dr, pr in zip(delay_rate, phase_rate)]
        await engine_client.request("delays", output.name, SYNC_TIME, *coeffs)

        first_timestamp = roundup(100 * recv_layout.chunk_samples, output.spectra_samples * output.spectra_per_heap)
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
            send_present=expected_spectra // output.spectra_per_heap - 1,
        )
        # Add a polarisation dimension to timestamps to simplify some
        # broadcasting computations below.
        atol = 4 * 0.5**send_sample_bits
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
        recv_layout = engine_server.recv_layout
        n_samples = 10 * recv_layout.chunk_samples
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
            assert_angles_allclose(phases, update_phases[i], atol=0.02)

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
    @pytest.mark.slow
    async def test_delay_slope(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        channels: int,
        default_gain: float,
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
        # Increase the gain to use the full dynamic range. We can't use a higher
        # tone_magnitude because it will lead to time-domain saturation.
        tone_magnitude = 60
        await engine_client.request("gain-all", output.name, 100 / tone_magnitude * default_gain)

        recv_layout = engine_server.recv_layout
        # Don't send the first chunk, to avoid complications with the step
        # change in the delay at SYNC_TIME.
        heap_samples = output.spectra_samples * output.spectra_per_heap
        first_timestamp = roundup(recv_layout.chunk_samples, heap_samples)
        n_samples = 20 * recv_layout.chunk_samples
        tone_timestamps = np.arange(n_samples) + first_timestamp

        rng = np.random.default_rng(123)
        n_tones = 10
        if isinstance(output, NarrowbandOutputNoDiscard):
            bandwidth = ADC_SAMPLE_RATE * 0.5
            pass_channels_half = int(output.pass_bandwidth / bandwidth * channels / 2)
            min_channel = channels // 2 - pass_channels_half
            max_channel = channels // 2 + pass_channels_half
            # + 1 so that max_channel is included. It would be more idiomatic
            # to pass endpoint=True, but numpy <2.2 has a bug in the type
            # annotations for that case.
            tone_channels = rng.integers(min_channel, max_channel + 1, size=n_tones)
        else:
            tone_channels = rng.integers(0, channels, size=n_tones)
        tone_channels[0] = channels // 2  # Ensure we test the intercept exactly
        tone_phases = rng.uniform(0, 2 * np.pi, size=n_tones)
        tones = [
            CW(frac_channel=frac_channel(output, channel), magnitude=tone_magnitude, phase=phase)
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
            send_present=expected_spectra // output.spectra_per_heap,
        )

        # Ensure we haven't saturated, but are also exploiting the full dynamic range
        assert 90 < np.max(np.abs(out_data.real)) < 127
        assert 90 < np.max(np.abs(out_data.imag)) < 127
        orig_phase = np.angle(out_data[tone_channels, :, 0])
        delayed_phase = np.angle(out_data[tone_channels, :, 1])
        channel_bw = ADC_SAMPLE_RATE / 2 / output.decimation / channels
        phase_ramp = -2 * np.pi * delay_s * channel_bw * (tone_channels - channels // 2)
        phase_ramp = phase_ramp[:, np.newaxis]
        # There is quite a lot of quantisation noise, so we need a large tolerance
        assert_angles_allclose(orig_phase + phase_ramp, delayed_phase, atol=4e-2)

    # Test with spectra_samples less than, equal to and greater than recv-packet-samples
    @pytest.mark.parametrize("channels", [64, 2048, 8192])
    # Use small jones-per-batch to get finer-grained testing of which spectra
    # were ditched. Fewer would be better, but there are internal alignment
    # requirements. --recv-chunk-samples needs to be increased (from
    # CHUNK_SAMPLES) to ensure narrowband windows fit.
    @pytest.mark.spectra_per_heap(32)
    @pytest.mark.cmdline_args("--recv-chunk-samples=8388608")
    @pytest.mark.slow
    async def test_missing_heaps(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
        channels: int,
        dig_rms_dbfs_window_chunks: int,
    ) -> None:
        """Test that the right output heaps are omitted when input heaps are missing.

        The test sends the same set of data twice, with gaps only in the first half.
        It then checks that the heaps successfully received in the first half match
        the heaps in the second half, up to a tolerance to account for dithering.
        """
        sensors = [engine_server.sensors[f"input{pol}.dig-rms-dbfs"] for pol in range(N_POLS)]
        sensor_update_dict = self._watch_sensors(sensors)
        spectra_per_heap = output.spectra_per_heap
        chunk_samples = engine_server.recv_layout.chunk_samples
        n_samples = 16 * chunk_samples
        # Half-open ranges of input heaps that are missing
        missing_ranges = [
            (8, 10),
            (15, 16),
            (117, 133),
            (6 * chunk_samples // PACKET_SAMPLES, 8 * chunk_samples // PACKET_SAMPLES),
        ]
        rng = np.random.default_rng(seed=1)
        dig_data = np.tile(rng.integers(-255, 255, size=(2, n_samples // 2), dtype=np.int16), 2)
        recv_present = np.ones((2, n_samples // PACKET_SAMPLES), bool)
        for a, b in missing_ranges:
            assert b < recv_present.shape[1]
            recv_present[:, a:b] = False
        # The data should have as many samples as the input, minus a reduction
        # from windowing, rounded down to a full batch.
        total_spectra = (n_samples - output.window) // output.spectra_samples
        total_batches = total_spectra // spectra_per_heap
        send_present = np.ones(total_batches, bool)
        # Compute which output batches should be missing. first_* and last_* are
        # both inclusive (b is exclusive)
        for a, b in missing_ranges:
            first_sample = a * PACKET_SAMPLES
            last_sample = b * PACKET_SAMPLES - 1  # -1 to make it inclusive
            assert last_sample < n_samples // 2  # Make sure gaps are restricted to first half
            first_spectrum = max(0, (first_sample - output.window + 1) // output.spectra_samples)
            last_spectrum = last_sample // output.spectra_samples
            first_batch = first_spectrum // spectra_per_heap
            last_batch = last_spectrum // spectra_per_heap
            send_present[first_batch : last_batch + 1] = False

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            out_data, timestamps = await self._send_data(
                mock_recv_stream,
                mock_send_stream,
                engine_server,
                output,
                dig_data,
                expected_first_timestamp=0,
                recv_present=recv_present,
                send_present=send_present,
            )
        # Position in send_present corresponding to the second half of dig_data.
        middle = (n_samples // 2) // (output.spectra_samples * spectra_per_heap)
        for i, p in enumerate(send_present):
            if p and i + middle < len(send_present):
                x = out_data[:, i * spectra_per_heap : (i + 1) * spectra_per_heap]
                y = out_data[:, (i + middle) * spectra_per_heap : (i + middle + 1) * spectra_per_heap]
                # For narrowband they're only guaranteed to be equal because
                # the time difference is a multiple of the mixer wavelength.
                # The tolerance allows for a difference of 1 on real and imag.
                np.testing.assert_allclose(x, y, atol=1.5)

        for pol in range(N_POLS):
            # Check prometheus counter
            input_missing_heaps = np.sum(~recv_present[pol])
            assert prom_diff.diff("input_missing_heaps_total", {"pol": str(pol)}) == input_missing_heaps

        n_substreams = len(mock_send_stream)
        output_heaps = np.sum(send_present) * n_substreams
        prom_diff = prom_diff.with_labels({"stream": output.name})
        assert prom_diff.diff("output_heaps_total") == output_heaps
        batch_samples = channels * spectra_per_heap * N_POLS
        batch_size = batch_samples * COMPLEX * np.dtype(np.int8).itemsize
        assert prom_diff.diff("output_bytes_total") == np.sum(send_present) * batch_size
        assert prom_diff.diff("output_samples_total") == np.sum(send_present) * batch_samples
        assert prom_diff.diff("output_skipped_heaps_total") == np.sum(~send_present) * n_substreams

        # Sensor is not present in the narrowband mode.
        if output.decimation == 1:

            class Update(Enum):
                NORMAL = 1
                FAILURE = 2
                OPTIONAL_FAILURE = 3

            expected_updates = []
            spectra_per_output_chunk = engine_server.chunk_jones // output.channels
            batches_per_output_chunk = spectra_per_output_chunk // spectra_per_heap
            batches_per_window = batches_per_output_chunk * dig_rms_dbfs_window_chunks
            window_timestamp_step = spectra_per_output_chunk * output.spectra_samples * dig_rms_dbfs_window_chunks

            total_chunks = (total_batches + batches_per_output_chunk - 1) // batches_per_output_chunk
            # The last windows is only emitted if we observe its final chunk,
            # so we round down here.
            total_windows = total_chunks // dig_rms_dbfs_window_chunks
            for i in range(total_windows):
                start_batch = i * batches_per_window
                stop_batch = (i + 1) * batches_per_window
                n_present = np.sum(send_present[start_batch:stop_batch])
                # The sensor timestamp is the end of the window
                sensor_timestamp = TIME_CONVERTER.adc_to_unix(
                    timestamps[start_batch * spectra_per_heap] + window_timestamp_step
                )
                if n_present == batches_per_window:
                    expected_updates.append((sensor_timestamp, Update.NORMAL))
                elif n_present > 0:
                    expected_updates.append((sensor_timestamp, Update.FAILURE))
                else:
                    # If a window is completely missing, it could indicate that
                    # there was a break in the input (and no OutQueueItems were
                    # generated), in which case there will be no sensor update.
                    # On the other hand, there might have been some OutQueueItems
                    # present but none of them had valid heaps.
                    expected_updates.append((sensor_timestamp, Update.OPTIONAL_FAILURE))

            for pol in range(N_POLS):
                p = 0
                actual = sensor_update_dict[f"input{pol}.dig-rms-dbfs"]
                for timestamp, update in expected_updates:
                    match update:
                        case Update.NORMAL:
                            assert actual[p].timestamp == timestamp
                            assert actual[p].status in {aiokatcp.Sensor.Status.NOMINAL, aiokatcp.Sensor.Status.WARN}
                            p += 1
                        case Update.FAILURE:
                            assert actual[p].timestamp == timestamp
                            assert actual[p].status == aiokatcp.Sensor.Status.FAILURE
                            p += 1
                        case Update.OPTIONAL_FAILURE:
                            if p < len(actual) and actual[p].timestamp == timestamp:
                                assert actual[p].status == aiokatcp.Sensor.Status.FAILURE
                                p += 1
                assert p == len(actual)

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
        saturation_value = 2 ** (engine_server.recv_layout.sample_bits - 1) - 1
        dig_data[0, 10000:15000] = saturation_value
        dig_data[1, 2 * CHUNK_SAMPLES + 50000 : 2 * CHUNK_SAMPLES + 60000] = -saturation_value
        await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        expected_timestamps = [TIME_CONVERTER.adc_to_unix(t * CHUNK_SAMPLES) for t in range(1, 10)]
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
        "input_voltage,output_power_dbfs", [(0, np.finfo(np.float64).min), (100, pytest.approx(-11.158118046054))]
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
        dig_rms_dbfs_window_samples: list[int],
    ) -> None:
        """Test that the ``dig-rms-dbfs`` sensors are set correctly."""
        sensors = [engine_server.sensors[f"input{pol}.dig-rms-dbfs"] for pol in range(N_POLS)]
        sensor_update_dict = self._watch_sensors(sensors)
        n_samples = 10 * CHUNK_SAMPLES
        dig_data = np.full((2, n_samples), input_voltage, np.int16)
        # Unpack the single-element list that was populated by
        # mock_dig_rms_dbfs_window_samples.
        window_size = dig_rms_dbfs_window_samples[0]

        await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        expected_timestamps = [TIME_CONVERTER.adc_to_unix(t) for t in range(window_size, n_samples, window_size)]
        assert len(expected_timestamps) > 0
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

        recv_layout = engine_server.recv_layout
        n_samples = 20 * recv_layout.chunk_samples
        dig_data = self._make_tone(np.arange(n_samples), tone, tone_pol)
        with PromDiff(namespace=METRIC_NAMESPACE, labels={"stream": output.name}) as prom_diff:
            _, timestamps = await self._send_data(
                mock_recv_stream,
                mock_send_stream,
                engine_server,
                output,
                dig_data,
            )

        assert prom_diff.diff("output_clipped_samples_total", {"pol": f"{tone_pol}"}) == len(timestamps)
        assert prom_diff.diff("output_clipped_samples_total", {"pol": f"{1 - tone_pol}"}) == 0

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
                if counter == 11:
                    await engine_client.request(*request)
                    timestamp.append(await engine_client.sensor_value("steady-state-timestamp", int))
                counter += 1
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
        n_samples = max(20 * CHUNK_SAMPLES, output.spectra_samples * output.spectra_per_heap * 3)
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
        n_samples = max(20 * CHUNK_SAMPLES, output.spectra_samples * output.spectra_per_heap * 3)
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

    async def test_incoherent_gain(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: list[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        output: Output,
    ) -> None:
        """Test that a ``?gain`` of 1 gives incoherent gain of 1."""
        await engine_client.request("gain-all", output.name, 1.0)
        n_samples = max(CHUNK_SAMPLES, 2 * output.spectra_samples * output.spectra_per_heap)
        rng = np.random.default_rng(seed=1)
        # Should be low enough to limit saturation but high enough to minimise
        # impact of quantisation noise.
        dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
        sigma = 40.0
        dig_data = np.rint(rng.normal(scale=sigma, size=(2, n_samples)).clip(-dig_max, dig_max)).astype(int)
        out_data, timestamps = await self._send_data(
            mock_recv_stream,
            mock_send_stream,
            engine_server,
            output,
            dig_data,
        )
        if isinstance(output, NarrowbandOutputNoDiscard):
            # Exclude the roll-off, since it will have less power
            pass_fraction = output.pass_bandwidth / (0.5 * ADC_SAMPLE_RATE)
            lo = math.ceil(output.channels * (1 - pass_fraction) * 0.5)
            hi = math.floor(output.channels * (1 + pass_fraction) * 0.5) + 1
            out_data = out_data[lo:hi]
        # Compute sqrt of average power
        out_sigma = np.sqrt(np.mean(np.square(np.abs(out_data))))
        # The tolerance is quite loose because there is lots of noise
        # (both from the random input data and from quantisation), and we're
        # mostly worried about major errors like a missing factor of sqrt(2).
        assert out_sigma == pytest.approx(sigma, rel=0.01)
