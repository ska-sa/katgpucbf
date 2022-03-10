################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import aiokatcp
import numpy as np
import pytest
import spead2.send
from numpy.typing import ArrayLike

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.fgpu import SAMPLE_BITS, send
from katgpucbf.fgpu.delay import wrap_angle
from katgpucbf.fgpu.engine import Engine
from katgpucbf.spead import IMMEDIATE_FORMAT

from .test_recv import gen_heaps

pytestmark = [pytest.mark.cuda_only, pytest.mark.asyncio]
# Command-line arguments
SYNC_EPOCH = 1632561921
CHANNELS = 1024
SPECTRA_PER_HEAP = 256
# Lower than the default to make tests quicker, and smaller than
# CHUNK_JONES to check that this case works.
# TODO: use a number that's not a multiple of the number of channels,
# once _send_data can handle partial chunks.
CHUNK_SAMPLES = 524288
CHUNK_JONES = 1048576
MAX_DELAY_DIFF = 16384  # Needs to be lowered because CHUNK_SAMPLES is lowered
TAPS = 16
FENG_ID = 42
ADC_SAMPLE_RATE = 1712e6
GAIN = np.float32(0.001)


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
        Frequency, as a channel number divided by the number of
        channels (e.g., 0.5 means the centre frequency)
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


class TestEngine:
    r"""Grouping of unit tests for :class:`.Engine`\'s various functionality."""

    engine_arglist = [
        "--katcp-host=127.0.0.1",
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        f"--channels={CHANNELS}",
        f"--sync-epoch={SYNC_EPOCH}",
        f"--src-chunk-samples={CHUNK_SAMPLES}",
        f"--dst-chunk-jones={CHUNK_JONES}",
        f"--max-delay-diff={MAX_DELAY_DIFF}",
        f"--spectra-per-heap={SPECTRA_PER_HEAP}",
        f"--feng-id={FENG_ID}",
        f"--taps={TAPS}",
        f"--adc-sample-rate={ADC_SAMPLE_RATE}",
        f"--gain={GAIN}",
        "--send-rate-factor=0",  # Infinitely fast
        "239.10.10.0+7:7149",  # src1
        "239.10.10.8+7:7149",  # src2
        "239.10.11.0+15:7149",  # dst
    ]

    def test_engine_required_arguments(self, engine_server: Engine) -> None:
        """Test proper setting of required arguments.

        .. note::

          This doesn't test if the functionality described by these is in any
          way correct, just whether or not the member variables are being
          correctly populated.
        """
        assert engine_server._port == 0
        assert engine_server._src_interface == "127.0.0.1"
        # TODO: `dst_interface` goes to the _sender member, which doesn't have anything we can query.
        assert engine_server._processor.channels == CHANNELS
        assert engine_server.sync_epoch == SYNC_EPOCH
        assert engine_server._srcs == [
            [
                ("239.10.10.0", 7149),
                ("239.10.10.1", 7149),
                ("239.10.10.2", 7149),
                ("239.10.10.3", 7149),
                ("239.10.10.4", 7149),
                ("239.10.10.5", 7149),
                ("239.10.10.6", 7149),
                ("239.10.10.7", 7149),
            ],
            [
                ("239.10.10.8", 7149),
                ("239.10.10.9", 7149),
                ("239.10.10.10", 7149),
                ("239.10.10.11", 7149),
                ("239.10.10.12", 7149),
                ("239.10.10.13", 7149),
                ("239.10.10.14", 7149),
                ("239.10.10.15", 7149),
            ],
        ]
        # TODO: same problem for `dst` itself.

    def _make_digitiser(self, queues: List[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
        """Create send stream for a fake digitiser.

        The resulting stream has one sub-stream per polarisation.
        """
        config = spead2.send.StreamConfig(max_packet_size=9000)  # Just needs to be bigger than the heaps
        return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)

    def _pack_samples(self, samples: ArrayLike) -> np.ndarray:
        """Pack 16-bit digitiser sample data down to SAMPLE_BITS bits.

        Parameters
        ----------
        samples
            A 2xN array of sample data
        """
        # Force to int16, and big endian so the bits come out in the right order
        samples_int16 = np.asarray(samples, dtype=">i2")
        # Unpack the bits into a new axis, so that we can toss out the top 6
        bits = np.unpackbits(samples_int16.view(np.uint8)).reshape(samples_int16.shape + (16,))
        # Put all the bits back into bytes. packbits automatically flattens
        # the array, so we have to restore the desired shape.
        return np.packbits(bits[..., -SAMPLE_BITS:]).reshape(samples_int16.shape[0], -1)

    def _make_tone(self, n_samples: int, tone: CW, pol: int) -> np.ndarray:
        """Synthesize digitiser data containing a tone.

        Only one polarisation (`pol`) contains the tone; the other is all zeros.

        The result includes random dithering, but with a fixed seed, so it will
        be the same for all calls with the same number of samples.

        Parameters
        ----------
        n_samples
            Number of samples to generate per polarisation
        tone
            The cosine wave to synthesize
        pol
            The polarisation containing the tone
        """
        rng = np.random.default_rng(1)
        t = np.arange(n_samples)
        data = tone(t)
        # Dither the signal to reduce quantisation artifacts, then quantise
        data += rng.random(size=data.shape)
        data = np.floor(data).astype(np.int16)
        # Fill in zeros for the other pol
        out = np.zeros((N_POLS, data.size), data.dtype)
        out[pol] = data
        return out

    async def _send_data(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine: Engine,
        dig_data: np.ndarray,
        first_timestamp: int = 0,
        expected_first_timestamp: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Send a contiguous stream of data to the engine and retrieve results.

        `dig_data` must contain integer values rather than packed 10-bit samples.

        Parameters
        ----------
        mock_recv_streams, mock_send_stream, engine
            Fixtures
        dig_data
            2xN array of samples (not yet packed), which must currently be a
            whole number of chunks
        first_timestamp
            Timestamp to send with the first sample
        expected_first_timestamp
            Timestamp expected for the first output heap; if none is provided
            the first timestamp in the data is not checked.

        Returns
        -------
        data
            Array of shape channels × times × pols
        timestamps
            Labels for the time axis of `data`
        """
        # Reshape into heap-size pieces (now has indices pol, heap, offset)
        src_layout = engine._src_layout
        assert dig_data.shape[0] == N_POLS
        assert dig_data.shape[1] % src_layout.chunk_samples == 0, "samples must be a whole number of chunks"
        dig_data = self._pack_samples(dig_data)
        dig_stream = self._make_digitiser(mock_recv_streams)
        heap_gens = [gen_heaps(src_layout, pol_data, first_timestamp, pol) for pol, pol_data in enumerate(dig_data)]

        for cur_heaps in zip(*heap_gens):
            for pol in range(N_POLS):
                await dig_stream.async_send_heap(cur_heaps[pol], substream_index=pol)
        for queue in mock_recv_streams:
            queue.stop()

        n_out_streams = len(mock_send_stream)
        assert n_out_streams == 16, "Number of output streams does not match command line"
        out_config = spead2.recv.StreamConfig()
        out_tp = spead2.ThreadPool()
        heaps = []
        for i, queue in enumerate(mock_send_stream):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(queue)
            ig = spead2.ItemGroup()
            # We don't have descriptors yet, so we have to build the Items manually
            raw_shape = (CHANNELS // n_out_streams, SPECTRA_PER_HEAP, N_POLS, COMPLEX)
            ig.add_item(send.TIMESTAMP_ID, "timestamp", "", shape=(), format=IMMEDIATE_FORMAT)
            ig.add_item(send.FENG_ID_ID, "feng_id", "", shape=(), format=IMMEDIATE_FORMAT)
            ig.add_item(send.FREQUENCY_ID, "frequency", "", shape=(), format=IMMEDIATE_FORMAT)
            ig.add_item(send.FENG_RAW_ID, "feng_raw", "", shape=raw_shape, dtype=np.int8)
            expected_timestamp = expected_first_timestamp
            timestamp_step = SPECTRA_PER_HEAP * CHANNELS * 2  # TODO not valid for narrowband
            row = []
            async for heap in stream:
                assert set(ig.update(heap)) == {"timestamp", "feng_id", "frequency", "feng_raw"}
                assert ig["feng_id"].value == FENG_ID
                if expected_timestamp is not None:
                    assert ig["timestamp"].value == expected_timestamp
                else:
                    expected_timestamp = expected_first_timestamp = ig["timestamp"].value
                assert ig["frequency"].value == i * raw_shape[0]
                expected_timestamp += timestamp_step
                row.append(ig["feng_raw"].value.copy())
            # Glue all the heaps together along the time axis
            heaps.append(np.concatenate(row, axis=1))
        # Glue the parts of the band together along the channel axis. This
        # also ensures that there were the same number of heaps per channel.
        data = np.concatenate(heaps, axis=0)
        # Convert to complex for analysis
        data = data[..., 0] + 1j * data[..., 1]
        timestamps = np.arange(data.shape[1], dtype=np.int64) * (CHANNELS * 2) + expected_first_timestamp
        return data, timestamps

    # One delay value is tested with vkgdr, another with smaller output chunks
    @pytest.mark.parametrize(
        "delay_samples",
        [
            (0.0, 0.0),
            (8192.0, 234.5),
            (42.0, 58.0),
            (42.4, 24.2),
            pytest.param((42.7, 24.9), marks=[pytest.mark.cmdline_args("--dst-chunk-jones=65536")]),
            pytest.param((42.8, 24.5), marks=[pytest.mark.use_vkgdr]),
        ],
    )
    async def test_channel_centre_tones(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
        delay_samples: Tuple[float, float],
    ) -> None:
        """Put in tones at channel centre frequencies, with delays and gains, and check the result."""
        # Delay the tone by a negative amount, then compensate with a positive delay.
        # (delay_samples and delay_s are correction terms).
        # The tones are placed in the second Nyquist zone (the "1 +" in
        # frac_channel) then down-converted to baseband, simulating what
        # happens in MeerKAT L-band.
        tone_channels = [64, 271]
        tones = [
            CW(frac_channel=1 + tone_channels[0] / CHANNELS, magnitude=80.0, delay=-delay_samples[0]),
            CW(frac_channel=1 + tone_channels[1] / CHANNELS, magnitude=110.0, phase=1.23, delay=-delay_samples[1]),
        ]
        delay_s = np.array(delay_samples) / ADC_SAMPLE_RATE
        sky_centre_frequency = 0.75 * ADC_SAMPLE_RATE
        # Compute phase correction to compensate for the down-conversion.
        # (delay_s is negated here because in the original it is the signal
        # delay rather than the correction).
        # Based on katpoint.delay.DelayCorrection.corrections
        phase = -2.0 * np.pi * sky_centre_frequency * -delay_s
        phase_correction = -phase
        coeffs = [f"{d},0.0:{p},0.0" for d, p in zip(delay_s, phase_correction)]
        await engine_client.request("delays", SYNC_EPOCH, *coeffs)

        # Use constant-magnitude gains to avoid throwing off the magnitudes
        rng = np.random.default_rng(123)
        gain_phase = rng.uniform(0, 2 * np.pi, (CHANNELS, N_POLS))
        gains = GAIN * np.exp(1j * gain_phase).astype(np.complex64)
        for pol in range(N_POLS):
            await engine_client.request("gain", pol, *(str(gain) for gain in gains[:, pol]))

        src_layout = engine_server._src_layout
        n_samples = 20 * src_layout.chunk_samples
        dig_data = self._make_tone(n_samples, tones[0], 0) + self._make_tone(n_samples, tones[1], 1)
        dig_data[:, 1::2] *= -1  # Down-convert to baseband

        # Don't send the first chunk, to avoid complications with the step
        # change in the delay at SYNC_EPOCH.
        first_timestamp = src_layout.chunk_samples
        expected_first_timestamp = first_timestamp
        # The data should have as many samples as the input, minus a reduction
        # from PFB windowing, rounded down to a full heap.
        expected_spectra = (n_samples + min(delay_samples)) // (CHANNELS * 2) - (TAPS - 1)
        expected_spectra = expected_spectra // SPECTRA_PER_HEAP * SPECTRA_PER_HEAP
        if max(delay_samples) > 0:
            # The first output heap would require data from before the first
            # timestamp, so it does not get produced
            expected_first_timestamp += CHANNELS * 2 * SPECTRA_PER_HEAP
            expected_spectra -= SPECTRA_PER_HEAP
        out_data, _ = await self._send_data(
            mock_recv_streams,
            mock_send_stream,
            engine_server,
            dig_data,
            first_timestamp=first_timestamp,
            expected_first_timestamp=expected_first_timestamp,
        )
        assert out_data.shape[1] == expected_spectra

        # Check for the tones
        for pol in range(2):
            tone_data = out_data[tone_channels[pol], :, pol]
            expected_mag = tones[pol].magnitude * CHANNELS * GAIN
            assert 50 <= expected_mag < 127, "Magnitude is outside of good range for testing"
            np.testing.assert_equal(np.abs(tone_data), pytest.approx(expected_mag, 2))
            # The frequency corresponds to an integer number of cycles per
            # spectrum, so the phase will be consistent across spectra.
            # The accuracy is limited by the quantisation.
            expected_phase = wrap_angle(tones[pol].phase + gain_phase[tone_channels[pol], pol])
            np.testing.assert_equal(np.angle(tone_data), pytest.approx(expected_phase, abs=0.01))
            # Suppress the tone and check that everything is now zero (the
            # spectral leakage should be below the quantisation threshold).
            tone_data.fill(0)
            np.testing.assert_equal(out_data[..., pol], 0)

    async def test_spectral_leakage(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
    ) -> None:
        """Test leakage from tones that are not in the frequency centre."""
        # Rather than parametrize the test (which would be slow), send in
        # lots of different tones at different times. Each tone is maintained
        # for a full PFB window, and we just discard the outputs corresponding
        # to times that mix the tones. The tones are all placed in the centre
        # channel, but linearly spaced over the frequencies in that channel's
        # frequency bin.
        n_tones = 1024  # Note: must lead to sending a whole number of chunks
        tones = [
            CW(frac_channel=(CHANNELS // 2 - 0.5 + (i + 0.5) / n_tones) / CHANNELS, magnitude=500)
            for i in range(n_tones)
        ]
        pfb_window = CHANNELS * 2 * TAPS
        dig_data = np.concatenate([self._make_tone(pfb_window, tone, 0) for tone in tones], axis=1)
        # Add some extra data to fill out the last output heap
        padding = np.zeros((2, engine_server._src_layout.chunk_samples), dig_data.dtype)
        dig_data = np.concatenate([dig_data, padding], axis=1)

        # Crank up the gain so that leakage is measurable
        gain = 100 / CHANNELS
        for pol in range(N_POLS):
            await engine_client.request("gain", pol, gain)
        # CBF-REQ-0126: The CBF shall perform channelisation such that the 53 dB
        # attenuation is ≤ 2x (twice) the pass band width.
        #
        # The division by 20 (not 10) is because we're dealing with voltage,
        # not power.
        tol = 10 ** (-53 / 20) * (tones[0].magnitude * CHANNELS) * gain

        out_data, _ = await self._send_data(
            mock_recv_streams,
            mock_send_stream,
            engine_server,
            dig_data,
        )
        for i in range(n_tones):
            # Get the data for the PFB window that holds the tone
            data = out_data[:, i * TAPS, 0]
            # Blank out the channel that is expected to have the tone, and
            # the nearer adjacent one (with is within the 2x tolerance).
            data[CHANNELS // 2] = 0
            if i < n_tones // 2:
                data[CHANNELS // 2 - 1] = 0
            else:
                data[CHANNELS // 2 + 1] = 0
            np.testing.assert_equal(data, pytest.approx(0, abs=tol))

    async def test_delay_phase_rate(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
    ) -> None:
        """Test that delay rate and phase rate setting works."""
        # One tone at centre frequency to test the absolute phase, and one at another
        # frequency to test the slope across the band.
        tone_channels = [CHANNELS // 2, CHANNELS - 123]
        tones = [CW(frac_channel=channel / CHANNELS, magnitude=110) for channel in tone_channels]
        src_layout = engine_server._src_layout
        n_samples = 10 * src_layout.chunk_samples
        dig_data = np.sum([self._make_tone(n_samples, tone, 0) for tone in tones], axis=0)
        dig_data[1] = dig_data[0]  # Copy data from pol 0 to pol 1

        # Should be high enough to cause multiple coarse delay changes per chunk
        delay_rate = np.array([1e-5, 1.2e-5])
        # Should wrap multiple times over the test
        phase_rate_per_sample = np.array([30, 32.5]) / n_samples
        phase_rate = phase_rate_per_sample * ADC_SAMPLE_RATE
        coeffs = [f"0.0,{dr}:0.0,{pr}" for dr, pr in zip(delay_rate, phase_rate)]
        await engine_client.request("delays", SYNC_EPOCH, *coeffs)

        first_timestamp = 100 * src_layout.chunk_samples
        out_data, timestamps = await self._send_data(
            mock_recv_streams,
            mock_send_stream,
            engine_server,
            dig_data,
            first_timestamp=first_timestamp,
        )
        # Add a polarisation dimension to timestamps to simplify some
        # broadcasting computations below.
        timestamps = timestamps[:, np.newaxis]
        # Invert the delay model to find the corresponding input timestamps,
        # which in turn tells us the delay and phase rate applied at those times.
        # If t_i is the input timestamp and t_o is the output timestamp, then the
        # delay is t_i * delay_rate and hence t_o = t_i * (1 + delay_rate).
        timestamps = timestamps / (1 + delay_rate)  # shape Nx2
        expected_phase = wrap_angle(phase_rate_per_sample * timestamps)
        np.testing.assert_equal(
            wrap_angle(np.angle(out_data[tone_channels[0]]) - expected_phase), pytest.approx(0.0, abs=0.01)
        )

        # Adjust expected phase from the centre frequency to the other channel
        expected_phase -= np.pi * (tone_channels[1] - tone_channels[0]) / CHANNELS * delay_rate * timestamps
        np.testing.assert_equal(
            wrap_angle(np.angle(out_data[tone_channels[1]]) - expected_phase), pytest.approx(0.0, abs=0.01)
        )

    async def test_delay_changes(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine_server: Engine,
        engine_client: aiokatcp.Client,
    ) -> None:
        """Test loading several future delay models."""
        # Set up infrastructure for testing delay sensor updates
        delay_sensors = [engine_server.sensors[f"input{pol}-delay"] for pol in range(N_POLS)]
        sensor_updates_dict: Dict[str, List] = {delay_sensor.name: [] for delay_sensor in delay_sensors}

        def sensor_observer(
            delay_sensor: aiokatcp.Sensor, sensor_reading: aiokatcp.Reading, *, updates_list: List
        ) -> None:
            """Populate a list to compare at the end of this unit-test."""
            updates_list.append(sensor_reading.value)

        for delay_sensor in delay_sensors:
            delay_sensor.attach(partial(sensor_observer, updates_list=sensor_updates_dict[delay_sensor.name]))

        # To keep things simple, we'll just use phase, not delay.
        tone_channel = CHANNELS // 2
        tone = CW(frac_channel=0.5, magnitude=110)
        src_layout = engine_server._src_layout
        n_samples = 10 * src_layout.chunk_samples
        dig_data = self._make_tone(n_samples, tone, 0)

        # Load some delay models for the future (the last one beyond the end of the data)
        update_times = [0, 123456, 400000, 1234567, 1234567890]  # in samples
        update_phases = [1.0, 0.2, -0.2, -2.0, 0.0]
        for time, phase in zip(update_times, update_phases):
            coeffs = f"0.0,0.0:{phase},0.0"
            await engine_client.request("delays", SYNC_EPOCH + time / ADC_SAMPLE_RATE, coeffs, coeffs)

        out_data, timestamps = await self._send_data(
            mock_recv_streams,
            mock_send_stream,
            engine_server,
            dig_data,
        )
        out_data = out_data[tone_channel, :, 0]  # Only pol 0, centre channel matters

        for i in range(len(update_times) - 1):
            # Check which timestamps this delay model applies to
            valid = (update_times[i] <= timestamps) & (timestamps < update_times[i + 1])
            assert np.any(valid)
            phases = np.angle(out_data[valid])
            np.testing.assert_equal(wrap_angle(phases - update_phases[i]), pytest.approx(0, abs=0.01))

        for delay_sensor in delay_sensors:
            for expected_time, expected_phase, sensor_update in zip(
                update_times, update_phases, sensor_updates_dict[delay_sensor.name]
            ):
                sensor_values = sensor_update[1:-1].split(",")  # (timestamp, delay, delay_rate, phase, phase_rate)
                sensor_values = [float(field.strip()) for field in sensor_values]
                # NOTE: This tolerance is in place as the ADC timestamp gets
                # converted to a UNIX time and back again, losing some precision
                # during the conversion process.
                np.testing.assert_allclose(int(sensor_values[0]), expected_time, atol=200)
                # NOTE: Using the default relative tolerance of 1e-07
                np.testing.assert_allclose(sensor_values[3], expected_phase)
