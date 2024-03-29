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

"""Unit tests for XBEngine module."""

from typing import AbstractSet, Any, AsyncGenerator, Callable, Final, Sequence

import aiokatcp
import async_timeout
import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
import spead2.send
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext
from katsdpsigproc.accel import roundup
from numba import njit

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.fgpu.send import PREAMBLE_SIZE
from katgpucbf.xbgpu import METRIC_NAMESPACE, bsend
from katgpucbf.xbgpu.correlation import Correlation, device_filter
from katgpucbf.xbgpu.engine import BPipeline, RxQueueItem, XBEngine, XPipeline
from katgpucbf.xbgpu.main import make_engine, parse_args, parse_beam, parse_corrprod
from katgpucbf.xbgpu.output import BOutput, XOutput

from .. import PromDiff, get_sensor
from . import test_parameters
from .test_recv import gen_heap

pytestmark = [pytest.mark.device_filter.with_args(device_filter)]

get_baseline_index = njit(Correlation.get_baseline_index)

ADC_SAMPLE_RATE: Final[float] = 1712e6  # L-band
HEAPS_PER_FENGINE_PER_CHUNK: Final[int] = 2
SEND_RATE_FACTOR: Final[float] = 1.1
SAMPLE_BITWIDTH: Final[int] = 8
# Mark that can be applied to a test that just needs one set of parameters
DEFAULT_PARAMETERS = pytest.mark.parametrize(
    "n_ants, n_channels_total, n_spectra_per_heap, heap_accumulation_threshold",
    [(4, 1024, 256, [300, 300])],
)


@njit
def bounded_int8(val):
    """Create an int8 value bounded to the range [-127, 127]."""
    val = np.int8(val)
    if val == -128:
        val += 1
    return val


@njit
def feng_sample(batch: int, channel: int, antenna: int, out: np.ndarray) -> None:
    """Compute a dummy F-engine dual-pol complex sample.

    This is done in a deterministic way so that the expected result of the
    correlation can be easily determined. The return value is integer, with
    an extra axis for the real/imaginary parts of the complex numbers.

    Each dual-pol complex sample is assigned as follows:

    .. code-block:: python

        pol0_real = sign * batch
        pol0_imag = sign * channel
        pol1_real = -sign * antenna
        pol1_imag = -sign * channel

    The sign value is 1 for even batch indices and -1 for odd ones, for an even
    spread of positive and negative numbers. An added nuance is that these
    8-bit values are first cast to np.int8, then clamped to -127 as -128 is
    not supported by the Tensor Cores.
    """
    sign = 1 if batch % 2 == 0 else -1
    out[0, 0] = bounded_int8(sign * batch)
    out[0, 1] = bounded_int8(sign * channel)
    out[1, 0] = bounded_int8(-sign * antenna)
    out[1, 1] = bounded_int8(-sign * channel)


@njit
def feng_samples(batch: int, antenna: int, n_channels: int) -> np.ndarray:
    """Compute dummy F-engine samples.

    This calls :func:`feng_sample` across a range of channels and collects
    the results in one array.
    """
    samples = np.empty((n_channels, N_POLS, COMPLEX), np.int8)
    for channel in range(n_channels):
        feng_sample(batch, channel, antenna, samples[channel, :, :])
    return samples


@njit
def cmult_and_scale(a, b, c, out):
    """Multiply ``a`` and ``conj(b)``, scale the result by ``c``, and add to ``out``.

    Both ``a`` and ``b`` inputs and the output are 2-element arrays of
    np.int32, representing the real and imaginary components. ``c`` is a
    scalar.
    """
    out[0] += (a[0] * b[0] + a[1] * b[1]) * c
    out[1] += (a[1] * b[0] - a[0] * b[1]) * c


@njit
def generate_expected_corrprods(
    batch_start_idx: int,
    num_batches: int,
    heap_accumulation_threshold: int,
    channels: int,
    antennas: int,
    n_spectra_per_heap: int,
    missing_antenna: int | None,
) -> np.ndarray:
    """Calculate the expected correlator output.

    This doesn't do a full correlator. It calculates the results according to
    what is expected from the specific input generated in
    :meth:`TestEngine._create_heaps`.
    """
    baselines = antennas * (antennas + 1) * 2
    output_array = np.zeros((channels, baselines, COMPLEX), dtype=np.int32)
    if num_batches < heap_accumulation_threshold:
        # The accumulation is incomplete, and therefore completely marked
        # by the XBEngine
        output_array[..., 0] = -(2**31)
        output_array[..., 1] = 1
        return output_array
    for b in range(batch_start_idx, batch_start_idx + num_batches):
        for c in range(channels):
            # This is allocated as int32 so that cmult_and_scale won't overflow. The actual
            # stored values are in the range -127..127.
            in_data = np.empty((antennas, N_POLS, COMPLEX), np.int32)
            for a in range(antennas):
                feng_sample(b, c, a, in_data[a])
            for a2 in range(antennas):
                for a1 in range(a2 + 1):
                    bl_idx = get_baseline_index(a1, a2)
                    output_piece = output_array[c, 4 * bl_idx : 4 * bl_idx + 4, :]
                    cmult_and_scale(in_data[a1, 0], in_data[a2, 0], n_spectra_per_heap, output_piece[0])
                    cmult_and_scale(in_data[a1, 1], in_data[a2, 0], n_spectra_per_heap, output_piece[1])
                    cmult_and_scale(in_data[a1, 0], in_data[a2, 1], n_spectra_per_heap, output_piece[2])
                    cmult_and_scale(in_data[a1, 1], in_data[a2, 1], n_spectra_per_heap, output_piece[3])

    # Flag missing data
    for a2 in range(antennas):
        for a1 in range(a2 + 1):
            bl_idx = get_baseline_index(a1, a2)
            if a1 == missing_antenna or a2 == missing_antenna:
                output_array[:, 4 * bl_idx : 4 * bl_idx + 4, 0] = -(2**31)
                output_array[:, 4 * bl_idx : 4 * bl_idx + 4, 1] = 1

    return output_array


@njit
def generate_expected_beams(
    batch_start_idx: int,
    num_batches: int,
    channels: int,
    antennas: int,
    n_spectra_per_heap: int,
    missing_antenna: int | None,
    beam_pols: np.ndarray,
    weights: np.ndarray,
    delays: np.ndarray,
    quant_gains: np.ndarray,
    channel_spacing: float,
    centre_channel: int,
) -> np.ndarray:
    """Calculate the expected beamformer output.

    It calculates the results according to what is expected from the specific
    input generated in :meth:`TestEngine._create_heaps`.

    Parameters
    ----------
    batch_start_idx
        First batch index to output.
    num_batches
        Number of consecutive batches to emit.
    channels
        Number of channels.
    antennas
        Number of antennas.
    n_spectra_per_heap
        Number of spectra in each batch.
    missing_antenna
        If not None, data for this antenna is excluded from the beam.
    beam_pols
        Indicates, for each beam, which polarisation is used to form the beam.
    weights
        Real-valued weights for summing the beams, with shape (beams, antennas).
    delays
        Real-valued delay model, with shape (beams, antennas, 2). In each pair,
        the first element is the delay in seconds and the second is the phase
        to be applied at the centre frequency.
    channel_spacing
        Frequency difference between adjacent channels, in Hz.
    centre_channel
        Index of the centre channel of the whole stream, relative to the first
        channel processed by this engine.
    """
    out = np.empty((len(beam_pols), num_batches, channels, n_spectra_per_heap, COMPLEX), bsend.SEND_DTYPE)
    accum = np.zeros((len(beam_pols), num_batches, channels), np.complex64)
    sample = np.empty((N_POLS, COMPLEX), np.int8)
    sample_fp = np.empty(N_POLS, np.complex64)
    for batch in range(num_batches):
        for channel in range(channels):
            # Compute scale factor for turning a delay into a phase
            delay_to_phase = -2 * np.pi * channel_spacing * (channel - centre_channel)
            for antenna in range(antennas):
                if antenna == missing_antenna:
                    continue
                feng_sample(batch + batch_start_idx, channel, antenna, sample)
                sample_fp[0] = sample[0, 0] + np.complex64(1j) * sample[0, 1]
                sample_fp[1] = sample[1, 0] + np.complex64(1j) * sample[1, 1]
                for beam, pol in enumerate(beam_pols):
                    phase = delay_to_phase * delays[beam, antenna, 0] + delays[beam, antenna, 1]
                    rotation = np.exp(1j * phase)
                    accum[beam, batch, channel] += sample_fp[pol] * weights[beam, antenna] * rotation
            for beam in range(len(beam_pols)):
                value = accum[beam, batch, channel] * quant_gains[beam]
                sample[0, 0] = np.fmin(np.fmax(np.rint(value.real), -127), 127)
                sample[0, 1] = np.fmin(np.fmax(np.rint(value.imag), -127), 127)
                # Copy to all spectra in the batch
                out[beam, batch, channel] = sample[0]

    return out


def cancel_delays(n_ants: int) -> tuple[str, ...]:
    """Generate beam delays that, given identical input signals, will cause a zero output.

    This is implemented by using no delays but with phase corrections
    evenly spread around the unit circle. This requires more than one
    antenna to work.
    """
    assert n_ants > 0
    angles = np.arange(n_ants) / n_ants * 2 * np.pi
    return tuple(f"0:{a}" for a in angles)


def valid_end_to_end_combination(combo: dict) -> bool:
    """Check whether a combination for :meth:`TestEngine.test_xengine_end_to_end` is valid."""
    n_ants = combo["n_ants"]
    missing_antenna = combo["missing_antenna"]
    if missing_antenna is None:
        return True
    # Don't want to delete all the data, or an out-of-range antenna
    return n_ants > 1 and missing_antenna < n_ants


def verify_corrprod_sensors(
    *,
    xpipelines: list[XPipeline],
    corrprod_results: list[np.ndarray],
    prom_diff: PromDiff,
    actual_sensor_updates: dict[str, list[tuple[Any, aiokatcp.Sensor.Status]]],
    incomplete_accumulation_counters: list[int],
    n_channels_per_substream: int,
    n_baselines: int,
    missing_antenna: int | None,
):
    """Verify katcp and Prometheus sensors for processed XPipeline data.

    Parameters
    ----------
    xpipelines
        List of :class:`XPipeline` that are part of the unit test.
    corrprod_results
        List of arrays of all GPU-generated data. One output array per
        corrprod_output, where each array has shape
        (n_accumulations, n_channels_per_substream, n_baselines, COMPLEX).
    prom_diff
        Collection of Prometheus metrics observed during the XBEngine's
    actual_sensor_updates
        Dictionary of lists of sensor updates. They dictionary keys are sensor
        names, the values are a list of tuples for each sensor update captured
        via the callback attached to :class:`XPipeline` sensors. Accommodating
        for three value types as there are three different types of sensors in
        the XBEngine.
        processing of data stimulus.
    incomplete_accumulation_counters
        List of counts of incomplete accumulations for the unit test. This is
        dictated in part by `missing_antenna`.
    n_channels_per_substream
        Unit test fixture.
    n_baselines
        Number of baselines for the array size in the unit test.
    missing_antenna
        Index of the antenna missing, if any, during the XBEngine's processing
        of data.
    """
    expected_xsensor_updates: list[tuple[bool, aiokatcp.Sensor.Status]] = []
    for xpipeline, corrprod_result, incomplete_accums_counter in zip(
        xpipelines, corrprod_results, incomplete_accumulation_counters
    ):
        output_name = xpipeline.output.name
        n_accumulations_completed = corrprod_result.shape[0]
        assert (
            prom_diff.get_sample_diff("output_x_incomplete_accs_total", {"stream": output_name})
            == incomplete_accums_counter
        )
        assert prom_diff.get_sample_diff("output_x_heaps_total", {"stream": output_name}) == n_accumulations_completed
        # Could manually calculate it here, but it's available inside the send_stream
        assert prom_diff.get_sample_diff("output_x_bytes_total", {"stream": output_name}) == (
            xpipeline.send_stream.heap_payload_size_bytes * n_accumulations_completed
        )
        assert prom_diff.get_sample_diff("output_x_visibilities_total", {"stream": output_name}) == (
            n_channels_per_substream * n_baselines * n_accumulations_completed
        )
        assert prom_diff.get_sample_diff("output_x_clipped_visibilities_total", {"stream": output_name}) == 0

        # Verify sensor updates while we're here
        xsync_sensor_name = f"{xpipeline.output.name}.rx.synchronised"
        # As per the explanation in :func:`~send_data`, the first accumulation
        # is expected to be incomplete.
        expected_xsensor_updates.append((False, aiokatcp.Sensor.Status.ERROR))
        # Depending on the `missing_antenna` parameter, the full accumulations
        # will either be all complete or incomplete.
        if missing_antenna is not None:
            expected_xsensor_updates += [(False, aiokatcp.Sensor.Status.ERROR)] * (incomplete_accums_counter - 1)
        else:
            expected_xsensor_updates += [(True, aiokatcp.Sensor.Status.NOMINAL)] * (n_accumulations_completed - 1)

        assert actual_sensor_updates[xsync_sensor_name] == expected_xsensor_updates
        # Just to be sure
        expected_xsensor_updates.clear()


def verify_beam_sensors(
    *,
    beam_outputs: list[BOutput],
    beam_results_shape: tuple[int, ...],
    beam_dtype: np.dtype,
    prom_diff: PromDiff,
    actual_sensor_updates: dict[str, list[tuple[Any, aiokatcp.Sensor.Status]]],
    first_timestamp: int,
    last_timestamp: int,
    weights: np.ndarray,
    quant_gains: np.ndarray,
    delays: np.ndarray,
) -> None:
    """Verify katcp sensors and Prometheus counters for BPipeline data.

    Parameters
    ----------
    beam_outputs
        Output beam configurations parsed into BOutput objects.
    beam_results_shape
        The shape of the verified beam data for all beams with shape
        (len(beam_outputs), n_beam_heaps_sent, n_channels_per_substream,
        n_samples_between_spectra, COMPLEX).
    beam_dtype
        The numpy data type of the beam data, used to calculate the number of
        bytes in each heap.
    prom_diff
        Collection of Prometheus metrics observed during the XBEngine's
        processing of data stimulus.
    actual_sensor_updates
        Dictionary of lists of sensor updates. They dictionary keys are sensor
        names, the values are a list of tuples for each sensor update captured
        via the callback attached to :class:`BPipeline` sensors. Accommodating
        for three value types as there are three different types of sensors in
        the XBEngine.
    first_timestamp, last_timestamp
        Two timestamps indicating the start and end of data processing
        by the :class:`BPipeline`.
    weights, quant_gains, delays
        The beam weights, quantiser-gains and delays applied to each input of
        the beam data product. These are real floating-point values generated
        for the unit test.
    """
    # Get the number of total heaps transmitted by each beam output
    n_beam_heaps_sent = beam_results_shape[1]
    heap_shape = beam_results_shape[2:]
    heap_bytes = np.prod(heap_shape) * beam_dtype.itemsize
    # We get rid of the final dimension in the beam data as we need the total
    # number of (COMPLEX) samples.
    heap_samples = np.prod(heap_shape[:-1])
    for i, beam_output in enumerate(beam_outputs):
        # The assert statements are mainly to force mypy to realise the
        # prom_diff values obtained are the expected data type
        prom_output_b_heaps_total = prom_diff.get_sample_diff("output_b_heaps_total", {"stream": beam_output.name})
        assert prom_output_b_heaps_total is not None, "output_b_heaps counter is None"
        prom_output_b_bytes_total = prom_diff.get_sample_diff("output_b_bytes_total", {"stream": beam_output.name})
        assert prom_output_b_bytes_total is not None, "output_b_bytes counter is None"
        prom_output_b_samples_total = prom_diff.get_sample_diff("output_b_samples_total", {"stream": beam_output.name})
        assert prom_output_b_samples_total is not None, "output_b_samples counter is None"
        assert prom_output_b_heaps_total == n_beam_heaps_sent
        assert prom_output_b_bytes_total == n_beam_heaps_sent * heap_bytes
        assert prom_output_b_samples_total == n_beam_heaps_sent * heap_samples
        # TODO: NGC-1173 Add check for `output_b_clipped_samples`

        assert first_timestamp < last_timestamp, (
            "Timestamp before katcp requests is not less than timestamp after data"
            f"has been processed: {first_timestamp} >= {last_timestamp}"
        )
        assert actual_sensor_updates[f"{beam_output.name}.weight"] == [
            (str(list(weights[i])), aiokatcp.Sensor.Status.NOMINAL)
        ]
        assert actual_sensor_updates[f"{beam_output.name}.quantiser-gain"] == [
            (quant_gains[i], aiokatcp.Sensor.Status.NOMINAL)
        ]

        delay_updates_str = ", ".join(f"{delay}, {phase}" for delay, phase in delays[i])
        # The ?beam-delay request is submitted before the xbengine starts
        # receiving/processing data, so the `loadmcnt` is zero (it is
        # applied immediately).
        assert actual_sensor_updates[f"{beam_output.name}.delay"] == [
            (f"({first_timestamp}, {delay_updates_str})", aiokatcp.Sensor.Status.NOMINAL)
        ]


class TestEngine:
    r"""Grouping of unit tests for :class:`.XBEngine`\'s various functionality."""

    @pytest.fixture
    def frequency(self, n_engines: int, n_channels_per_substream: int):
        """Return the start frequency for the XBEngine.

        We arbitrarily choose to pretend to be an engine in the middle of the
        range for the given array size.
        """
        engine_number = n_engines // 2
        return engine_number * n_channels_per_substream

    @pytest.fixture
    def corrprod_args(self, heap_accumulation_threshold: tuple[int, int]) -> list[str]:
        """Arguments to pass to the command-line parser for multiple --corrprods."""

        return [
            f"name=bcp1,dst=239.10.11.0:7148,heap_accumulation_threshold={heap_accumulation_threshold[0]}",
            f"name=bcp2,dst=239.10.11.1:7148,heap_accumulation_threshold={heap_accumulation_threshold[1]}",
        ]

    @pytest.fixture
    def beam_args(self) -> list[str]:
        """Arguments to pass to the command-line parser for multiple beams."""
        return [
            "name=beam_0x,dst=239.10.12.0:7148,pol=0",
            "name=beam_0y,dst=239.10.12.1:7148,pol=1",
        ]

    @pytest.fixture
    def corrprod_outputs(self, corrprod_args: list[str]) -> list[XOutput]:
        """The outputs to run correlation tests against."""
        return [parse_corrprod(corrprod_arg) for corrprod_arg in corrprod_args]

    @pytest.fixture
    def beam_outputs(self, beam_args: list[str]) -> list[BOutput]:
        """The outputs to run beamforming against."""
        return [parse_beam(beam_arg) for beam_arg in beam_args]

    @staticmethod
    def _create_heaps(
        timestamp: int,
        batch_index: int,
        n_ants: int,
        n_channels_per_substream: int,
        frequency: int,
        n_spectra_per_heap: int,
        missing_antennas: AbstractSet[int] = frozenset(),
    ) -> list[spead2.send.HeapReference]:
        """Generate a deterministic input for sending to the XBEngine.

        One heap is generated per antenna in the array. All heaps will have the
        same timestamp. A heap is composed of multiple channels. Per channel,
        all values are kept constant. This makes for faster verification with
        the downside being that if samples within the channel range get mixed
        up, this will not be detected. See :func:`feng_sample` for the formula
        used.

        This results in a deterministic expected output value without the need
        for a full CPU-side correlator.

        Parameters
        ----------
        timestamp
            The timestamp that will be assigned to all heaps.
        batch_index
            Represents the index of this collection of generated heaps. Value is
            used to encode sample data.
        n_ants
            The number of antennas that data will be received from. A seperate heap
            will be generated per antenna.
        n_channels_per_substream
            The number of frequency channels contained in a heap.
        frequency
            The first channel in the range handled by this XBEngine.
        n_spectra_per_heap
            The number of time samples per frequency channel.
        missing_antennas
            The desired antennas whose heaps will be removed from the created
            list, indexed from zero (0).

        Returns
        -------
        heaps
            A list of HeapReference objects as accepted by :func:`.send_heaps`.
        """
        # Generate all the heaps for the different antennas.
        heaps: list[spead2.send.HeapReference] = []
        for ant_index in range(n_ants):
            if ant_index in missing_antennas:
                continue
            sample_array = feng_samples(batch_index, ant_index, n_channels_per_substream)
            # Replicate the value to all spectra in the heap
            sample_array = sample_array[:, np.newaxis, :, :].repeat(n_spectra_per_heap, axis=1)

            # Create the heap, add it to a list of HeapReferences.
            heap = gen_heap(timestamp, ant_index, frequency, sample_array)
            heaps.append(spead2.send.HeapReference(heap))

        return heaps

    def _make_feng(
        self, queues: list[spead2.InprocQueue], max_packet_size: int, max_heaps: int
    ) -> "spead2.send.asyncio.AsyncStream":
        """Create send stream for a fake F-Engine."""
        feng_stream_config = spead2.send.StreamConfig(
            max_packet_size=max_packet_size,
            max_heaps=max_heaps,
        )
        return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, feng_stream_config)

    async def _send_data(
        self,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        corrprod_outputs: list[XOutput],
        beam_outputs: list[BOutput],
        *,
        heap_factory: Callable[[int], list[spead2.send.HeapReference]],
        batch_indices: Sequence[int],
        timestamp_step: int,
        n_ants: int,
        n_channels_per_substream: int,
        frequency: int,
        n_spectra_per_heap: int,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Send a contiguous stream of data to the engine and retrieve the results.

        Each full accumulation (for each corrprod-output) requires
        `heap_accumulation_threshold` batches of heaps. However, `batch_indices`
        is not required to contain full accumulations.

        Results are returned for both correlation products and beams.

        Parameters
        ----------
        mock_recv_stream
            Fixture
        mock_send_stream
            Fixture
        corrprod_outputs
            Fixture
        heap_factory
            Callback that takes a batch index and returns the heaps for that index.
        batch_indices
            Indices of the batches to send. These must be strictly increasing,
            but need not be contiguous.
        timestamp_step
            Timestamp step between each received heap processed.
        n_ants, n_channels_per_substream, n_spectra_per_heap, frequency
            See :meth:`_create_heaps` for more info.

        Returns
        -------
        corrprod_results
            List of arrays of all GPU-generated data. One output array per
            corrprod_output, where each array has shape
            (n_accumulations, n_channels_per_substream, n_baselines, COMPLEX).
        beam_results
            Beamformer output, with shape (n_beams, n_frames,
            n_channels_per_substream, n_spectra_per_heap, COMPLEX).
        """
        max_packet_size = n_spectra_per_heap * N_POLS * COMPLEX * SAMPLE_BITWIDTH // 8 + PREAMBLE_SIZE
        max_heaps = n_ants * HEAPS_PER_FENGINE_PER_CHUNK * 10
        feng_stream = self._make_feng(mock_recv_streams, max_packet_size, max_heaps)

        acc_indices: list[set[int]] = [set() for _ in corrprod_outputs]
        for batch_index in batch_indices:
            for i, corrprod_output in enumerate(corrprod_outputs):
                acc_index = batch_index // corrprod_output.heap_accumulation_threshold
                acc_indices[i].add(acc_index)
            heaps = heap_factory(batch_index)
            await feng_stream.async_send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

        for queue in mock_recv_streams:
            queue.stop()

        n_baselines = n_ants * (n_ants + 1) * 2
        corrprod_results = [
            np.zeros(
                shape=(
                    len(acc_index_set),  # n_accumulations for this XPipeline
                    n_channels_per_substream,
                    n_baselines,
                    COMPLEX,
                ),
                dtype=np.int32,
            )
            for acc_index_set in acc_indices
        ]

        out_config = spead2.recv.StreamConfig(max_heaps=100)
        out_tp = spead2.ThreadPool()

        for i, corrprod_output in enumerate(corrprod_outputs):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(mock_send_stream[i])
            # It is expected that the first packet will be a descriptor.
            ig_recv = spead2.ItemGroup()
            heap = await stream.get()
            items = ig_recv.update(heap)
            assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

            for j, accumulation_index in enumerate(sorted(acc_indices[i])):
                # Wait for heap to be ready and then update out item group
                # with the new values.
                heap = await stream.get()

                while (updated_items := set(ig_recv.update(heap))) == set():
                    # Test has gone on long enough that we've received another descriptor
                    heap = await stream.get()
                assert updated_items == {"frequency", "timestamp", "xeng_raw"}
                # Ensure that the timestamp from the heap is what we expect.
                assert (
                    ig_recv["timestamp"].value % (timestamp_step * corrprod_output.heap_accumulation_threshold) == 0
                ), "Output timestamp is not a multiple of timestamp_step * heap_accumulation_threshold."

                assert (
                    ig_recv["timestamp"].value
                    == accumulation_index * timestamp_step * corrprod_output.heap_accumulation_threshold
                ), (
                    "Output timestamp is not correct. "
                    f"Expected: "
                    f"{hex(accumulation_index * timestamp_step * corrprod_output.heap_accumulation_threshold)}, "
                    f"actual: {hex(ig_recv['timestamp'].value)}."
                )

                assert ig_recv["frequency"].value == frequency, (
                    "Output channel offset not correct. "
                    f"Expected: {frequency}, "
                    f"actual: {ig_recv['frequency'].value}."
                )

                corrprod_results[i][j] = ig_recv["xeng_raw"].value

        # TODO: NGC-1172 The tweaks to process beam data below rely on
        # `batch_indices` to be contiguous and for the zeroth (and minimum)
        # value to be a multiple of `HEAPS_PER_FENGINE_PER_CHUNK`. The check
        # below is temporary until the BPipeline is able to handle missing
        # data.
        assert list(batch_indices) == (
            list(range(min(batch_indices), max(batch_indices) + 1))
        ), "Batch indices need to be contiguous for testing beam data"
        assert batch_indices[0] % HEAPS_PER_FENGINE_PER_CHUNK == 0, (
            "Need to start data transmission with a batch index that is a multiple "
            f"of HEAPS_PER_FENGINE_PER_CHUNK ({HEAPS_PER_FENGINE_PER_CHUNK})"
        )

        # NOTE: Update `batch_indices` to end on a multiple of
        # `HEAPS_PER_FENGINE_PER_CHUNK`, but only for the beam_outputs because
        # they currently send `HEAPS_PER_FENGINE_PER_CHUNK` heaps all the time.
        # This does not mean the final heap (for each beam_output) has sane
        # data in it. In fact, ensure you verify data for values in
        # `batch_indices`.
        n_beam_heaps = roundup(len(batch_indices), HEAPS_PER_FENGINE_PER_CHUNK)
        beam_batch_indices = range(batch_indices[0], batch_indices[0] + n_beam_heaps)
        beam_results = np.zeros(
            (len(beam_outputs), n_beam_heaps, n_channels_per_substream, n_spectra_per_heap, COMPLEX),
            bsend.SEND_DTYPE,
        )
        for i in range(len(beam_outputs)):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(mock_send_stream[i + len(corrprod_outputs)])
            # It is expected that the first packet will be a descriptor.
            ig_recv = spead2.ItemGroup()
            heap = await stream.get()
            items = ig_recv.update(heap)
            assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

            for j, index in enumerate(beam_batch_indices):
                heap = await stream.get()
                while (updated_items := set(ig_recv.update(heap))) == set():
                    # Test has gone on long enough that we've received another descriptor
                    heap = await stream.get()

                assert updated_items == {"frequency", "timestamp", "bf_raw"}
                assert ig_recv["timestamp"].value == index * timestamp_step
                assert ig_recv["frequency"].value == frequency
                beam_results[i, j, ...] = ig_recv["bf_raw"].value

        return corrprod_results, beam_results

    @pytest.fixture
    def n_engines(self, n_ants: int) -> int:
        """Get a realistic number of engines by rounding up to the next power of 2."""
        n_engines = 1
        while n_engines < n_ants:
            n_engines *= 2
        return n_engines

    @pytest.fixture
    def n_channels_per_substream(self, n_channels_total: int, n_engines: int) -> int:  # noqa: D102
        return n_channels_total // n_engines

    @pytest.fixture
    def n_samples_between_spectra(self, n_channels_total: int) -> int:  # noqa: D102
        # NOTE: Multiply by 8 to account for a decimation factor in the
        # Narrowband case. It is also included to ensure we don't rely on the
        # assumption that `n_samples_between_spectra == 2 * n_channels_total`.
        return 2 * n_channels_total * 8

    @pytest.fixture
    def engine_arglist(
        self,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        corrprod_args: list[str],
        beam_args: list[str],
    ) -> list[str]:
        args = [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--array-size={n_ants}",
            f"--channels={n_channels_total}",
            f"--channels-per-substream={n_channels_per_substream}",
            f"--samples-between-spectra={n_samples_between_spectra}",
            f"--channel-offset-value={frequency}",
            f"--spectra-per-heap={n_spectra_per_heap}",
            f"--heaps-per-fengine-per-chunk={HEAPS_PER_FENGINE_PER_CHUNK}",
            "--sync-epoch=1234567890",
            "--src-interface=lo",
            "--dst-interface=lo",
            "--tx-enabled",
            "239.10.11.4:7149",  # src
        ]
        for corrprod in corrprod_args:
            args.append(f"--corrprod={corrprod}")
        for beam in beam_args:
            args.append(f"--beam={beam}")
        return args

    @pytest.fixture
    async def xbengine(
        self,
        context: AbstractContext,
        engine_arglist: list[str],
    ) -> AsyncGenerator[XBEngine, None]:
        """Create and start an engine based on the fixture values."""
        args = parse_args(engine_arglist)
        xbengine, _ = make_engine(context, args)
        await xbengine.start()

        yield xbengine

        await xbengine.stop()

    @pytest.fixture
    async def client(self, xbengine: XBEngine) -> AsyncGenerator[aiokatcp.Client, None]:
        host, port = xbengine.sockets[0].getsockname()[:2]
        async with async_timeout.timeout(5):  # To fail the test quickly if unable to connect
            client = await aiokatcp.Client.connect(host, port)

        yield client

        client.close()
        await client.wait_closed()

    @pytest.mark.combinations(
        "n_ants, n_channels_total, n_spectra_per_heap, missing_antenna, heap_accumulation_threshold",
        test_parameters.array_size,
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
        [None, 0, 3],
        [(3, 7), (4, 8), (5, 9)],
        filter=valid_end_to_end_combination,
    )
    async def test_engine_end_to_end(
        self,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        xbengine: XBEngine,
        client: aiokatcp.Client,
        n_ants: int,
        n_spectra_per_heap: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        corrprod_outputs: list[XOutput],
        beam_outputs: list[BOutput],
        missing_antenna: int | None,
    ):
        """
        End-to-end test for the XBEngine.

        Simulated input data is generated and passed to the XBEngine, yielding
        output results which are then verified.

        The simulated data is not random but is encoded based on certain
        parameters. This allows the verification function to generate the
        correct data to compare to the xbengine without performing the full
        correlation algorithm, greatly improving processing time.

        This test simulates an incomplete accumulation at the start of transmission
        to ensure that the auto-resync logic works correctly. Data is also
        generated from a timestamp starting after the first accumulation
        boundary to more accurately test the setting of the first output
        packet's timestamp (to be non-zero).
        """
        # NOTE: `HEAPS_PER_FENGINE_PER_CHUNK` and the `heap_accumulation_threshold`s
        # are chosen carefully for this test. We simulate the first accumulation
        # being incomplete by ensuring there is only one chunk of data present.
        # In doing so, we must ensure `HEAPS_PER_FENGINE_PER_CHUNK` is smaller
        # than either pipeline's `heap_accumulation_threshold`. The logic of the
        # test has *not* been verified if that constraint isn't met, it merely
        # `assert`'s.
        heap_accumulation_thresholds = [
            corrprod_output.heap_accumulation_threshold for corrprod_output in corrprod_outputs
        ]
        assert min(heap_accumulation_thresholds) > HEAPS_PER_FENGINE_PER_CHUNK

        n_baselines = n_ants * (n_ants + 1) * 2

        timestamp_step = n_samples_between_spectra * n_spectra_per_heap
        missing_antennas = set() if missing_antenna is None else {missing_antenna}

        range_start = frequency
        range_end = range_start + n_channels_per_substream - 1
        for corrprod_output in corrprod_outputs:
            assert xbengine.sensors[f"{corrprod_output.name}.chan-range"].value == f"({range_start},{range_end})"
        for beam_output in beam_outputs:
            assert xbengine.sensors[f"{beam_output.name}.chan-range"].value == f"({range_start},{range_end})"

        # Need a method of capturing synchronised aiokatcp.Sensor updates as
        # they happen in the XBEngine
        dynamic_bsensor_names = ["delay", "quantiser-gain", "weight"]
        actual_sensor_updates: dict[str, list[tuple[Any, aiokatcp.Sensor.Status]]] = {
            f"{beam_output.name}.{dynamic_bsensor_name}": list()
            for beam_output in beam_outputs
            for dynamic_bsensor_name in dynamic_bsensor_names
        }
        actual_sensor_updates.update(
            (f"{corrprod_output.name}.rx.synchronised", list()) for corrprod_output in corrprod_outputs
        )

        def sensor_observer(sensor: aiokatcp.Sensor, sensor_reading: aiokatcp.Reading):
            """Record sensor updates in a list for later comparison."""
            actual_sensor_updates[sensor.name].append((sensor_reading.value, sensor_reading.status))

        for corrprod_output in corrprod_outputs:
            xbengine.sensors[f"{corrprod_output.name}.rx.synchronised"].attach(sensor_observer)

        for beam_output in beam_outputs:
            for dynamic_bsensor_name in dynamic_bsensor_names:
                xbengine.sensors[f"{beam_output.name}.{dynamic_bsensor_name}"].attach(sensor_observer)

        def heap_factory(batch_index: int) -> list[spead2.send.HeapReference]:
            timestamp = batch_index * timestamp_step
            return self._create_heaps(
                timestamp,
                batch_index,
                n_ants,
                n_channels_per_substream,
                frequency,
                n_spectra_per_heap,
                missing_antennas=missing_antennas,
            )

        first_timestamp = last_timestamp = 0
        # Also need to access the request arguments later when generating expected sensor updates
        rng = np.random.default_rng(seed=1)
        weights = rng.uniform(0.5, 2.0, size=(len(beam_outputs), n_ants))
        quant_gains = rng.uniform(0.5, 2.0, size=(len(beam_outputs)))
        delays = np.zeros((len(beam_outputs), n_ants, 2), np.float64)
        # Delay is in seconds, so needs to be very small
        delays[..., 0] = rng.uniform(-1e-9, 1e-9, size=(len(beam_outputs), n_ants))
        # Phase is in radians
        delays[..., 1] = rng.uniform(-2 * np.pi, 2 * np.pi, size=(len(beam_outputs), n_ants))
        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            # NOTE: The product of `heap_accumulation_thresholds` is used in
            # two ways below. Both uses are to ensure there is a whole number
            # of accumulations for *both* XPipelines. The first usage is
            # dual-purpose once more:
            # - In addition to the above, the arbitrary start position for data
            #   in this test dictates the first received chunk to be in the
            #   middle of an accumulation.
            # - More accurately, it forces the first accumulation (to be
            #   processed) to only contain one chunk of data. This ensures
            #   we test that output dumps are aligned correctly, despite
            #   the first data processed not being on an accumulation
            #   boundary.
            # Explicitly cast to python int as the np.int64 returned wasn't
            # playing nice with `last_timestamp`
            n_heaps = int(np.prod(heap_accumulation_thresholds))
            batch_start_index = 12 * n_heaps  # Somewhere arbitrary that isn't zero
            batch_end_index = batch_start_index + n_heaps
            # Add an extra chunk before the first full accumulation
            batch_start_index -= HEAPS_PER_FENGINE_PER_CHUNK

            for i, output in enumerate(beam_outputs):
                # We only capture the timestamps before and after all katcp
                # requests are executed as we only need to ensure it has
                # increased across all three requests (not in between).
                # The first timestamp should be zero as the xbengine has not
                # been given data to process yet. That is, the xbengine is
                # currently at idle.
                first_timestamp = 0
                await client.request("beam-weights", output.name, *weights[i])
                await client.request("beam-quant-gains", output.name, quant_gains[i])
                await client.request("beam-delays", output.name, *[f"{d[0]}:{d[1]}" for d in delays[i]])

            corrprod_results, beam_results = await self._send_data(
                mock_recv_streams,
                mock_send_stream,
                corrprod_outputs=corrprod_outputs,
                beam_outputs=beam_outputs,
                batch_indices=range(batch_start_index, batch_end_index),
                heap_factory=heap_factory,
                timestamp_step=timestamp_step,
                n_ants=n_ants,
                n_channels_per_substream=n_channels_per_substream,
                frequency=frequency,
                n_spectra_per_heap=n_spectra_per_heap,
            )
            last_timestamp = batch_end_index * timestamp_step

        incomplete_accums_counters = []
        for i, corrprod_output in enumerate(corrprod_outputs):
            # Or assert if incomplete_accs_total == incomplete_accums_counter * len(xbengine._pipelines)
            incomplete_accums_counter = 0
            base_batch_index = batch_start_index
            for j, corrprod_result in enumerate(corrprod_results[i]):
                # The first heap is an incomplete accumulation containing a
                # single batch, we need to make sure that this is taken into
                # account by the verification function.
                if j == 0:
                    # This is to handle the first accumulation processed. The value
                    # checked here is simply the first in the range.
                    # - Even though :func:`generate_expected_corrprods` returns a
                    #   zeroed array for an incomplete accumulation, we still need
                    #   to maintain programmatic sense in the values generated here.
                    num_batches_in_current_accumulation = HEAPS_PER_FENGINE_PER_CHUNK
                    incomplete_accums_counter += 1
                else:
                    num_batches_in_current_accumulation = corrprod_output.heap_accumulation_threshold
                    if missing_antenna is not None:
                        incomplete_accums_counter += 1

                expected_output = generate_expected_corrprods(
                    base_batch_index,
                    num_batches_in_current_accumulation,
                    corrprod_output.heap_accumulation_threshold,
                    n_channels_per_substream,
                    n_ants,
                    n_spectra_per_heap,
                    missing_antenna,
                )
                base_batch_index += num_batches_in_current_accumulation
                np.testing.assert_equal(expected_output, corrprod_result)
            incomplete_accums_counters.append(incomplete_accums_counter)

        xpipelines: list[XPipeline] = [pipeline for pipeline in xbengine._pipelines if isinstance(pipeline, XPipeline)]
        verify_corrprod_sensors(
            xpipelines=xpipelines,
            corrprod_results=corrprod_results,
            prom_diff=prom_diff,
            actual_sensor_updates=actual_sensor_updates,
            incomplete_accumulation_counters=incomplete_accums_counters,
            n_channels_per_substream=n_channels_per_substream,
            n_baselines=n_baselines,
            missing_antenna=missing_antenna,
        )

        channel_spacing = xbengine.bandwidth_hz / xbengine.n_channels_total
        expected_beams = generate_expected_beams(
            batch_start_index,
            batch_end_index - batch_start_index,
            n_channels_per_substream,
            n_ants,
            n_spectra_per_heap,
            missing_antenna,
            np.array([beam_output.pol for beam_output in beam_outputs]),
            weights=weights,
            delays=delays,
            quant_gains=quant_gains,
            channel_spacing=channel_spacing,
            centre_channel=n_channels_total // 2 - frequency,
        )
        # assert_allclose converts to float, which bloats memory usage.
        # To keep it manageable, compare a batch at a time.
        for i in range(expected_beams.shape[0]):
            # NOTE: As per the explanation at the end of `_send_data`, we
            # only verify data in the range of `batch_indices` for each
            # `beam_result` as any heaps sent afterwards are sent by default
            # - not because they are expected to have sane data in them.
            for j in range(expected_beams.shape[1]):
                np.testing.assert_allclose(expected_beams[i, j], beam_results[i, j], atol=1)

        # `beam_results` holds results for each heap transmitted by a
        # `beam_output` for all `beam_outputs`. We can reuse its dimensions in
        # the sensor verification below.
        verify_beam_sensors(
            beam_outputs=beam_outputs,
            beam_results_shape=beam_results.shape,
            beam_dtype=beam_results.dtype,
            prom_diff=prom_diff,
            actual_sensor_updates=actual_sensor_updates,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            weights=weights,
            quant_gains=quant_gains,
            delays=delays,
        )

    @DEFAULT_PARAMETERS
    async def test_saturation(
        self,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        xbengine: XBEngine,
        n_ants: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        heap_accumulation_threshold: list[int],
        corrprod_outputs: list[XOutput],
        beam_outputs: list[BOutput],
    ):
        """Test saturation statistics.

        .. todo::

           After the implementation is updated to avoid counting missing data
           as saturated, extend the test to check that.
        """
        timestamp_step = n_samples_between_spectra * n_spectra_per_heap
        n_baselines = n_ants * (n_ants + 1) * 2

        def heap_factory(batch_index: int) -> list[spead2.send.HeapReference]:
            timestamp = batch_index * timestamp_step
            data = np.full((n_channels_per_substream, n_spectra_per_heap, N_POLS, COMPLEX), 127, np.int8)
            return [
                spead2.send.HeapReference(gen_heap(timestamp, ant_index, frequency, data))
                for ant_index in range(n_ants)
            ]

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            await self._send_data(
                mock_recv_streams,
                mock_send_stream,
                corrprod_outputs,
                beam_outputs,
                batch_indices=range(0, heap_accumulation_threshold[0]),
                heap_factory=heap_factory,
                timestamp_step=timestamp_step,
                n_ants=n_ants,
                n_channels_per_substream=n_channels_per_substream,
                frequency=frequency,
                n_spectra_per_heap=n_spectra_per_heap,
            )

            await xbengine.stop()

        assert (
            prom_diff.get_sample_diff("output_x_visibilities_total", {"stream": "bcp1"})
            == n_channels_per_substream * n_baselines
        )
        assert (
            prom_diff.get_sample_diff("output_x_clipped_visibilities_total", {"stream": "bcp1"})
            == n_channels_per_substream * n_baselines
        )

    def _patch_get_rx_item(
        self, monkeypatch: pytest.MonkeyPatch, count: int, client: aiokatcp.Client, *request
    ) -> list[int]:
        """Patch :meth:`~.BPipeline._get_rx_item` to make a request partway through the stream.

        The returned list will be populated with the value of the
        ``steady-state-timestamp`` sensor immediately after executing the
        request.
        """
        counter = 0
        timestamp = []
        orig_get_rx_item = BPipeline._get_rx_item

        async def get_rx_item(self: BPipeline) -> RxQueueItem | None:
            nonlocal counter
            counter += 1
            if counter == count:
                await client.request(*request)
                _, informs = await client.request("sensor-value", "steady-state-timestamp")
                timestamp.append(int(informs[0].arguments[4]))
            return await orig_get_rx_item(self)

        monkeypatch.setattr(BPipeline, "_get_rx_item", get_rx_item)
        return timestamp

    @DEFAULT_PARAMETERS
    @pytest.mark.parametrize(
        "request_factory",
        [
            pytest.param(lambda name, n_ants: ("beam-quant-gains", name, 0.0), id="beam-quant-gains"),
            pytest.param(lambda name, n_ants: ("beam-weights", name) + (0.0,) * n_ants, id="beam-weights"),
            pytest.param(lambda name, n_ants: ("beam-delays", name) + cancel_delays(n_ants), id="beam-delays"),
        ],
    )
    async def test_steady_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        xbengine: XBEngine,
        client: aiokatcp.Client,
        n_ants: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        heap_accumulation_threshold: list[int],
        corrprod_outputs: list[XOutput],
        beam_outputs: list[BOutput],
        request_factory: Callable[[str, int], tuple],
    ):
        """Test that the steady-state-timestamp sensor works."""
        assert (await get_sensor(client, "steady-state-timestamp")) == 0

        timestamp_step = n_samples_between_spectra * n_spectra_per_heap

        def heap_factory(batch_index: int) -> list[spead2.send.HeapReference]:
            timestamp = batch_index * timestamp_step
            data = np.full((n_channels_per_substream, n_spectra_per_heap, N_POLS, COMPLEX), 10, np.int8)
            return [
                spead2.send.HeapReference(gen_heap(timestamp, ant_index, frequency, data))
                for ant_index in range(n_ants)
            ]

        request = request_factory(beam_outputs[0].name, n_ants)
        timestamp_list = self._patch_get_rx_item(monkeypatch, 4, client, *request)
        n_batches = heap_accumulation_threshold[0]
        _, data = await self._send_data(
            mock_recv_streams,
            mock_send_stream,
            corrprod_outputs,
            beam_outputs,
            batch_indices=range(n_batches),
            heap_factory=heap_factory,
            timestamp_step=timestamp_step,
            n_ants=n_ants,
            n_channels_per_substream=n_channels_per_substream,
            frequency=frequency,
            n_spectra_per_heap=n_spectra_per_heap,
        )
        await xbengine.stop()
        assert len(timestamp_list) == 1
        steady_state_timestamp = timestamp_list[0]
        # Not technically required by the definition, but ought to be true
        # in practice, and it makes the rest of the test simpler to write
        assert steady_state_timestamp % timestamp_step == 0
        steady_state_batch = steady_state_timestamp // timestamp_step
        assert 0 < steady_state_batch < n_batches
        # Should be all zeros after the steady state, but not before
        np.testing.assert_equal(data[0, :steady_state_batch] != 0, True)
        np.testing.assert_equal(data[0, steady_state_batch:], 0)

    @DEFAULT_PARAMETERS
    async def test_bad_requests(self, client: aiokatcp.Client, n_ants: int) -> None:
        # Trying to use beamformer request on wrong stream type
        with pytest.raises(aiokatcp.FailReply, match=r"not a tied-array-channelised-voltage stream"):
            await client.request("beam-quant-gains", "bcp1", 1.0)
        with pytest.raises(aiokatcp.FailReply, match=r"not a tied-array-channelised-voltage stream"):
            await client.request("beam-weights", "bcp1", *([1.0] * n_ants))
        with pytest.raises(aiokatcp.FailReply, match=r"not a tied-array-channelised-voltage stream"):
            await client.request("beam-delays", "bcp1", *(["0.0:0.0"] * n_ants))

        # Vector requests with wrong number of parameters
        with pytest.raises(aiokatcp.FailReply, match=r"Incorrect number of weights \(expected 4, received 3\)"):
            await client.request("beam-weights", "beam_0x", 1.0, 2.0, 3.0)
        with pytest.raises(aiokatcp.FailReply, match=r"Incorrect number of delays \(expected 4, received 5\)"):
            await client.request("beam-delays", "beam_0x", "0:0", "1:1", "2:2", "3:3", "4:4")

        # Bad delay formatting
        with pytest.raises(aiokatcp.FailReply):
            await client.request("beam-delays", "beam_0x", "0", "1", "2", "3")  # Missing ":"
        with pytest.raises(aiokatcp.FailReply):
            await client.request("beam-delays", "beam_0x", "0:0", "1:1", "2:2", "3:3:3")  # Too many :'s
        with pytest.raises(aiokatcp.FailReply):
            await client.request("beam-delays", "beam_0x", "0:0", "1:1", "2:2", "3:2j")  # Not float
