################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

from collections.abc import Iterable
from typing import AbstractSet, AsyncGenerator, Callable, Final

import aiokatcp
import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
import spead2.send
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext
from numba import njit

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.fgpu.send import PREAMBLE_SIZE
from katgpucbf.xbgpu import METRIC_NAMESPACE
from katgpucbf.xbgpu.correlation import Correlation, device_filter
from katgpucbf.xbgpu.engine import XBEngine
from katgpucbf.xbgpu.main import make_engine, parse_args

from .. import PromDiff
from . import test_parameters
from .test_recv import gen_heap

pytestmark = [pytest.mark.device_filter.with_args(device_filter)]

get_baseline_index = njit(Correlation.get_baseline_index)

ADC_SAMPLE_RATE: Final[float] = 1712e6  # L-band
HEAPS_PER_FENGINE_PER_CHUNK: Final[int] = 2
SEND_RATE_FACTOR: Final[float] = 1.1
SAMPLE_BITWIDTH: Final[int] = 8
CHANNEL_OFFSET: Final[int] = 4  # Selected fairly arbitrarily, just to be something.


@njit
def bounded_int8(val):
    """Create an int8 value bounded to the range [-127, 127].

    Returns
    -------
    np.int32
        This datatype is used to avoid overflow issues when the output is used
        in multiplication operations, but the value is bounded by the closed
        interval described above.
    """
    val = np.int8(val)
    if val == -128:
        val += 1
    return np.int32(val)


@njit
def feng_sample(batch: int, channel: int, antenna: int) -> np.ndarray:
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
    return np.array(
        [
            [bounded_int8(sign * batch), bounded_int8(sign * channel)],
            [bounded_int8(-sign * antenna), bounded_int8(-sign * channel)],
        ],
        dtype=np.int8,
    )


@njit
def cmult_and_scale(a, b, c):
    """Multiply ``a`` and ``conj(b)``, and scale the result by ``c``.

    Both ``a`` and ``b`` inputs and the output are 2-element arrays of
    np.int32, representing the real and imaginary components. ``c`` is a
    scalar.
    """
    result = np.empty((2,), dtype=np.int32)
    result[0] = a[0] * b[0] + a[1] * b[1]
    result[1] = a[1] * b[0] - a[0] * b[1]
    result *= c
    return result


@njit
def generate_expected_output(
    batch_start_idx,
    num_batches,
    heap_accumulation_threshold,
    channels,
    antennas,
    n_spectra_per_heap,
    missing_antenna,
):
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
            in_data = np.empty((antennas, N_POLS, COMPLEX), np.int32)
            for a in range(antennas):
                in_data[a] = feng_sample(b, c, a)
            for a2 in range(antennas):
                for a1 in range(a2 + 1):
                    bl_idx = get_baseline_index(a1, a2)
                    output_piece = output_array[c, 4 * bl_idx : 4 * bl_idx + 4, :]
                    output_piece[0] += cmult_and_scale(in_data[a1, 0], in_data[a2, 0], n_spectra_per_heap)
                    output_piece[1] += cmult_and_scale(in_data[a1, 1], in_data[a2, 0], n_spectra_per_heap)
                    output_piece[2] += cmult_and_scale(in_data[a1, 0], in_data[a2, 1], n_spectra_per_heap)
                    output_piece[3] += cmult_and_scale(in_data[a1, 1], in_data[a2, 1], n_spectra_per_heap)

    # Flag missing data
    for a2 in range(antennas):
        for a1 in range(a2 + 1):
            bl_idx = get_baseline_index(a1, a2)
            if a1 == missing_antenna or a2 == missing_antenna:
                output_array[:, 4 * bl_idx : 4 * bl_idx + 4, 0] = -(2**31)
                output_array[:, 4 * bl_idx : 4 * bl_idx + 4, 1] = 1

    return output_array


def valid_end_to_end_combination(combo: dict) -> bool:
    """Check whether a combination for :meth:`TestEngine.test_xengine_end_to_end` is valid."""
    n_ants = combo["n_ants"]
    missing_antenna = combo["missing_antenna"]
    if missing_antenna is None:
        return True
    # Don't want to delete all the data, or an out-of-range antenna
    return n_ants > 1 and missing_antenna < n_ants


class TestEngine:
    r"""Grouping of unit tests for :class:`.XBEngine`\'s various functionality."""

    @staticmethod
    def _create_heaps(
        timestamp: int,
        batch_index: int,
        n_ants: int,
        n_channels_per_stream: int,
        n_spectra_per_heap: int,
        missing_antennas: AbstractSet[int] = frozenset(),
    ) -> list[spead2.send.HeapReference]:
        """Generate a deterministic input for sending to the XBEngine.

        One heap is generated per antenna in the array. All heaps will have the
        same timestamp. A heap is composed of multiple channels. Per channel,
        all values are kept constant. This makes for faster verification with
        the downside being that if samples within the channel range get mixed
        up, this will not be detected. See :meth:`feng_sample` for the formula
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
        n_channels_per_stream
            The number of frequency channels contained in a heap.
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
        # Define heap shapes needed to generate simulated data.
        heap_shape = (n_channels_per_stream, n_spectra_per_heap, N_POLS, COMPLEX)

        # Generate all the heaps for the different antennas.
        heaps: list[spead2.send.HeapReference] = []
        for ant_index in range(n_ants):
            if ant_index in missing_antennas:
                continue
            sample_array = np.zeros(heap_shape, np.int8)

            # Generate the data for the heap iterating assigning a different value to each channel.
            for chan_index in range(n_channels_per_stream):
                sample_array[chan_index] = feng_sample(batch_index, chan_index, ant_index)

            # Create the heap, add it to a list of HeapReferences.
            heap = gen_heap(timestamp, ant_index, n_channels_per_stream * CHANNEL_OFFSET, sample_array)
            heaps.append(spead2.send.HeapReference(heap))

        return heaps

    def _make_feng(
        self, queue: spead2.InprocQueue, max_packet_size: int, max_heaps: int
    ) -> "spead2.send.asyncio.AsyncStream":
        """Create send stream for a fake F-Engine."""
        feng_stream_config = spead2.send.StreamConfig(
            max_packet_size=max_packet_size,
            max_heaps=max_heaps,
        )
        return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), [queue], feng_stream_config)

    async def _send_data(
        self,
        mock_recv_stream: spead2.InprocQueue,
        recv_stream: spead2.recv.asyncio.Stream,
        *,
        heap_factory: Callable[[int], list[spead2.send.HeapReference]],
        heap_accumulation_threshold: int,
        batch_indices: Iterable[int],
        timestamp_step: int,
        n_ants: int,
        n_channels_per_stream: int,
        n_spectra_per_heap: int,
    ) -> np.ndarray:
        """Send a contiguous stream of data to the engine and retrieve the results.

        Each full accumulation requires `heap_accumulation_threshold` batches of
        heaps. However, `batch_indices` is not required to contain full
        accumulations.

        Parameters
        ----------
        mock_recv_stream
            Fixture
        recv_stream
            InprocStream to receive data output by XBEngine.
        heap_factory
            Callback that takes a batch index and returns the heaps for that index.
        heap_accumulation_threshold
            Number of consecutive heaps to process in a single accumulation.
        batch_indices
            Indices of the batches to send. These must be strictly increasing,
            but need not be contiguous.
        timestamp_step
            Timestamp step between each received heap processed.
        n_ants, n_channels_per_stream, n_spectra_per_heap
            See :meth:`_create_heaps` for more info.

        Returns
        -------
        device_results
            Array of all GPU-generated data of shape
            - (n_total_accumulations, n_channels_per_stream, n_baselines, COMPLEX)
        """
        max_packet_size = n_spectra_per_heap * N_POLS * COMPLEX * SAMPLE_BITWIDTH // 8 + PREAMBLE_SIZE
        max_heaps = n_ants * HEAPS_PER_FENGINE_PER_CHUNK * 10
        feng_stream = self._make_feng(mock_recv_stream, max_packet_size, max_heaps)

        accumulation_indices_set = set()
        for batch_index in batch_indices:
            heaps = heap_factory(batch_index)
            accumulation_indices_set.add(batch_index // heap_accumulation_threshold)
            await feng_stream.async_send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

        mock_recv_stream.stop()

        # It is expected that the first packet will be a descriptor.
        ig_recv = spead2.ItemGroup()
        heap = await recv_stream.get()
        items = ig_recv.update(heap)
        assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

        accumulation_indices = sorted(accumulation_indices_set)
        n_baselines = n_ants * (n_ants + 1) * 2
        device_results = np.zeros(
            shape=(len(accumulation_indices), n_channels_per_stream, n_baselines, COMPLEX), dtype=np.int32
        )
        for i, accumulation_index in enumerate(accumulation_indices):
            # Wait for heap to be ready and then update out item group
            # with the new values.
            heap = await recv_stream.get()

            while (updated_items := set(ig_recv.update(heap))) == set():
                # Test has gone on long enough that we've received another descriptor
                heap = await recv_stream.get()
            assert updated_items == {"frequency", "timestamp", "xeng_raw"}
            # Ensure that the timestamp from the heap is what we expect.
            assert (
                ig_recv["timestamp"].value % (timestamp_step * heap_accumulation_threshold) == 0
            ), "Output timestamp is not a multiple of timestamp_step * heap_accumulation_threshold."

            assert ig_recv["timestamp"].value == accumulation_index * timestamp_step * heap_accumulation_threshold, (
                "Output timestamp is not correct. "
                f"Expected: {hex(accumulation_index * timestamp_step * heap_accumulation_threshold)}, "
                f"actual: {hex(ig_recv['timestamp'].value)}."
            )

            assert ig_recv["frequency"].value == n_channels_per_stream * CHANNEL_OFFSET, (
                "Output channel offset not correct. "
                f"Expected: {n_channels_per_stream * CHANNEL_OFFSET}, "
                f"actual: {ig_recv['frequency'].value}."
            )

            device_results[i] = ig_recv["xeng_raw"].value

        return device_results

    @pytest.fixture
    def n_engines(self, n_ants: int) -> int:
        """Get a realistic number of engines by rounding up to the next power of 2."""
        n_engines = 1
        while n_engines < n_ants:
            n_engines *= 2
        return n_engines

    @pytest.fixture
    def n_channels_per_stream(self, n_channels_total: int, n_engines: int) -> int:  # noqa: D102
        return n_channels_total // n_engines

    @pytest.fixture
    def n_samples_between_spectra(self, n_channels_total: int) -> int:  # noqa: D102
        # Will need to be updated for narrowband
        return 2 * n_channels_total

    @pytest.fixture
    def recv_stream(self, mock_send_stream: spead2.InprocQueue) -> spead2.recv.asyncio.Stream:
        """Stream on the receive end of the ``mock_send_stream`` fixture."""
        stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(), spead2.recv.StreamConfig(max_heaps=100))
        stream.add_inproc_reader(mock_send_stream)
        return stream

    @pytest.fixture
    async def xbengine(
        self,
        context: AbstractContext,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_stream: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        heap_accumulation_threshold: int,
    ) -> AsyncGenerator[XBEngine, None]:
        """Create and start an engine based on the fixture values."""
        arglist = [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            "--corrprod=name=bcp1,"
            f"heap_accumulation_threshold={heap_accumulation_threshold},"
            "dst=239.21.11.4:7149",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--array-size={n_ants}",
            f"--channels={n_channels_total}",
            f"--channels-per-substream={n_channels_per_stream}",
            f"--samples-between-spectra={n_samples_between_spectra}",
            f"--channel-offset-value={n_channels_per_stream * CHANNEL_OFFSET}",
            f"--spectra-per-heap={n_spectra_per_heap}",
            f"--heaps-per-fengine-per-chunk={HEAPS_PER_FENGINE_PER_CHUNK}",
            "--sync-epoch=1234567890",
            "--src-interface=lo",
            "--dst-interface=lo",
            "--tx-enabled",
            "239.10.11.4:7149",  # src
        ]
        args = parse_args(arglist)
        xbengine, _ = make_engine(context, args)
        await xbengine.start()

        yield xbengine

        await xbengine.stop()

    @pytest.mark.combinations(
        "n_ants, n_channels_total, n_spectra_per_heap, missing_antenna",
        test_parameters.array_size,
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
        [None, 0, 3],
        filter=valid_end_to_end_combination,
    )
    @pytest.mark.parametrize("heap_accumulation_threshold", [4])
    async def test_xengine_end_to_end(
        self,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: spead2.InprocQueue,
        recv_stream: spead2.recv.asyncio.Stream,
        xbengine: XBEngine,
        n_ants: int,
        n_spectra_per_heap: int,
        n_channels_total: int,
        n_channels_per_stream: int,
        n_samples_between_spectra: int,
        heap_accumulation_threshold: int,
        missing_antenna: int | None,
    ):
        """
        End-to-end test for the XBEngine.

        Simulated input data is generated and passed to the XBEngine, yielding
        output results which are then verified.

        The simulated data is not random, it is encoded based on certain
        parameters, this allows the verification function to generate the
        correct data to compare to the xbengine without performing the full
        correlation algorithm, greatly improving processing time.

        This test simulates an incomplete accumulation at the start of transmission
        to ensure that the auto-resync logic works correctly. Data is also
        generated from a timestamp starting after the first accumulation
        boundary to more accurately test the setting of the first output
        packet's timestamp (to be non-zero).
        """
        n_baselines = n_ants * (n_ants + 1) * 2
        first_accumulation_index = 123
        n_full_accumulations = 3
        n_total_accumulations = n_full_accumulations + 1
        timestamp_step = n_samples_between_spectra * n_spectra_per_heap
        missing_antennas = set() if missing_antenna is None else {missing_antenna}

        range_start = n_channels_per_stream * CHANNEL_OFFSET
        range_end = range_start + n_channels_per_stream - 1
        assert xbengine.sensors["chan-range"].value == f"({range_start},{range_end})"

        # Need a method of capturing synchronised aiokatcp.Sensor updates
        # as they happen in the XBEngine
        actual_sensor_updates: list[tuple[bool, aiokatcp.Sensor.Status]] = []

        def sensor_observer(sync_sensor: aiokatcp.Sensor, sensor_reading: aiokatcp.Reading):
            """Record sensor updates in a list for later comparison."""
            actual_sensor_updates.append((sensor_reading.value, sensor_reading.status))

        xbengine.sensors["rx.synchronised"].attach(sensor_observer)

        def heap_factory(batch_index: int) -> list[spead2.send.HeapReference]:
            timestamp = batch_index * timestamp_step
            return self._create_heaps(
                timestamp,
                batch_index,
                n_ants,
                n_channels_per_stream,
                n_spectra_per_heap,
                missing_antennas=missing_antennas,
            )

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            # Generate one extra chunk to simulate an incomplete accumulation
            # to check that dumps are aligned correctly - even if the first
            # received batch is from the middle of an accumulation.
            batch_start_index = (
                first_accumulation_index + 1
            ) * heap_accumulation_threshold - HEAPS_PER_FENGINE_PER_CHUNK
            batch_end_index = (first_accumulation_index + 1 + n_full_accumulations) * heap_accumulation_threshold
            device_results = await self._send_data(
                mock_recv_stream,
                recv_stream,
                heap_accumulation_threshold=heap_accumulation_threshold,
                batch_indices=range(batch_start_index, batch_end_index),
                heap_factory=heap_factory,
                timestamp_step=timestamp_step,
                n_ants=n_ants,
                n_channels_per_stream=n_channels_per_stream,
                n_spectra_per_heap=n_spectra_per_heap,
            )

        incomplete_accums_counter = 0
        base_batch_index = batch_start_index
        for i in range(n_total_accumulations):
            # The first heap is an incomplete accumulation containing a
            # single batch, we need to make sure that this is taken into
            # account by the verification function.
            if i == 0:
                # This is to handle the first accumulation processed. The value
                # checked here is simply the first in the range.
                # - Even though :meth:`generate_expected_output` returns a
                #   zeroed array for an incomplete accumulation, we still need
                #   to maintain programmatic sense in the values generated here.
                num_batches_in_current_accumulation = HEAPS_PER_FENGINE_PER_CHUNK
                incomplete_accums_counter += 1
            else:
                num_batches_in_current_accumulation = heap_accumulation_threshold
                if missing_antenna is not None:
                    incomplete_accums_counter += 1

            expected_output = generate_expected_output(
                base_batch_index,
                num_batches_in_current_accumulation,
                heap_accumulation_threshold,
                n_channels_per_stream,
                n_ants,
                n_spectra_per_heap,
                missing_antenna,
            )
            base_batch_index += num_batches_in_current_accumulation

            np.testing.assert_equal(expected_output, device_results[i])

        assert prom_diff.get_sample_diff("output_x_incomplete_accs_total") == incomplete_accums_counter
        assert prom_diff.get_sample_diff("output_x_heaps_total") == n_total_accumulations
        # Could manually calculate it here, but it's available inside the send_stream
        assert prom_diff.get_sample_diff("output_x_bytes_total") == (
            xbengine.send_stream.heap_payload_size_bytes * n_total_accumulations
        )
        assert prom_diff.get_sample_diff("output_x_visibilities_total") == (
            n_channels_per_stream * n_baselines * n_total_accumulations
        )
        assert prom_diff.get_sample_diff("output_x_clipped_visibilities_total") == 0

        expected_sensor_updates: list[tuple[bool, aiokatcp.Sensor.Status]] = []
        # As per the explanation in :func:`~send_data`, the first accumulation
        # is expected to be incomplete.
        expected_sensor_updates.append((False, aiokatcp.Sensor.Status.ERROR))
        # Depending on the `missing_antenna` parameter, the full accumulations
        # will either be all complete or incomplete.
        if missing_antenna is not None:
            expected_sensor_updates += [(False, aiokatcp.Sensor.Status.ERROR)] * n_full_accumulations
        else:
            expected_sensor_updates += [(True, aiokatcp.Sensor.Status.NOMINAL)] * n_full_accumulations

        assert actual_sensor_updates == expected_sensor_updates

    # This uses parametrize to set fixture values for the test rather than to
    # create multiple tests.
    @pytest.mark.parametrize("n_ants", [4])
    @pytest.mark.parametrize("n_channels_total", [1024])
    @pytest.mark.parametrize("n_spectra_per_heap", [256])
    @pytest.mark.parametrize("heap_accumulation_threshold", [300])
    async def test_saturation(
        self,
        context: AbstractContext,
        mock_recv_stream: spead2.InprocQueue,
        mock_send_stream: spead2.InprocQueue,
        recv_stream: spead2.recv.asyncio.Stream,
        xbengine: XBEngine,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_stream: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        heap_accumulation_threshold: int,
    ):
        """Test saturation statistics.

        .. todo::

           After the implementation is updated to avoid counting missing data
           as saturated, extend the test to check that.
        """
        timestamp_step = n_samples_between_spectra * n_spectra_per_heap
        n_baselines = n_ants * (n_ants + 1) * 2

        def heap_factory(batch_index: int) -> list[spead2.send.HeapReference]:
            timestamp = timestamp_step * batch_index
            data = np.full((n_channels_per_stream, n_spectra_per_heap, N_POLS, COMPLEX), 127, np.int8)
            return [
                spead2.send.HeapReference(gen_heap(timestamp, ant_index, n_channels_per_stream * CHANNEL_OFFSET, data))
                for ant_index in range(n_ants)
            ]

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            await self._send_data(
                mock_recv_stream,
                recv_stream,
                heap_accumulation_threshold=heap_accumulation_threshold,
                batch_indices=range(0, heap_accumulation_threshold),
                heap_factory=heap_factory,
                timestamp_step=timestamp_step,
                n_ants=n_ants,
                n_channels_per_stream=n_channels_per_stream,
                n_spectra_per_heap=n_spectra_per_heap,
            )
            await xbengine.stop()

        assert prom_diff.get_sample_diff("output_x_visibilities_total") == n_channels_per_stream * n_baselines
        assert prom_diff.get_sample_diff("output_x_clipped_visibilities_total") == n_channels_per_stream * n_baselines
