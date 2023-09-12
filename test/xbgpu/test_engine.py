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

from typing import AbstractSet, AsyncGenerator, Callable, Final, Iterable

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
from katgpucbf.xbgpu.engine import XBEngine, XOutput, XPipeline
from katgpucbf.xbgpu.main import make_engine, parse_args, parse_corrprod

from .. import PromDiff
from . import test_parameters
from .test_recv import gen_heap

pytestmark = [pytest.mark.device_filter.with_args(device_filter)]

get_baseline_index = njit(Correlation.get_baseline_index)

ADC_SAMPLE_RATE: Final[float] = 1712e6  # L-band
HEAPS_PER_FENGINE_PER_CHUNK: Final[int] = 2
SEND_RATE_FACTOR: Final[float] = 1.1
SAMPLE_BITWIDTH: Final[int] = 8


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
    def corrprod_outputs(self, corrprod_args: list[str]) -> list[XOutput]:
        """The outputs to run tests against."""
        return [parse_corrprod(corrprod_arg) for corrprod_arg in corrprod_args]

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
        # Define heap shapes needed to generate simulated data.
        heap_shape = (n_channels_per_substream, n_spectra_per_heap, N_POLS, COMPLEX)

        # Generate all the heaps for the different antennas.
        heaps: list[spead2.send.HeapReference] = []
        for ant_index in range(n_ants):
            if ant_index in missing_antennas:
                continue
            sample_array = np.zeros(heap_shape, np.int8)

            # Generate the data for the heap iterating assigning a different value to each channel.
            for chan_index in range(n_channels_per_substream):
                sample_array[chan_index] = feng_sample(batch_index, chan_index, ant_index)

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
        *,
        heap_factory: Callable[[int], list[spead2.send.HeapReference]],
        batch_indices: Iterable[int],
        timestamp_step: int,
        n_ants: int,
        n_channels_per_substream: int,
        frequency: int,
        n_spectra_per_heap: int,
    ) -> list[np.ndarray]:
        """Send a contiguous stream of data to the engine and retrieve the results.

        Each full accumulation (for each corrprod-output) requires
        `heap_accumulation_threshold` batches of heaps. However, `batch_indices`
        is not required to contain full accumulations.

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
        device_results
            List of arrays of all GPU-generated data. One output array per
            corrprod_output, each array has shape
            - (n_accumulations, n_channels_per_substream, n_baselines, COMPLEX)
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
        device_results = [
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

                device_results[i][j] = ig_recv["xeng_raw"].value

        return device_results

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
    ) -> list[str]:
        return [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            f"--corrprod={corrprod_args[0]}",
            f"--corrprod={corrprod_args[1]}",
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

    @pytest.mark.combinations(
        "n_ants, n_channels_total, n_spectra_per_heap, missing_antenna",
        test_parameters.array_size,
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
        [None, 0, 3],
        filter=valid_end_to_end_combination,
    )
    @pytest.mark.parametrize("heap_accumulation_threshold", [(3, 7), (4, 8), (5, 9)])
    async def test_xengine_end_to_end(
        self,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        xbengine: XBEngine,
        n_ants: int,
        n_spectra_per_heap: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        corrprod_outputs: list[XOutput],
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

        timestamp_step = n_samples_between_spectra * n_spectra_per_heap
        missing_antennas = set() if missing_antenna is None else {missing_antenna}

        range_start = frequency
        range_end = range_start + n_channels_per_substream - 1
        for corrprod_output in corrprod_outputs:
            assert xbengine.sensors[f"{corrprod_output.name}.chan-range"].value == f"({range_start},{range_end})"

        # Need a method of capturing synchronised aiokatcp.Sensor updates
        # as they happen in the XBEngine
        actual_sensor_updates: dict[str, list[tuple[bool, aiokatcp.Sensor.Status]]]
        actual_sensor_updates = {
            f"{corrprod_output.name}.rx.synchronised": list() for corrprod_output in corrprod_outputs
        }
        expected_sensor_updates: dict[str, list[tuple[bool, aiokatcp.Sensor.Status]]]
        expected_sensor_updates = {sensor_name: list() for sensor_name in actual_sensor_updates.keys()}

        def sensor_observer(sync_sensor: aiokatcp.Sensor, sensor_reading: aiokatcp.Reading):
            """Record sensor updates in a list for later comparison."""
            actual_sensor_updates[sync_sensor.name].append((sensor_reading.value, sensor_reading.status))

        for corrprod_output in corrprod_outputs:
            xbengine.sensors[f"{corrprod_output.name}.rx.synchronised"].attach(sensor_observer)

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

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            # NOTE: The product of `heap_acc_thresholds` is used in two ways
            # below. Both uses are to ensure there is a whole number of
            # accumulations for *both* XPipelines. The first usage is
            # dual-purpose once more:
            # - In addition to the above, the arbitrary start position for data
            #   in this test dictates the first received batch to be in the
            #   middle of an accumulation.
            # - More accurately, it forces the first accumulation (to be
            #   processed) to only contain one batch of data. This ensures
            #   we test that output dumps are aligned correctly, despite
            #   the first data processed not being on an accumulation
            #   boundary.
            heap_acc_thresholds = [corrprod_output.heap_accumulation_threshold for corrprod_output in corrprod_outputs]
            n_heaps = np.product(heap_acc_thresholds)
            batch_start_index = 12 * n_heaps  # Somewhere arbitrary that isn't zero
            batch_end_index = batch_start_index + n_heaps
            # Add an extra batch before the first full accumulation
            batch_start_index -= HEAPS_PER_FENGINE_PER_CHUNK

            device_results = await self._send_data(
                mock_recv_streams,
                mock_send_stream,
                corrprod_outputs=corrprod_outputs,
                batch_indices=range(batch_start_index, batch_end_index),
                heap_factory=heap_factory,
                timestamp_step=timestamp_step,
                n_ants=n_ants,
                n_channels_per_substream=n_channels_per_substream,
                frequency=frequency,
                n_spectra_per_heap=n_spectra_per_heap,
            )

        incomplete_accums_counters = []
        for i, corrprod_output in enumerate(corrprod_outputs):
            # Or assert if incomplete_accs_total == incomplete_accums_counter * len(xbengine._pipelines)
            incomplete_accums_counter = 0
            base_batch_index = batch_start_index
            for j, device_result in enumerate(device_results[i]):
                # The first heap is an incomplete accumulation containing a
                # single batch, we need to make sure that this is taken into
                # account by the verification function.
                if j == 0:
                    # This is to handle the first accumulation processed. The value
                    # checked here is simply the first in the range.
                    # - Even though :meth:`generate_expected_output` returns a
                    #   zeroed array for an incomplete accumulation, we still need
                    #   to maintain programmatic sense in the values generated here.
                    num_batches_in_current_accumulation = HEAPS_PER_FENGINE_PER_CHUNK
                    incomplete_accums_counter += 1
                else:
                    num_batches_in_current_accumulation = corrprod_output.heap_accumulation_threshold
                    if missing_antenna is not None:
                        incomplete_accums_counter += 1

                expected_output = generate_expected_output(
                    base_batch_index,
                    num_batches_in_current_accumulation,
                    corrprod_output.heap_accumulation_threshold,
                    n_channels_per_substream,
                    n_ants,
                    n_spectra_per_heap,
                    missing_antenna,
                )
                base_batch_index += num_batches_in_current_accumulation
                np.testing.assert_equal(expected_output, device_result)
            incomplete_accums_counters.append(incomplete_accums_counter)

        xpipelines: list[XPipeline] = [pipeline for pipeline in xbengine._pipelines if isinstance(pipeline, XPipeline)]
        for pipeline, device_result, incomplete_accums_counter in zip(
            xpipelines, device_results, incomplete_accums_counters
        ):
            output_name = pipeline.output.name
            n_accumulations_completed = device_result.shape[0]
            assert (
                prom_diff.get_sample_diff("output_x_incomplete_accs_total", {"stream": output_name})
                == incomplete_accums_counter
            )
            assert (
                prom_diff.get_sample_diff("output_x_heaps_total", {"stream": output_name}) == n_accumulations_completed
            )
            # Could manually calculate it here, but it's available inside the send_stream
            assert prom_diff.get_sample_diff("output_x_bytes_total", {"stream": output_name}) == (
                pipeline.send_stream.heap_payload_size_bytes * n_accumulations_completed
            )
            assert prom_diff.get_sample_diff("output_x_visibilities_total", {"stream": output_name}) == (
                n_channels_per_substream * n_baselines * n_accumulations_completed
            )
            assert prom_diff.get_sample_diff("output_x_clipped_visibilities_total", {"stream": output_name}) == 0

            # Verify sensor updates while we're here
            sensor_name = f"{pipeline.output.name}.rx.synchronised"
            # As per the explanation in :func:`~send_data`, the first accumulation
            # is expected to be incomplete.
            expected_sensor_updates[sensor_name].append((False, aiokatcp.Sensor.Status.ERROR))
            # Depending on the `missing_antenna` parameter, the full accumulations
            # will either be all complete or incomplete.
            if missing_antenna is not None:
                expected_sensor_updates[sensor_name] += [(False, aiokatcp.Sensor.Status.ERROR)] * (
                    incomplete_accums_counter - 1
                )
            else:
                expected_sensor_updates[sensor_name] += [(True, aiokatcp.Sensor.Status.NOMINAL)] * (
                    n_accumulations_completed - 1
                )

        assert actual_sensor_updates == expected_sensor_updates

    # This uses parametrize to set fixture values for the test rather than to
    # create multiple tests.
    @pytest.mark.parametrize("n_ants", [4])
    @pytest.mark.parametrize("n_channels_total", [1024])
    @pytest.mark.parametrize("n_spectra_per_heap", [256])
    @pytest.mark.parametrize("heap_accumulation_threshold", [[300, 300]])
    async def test_saturation(
        self,
        context: AbstractContext,
        mock_recv_streams: list[spead2.InprocQueue],
        mock_send_stream: list[spead2.InprocQueue],
        xbengine: XBEngine,
        n_ants: int,
        n_channels_total: int,
        n_channels_per_substream: int,
        frequency: int,
        n_samples_between_spectra: int,
        n_spectra_per_heap: int,
        heap_accumulation_threshold: list[int],
        corrprod_outputs: list[XOutput],
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
