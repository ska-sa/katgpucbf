################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

from typing import Final, List

import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
import spead2.send
import spead2.send.asyncio
from numba import njit

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.xbgpu.correlation import Correlation, device_filter
from katgpucbf.xbgpu.main import make_engine, parse_args

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
def generate_expected_output(batch_start_idx, num_batches, channels, antennas, n_spectra_per_heap):
    """Calculate the expected correlator output.

    This doesn't do a full correlator. It calculates the results according to
    what is expected from the specific input generated in
    :func:`create_heaps`.
    """
    baselines = antennas * (antennas + 1) * 2
    output_array = np.zeros((channels, baselines, COMPLEX), dtype=np.int32)
    for b in range(batch_start_idx, batch_start_idx + num_batches):
        sign = pow(-1, b)
        for c in range(channels):
            h = np.empty((antennas, 2), np.int32)
            v = np.empty((antennas, 2), np.int32)
            for a in range(antennas):
                # This process is a bit non-intuitive. Numba can handle Python's
                # complex numbers, BUT, they are represented as floating-point,
                # not integer. So we have helper functions here.
                h[a, 0] = bounded_int8(sign * b)
                h[a, 1] = bounded_int8(sign * c)
                v[a, 0] = bounded_int8(-sign * a)
                v[a, 1] = bounded_int8(-sign * c)
            for a2 in range(antennas):
                for a1 in range(a2 + 1):
                    bl_idx = get_baseline_index(a1, a2)
                    output_array[c, 4 * bl_idx + 0, :] += cmult_and_scale(h[a1], h[a2], n_spectra_per_heap)
                    output_array[c, 4 * bl_idx + 1, :] += cmult_and_scale(v[a1], h[a2], n_spectra_per_heap)
                    output_array[c, 4 * bl_idx + 2, :] += cmult_and_scale(h[a1], v[a2], n_spectra_per_heap)
                    output_array[c, 4 * bl_idx + 3, :] += cmult_and_scale(v[a1], v[a2], n_spectra_per_heap)

    return output_array


class TestEngine:
    r"""Grouping of unit tests for :class:`.XBEngine`\'s various functionality."""

    @staticmethod
    def _create_heaps(
        timestamp: int,
        batch_index: int,
        n_ants: int,
        n_channels_per_stream: int,
        n_spectra_per_heap: int,
    ) -> List[spead2.send.HeapReference]:
        """Generate a deterministic input for sending to the XBEngine.

        One heap is generated per antenna in the array. All heaps will have the
        same timestamp. The 8-bit complex samples for both pols are grouped
        together and encoded as a single 32-bit unsigned integer value. A heap is
        composed of multiple channels. Per channel, all 32-bit values are kept
        constant. This makes for faster verification with the downside being that
        if samples within the channel range get mixed up, this will not be
        detected.

        The coded 32-bit value is a combination of the antenna index, batch_index
        and channel index. The sample value is equal to the following:

        .. code-block:: python

            coded_sample_value = (np.uint8(-sign * chan_index) << 24) + (np.uint8(-sign * ant_index) << 16) +
                                 (np.uint8(sign * chan_index) << 8) + np.uint8(sign * batch_index)

        The sign value is 1 for even batch indices and -1 for odd ones for an even
        spread of positive and negative numbers. An added nuance is that these
        8-bit values are clamped to -127 as -128 is not supported by the Tensor
        Cores.

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

        Returns
        -------
        heaps
            A list of HeapReference objects as accepted by :func:`.send_heaps`.
        """
        # 1. Define heap shapes that will be needed to generate simulated data.
        heap_shape = (n_channels_per_stream, n_spectra_per_heap, N_POLS, COMPLEX)
        # The heaps shape has been modified with the COMPLEX dimension and n_pols
        # dimension equal to 1 instead of 2. This is because we treat the two
        # 8-bit complex samples for both pols as a single 32-bit value when
        # generating the simulated data. We correct the shape before sending.
        modified_heap_shape = (n_channels_per_stream, n_spectra_per_heap, N_POLS // 2, COMPLEX // 2)

        # 2. Generate all the heaps for the different antennas.
        heaps: List[spead2.send.HeapReference] = []
        for ant_index in range(n_ants):
            sample_array = np.zeros(modified_heap_shape, np.uint32)

            # 2.1 Generate the data for the heap iterating assigning a different value to each channel.
            for chan_index in range(n_channels_per_stream):

                # 2.1.1 Determine the sign modifier value
                sign = 1 if batch_index % 2 == 0 else -1

                def clamp_to_127(input: int) -> np.int8:
                    """Clamp the output to [-127, 127] to support Tensor Cores."""
                    retval = np.int8(input)
                    if retval == -128:
                        return np.int8(-127)
                    else:
                        return retval

                # 2.1.1 Generate the samples to combine into a code word.
                pol0_real = clamp_to_127(sign * batch_index)
                pol0_imag = clamp_to_127(sign * chan_index)
                pol1_real = clamp_to_127(-sign * ant_index)
                pol1_imag = clamp_to_127(-sign * chan_index)

                # 2.1.2 Combine values into a code word. The values are all cast
                # to uint8s as when I was casting them to int8s, the sign
                # extension would behave strangely and what I expected to be in
                # the code word would be one bit off.
                coded_sample_value = np.uint32(
                    (np.uint8(pol1_imag) << 24)
                    + (np.uint8(pol1_real) << 16)
                    + (np.uint8(pol0_imag) << 8)
                    + (np.uint8(pol0_real) << 0)
                )

                # 2.1.3 Set each sample in this channel to contain the same value.
                sample_array[chan_index][:] = coded_sample_value

            # 2.2 Change dtype and shape of the array back to the correct values
            # required by the receiver. The data itself is not modified, its just
            # how it is intepreted that is changed.
            sample_array = sample_array.view(np.int8)
            sample_array = np.reshape(sample_array, heap_shape)

            # Create the heap, configure it to send immediate values in each
            # packet, add it to a list of HeapReferences.
            heap = gen_heap(timestamp, ant_index, n_channels_per_stream * CHANNEL_OFFSET, sample_array)
            heap.repeat_pointers = True
            heaps.append(spead2.send.HeapReference(heap))

        return heaps

    @staticmethod
    async def _recv_process(
        recv_stream: spead2.recv.asyncio.Stream,
        n_ants: int,
        n_channels_per_stream: int,
        n_spectra_per_heap: int,
        n_accumulations: int,
        heap_accumulation_threshold: int,
        timestamp_step,
    ) -> None:
        """Receive and verify XBEngine output data."""
        # It is expected that the first packet will be a descriptor.
        ig_recv = spead2.ItemGroup()
        heap = await recv_stream.get()
        items = ig_recv.update(heap)
        assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

        # We expect to receive (n_accumulations + 1) output heaps. Each of
        # these heaps is verified for correctness
        for i in range(n_accumulations + 1):
            # Wait for heap to be ready and then update out item group
            # with the new values.
            heap = await recv_stream.get()
            items = ig_recv.update(heap)

            # The first heap is an incomplete accumulation containing a
            # single batch, we need to make sure that this is taken into
            # account by the verification function.
            if i == 0:
                num_batches_in_current_accumulation = 1
                base_batch_index = heap_accumulation_threshold - 1
            else:
                num_batches_in_current_accumulation = heap_accumulation_threshold
                base_batch_index = i * heap_accumulation_threshold

            # Ensure that the timestamp from the heap is what we expect.
            assert (
                ig_recv["timestamp"].value % (timestamp_step * heap_accumulation_threshold) == 0
            ), "Output timestamp is not a multiple of timestamp_step * heap_accumulation_threshold."

            assert ig_recv["timestamp"].value == timestamp_step * heap_accumulation_threshold * i, (
                "Output timestamp is not correct. "
                f"Expected: {hex(timestamp_step * heap_accumulation_threshold * i)}, "
                f"actual: {hex(ig_recv['timestamp'].value)}."
            )

            assert ig_recv["frequency"].value == n_channels_per_stream * CHANNEL_OFFSET, (
                "Output channel offset not correct. "
                f"Expected: {n_channels_per_stream * CHANNEL_OFFSET}, "
                f"actual: {ig_recv['frequency'].value}."
            )

            expected_output = generate_expected_output(
                base_batch_index,
                num_batches_in_current_accumulation,
                n_channels_per_stream,
                n_ants,
                n_spectra_per_heap,
            )

            gpu_result = ig_recv["xeng_raw"].value
            np.testing.assert_equal(expected_output, gpu_result)

    @pytest.mark.combinations(
        "n_ants, n_channels_total, n_spectra_per_heap",
        test_parameters.array_size,
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
    )
    async def test_xengine_end_to_end(
        self,
        context,
        n_ants,
        n_spectra_per_heap,
        n_channels_total,
        mock_recv_streams,
    ):
        """
        End-to-end test for the XBEngine.

        Simulated input data is generated and passed to the XBEngine, which
        produces an output which is then verified.

        The simulated data is not random, it is encoded based on certain
        parameters, this allows the verification function to generate the
        correct data to compare to the xbengine without performing the full
        correlation algorithm, greatly improving processing time.

        This test simulates an incomplete accumulation at the start of transmission
        to ensure that the auto-resync logic works correctly.
        """
        n_samples_between_spectra = 2 * n_channels_total

        # Get a realistic number of engines, round up to the next power of 2.
        n_engines = 1
        while n_engines < n_ants:
            n_engines *= 2
        n_channels_per_stream = n_channels_total // n_engines
        heap_accumulation_threshold = 4
        n_accumulations = 3

        # Header is 12 fields of 8 bytes each: So 96 bytes of header
        max_packet_size = n_spectra_per_heap * N_POLS * COMPLEX * SAMPLE_BITWIDTH // 8 + 96
        timestamp_step = n_samples_between_spectra * n_spectra_per_heap

        # Create source stream object - simulates received data.
        feng_stream_config = spead2.send.StreamConfig(
            max_packet_size=max_packet_size, max_heaps=n_ants * HEAPS_PER_FENGINE_PER_CHUNK * 10
        )
        feng_stream = spead2.send.asyncio.InprocStream(spead2.ThreadPool(), mock_recv_streams, feng_stream_config)
        queue = spead2.InprocQueue()
        recv_stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(), spead2.recv.StreamConfig(max_heaps=100))
        recv_stream.add_inproc_reader(queue)

        arglist = [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--array-size={n_ants}",
            f"--channels={n_channels_total}",
            f"--channels-per-substream={n_channels_per_stream}",
            f"--samples-between-spectra={n_samples_between_spectra}",
            f"--channel-offset-value={n_channels_per_stream * CHANNEL_OFFSET}",
            f"--spectra-per-heap={n_spectra_per_heap}",
            f"--heaps-per-fengine-per-chunk={HEAPS_PER_FENGINE_PER_CHUNK}",
            f"--heap-accumulation-threshold={heap_accumulation_threshold}",
            "--src-interface=lo",
            "--dst-interface=lo",
            "239.10.11.4:7149",  # src
            "239.21.11.4:7149",  # dst
        ]

        args = parse_args(arglist)
        xbengine, _ = make_engine(context, args)

        # 7. Add transports to xbengine.
        xbengine.add_inproc_sender_transport(queue)
        await xbengine.send_stream.send_descriptor_heap()

        await xbengine.start()

        # Generate Data to be sent to the receiver. We are performing
        # <n_accumulations> full accumulations. Each accumulation requires
        # heap_accumulation_threshold batches of heaps. Additionally, we generate
        # one extra batch to simulate an incomplete accumulation to check that
        # dumps are aligned correctly even if the first received batch is from the
        # middle of an accumulation.
        for i in range(heap_accumulation_threshold * n_accumulations + 1):
            # 6.1. Generate the batch index. By setting the first batch timestamp
            # value to timestamp_step * (heap_accumulation_threshold - 1) we
            # generate only a single batch for the first accumulation as the
            # accumulations are aligned to integer multiples of
            # heap_accumulation_threshold * timestamp_step
            batch_index = i + (heap_accumulation_threshold - 1)
            timestamp = batch_index * timestamp_step
            heaps = self._create_heaps(timestamp, batch_index, n_ants, n_channels_per_stream, n_spectra_per_heap)
            await feng_stream.async_send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        for q in mock_recv_streams:
            q.stop()

        await self._recv_process(
            recv_stream,
            n_ants,
            n_channels_per_stream,
            n_spectra_per_heap,
            n_accumulations,
            heap_accumulation_threshold,
            timestamp_step,
        )
        await xbengine.stop()
