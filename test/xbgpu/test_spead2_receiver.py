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

"""
Module for performing unit tests on the xbgpu SPEAD2 receiver.

Testing network code is difficult to do on a single thread. SPEAD2 has the concept of transports. A transport generally
receives data from a network. SPEAD2 provides two other transports that can receive simulated network data - these can
be used for testing the receiver in once process. The two transports are an inproc and a buffer transport. The
inproc transport is more flexible but requires porting the inproc code to xbgpu. So we use the buffer one instead.
It is more limited but easier to work with. One downside of the buffer transport is that it cannot interleave packets
from different antennas. This functionality has not yet been added to the buffer transport but it is available in the
inproc transport.

NOTE: A downside of this test is that it does not check that the packet formats are exactly correct. This test will
ensure that the packets are transmitted in a way that they are able to be assembled into a heap by any SPEAD2 receiver
or a full implementation of the SPEAD protocol. However, the exact packet size and the presence of repeat pointers
within the a packet are not checked. Some sort of external test should be done to check this. See the
display_fsim_multicast_packets.py script in the scratch folder of this repo as a starting point to check packet formats.

TODO: Turn create_test_objects() into a pytest fixture.
"""

import asyncio
import logging

import numpy as np
import pytest
import spead2
import spead2.send

from katgpucbf.spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, TIMESTAMP_ID
from katgpucbf.xbgpu.recv import Chunk, make_stream

from . import test_parameters

logging.basicConfig(level=logging.INFO)

# 3. Define Constants

# 3.2 Explicit note a complex sample has real and imaginay samples.
complexity = 2


def create_test_objects(
    n_ants: int,
    n_channels_per_stream: int,
    n_spectra_per_heap: int,
    n_pols: int,
    sample_bits: int,
    heaps_per_fengine_per_chunk: int,
    max_active_chunks: int,
    timestamp_step: int,
):
    """Create all objects required to run a SPEAD receiver test.

    This function exists so that it can be called in multiple different types of tests without having to duplicate code.
    It could potentially be an asyncio fixture but I have not looked further into that.

    Parameters
    ----------
    n_ants: int
        The number of antennas that data will be received from
    n_channels_per_stream: int
        The number of frequency channels contained in the stream.
    n_spectra_per_heap: int
        The number of time samples received per frequency channel.
    n_pols: int
        The number of pols per antenna. Expected to always be 2 at the moment.
    sample_bits: int
        The number of bits per sample. Only 8 bits is supported at the moment.
    heaps_per_fengine_per_chunk: int
        Each chunk out of the SPEAD2 receiver will contain multiple heaps from
        each antenna. This parameters specifies the number of heaps per antenna
        that each chunk will contain.
    max_active_chunks: int
        The maximum number of chunks that can be received concurrently.
    timestamp_step: int
        Each heap contains a timestamp. The timestamp between consecutive heaps
        changes depending on the FFT size and the number of time samples per
        channel. This parameter defines the difference in timestamp values
        between consecutive heaps. This parameter can be calculated from the
        array configuration parameters for power-of-two array sizes, but is
        configurable to allow for greater flexibility during testing.

    Returns
    -------
    source_stream: spead2.send.BytesStream
        Source SPEAD2 object that will generate the byte array representing
        simulated data.
    ig: spead2.send.ItemGroup
        The ig is used to generate heaps that will  be passed to the source
        stream.
    receiver_stream: spead2.recv.ChunkRingStream
        The receiver under test - will receive data from the source_stream.
    async_ringbuffer: spead2.recv.asyncio.Ringbuffer
        Wraps the receiver_stream ringbuffer so that it can be called using
        asyncio in python.
    """
    # 1. Calculate important parameters.
    max_packet_size = (
        n_spectra_per_heap * n_pols * complexity * sample_bits // 8 + 96
    )  # Header is 12 fields of 8 bytes each: So 96 bytes of header
    heap_shape = (n_channels_per_stream, n_spectra_per_heap, n_pols, complexity)

    # 2. Create source_stream object - transforms "transmitted" heaps into a byte array to simulate received data.
    thread_pool = spead2.ThreadPool()
    source_stream = spead2.send.BytesStream(
        thread_pool,
        spead2.send.StreamConfig(max_packet_size=max_packet_size, max_heaps=n_ants * heaps_per_fengine_per_chunk),
    )

    # 3. Create ItemGroup and add all the required fields.
    ig = spead2.send.ItemGroup(flavour=FLAVOUR)
    ig.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=[],
        format=[("u", FLAVOUR.heap_address_bits)],
    )
    ig.add_item(
        FENG_ID_ID,
        "feng_id",
        "F-Engine heap is received from.",
        shape=[],
        format=[("u", FLAVOUR.heap_address_bits)],
    )
    ig.add_item(
        FREQUENCY_ID,
        "frequency",
        "Value of first channel in collections stored here.",
        shape=[],
        format=[("u", FLAVOUR.heap_address_bits)],
    )
    ig.add_item(FENG_RAW_ID, "feng_raw", "Raw Channelised data", shape=heap_shape, dtype=np.int8)

    # 4. Configure receiver

    # 4.2 Create ringbuffer that all received chunks will be placed on.
    ringbuffer_capacity = 10
    ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(ringbuffer_capacity)

    # 4.3 Create Receiver
    thread_affinity = 2  # This ties the thread to the CPU core. 2 has been chosen at random.
    receiver_stream = make_stream(
        n_ants,
        n_channels_per_stream,
        n_spectra_per_heap,
        sample_bits,
        timestamp_step,
        heaps_per_fengine_per_chunk,
        max_active_chunks,
        ringbuffer,
        thread_affinity,
    )

    # 4.4 Create empty chunks and add them to the receiver empty queue.
    src_chunks_per_stream = max_active_chunks + 1  # Make sure it works with the minimum sane value
    chunk_heaps = n_ants * heaps_per_fengine_per_chunk
    chunk_bytes = chunk_heaps * n_channels_per_stream * n_spectra_per_heap * sample_bits * n_pols * complexity // 8
    for _ in range(src_chunks_per_stream):
        buf = np.empty((chunk_bytes,), np.uint8)
        present = np.zeros((chunk_heaps,), np.uint8)
        chunk = Chunk(data=buf, present=present)
        receiver_stream.add_free_chunk(chunk)

    # 6. Return relevant objects
    return source_stream, ig, receiver_stream, ringbuffer


def create_heaps(
    timestamp: int,
    id: int,
    n_ants: int,
    n_channels_per_stream: int,
    n_spectra_per_heap: int,
    n_pols: int,
    ig: spead2.send.ItemGroup,
):
    """
    Generate a list of heaps to send via the source_stream.

    One heap is generated per antenna in the array. All heaps will have the same timestamp. The 8-bit complex samples
    are treated as a single 16-bit value. Per heap, all sample values are the same. This makes for faster verification
    (The downside is that if the packets in a heap get mixed up, this will not be detected - however this is something
    that is expected to be picked up in the SPEAD2 unit tests). The coded sample is a combination of the antenna index
    and a unique 8-bit ID that can is passed to this function. The sample value is equal to the following:

    coded_sample_value = (np.uint8(id) << 8) + np.uint8(ant_index)

    Parameters
    ----------
    timestamp: int
        The timestamp that will be assigned to all heaps.
    id: int
        8-bit value that will be encoded into all samples in this set of generated heaps.
    n_ants: int
        The number of antennas that data will be received from. A seperate heap will be generated per antenna.
    n_channels_per_stream: int
        The number of frequency channels contained in a heap.
    n_spectra_per_heap: int
        The number of time samples per frequency channel.
    n_pols: int
        The number of pols per antenna. Expected to always be 2 at the moment.
    ig: spead2.send.ItemGroup
        The ig is used to generate heaps that will be passed to the source stream. This ig is expected to have been
        configured correctly using the create_test_objects function.

    Returns
    -------
    heaps: [spead2.send.HeapReference]
        The required heaps are stored in an array. EAch heap is wrapped in a HeapReference object is this is what is
        required by the SPEAD2 send_heaps() function.
    """
    # The heaps shape has been modified with the complexity dimension equal to 1 instead of 2. This is because we treat
    # the two 8-bit complex samples
    modified_heap_shape = (
        n_channels_per_stream,
        n_spectra_per_heap,
        n_pols,
        complexity // 2,
    )
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        coded_sample_value = (np.uint8(id) << 8) + np.uint8(ant_index)
        sample_array = np.full(modified_heap_shape, coded_sample_value, np.uint16)
        # Here we change the dtype of the array from uint16 back to int8. This does not modify the actual data in the
        # array. It just changes the shape back to what we expect. (The complexity dimension is now back to 2 from 1).
        sample_array = sample_array.view(np.int8)

        ig["timestamp"].value = timestamp
        ig["feng_id"].value = ant_index
        ig["frequency"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
        ig["feng_raw"].value = sample_array
        heap = ig.get_heap(descriptors="none", data="all")  # We dont want to deal with descriptors

        # This function makes sure that the immediate values in each heap are transmitted per packet in the heap. By
        # default these values are only transmitted once. These immediate values are required as this is how data is
        # received from the MeerKAT SKARAB F-Engines.
        heap.repeat_pointers = True

        # NOTE: The substream_index is set to zero as the SPEAD BytesStream transport has not had the concept of
        # substreams introduced. It has not been updated along with the rest of the transports. As such the unit test
        # cannot yet test that packet interleaving works correctly. I am not sure if this feature is planning to be
        # added. If it is, then set `substream_index=ant_index`. If this starts becoming an issue, then we will need to
        # look at using the inproc transport. The inproc transport would be much better, but requires porting a bunch
        # of things from SPEAD2 python to xbgpu python. This will require much more work.
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=0))
    return heaps


@pytest.mark.combinations(
    "num_ants, num_channels, num_spectra_per_heap",
    test_parameters.array_size,
    test_parameters.num_channels,
    test_parameters.num_spectra_per_heap,
)
def test_recv_simple(event_loop, num_ants, num_spectra_per_heap, num_channels):
    """Tests the xbgpu SPEAD2 reciever.

    This test is run using simulated packets that are passed to xbgpu receiver as a ByteArray. This test is useful
    for determining that the receiver is doing what is expected when receiving the correct data. It is not able to
    simulate real network conditions.

    This test checks a number of things:
    1. Simple test - will heaps transmitted in order be received correctly. This is carried out on the first 5
       chunks.
    2. Out of order heaps in same chunk - heaps destined to the same chunk are sent out of order to verify that
       they are placed correctly in the chunk at the receiver.
    3. Out of order heaps in a different chunk - heaps destined to two different chunks are sent slightly out of
       order to check that two chunks can assembled in parallel.

    It may be better for clarity to have each of these tests run in a different test functions. However that would
    require dupicating lots of code and the tests would take much longer to run. Also, I am lazy.

    This test does not generate random data as it will take a bit more effort to check that the random data is received
    correctly. This

    Parameters
    ----------
    event_loop: AsyncIO Event Loop
        The event loop that the async events will be placed on. When running a unit test, this is a fixture provided
        by the pytest-asyncio module.
    num_ants: int
        The number of antennas that data will be received from.
    num_spectra_per_heap: int
        The number of time samples per frequency channel.
    num_channels: int
        The number of frequency channels out of the FFT. NB: This is not the number of FFT channels per stream. The
        number of channels per stream is calculated from this value.
    """
    # 1. Configuration parameters
    n_ants = num_ants
    n_channels_total = num_channels

    # This integer division is so that when n_ants % num_channels !=0 then the remainder will be dropped. This will
    # only occur in the MeerKAT Extension correlator. Technically we will also need to consider the case where we round
    # up as some X-Engines will need to do this to capture all the channels, however that is not done in this test.
    n_channels_per_stream = num_channels // n_ants // 4
    n_spectra_per_heap = num_spectra_per_heap
    n_pols = 2
    sample_bits = 8
    heaps_per_fengine_per_chunk = 8
    max_active_chunks = 8

    # Multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier
    # transform.
    timestamp_step = n_channels_total * 2 * n_spectra_per_heap

    total_chunks = 10
    n_heaps_in_flight_per_antenna = heaps_per_fengine_per_chunk * total_chunks

    # 2. Create all required test objects.
    source_stream, ig, receiver_stream, async_ringbuffer = create_test_objects(
        n_ants,
        n_channels_per_stream,
        n_spectra_per_heap,
        n_pols,
        sample_bits,
        heaps_per_fengine_per_chunk,
        max_active_chunks,
        timestamp_step,
    )

    # 3. "Transmit" mutiple simulated heaps. These heaps will placed in a single ByteArray that SPEAD2 can understand
    # decode. These heaps are tranmitted in such a way as to perform the different test mentioned in this function's
    # docstring.

    # 3.1 Transmit first 5 chunks completely in order
    heap_index = 0
    for _ in range(5):
        heaps = create_heaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
        )
        source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 3.2 For chunk 6, transmit the second collection of heaps in the chunk before the first to ensure that heaps
    # received out of order are processed correctly

    # 3.2.1 Transmit the second heap first
    heaps = create_heaps(
        timestamp_step * (heap_index + 1),
        (heap_index + 1),
        n_ants,
        n_channels_per_stream,
        n_spectra_per_heap,
        n_pols,
        ig,
    )
    source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 3.2.2 Transmit the first heap second
    heaps = create_heaps(
        timestamp_step * (heap_index), (heap_index), n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
    )
    source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    heap_index += 2

    # 3.2.3 Transmit the rest of the heaps in chunk 6 in order
    for _ in range(heaps_per_fengine_per_chunk - 2):
        heaps = create_heaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
        )
        source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 3.3 For chunk 7 and 8 transmit the first set of heaps of chunk 8 before the last set of heaps of chunk 7.

    # 3.3.1 Transmit all but the last collection of heaps of chunk 7
    for _ in range(heaps_per_fengine_per_chunk - 1):
        heaps = create_heaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
        )
        source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 3.3.2 Transmit the first collection of heaps of chunk 8
    heaps = create_heaps(
        timestamp_step * (heap_index + 1),
        (heap_index + 1),
        n_ants,
        n_channels_per_stream,
        n_spectra_per_heap,
        n_pols,
        ig,
    )
    source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 3.3.3 Transmit the last collection of heaps of chunk 7
    heaps = create_heaps(
        timestamp_step * (heap_index), (heap_index), n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
    )
    source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    heap_index += 2

    # 3.3.4 Transmit the rest of the heaps in chunk 8 in order
    for _ in range(heaps_per_fengine_per_chunk - 2):
        heaps = create_heaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
        )
        source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 3.4 Transmit the remaining chunks
    for _ in range(heap_index, n_heaps_in_flight_per_antenna):
        heaps = create_heaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_spectra_per_heap, n_pols, ig
        )
        source_stream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 4. Pass simulated buffer from source_stream to the receiver.
    buffer = source_stream.getvalue()
    receiver_stream.add_buffer_reader(buffer)

    # 5. Define function that will test all received data.
    async def get_chunks(
        async_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
        receiver_stream: spead2.recv.ChunkRingStream,
        total_chunks: int,
    ):
        """Iterate through chunks processed by the receiver.

        This function checks that all the data received is correct and contains all the assert statements in this test.
        """
        chunk_index = 0
        dropped = 0
        received = 0

        # 5.1 Iterate through complete chunks in the ringbuffer asynchronously
        async for chunk in async_ringbuffer:
            assert isinstance(chunk, Chunk)
            received += len(chunk.present)
            dropped += len(chunk.present) - int(np.sum(chunk.present))
            assert len(chunk.present) == n_ants * heaps_per_fengine_per_chunk, (
                "Incorrect number of heaps in chunk. "
                f"Expected: {n_ants*heaps_per_fengine_per_chunk}. actual: {len(chunk.present)}"
            )
            # Should not be dropping anything when just reading a buffer
            assert len(chunk.present) == sum(chunk.present), f"{sum(chunk.present)} dropped heaps in chunk"
            chunk.data = chunk.data.view(np.uint16)  # We read the real and imaginary samples together
            # print(
            #     f"Chunk: {chunk_index:>5} "
            #     f"Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. "
            #     f"All time dropped/received heaps: {dropped}/{received}. "
            #     f"Timestamp: {chunk.timestamp}, {chunk.timestamp/timestamp_step}, {chunk.data.shape}"
            # )

            # 5.2 Iterate through data in chunk to check that it contains the corrected data for each antenna and heap.
            for heap_index in range(heaps_per_fengine_per_chunk):
                for ant_index in range(n_ants):
                    expected_sample_value = (
                        np.uint8(chunk_index * heaps_per_fengine_per_chunk + heap_index) << 8
                    ) + np.uint8(ant_index)
                    fengine_start_index = (
                        (heap_index * n_ants + ant_index) * n_channels_per_stream * n_spectra_per_heap * n_pols
                    )
                    fengine_stop_index = fengine_start_index + n_channels_per_stream * n_spectra_per_heap * n_pols
                    assert np.all(chunk.data[fengine_start_index:fengine_stop_index] == expected_sample_value), (
                        f"Chunk {chunk_index}, heap {heap_index}, ant {ant_index}. "
                        f"Expected all values to equal: {hex(expected_sample_value)}"
                    )

            # 5.3 Give chunk back to receiver once we are done using it.
            receiver_stream.add_free_chunk(chunk)
            chunk_index += 1

            # 5.4 Exit condition
            # I am not sure if I am happy that this is here - some of my Jenkins unit tests fail when this is not here
            # throwing a "spead2._spead2.Stopped: ring buffer has been stopped" error. I think its still trying to
            # iterate through the async_ringbuffer once the buffer is "finished" but I dont know enough about the
            # internal workings of asyncio to be sure. Just going to leave it for now and can revisit it later if we
            # decide the coverage is not enough. It does make the assert (chunk_index == total_chunk) test below a
            # bit less useful.
            if chunk_index == total_chunks:
                break

        assert chunk_index == total_chunks, f"Expected to receive {total_chunks} chunks. Only received {chunk_index}"

    # 6. Run get_chunks() function
    event_loop.run_until_complete(get_chunks(async_ringbuffer, receiver_stream, total_chunks))


# A manual run useful when debugging the unit tests.
if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    loop = asyncio.get_event_loop()
    test_recv_simple(loop, 4, 256, 32768)
    test_recv_simple(loop, 8, 256, 32768)
    test_recv_simple(loop, 16, 256, 32768)
    test_recv_simple(loop, 32, 256, 32768)
    test_recv_simple(loop, 64, 256, 32768)
    print("Tests complete")
