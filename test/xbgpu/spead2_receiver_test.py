"""TODO: Add a comment."""

# import katxgpu
import test_parameters
import katxgpu._katxgpu.recv as recv
import katxgpu.monitor
import katxgpu.ringbuffer

import pytest
import logging
import asyncio
import numpy as np
import katsdpsigproc.accel as accel
import spead2
import spead2.send

logging.basicConfig(level=logging.INFO)

# 1. Define Constants
# SPEAD IDs
TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x4300

complexity = 2

# TO NOTE: Test on random data is missing, this means that if the packets within a heap are not interelaved properly, we wont be able to tell
# To NOTE: Interleaving does not take plce correctly.


def createTestObjects(
    n_ants: int,
    n_channels_per_stream: int,
    n_samples_per_channel: int,
    n_pols: int,
    sample_bits: int,
    heaps_per_fengine_per_chunk: int,
    complexity: int,
    timestamp_step: int,
):
    """TODO: Add a comment."""
    max_packet_size = (
        n_samples_per_channel * n_pols * complexity * sample_bits // 8 + 96
    )  # Header is 12 fields of 8 bytes each: So 96 bytes of header
    thread_pool = spead2.ThreadPool()
    sourceStream = spead2.send.BytesStream(
        thread_pool,
        spead2.send.StreamConfig(max_packet_size=max_packet_size, max_heaps=n_ants * heaps_per_fengine_per_chunk),
    )
    del thread_pool

    shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
    ig = spead2.send.ItemGroup(flavour=spead2.Flavour(4, 64, 48, 0))
    ig.add_item(TIMESTAMP_ID, "timestamp", "timestamp description", shape=[], format=[("u", 48)])
    ig.add_item(FENGINE_ID, "fengine id", "F-Engine heap is received from", shape=[], format=[("u", 48)])
    ig.add_item(
        CHANNEL_OFFSET,
        "channel offset",
        "Value of first channel in collections stored here",
        shape=[],
        format=[("u", 48)],
    )
    ig.add_item(DATA_ID, "feng_raw", "Raw Channelised data", shape=shape, dtype=np.int8)
    # Adding padding
    for i in range(3):
        ig.add_item(
            CHANNEL_OFFSET + 1 + i,
            f"padding {i}",
            "Padding field {i} to align header to 256-bit boundary.",
            shape=[],
            format=[("u", 48)],
        )
    ig.get_heap()  # Throw away first heap - need to get this as it contains a bunch of descriptor information that we dont want for the purposes of this test.

    # 6. Create all receiver data

    # 6.1 Create monitor for file
    monitor = katxgpu.monitor.NullMonitor()

    # 6.2 Create ringbuffer
    ringbuffer_capacity = 10
    ringbuffer = recv.Ringbuffer(ringbuffer_capacity)

    # 6.3 Create Receiver
    thread_affinity = 2
    receiverStream = recv.Stream(
        n_ants,
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        sample_bits,
        timestamp_step,
        heaps_per_fengine_per_chunk,
        ringbuffer,
        thread_affinity,
        monitor=monitor,
        use_gdrcopy=False,
    )

    # 6.4 Add free chunks to SPEAD2 receiver
    context = accel.create_some_context(device_filter=lambda x: x.is_cuda)
    src_chunks_per_stream = 8
    for i in range(src_chunks_per_stream):
        buf = accel.HostArray((receiverStream.chunk_bytes,), np.uint8, context=context)
        chunk = recv.Chunk(buf)
        receiverStream.add_chunk(chunk)

    asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
        receiverStream.ringbuffer, monitor, "recv_ringbuffer", "get_chunks"
    )

    return sourceStream, ig, receiverStream, asyncRingbuffer


def createHeaps(
    timestamp: int,
    id: int,
    n_ants: int,
    n_channels_per_stream: int,
    n_samples_per_channel: int,
    n_pols: int,
    ig: spead2.send.ItemGroup,
):
    """
    Generate a list of heaps to send in an interleaved manner.

    The list is of HeapReference objects which point to the heaps as this is what the send_heaps() function requires.

    TODO: Mention the ID
    """
    modified_shape = (
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        complexity // 2,
    )  # Comment on this divide by 2
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        coded_sample_value = (np.uint8(id) << 8) + np.uint8(ant_index)
        sample_array = np.full(modified_shape, coded_sample_value, np.uint16)
        sample_array.dtype = np.int8

        ig["timestamp"].value = timestamp
        ig["fengine id"].value = ant_index
        ig["channel offset"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
        ig["feng_raw"].value = sample_array
        ig["padding 0"].value = 0
        ig["padding 1"].value = 0
        ig["padding 2"].value = 0
        heap = ig.get_heap()
        heap.repeat_pointers = True

        # NOTE: The substream_index is set to zero as the SPEAD BytesStream transport has not had the concept of substreams introduced. It has not been updated along with the rest of the transports. As such the unit test cannot yet test that packet interleaving works correctly. I am not sure if this feature is planning to be added. If it is, then set `substream_index=ant_index`. If this starts becoming an issue, then we will need to lok at using the inproc transport. The inproc transport would be much better, but requires porting a bunch of things from SPEAD2 python to katxgpu python. So it will require much more work.
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=0))
    return heaps


# NOTE: Split tests here differently
@pytest.mark.parametrize("num_ants", test_parameters.array_size)
def test_recv_simple(event_loop, num_ants):
    """TODO: Add a comment."""
    # Configuration parameters
    n_ants = num_ants
    n_channels_total = 32768
    n_channels_per_stream = 128
    n_samples_per_channel = 256
    n_pols = 2
    sample_bits = 8
    heaps_per_fengine_per_chunk = 8

    timestamp_step = (
        n_channels_total * 2 * n_samples_per_channel
    )  # Multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.

    total_chunks = 10
    n_heaps_in_flight_per_antenna = heaps_per_fengine_per_chunk * total_chunks

    # 2. Set up test
    sourceStream, ig, receiverStream, asyncRingbuffer = createTestObjects(
        n_ants,
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        sample_bits,
        heaps_per_fengine_per_chunk,
        complexity,
        timestamp_step,
    )

    # 5. "Transmit" mutiple simulated heaps. These heaps will placed in a single ByteArray that SPEAD2 can understand
    # decode. NOTE: A few different things are tested here:
    #    1. Simple test - will heaps transmitted in order be received correctly. This is carried out on the first 5
    #       chunks.
    #    2. Out of order heaps in same chunk - heaps destined to the same chunk are sent out of order to verify that
    #       they are placed correctly in the chunk at the receiver.
    #    3. Out of order heaps in a different chunk - heaps destined to two different chunks are sent slightly out of
    #       order to check that two chunks can assembled in parallel.
    # It may be better for clarity to have each of these tests run in a different test functions. However that would
    # require dupicating lots of code and the tests would take much longer to run. Also, I am lazy.

    # 5.1 Transmit first 5 chunks completly in order
    heap_index = 0
    print("In order")
    print(heap_index)
    for i in range(5):
        heaps = createHeaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 5.2 For chunk 6, transmit the second collection of heaps before the first to ensure that heaps received out of order can be processed

    # 5.2.1 Transmit the second heap first
    heaps = createHeaps(
        timestamp_step * (heap_index + 1),
        (heap_index + 1),
        n_ants,
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        ig,
    )
    sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 5.2.2 Transmit the first heap second
    heaps = createHeaps(
        timestamp_step * (heap_index), (heap_index), n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
    )
    sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    heap_index += 2

    # 5.2.3 Transmit the rest of the heaps in chunk 6 in order
    for i in range(heaps_per_fengine_per_chunk - 2):
        heaps = createHeaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 5.3 For chunk 7 and 8 transmit the first set of heaps of chunk 8 before the last set of heaps of chunk 7.

    # 5.3.1 Transmit all but the last collection of heaps of chunk 7
    for i in range(heaps_per_fengine_per_chunk - 1):
        heaps = createHeaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 5.3.2 Transmit the first collection of heaps of chunk 8
    heaps = createHeaps(
        timestamp_step * (heap_index + 1),
        (heap_index + 1),
        n_ants,
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        ig,
    )
    sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 5.3.3 Transmit the last collection of heaps of chunk 7
    heaps = createHeaps(
        timestamp_step * (heap_index), (heap_index), n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
    )
    sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    heap_index += 2

    # 5.3.4 Transmit the rest of the heaps in chunk 8 in order
    for i in range(heaps_per_fengine_per_chunk - 2):
        heaps = createHeaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    # 5.4 Transmit the remaining chunks
    for i in range(heap_index, n_heaps_in_flight_per_antenna):
        heaps = createHeaps(
            timestamp_step * heap_index, heap_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)
        heap_index += 1

    buffer = sourceStream.getvalue()
    receiverStream.add_buffer_reader(buffer)

    async def get_chunks(
        asyncRingbuffer: katxgpu.ringbuffer.AsyncRingbuffer,
        receiverStream: katxgpu._katxgpu.recv.Stream,
        total_chunks: int,
    ):
        """TODO: Create docstring."""
        chunk_index = 0
        dropped = 0
        received = 0
        async for chunk in asyncRingbuffer:
            received += len(chunk.present)
            dropped += len(chunk.present) - sum(chunk.present)
            assert (
                len(chunk.present) == n_ants * heaps_per_fengine_per_chunk
            ), f"Incorrect number of heaps in chunk. Expected: {n_ants*heaps_per_fengine_per_chunk}. actual: {len(chunk.present)}"
            assert len(chunk.present) == sum(
                chunk.present
            ), f"{sum(chunk.present)} dropped heaps in chunk"  # Should not be dropping anything when just reading a buffer
            chunk.base.dtype = np.uint16  # We read the real and imaginary samples together
            print(
                f"Chunk: {chunk_index:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}. Timestamp: {chunk.timestamp}, {chunk.timestamp/timestamp_step}, {chunk.base.shape}"
            )

            print(chunk.present)

            for heap_index in range(heaps_per_fengine_per_chunk):
                for ant_index in range(n_ants):
                    expected_sample_value = (
                        np.uint8(chunk_index * heaps_per_fengine_per_chunk + heap_index) << 8
                    ) + np.uint8(ant_index)
                    fengine_start_index = (
                        (heap_index * n_ants + ant_index) * n_channels_per_stream * n_samples_per_channel * n_pols
                    )
                    fengine_stop_index = fengine_start_index + n_channels_per_stream * n_samples_per_channel * n_pols
                    # print(hex(expected_sample_value), chunk.base[fengine_start_index:fengine_stop_index], np.all(chunk.base[fengine_start_index:fengine_stop_index] == expected_sample_value)  ,sum(chunk.base[fengine_start_index:fengine_stop_index]))
                    assert np.all(
                        chunk.base[fengine_start_index:fengine_stop_index] == expected_sample_value
                    ), f"Chunk {chunk_index}, heap {heap_index}, ant {ant_index}. Expected all values to equal: {hex(expected_sample_value)}"

            # Give chunk back to receiver
            receiverStream.add_chunk(chunk)
            chunk_index += 1
        assert chunk_index == total_chunks, f"Expected to receive {total_chunks} chunks. Only received {chunk_index}"

    event_loop.run_until_complete(get_chunks(asyncRingbuffer, receiverStream, total_chunks))

    # Something is not being cleared properly at the end - if I do not delete these I get an error on the next test that is run
    del sourceStream, ig, receiverStream, asyncRingbuffer


if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    loop = asyncio.get_event_loop()
    test_recv_simple(loop, 4)
    # test_recv_simple(loop, 8)
    # test_recv_simple(loop, 16)
    # test_recv_simple(loop, 32)
    # test_recv_simple(loop, 64)
    print("Tests complete")
