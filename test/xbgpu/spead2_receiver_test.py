"""TODO: Add a comment."""

# import katxgpu
import katxgpu._katxgpu.recv as recv
import katxgpu.monitor
import katxgpu.ringbuffer
import spead2
import spead2.send

import logging
import asyncio
import numpy as np
import katsdpsigproc.accel as accel

logger = logging.getLogger(__name__)

# TODO: Check data format is correct
# TODO: Add katxgpu receiver to this

# 1. Define Constants
# SPEAD IDs
TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x4300

complexity = 2


def createTestObjects(
    n_ants,
    n_channels_per_stream,
    n_samples_per_channel,
    n_pols,
    sample_bits,
    heaps_per_fengine_per_chunk,
    complexity,
    timestamp_step,
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
    monitor.event_qsize("recv_ringbuffer", 0, ringbuffer_capacity)

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
    monitor.event_qsize("free_chunks", 0, src_chunks_per_stream)
    for i in range(src_chunks_per_stream):
        buf = accel.HostArray((receiverStream.chunk_bytes,), np.uint8, context=context)
        chunk = recv.Chunk(buf)
        receiverStream.add_chunk(chunk)

    asyncRingbuffer = katxgpu.ringbuffer.AsyncRingbuffer(
        receiverStream.ringbuffer, monitor, "recv_ringbuffer", "get_chunks"
    )

    return sourceStream, ig, receiverStream, asyncRingbuffer


def createHeaps(
    timestamp: int, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig: spead2.send.ItemGroup
):
    """
    Generate a list of heaps to send in an interleaved manner.

    The list is of HeapReference objects which point to the heaps as this is what the send_heaps() function requires.
    """
    shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        ig["timestamp"].value = timestamp
        ig["fengine id"].value = ant_index
        ig["channel offset"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
        ig["feng_raw"].value = np.full(shape, ant_index, np.int8)
        ig["padding 0"].value = 0
        ig["padding 1"].value = 0
        ig["padding 2"].value = 0
        heap = ig.get_heap()
        heap.repeat_pointers = True

        # NOTE: The substream_index is set to zero as the SPEAD BytesStream transport has not had the concept of substreams introduced. It has not been updated along with the rest of the transports. As such the unit test cannot yet test that packet interleaving works correctly. I am not sure if this feature is planning to be added. If it is, then set `substream_index=ant_index`. If this starts becoming an issue, then we will need to lok at using the inproc transport. The inproc transport would be much better, but requires porting a bunch of things from SPEAD2 python to katxgpu python. So it will require much more work.
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=0))
    return heaps


def test_recv_simple():
    """TODO: Add a comment."""
    # Configuration parameters
    n_ants = 64
    n_channels_total = 32768
    n_channels_per_stream = 128
    n_samples_per_channel = 256
    n_pols = 2
    sample_bits = 8
    heaps_per_fengine_per_chunk = 5

    timestamp_step = (
        n_channels_total * 2 * n_samples_per_channel
    )  # Multiply step by 2 to account for dropping half of the spectrum due to symmetric properties of the fourier transform.

    n_heaps_in_flight_per_antenna = 20

    # 2. Set up receiver
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

    # 3. Define Heap Format

    # 4. Create and array of heaps to send

    # 5. Transmit Data

    # 5.1 Transmit initial few heaps
    for i in range(n_heaps_in_flight_per_antenna):
        heaps = createHeaps(timestamp_step * i, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig)
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # # 5.2 Transmit data that skips a few heaps verifying that the pipeline can flush properly
    # offset_timestamp = timestamp_step * (n_heaps_in_flight_per_antenna + 3 * heaps_per_fengine_per_chunk)
    # for i in range(heaps_per_fengine_per_chunk):
    #     heaps = createHeaps(offset_timestamp + i * timestamp_step)
    #     sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # # 5.3 Transmit data backwards in time and check that it is dropped
    # offset_timestamp = timestamp_step * (n_heaps_in_flight_per_antenna + 2 * heaps_per_fengine_per_chunk)
    # for i in range(heaps_per_fengine_per_chunk):
    #     heaps = createHeaps(offset_timestamp + i * timestamp_step)
    #     sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    receiverStream.add_buffer_reader(sourceStream.getvalue())
    # receiverStream.add_udp_ibv_reader([("239.10.10.10", 7149)], "10.100.44.1", 10000000, 0)

    async def get_chunks():
        """TODO: Create docstring."""
        print("Main asyncio loop now running.")
        i = 0
        dropped = 0
        received = 0
        async for chunk in asyncRingbuffer:
            received += len(chunk.present)
            dropped += len(chunk.present) - sum(chunk.present)
            print(
                f"Chunk: {i:>5} Received: {sum(chunk.present):>4} of {len(chunk.present):>4} expected heaps. All time dropped/received heaps: {dropped}/{received}. Timestamp: {chunk.timestamp}, {chunk.timestamp/timestamp_step}"
            )
            receiverStream.add_chunk(chunk)
            i += 1

    print(f"Timestamp Step {timestamp_step}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_chunks())

    loop.close()

    print("Done")


test_recv_simple()
