"""TODO: Write this."""

# 1. Import local modules
import test_parameters
import katxgpu.ringbuffer
import katxgpu.xbengine

# 2. Import external modules
import os
import ctypes
import pytest
import logging
import asyncio
import numpy as np
import spead2
import spead2.send
import spead2.recv.asyncio

logging.basicConfig(level=logging.INFO)

# 3. Define Constants
# 3.1 SPEAD IDs
TIMESTAMP_ID = 0x1600
FENGINE_ID = 0x4101
CHANNEL_OFFSET = 0x4103
DATA_ID = 0x4300

default_spead_flavour = {"version": 4, "item_pointer_bits": 64, "heap_address_bits": 48, "bug_compat": 0}
complexity = 2


def createHeaps(
    timestamp: int,
    batch_index: int,
    n_ants: int,
    n_channels_per_stream: int,
    n_samples_per_channel: int,
    n_pols: int,
    ig: spead2.send.ItemGroup,
):
    """
    TODO: Update This.

    Generate a list of heaps to send via the sourceStream.

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
    n_samples_per_channel: int
        The number of time samples per frequency channel.
    n_pols: int
        The number of pols per antenna. Expected to always be 2 at the moment.
    ig: spead2.send.ItemGroup
        The ig is used to generate heaps that will be passed to the source stream. This ig is expected to have been
        configured correctly using the createTestObjects function.

    Returns
    -------
    heaps: [spead2.send.HeapReference]
        The required heaps are stored in an array. EAch heap is wrapped in a HeapReference object is this is what is
        required by tge SPEAD2 send_heaps() function.
    """
    heap_shape = (
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols,
        complexity,
    )
    # The heaps shape has been modified with the complexity dimension equal to 1 instead of 2. This is because we treat
    # the two 8-bit complex samples
    modified_heap_shape = (
        n_channels_per_stream,
        n_samples_per_channel,
        n_pols // 2,
        complexity // 2,
    )
    heaps = []  # Needs to be of type heap reference, not heap for substream transmission.
    for ant_index in range(n_ants):
        sample_array = np.zeros(modified_heap_shape, np.uint32)

        for chan_index in range(n_channels_per_stream):
            # coded_sample_value = (0 << 24) + (ant_index << 16) + (0 << 8) + np.uint8(ant_index)
            sign = 1 if batch_index % 2 == 0 else -1
            pol0Real = np.int8(sign * batch_index)
            pol0Imag = np.int8(sign * chan_index)
            pol1Real = np.int8(-sign * ant_index)
            pol1Imag = np.int8(-sign * chan_index)
            if pol0Real == -128:
                pol0Real = -127
            if pol0Imag == -128:
                pol0Imag = -127
            if pol1Real == -128:
                pol1Real = -127
            if pol1Imag == -128:
                pol1Imag = -127
            coded_sample_value = np.uint32(
                (np.uint8(pol1Imag) << 24)
                + (np.uint8(pol1Real) << 16)
                + (np.uint8(pol0Imag) << 8)
                + (np.uint8(pol0Real) << 0)
            )
            # if ant_index == 127 or ant_index == 0:
            #     print(ant_index, "Input numbers: ", pol0Real, pol0Imag, pol1Real, pol1Imag)
            #     print(hex(coded_sample_value))
            sample_array[chan_index][:] = np.full((n_samples_per_channel, 1, 1), coded_sample_value, np.uint32)

        # Here we change the dtype of the array from uint16 back to int8. This does not modify the actual data in the
        # array. It just changes the shape back to what we expect. (The complexity dimension is now back to 2 from 1).
        sample_array.dtype = np.int8
        sample_array = np.reshape(sample_array, heap_shape)

        ig["timestamp"].value = timestamp
        ig["fengine id"].value = ant_index
        ig["channel offset"].value = n_channels_per_stream * 4  # Arbitrary multiple for now
        ig["feng_raw"].value = sample_array
        ig["padding 0"].value = 0
        ig["padding 1"].value = 0
        ig["padding 2"].value = 0
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
        # of things from SPEAD2 python to katxgpu python. This will require much more work.
        heaps.append(spead2.send.HeapReference(heap, cnt=-1, substream_index=0))
    return heaps


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_samples_per_channel", test_parameters.num_samples_per_channel)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
def test_xbengine(event_loop, num_ants, num_samples_per_channel, num_channels):
    """TODO: Write this."""
    # 1. Configuration parameters
    n_ants = num_ants
    n_channels_total = num_channels

    # This integer division is so that when n_ants % num_channels !=0 then the remainder will be dropped. This will
    # only occur in the MeerKAT Extension correlator. Technically we will also need to consider the case where we round
    # up as some X-Engines will need to do this to capture all the channels, however that is not done in this test.
    n_channels_per_stream = num_channels // n_ants // 4
    n_samples_per_channel = num_samples_per_channel
    n_pols = 2
    sample_bits = 8
    heaps_per_fengine_per_chunk = 2
    heap_accumulation_threshold = 4
    n_accumulations = 3

    max_packet_size = (
        n_samples_per_channel * n_pols * complexity * sample_bits // 8 + 96
    )  # Header is 12 fields of 8 bytes each: So 96 bytes of header
    heap_shape = (n_channels_per_stream, n_samples_per_channel, n_pols, complexity)
    timestamp_step = n_channels_total * 2 * n_samples_per_channel

    # 2. Create sourceStream object - transforms "transmitted" heaps into a byte array to simulate received data.
    thread_pool = spead2.ThreadPool()
    sourceStream = spead2.send.BytesStream(
        thread_pool,
        spead2.send.StreamConfig(
            max_packet_size=max_packet_size, max_heaps=n_ants * heaps_per_fengine_per_chunk * 10
        ),  # Need a bigger buffer
    )

    # 2.1. Create ItemGroup and add all the required fields.
    ig_send = spead2.send.ItemGroup(flavour=spead2.Flavour(**default_spead_flavour))
    ig_send.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=[],
        format=[("u", default_spead_flavour["heap_address_bits"])],
    )
    ig_send.add_item(
        FENGINE_ID,
        "fengine id",
        "F-Engine heap is received from",
        shape=[],
        format=[("u", default_spead_flavour["heap_address_bits"])],
    )
    ig_send.add_item(
        CHANNEL_OFFSET,
        "channel offset",
        "Value of first channel in collections stored here",
        shape=[],
        format=[("u", default_spead_flavour["heap_address_bits"])],
    )
    ig_send.add_item(DATA_ID, "feng_raw", "Raw Channelised data", shape=heap_shape, dtype=np.int8)
    # 2.1 Adding padding to header so it is the required width.
    for i in range(3):
        ig_send.add_item(
            CHANNEL_OFFSET + 1 + i,
            f"padding {i}",
            "Padding field {i} to align header to 256-bit boundary.",
            shape=[],
            format=[("u", default_spead_flavour["heap_address_bits"])],
        )

    # 3. Create receiver
    queue = spead2.InprocQueue()
    thread_pool = spead2.ThreadPool()
    recvStream = spead2.recv.asyncio.Stream(thread_pool, spead2.recv.StreamConfig(max_heaps=100))
    recvStream.add_inproc_reader(queue)

    # 4. Create xbengine
    xbengine = katxgpu.xbengine.XBEngine(
        adc_sample_rate_Hz=1712000000,  # L-Band, not important
        n_ants=n_ants,
        n_channels_total=n_channels_total,
        n_channels_per_stream=n_channels_per_stream,
        n_samples_per_channel=n_samples_per_channel,
        n_pols=n_pols,
        sample_bits=sample_bits,
        heap_accumulation_threshold=heap_accumulation_threshold,
        channel_offset_value=0,
        rx_thread_affinity=0,
        batches_per_chunk=heaps_per_fengine_per_chunk,
    )

    # 6. Generate Data
    for i in range(heap_accumulation_threshold * n_accumulations + 1):
        batch_index = i + (heap_accumulation_threshold - 1)
        timestamp = batch_index * timestamp_step  # Say what this -1 is for
        # print(i, hex(timestamp))
        heaps = createHeaps(
            timestamp, batch_index, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig_send
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 5. Define C function for verification of data.
    verificationFunctionsLib_C = np.ctypeslib.load_library(
        libname="lib_verification_functions.so", loader_path=os.path.abspath("./test/")
    )
    verify_xbengine_C = verificationFunctionsLib_C.verify_xbengine

    baselines_products = n_ants * (n_ants + 1) // 2 * n_pols * n_pols * n_channels_per_stream
    verify_xbengine_C.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.uint64,  # Output data array
            shape=(baselines_products,),
            flags="C_CONTIGUOUS",
        ),
        ctypes.c_int,  # Batch Start Index
        ctypes.c_int,  # Batches to accumulat
        ctypes.c_int,  # Antennas
        ctypes.c_int,  # Channels
        ctypes.c_int,  # Samples-per-channel
        ctypes.c_int,  # Polarisations
    ]

    verify_xbengine_C.restype = ctypes.c_int

    # 7. Add transports
    xbengine.add_inproc_sender_transport(queue)
    xbengine.sendStream.send_descriptor_heap()

    buffer = sourceStream.getvalue()
    xbengine.add_buffer_receiver_transport(buffer)

    # 8. Function to get data from xb_engine
    @pytest.mark.asyncio
    async def recv_process():
        """TODO: Write this."""
        ig_recv = spead2.ItemGroup()
        heap = await recvStream.get()
        items = ig_recv.update(heap)
        assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

        for i in range(n_accumulations + 1):
            heap = await recvStream.get()
            items = ig_recv.update(heap)
            # assert len(list(items.values())) != 0, "This heap contains item values not just the expected descriptors."
            assert (
                ig_recv["timestamp"].value % (timestamp_step * heap_accumulation_threshold) == 0
            ), "Output timestamp is not a multiple of timestamp_step * heap_accumulation_threshold."

            # print(
            #     f"Received Timestamp: {hex(ig_recv['timestamp'].value)}, Value: {ig_recv['xeng_raw'].value[0][0][0][0]}"
            # )

            if i == 0:
                num_batches_in_current_accumulation = 1
                base_batch_index = heap_accumulation_threshold - 1
            else:
                num_batches_in_current_accumulation = heap_accumulation_threshold
                base_batch_index = i * heap_accumulation_threshold
            result = verify_xbengine_C(
                ig_recv["xeng_raw"].value.flatten(order="C"),
                base_batch_index,
                num_batches_in_current_accumulation,
                n_ants,
                n_channels_per_stream,
                n_samples_per_channel,
                n_pols,
            )
            # print("asdasd",np.int32(ig_recv["xeng_raw"].value[0][katxgpu.tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(127,0)][1][0]), np.int64(ig_recv["xeng_raw"].value[0][katxgpu.tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(127,0)][1][0])>>32)
            assert result, "Gosh darnit"
            # print(katxgpu.tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(4,3))

    # 8. Function that will launch the send_process() and xbengin loop
    @pytest.mark.asyncio
    async def run():
        """TODO: Write this."""
        task1 = event_loop.create_task(xbengine.run())
        task2 = event_loop.create_task(recv_process())
        await task2
        task1.cancel()

    # 9. Launch asyn functions and wait until completion
    event_loop.run_until_complete(run())
    xbengine.stop()


# A manual run useful when debugging the unit tests.
if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    loop = asyncio.get_event_loop()
    test_xbengine(loop, 130, 256, 4096)
    # test_xbengine(loop, 8, 1024, 32768)
    # test_xbengine(loop, 16, 1024, 32768)
    # test_xbengine(loop, 32, 1024, 32768)
    # test_xbengine(loop, 64, 1024, 32768)
    print("Tests complete")
