"""TODO: Write this."""

# 1. Import local modules
import test_parameters
import spead2_receiver_test
import katxgpu.ringbuffer
import katxgpu.xbengine_proc_loop

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
    xbengine_proc_loop = katxgpu.xbengine_proc_loop.XBEngineProcessingLoop(
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

    # 5. Generate Data
    for i in range(heap_accumulation_threshold * n_accumulations + 1):
        timestamp = (i + (heap_accumulation_threshold - 1)) * timestamp_step  # Say what this -1 is for
        print(i, hex(timestamp))
        heaps = spead2_receiver_test.createHeaps(
            timestamp, i, n_ants, n_channels_per_stream, n_samples_per_channel, n_pols, ig_send
        )
        sourceStream.send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    # 6. Define C function for verification of data.
    verificationFunctionsLib_C = np.ctypeslib.load_library(
        libname="lib_verification_functions.so", loader_path=os.path.abspath("./")
    )
    verify_xbengine_proc_loop_C = verificationFunctionsLib_C.verify_xbengine_proc_loop

    baselines_products = n_ants * (n_ants + 1) // 2 * n_pols * n_pols * n_channels_per_stream
    print("++++++++++++++++++", baselines_products, n_channels_per_stream, n_ants)
    verify_xbengine_proc_loop_C.argtypes = [
        np.ctypeslib.ndpointer(
            dtype=np.int64,  # Output data array
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

    verify_xbengine_proc_loop_C.restype = ctypes.c_int

    # 7. Add transports
    xbengine_proc_loop.add_inproc_sender_transport(queue)
    xbengine_proc_loop.sendStream.send_descriptor_heap()

    buffer = sourceStream.getvalue()
    xbengine_proc_loop.add_buffer_receiver_transport(buffer)

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

            print(
                f"Received Timestamp: {hex(ig_recv['timestamp'].value)}, Value: {ig_recv['xeng_raw'].value[0][0][0][0]}"
            )

            print(ig_recv["xeng_raw"].value.flatten(order="C"))
            print(len(ig_recv["xeng_raw"].value.flatten(order="C")))
            if i == 0:
                num_batches_in_current_accumulation = 1
                base_batch_index = heap_accumulation_threshold - 1
            else:
                num_batches_in_current_accumulation = heap_accumulation_threshold
                base_batch_index = i * heap_accumulation_threshold
            result = verify_xbengine_proc_loop_C(
                ig_recv["xeng_raw"].value.flatten(order="C"),
                base_batch_index,
                num_batches_in_current_accumulation,
                n_ants,
                n_channels_per_stream,
                n_samples_per_channel,
                n_pols,
            )
            print(f"========================== {result} ============================")

    # 8. Function that will launch the send_process() and xbengin loop
    @pytest.mark.asyncio
    async def run():
        """TODO: Write this."""
        task1 = event_loop.create_task(xbengine_proc_loop.run())
        task2 = event_loop.create_task(recv_process())
        await task2
        task1.cancel()

    # 9. Launch asyn functions and wait until completion
    event_loop.run_until_complete(run())
    xbengine_proc_loop.stop()


# A manual run useful when debugging the unit tests.
if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    loop = asyncio.get_event_loop()
    test_xbengine(loop, 64, 1024, 32768)
    # test_xbengine(loop, 8, 1024, 32768)
    # test_xbengine(loop, 16, 1024, 32768)
    # test_xbengine(loop, 32, 1024, 32768)
    # test_xbengine(loop, 64, 1024, 32768)
    print("Tests complete")
