"""TODO: Write this docstring - not testing samples are not mixed up, assumption is that speed two does that."""

# 1. Import local modules
import test_parameters
import katxgpu.xsend

# 2. Import external modules
import pytest
import spead2
import spead2.recv.asyncio
import asyncio
import numpy as np
import katsdpsigproc.accel as accel


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
def test_send_simple(event_loop, num_ants, num_channels):
    """TODO: Write this docstring."""
    heaps_to_send = 20

    n_channels_per_stream = num_channels // num_ants // 4
    n_pols = 2
    dump_rate_s = 0.05  # Normally 0.4 but we set it very low so that the test runs quickly
    complexity = 2
    sample_bits = 32

    n_baselines = (n_pols * num_ants + 1) * (num_ants * n_pols) // 2

    # 3. Create cuda context - all buffers created in the XEngineSPEADIbvSend object are created from this context.
    context = accel.create_some_context(device_filter=lambda x: x.is_cuda)

    queue = spead2.InprocQueue()

    sendStream = katxgpu.xsend.XEngineSPEADInprocSend(
        n_ants=num_ants,
        n_channels_per_stream=n_channels_per_stream,
        n_pols=n_pols,
        dump_rate_s=dump_rate_s,
        channel_offset=n_channels_per_stream * 4,  # Arbitrary for now
        context=context,
        queue=queue,
    )

    thread_pool = spead2.ThreadPool()
    recvStream = spead2.recv.asyncio.Stream(thread_pool, spead2.recv.StreamConfig(max_heaps=100))
    recvStream.add_inproc_reader(queue)
    del thread_pool

    async def send_process():
        """TODO: Write docstring."""
        num_sent = 0
        sendStream.send_descriptor_heap()

        while num_sent < heaps_to_send:
            # 5.1 Get a free buffer to store the next heap.
            buffer_wrapper = await sendStream.get_free_heap()

            # 5.2 Populate the buffer with dummy data - notice how we copy new values into the buffer, we dont overwrite
            # the buffer. Attempts to overwrite the buffer will throw an error. This is intended behavour as the memory
            # regions in the buffer have been configured for zero-copy sends.
            buffer_wrapper.buffer[:] = np.full(
                buffer_wrapper.buffer.shape, num_sent, np.uint8
            )  # [:] forces a copy, not an overwrite

            # 5.3 Give the buffer back to the sendStream to transmit out onto the network.
            sendStream.send_heap(num_sent * 0x1000000, buffer_wrapper)
            # print(f"Sent heap {num_sent-1}. Values: [{buffer_wrapper.buffer[0]}...{buffer_wrapper.buffer[0]}]")

            num_sent += 1

    async def recv_process():
        """TODO: Write docstring."""
        num_received = 0
        ig = spead2.ItemGroup()
        heap = await recvStream.get()  # We expect this to be the descriptor and as such there are no items in the heap
        items = ig.update(heap)

        assert len(list(items.values())) == 0, "This heap contains item values not just desciptors as expected"

        while num_received < heaps_to_send:
            heap = await recvStream.get()
            items = ig.update(heap)

            has_timestamp = False
            has_channel_offset = False
            has_xeng_raw = False

            for item in items.values():
                if item.id == 0x1600:
                    has_timestamp = True
                    assert (
                        item.value == num_received * 0x1000000
                    ), f"Timestamp value incorrect. Expected: {hex(num_received)}, actual {hex(item.value)}"

                if item.id == 0x4103:
                    has_channel_offset = True
                    assert (
                        item.value == n_channels_per_stream * 4
                    ), f"Channel offset incorrect. Expected: {hex(n_channels_per_stream * 4)}, actual {hex(item.value)}"

                if item.id == 0x1800:
                    has_xeng_raw = True
                    data_length_bytes = n_baselines * n_channels_per_stream * complexity * sample_bits // 8
                    assert (
                        len(item.value) == data_length_bytes
                    ), f"xeng_raw data not correct size. Expected: {data_length_bytes} bytes, actual {data_length_bytes} bytes"
                    assert np.all(item.value == num_received)

            assert (
                has_timestamp
            ), f"Received heap is missing timestamp item with ID {hex(katxgpu.xsend.XEngineSPEADAbstractSend.TIMESTAMP_ID)}"
            assert (
                has_channel_offset
            ), f"Received heap is missing channel offset item with ID {hex(katxgpu.xsend.XEngineSPEADAbstractSend.CHANNEL_OFFSET)}"
            assert (
                has_xeng_raw
            ), f"Received heap is missing xeng_raw data buffer item with ID {hex(katxgpu.xsend.XEngineSPEADAbstractSend.DATA_ID)}"

            num_received += 1

    async def run():
        """TODO: Write docstring."""
        task1 = event_loop.create_task(send_process())
        task2 = event_loop.create_task(recv_process())
        await task1
        await task2

    event_loop.run_until_complete(run())


# A manual run useful when debugging the unit tests.
if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    event_loop = asyncio.get_event_loop()
    test_send_simple(event_loop, num_ants=64, num_channels=32768)
    print("Tests complete")
