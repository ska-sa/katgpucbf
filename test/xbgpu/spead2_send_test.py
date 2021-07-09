"""
Module for performing unit tests on the XEngineSPEADAbstractSend class in the xsend.py module.

Testing network code is difficult to do on a single thread. SPEAD2 has the concept of transports. A transport generally
receives data from a network. SPEAD2 provides two other transports that can receive simulated network data - these can
be used for testing the receiver in once process. The two transports are an inproc and a buffer transport. The
inproc transport is more flexible and as such is used here.

NOTE: A downside of this test is that it does not check that the packet formats are exactly correct. This test will
ensure that the packets are transmitted in a way that they are able to be assembled into a heap by any SPEAD2 receiver
or a full implementation of the SPEAD protocol. However, the exact packet size and the presence of repeat pointers
within the a packet are not checked. Some sort of external test should be done to check this. See the
display_xengine_multicast_packets.py script in the scratch folder of this repo as a starting point to check packet
formats. UPDATE: If we wanted to actually process real network packets, we could use the SPEAD2 PCAP transport. This
allows SPEAD2 to read a PCAP file. We could generate network data in a manner that duplicates real traffic and then
store that in a PCAP file to be used in unit tests. This data could be interleaved and out of order if necessary for a
more robust test. The downside to a PCAP test is that the PCAP file could get quite large quite quickly. This could
lead to the repo getting very large which is to be avoided. I think the best use a single PCAP file of a single array
configuration and run a single unit test and then use the inproc readers as done below for the remaining configurations.
This will ensure that we run a test on "real" data while still testing all configurations without the repo growing too
large.

TODO: Implement a pcap test as described in the above note.
TODO: Review the xsend.py class to see if some functionality has not been covered in all of these tests.
"""

import asyncio

import katsdpsigproc.accel as accel
import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
import test_parameters

import katgpucbf.xbgpu.xsend


@pytest.mark.parametrize("num_ants", test_parameters.array_size)
@pytest.mark.parametrize("num_channels", test_parameters.num_channels)
def test_send_simple(event_loop, num_ants, num_channels):
    """
    Tests the XEngineSPEADAbstractSend class in the xsend.py module.

    This test transmits a number of heaps from a XEngineSPEADAbstractSend object over a SPEAD2 inproc transport. A
    SPEAD2 receiver object then examines the transmitted heaps to ensure that the correct fields are transmitted with
    the correct values.

    This test does not generate random data as it will take much more compute to check that the random data is received
    correctly. I do not think that random data is necessary, as that would be checking that SPEAD2 is assembling heaps
    correctly which is a function of SPEAD2 not the XEngineSPEADAbstractSend class. Mangled heaps should be picked up
    in the SPEAD2 unit tests.

    Parameters
    ----------
    event_loop: AsyncIO Event Loop
        The event loop that the async events will be placed on. When running a unit test, this is a fixture provided
        by the pytest-asyncio module.
    num_ants: int
        The number of antennas that have been correlated.
    num_channels: int
        The number of frequency channels out of the FFT. NB: This is not the number of FFT channels per stream. The
        number of channels per stream is calculated from this value.
    """
    # 1. Configuration Parameters
    # 1.1 Fixed Parameters
    heaps_to_send = (
        20  # Number of heaps to transmit in the this test. I do not see a need for this number to be larger.
    )
    dump_interval_s = 0  # Normally 0.4 but we set it to as fast as possible so things run quickly.

    n_pols = 2
    complexity = 2
    sample_bits = 32

    # 1.2 Derived parameters

    # This integer division is so that when n_ants % num_channels !=0 then the remainder will be dropped. This will
    # only occur in the MeerKAT Extension correlator. Technically we will also need to consider the case where we round
    # up as some X-Engines will need to do this to capture all the channels, however that is not done in this test.
    # The // 4 is here because in the MeerKAT case, there are 4*num_ants multicast streams.
    n_channels_per_stream = num_channels // num_ants // 4
    n_baselines = (num_ants + 1) * (num_ants) // 2

    # 2. Create cuda context - all buffers created in the XEngineSPEADInprocSend object are created from this context.
    context = accel.create_some_context(device_filter=lambda x: x.is_cuda)

    # 3. Initialise SPEAD2 sender and receiver objects and link them together.

    # 3.1 Create the queue that will link the sender and receiver together.
    queue = spead2.InprocQueue()

    # 3.2 Create katgpucbf.xbgpu.xsend.XEngineSPEADInprocSend that will wrap a SPEAD2 send stream.
    sendStream = katgpucbf.xbgpu.xsend.XEngineSPEADInprocSend(
        n_ants=num_ants,
        n_channels_per_stream=n_channels_per_stream,
        n_pols=n_pols,
        dump_interval_s=dump_interval_s,
        channel_offset=n_channels_per_stream * 4,  # Arbitrary for now
        context=context,
        queue=queue,
    )

    # 3.3 Create a generic SPEAD2 receiver that will receive heaps from the XEngineSPEADInprocSend over the queue.
    thread_pool = spead2.ThreadPool()
    recvStream = spead2.recv.asyncio.Stream(thread_pool, spead2.recv.StreamConfig(max_heaps=100))
    recvStream.add_inproc_reader(queue)

    # 4. Define an async function to manage sending of X-Engine heaps. This function generates the heaps to be tested.
    async def send_process():
        """
        Run the transmit code asynchronously.

        This process sends a fixed number of heaps and then exits.
        """
        num_sent = 0

        # 4.1 Send the descriptor as the recvStream object needs it to interpret the received heaps correctly.
        sendStream.send_descriptor_heap()

        # 4.2 Run until a set number of heaps have been transferred.
        while num_sent < heaps_to_send:
            # 4.2.1 Get a free buffer to store the next heap - there is not always a free buffer available. This
            # function yields until one it available.
            buffer_wrapper = await sendStream.get_free_heap()

            # 4.2.2 Populate the buffer with dummy data - notice how we copy new values into the buffer, we dont
            # overwrite the buffer. Attempts to overwrite the buffer will throw an error. This is intended behavour as
            # the memory regions in the buffer have been configured for zero-copy sends.
            # The [:] syntax forces the data to be copied to the buffer. Without this, the buffer_wrapper variable
            # would point to the new created numpy array which is located in new location in memory and the old buffer
            # would be garbage collected.
            buffer_wrapper.buffer[:] = np.full(buffer_wrapper.buffer.shape, num_sent, np.uint8)

            # 4.2.3 Give the buffer back to the sendStream to transmit out onto the network. The timestamp is
            # multiplied by 0x1000000 so that its value is different from the values in the buffer_wrapper array.
            sendStream.send_heap(num_sent * 0x1000000, buffer_wrapper)
            num_sent += 1

    # 5. Define an async function to manage receiving of X-Engine heaps. This function checks that the data is correct.
    async def recv_process():
        """
        Run the receiver process asynchronously.

        This process receives the data transmitted by the send_process() funtion and ensures that it is received
        correctly.
        """
        num_received = 0
        ig = spead2.ItemGroup()

        # 5.1 Wait for the first packet to arrive - it is expected to be the SPEAD descriptor. Without the desciptor
        # the recvStream cannot interpret the heaps correctly.
        heap = await recvStream.get()
        items = ig.update(heap)
        assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

        # 5.2 Wait for the rest of the heaps to arrive and then end the function.
        while num_received < heaps_to_send:
            # 5.2.1 The next heap may not be available immediatly. This function waits asynchronously until a
            # heap arrives.
            heap = await recvStream.get()

            items = ig.update(heap)
            has_timestamp = False
            has_channel_offset = False
            has_xeng_raw = False

            for item in items.values():
                # 5.2.2 Check that the received heap has a timestamp item with the correct expected value.
                if item.id == 0x1600:
                    has_timestamp = True
                    assert (
                        item.value == num_received * 0x1000000
                    ), f"Timestamp value incorrect. Expected: {hex(num_received)}, actual {hex(item.value)}"

                # 5.2.2 Check that the received heap has a channel offset item with the correct expected value.
                if item.id == 0x4103:
                    has_channel_offset = True
                    assert (
                        item.value == n_channels_per_stream * 4
                    ), f"Channel offset incorrect. Expected: {hex(n_channels_per_stream * 4)}, actual: {hex(item.value)}"

                # 5.2.2 Check that the received heap has an xeng_raw data buffer item. Check that the buffer is the
                # correct size and that the values are all the expected value.
                if item.id == 0x1800:
                    has_xeng_raw = True
                    data_length_bytes = (
                        n_baselines * n_channels_per_stream * n_pols * n_pols * complexity * sample_bits // 8
                    )
                    assert (
                        item.value.size * 8 == data_length_bytes  # *8 as there are 64 bytes in a sample
                    ), f"xeng_raw data not correct size. Expected: {data_length_bytes} bytes, actual: {item.value.size} bytes."
                    assert (
                        item.value.dtype == np.uint64
                    ), f"xeng_raw dtype is {(item.value.dtype)}, dtype of uint64 expected."
                    assert np.all(item.value == num_received)

            assert (
                has_timestamp
            ), f"Received heap is missing timestamp item with ID {hex(katgpucbf.xbgpu.xsend.XEngineSPEADAbstractSend.TIMESTAMP_ID)}"
            assert (
                has_channel_offset
            ), f"Received heap is missing channel offset item with ID {hex(katgpucbf.xbgpu.xsend.XEngineSPEADAbstractSend.CHANNEL_OFFSET)}"
            assert (
                has_xeng_raw
            ), f"Received heap is missing xeng_raw data buffer item with ID {hex(katgpucbf.xbgpu.xsend.XEngineSPEADAbstractSend.DATA_ID)}"

            num_received += 1

    # 6. Function that will launch the send_process() and recv_process() in a parallel on a single asyncio loop.
    async def run():
        """Launch the send_process() function and recv_process() function in parallel in a single asyncio loop."""
        task1 = event_loop.create_task(send_process())
        task2 = event_loop.create_task(recv_process())
        await task1
        await task2

    # 7. Start the IO loop.
    event_loop.run_until_complete(run())


# A manual run useful when debugging the unit tests.
if __name__ == "__main__":
    np.set_printoptions(formatter={"int": hex})
    print("Running tests")
    event_loop = asyncio.get_event_loop()
    test_send_simple(event_loop, num_ants=64, num_channels=32768)
    print("Tests complete")
