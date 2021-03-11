#include "recv.h"
#include "py_common.h"

#include <cassert>
#include <iostream> //For debugging
#include <map>
#include <spead2/common_endian.h>
#include <spead2/common_logging.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_mem.h>
#include <spead2/recv_udp_pcap.h>
#include <stdexcept>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace katxgpu::recv
{

static constexpr int HEAP_OFFSET_ID = 0x0003;
static constexpr int TIMESTAMP_ID = 0x1600;
static constexpr int FENGINE_ID = 0x4101;
static constexpr int DATA_ID = 0x4300;

allocator::allocator(stream &recv) : m_receiverStream(recv)
{
}

auto allocator::allocate(std::size_t size, void *hint) -> pointer
{
    if (hint)
    {
        void *ptr = m_receiverStream.allocate(size, *reinterpret_cast<spead2::recv::packet_header *>(hint));
        if (ptr)
            return pointer(reinterpret_cast<std::uint8_t *>(ptr),
                           deleter(shared_from_this(), (void *)std::uintptr_t(true)));
    }
    return spead2::memory_allocator::allocate(size, hint);
}

void allocator::free(std::uint8_t *ptr, void *user)
{
    if (!user)
        delete[] ptr;
}

stream::stream(int iNumAnts, int iNumChannels, int iNumSamplesPerChannel, int iNumPols, int iSampleBits,
               int iTimestampStep, size_t iHeapsPerFenginePerChunk, ringbuffer_t &completedChunksRingbuffer,
               int iThreadAffinity, bool bUseGDRCopy)
    : spead2::thread_pool(1, iThreadAffinity < 0 ? std::vector<int>{} : std::vector<int>{iThreadAffinity}),
      spead2::recv::stream(*static_cast<thread_pool *>(this),
                           spead2::recv::stream_config()
                               .set_max_heaps(iNumAnts * iHeapsPerFenginePerChunk * 10)
                               .set_allow_unsized_heaps(false)
                               .set_memory_allocator(std::make_shared<katxgpu::recv::allocator>(*this))
                               .set_memcpy(bUseGDRCopy ? spead2::MEMCPY_NONTEMPORAL : spead2::MEMCPY_STD)
                               .set_allow_out_of_order(true)),
      m_iNumAnts(iNumAnts), m_iNumChannels(iNumChannels), m_iNumSamplesPerChannel(iNumSamplesPerChannel),
      m_iNumPols(iNumPols), m_iSampleBits(iSampleBits), m_iTimestampStep(iTimestampStep),
      m_iHeapsPerFenginePerChunk(iHeapsPerFenginePerChunk),
      m_ulPacketSize_bytes(iNumSamplesPerChannel * iNumPols * m_iComplexity / 8 * iSampleBits),
      m_ulChunkSize_bytes(m_ulPacketSize_bytes * iNumChannels * iNumAnts * iHeapsPerFenginePerChunk),
      m_completedChunksRingbuffer(completedChunksRingbuffer)
{
    if (iSampleBits != 8)
        throw std::invalid_argument("iSampleBits must equal 8 - logic for other sample sizes has not been tested.");
    if (iNumPols != 2)
        throw std::invalid_argument(
            "n_pols must equal 8 - logic for other types of polarisations has not been added yet.");
    if (iNumAnts <= 0)
        throw std::invalid_argument("iNumAnts must be greater than 0");
    if (iNumChannels <= 0)
        throw std::invalid_argument("iNumChannels must be greater than 0");
    if (iNumSamplesPerChannel <= 0)
        throw std::invalid_argument("iNumSamplesPerChannel must be greater than 0");
    if (m_ulPacketSize_bytes <= 0)
        throw std::invalid_argument("m_ulPacketSize_bytes must be greater than 0");
}

stream::~stream()
{
    this->stop();
}

void stream::add_chunk(std::unique_ptr<chunk> &&pChunk)
{
    if (buffer_size(pChunk->m_storage) != m_ulChunkSize_bytes)
        throw std::invalid_argument("Chunk has incorrect size");

    pChunk->m_vbPacketPresent.clear();
    pChunk->m_vbPacketPresent.resize(m_iNumAnts * m_iHeapsPerFenginePerChunk);
    {
        std::lock_guard<std::mutex> lock(m_freeChunksLock);
        m_freeChunksStack.push(std::move(pChunk));
    }
    m_freeChunksSemaphore.put();
}

void stream::grab_chunk(std::int64_t i64Timestamp)
{
    pre_wait_chunk();
    semaphore_get(m_freeChunksSemaphore);
    post_wait_chunk();
    {
        std::lock_guard<std::mutex> lock(m_freeChunksLock);
        assert(!m_freeChunksStack.empty());
        m_activeChunksQueue.push_back(std::move(m_freeChunksStack.top()));
        m_freeChunksStack.pop();
    }
    m_activeChunksQueue.back()->m_i64timestamp = i64Timestamp;
}

bool stream::flush_chunk()
{
    assert(!m_activeChunksQueue.empty());
    try
    {
        pre_ringbuffer_push();
        // Take the oldest active chunk and add it to the ringbuffer.
        m_completedChunksRingbuffer.push(std::move(m_activeChunksQueue[0]));
        post_ringbuffer_push();
        // Once the chunk has been put on the ringbuffer, remove it from the active chunks queue.
        m_activeChunksQueue.pop_front();
        return true;
    }
    catch (spead2::ringbuffer_stopped &e)
    {
        // TODO: add to counter
        return false;
    }
}

std::tuple<void *, chunk *, std::size_t> stream::calculate_packet_destination(std::int64_t i64Timestamp,
                                                                              std::int64_t i64FengineID, chunk &chunk)
{
    // The timestamp index is a number of  m_timestamp_steps from the chunks base timestamp.
    std::size_t ulTimestampIndex = (i64Timestamp - chunk.m_i64timestamp) / (m_iTimestampStep);

    // The fengine index is very simple to calculate
    std::size_t ulFengineIndex = i64FengineID;

    // The heap index in the chunk is a combination of the heap timestamp and f-engine id.
    std::size_t ulHeapIndex = ulTimestampIndex * m_iNumAnts + ulFengineIndex;

    // The buffer offset is the number of bytes the current heap is offset from the chunk buffer's base address. It is
    // basically the heap index multiplied by the heap size (m_iNumChannels * m_ulPacketSize_bytes).
    std::size_t ulBufferOffset_bytes = ulHeapIndex * m_iNumChannels * m_ulPacketSize_bytes;

    // Use the buffer offset to calculate the actual pointer to allocate the heap to.
    void *pBufferPointer = boost::asio::buffer_cast<std::uint8_t *>(chunk.m_storage) + ulBufferOffset_bytes;

    // Return all pertinent values in a tuple.
    return std::make_tuple(pBufferPointer, &chunk, ulHeapIndex);
}

std::tuple<void *, chunk *, std::size_t> stream::calculate_packet_destination(std::int64_t i64Timestamp,
                                                                              std::int64_t i64FengineID)
{
    // 1. The very first heap is used to populate the first timestamp.
    if (m_i64FirstTimestamp == -1)
    {
        m_i64FirstTimestamp = i64Timestamp;
        grab_chunk(m_i64FirstTimestamp);
    }
    // 2. We check that the heap timestamp is a multiple of m_iTimestampStep from the first timestamp. If not, then
    // something has gone very wrong.
    if ((i64Timestamp - m_i64FirstTimestamp) % m_iTimestampStep != 0)
    {
        spead2::log_info("Timestamp is broken");
        return std::make_tuple(nullptr, nullptr, 0);
    }

    // 3. We search through all active chunks in the m_activeChunksQueue queue to see if the new heap falls within one
    // of these chunks.
    std::int64_t i64BaseTimestamp = m_activeChunksQueue[0]->m_i64timestamp;
    for (const auto &chunk : m_activeChunksQueue)
    {
        if (i64Timestamp >= chunk->m_i64timestamp &&
            i64Timestamp < chunk->m_i64timestamp + m_iTimestampStep * m_iHeapsPerFenginePerChunk)
        {
            // 3.1. If the heap falls within a specific active chunk we use the timestamp and the fengine_id to
            // determine the exact destination in the timestamp by calling the overloaded calculate_packet_destination()
            // function.
            return calculate_packet_destination(i64Timestamp, i64FengineID, *chunk);
        }
    }

    // 4. If the heap does not fall within the active chunks, then we need to decide what to do with it.
    // 4.1 If the heap is older than the oldest active heap then we drops it.
    if (i64Timestamp < i64BaseTimestamp)
    {
        spead2::log_info("Seem to have gone backwards in time");
        return std::make_tuple(nullptr, nullptr, 0);
    }
    else
    {
        // 4.2. If the heap is newer in time than the last active heap, then we need to add a new chunk to the active
        // chunks list. We can only have ulMaxActiveChunks numbers of chunks on the queue at any one time. So we need to
        // make room if needed. Usually the next chunk will be the next sequential chunk. If not, we flush all chunks
        // rather than leaving m_activeChunksQueue discontiguous.
        //
        // This discontiunity will happen very rarely as data is being received from multiple senders and the chance of
        // all senders being down is negligible. The most likely cause of this issue would be an interruption in the
        // link between the katxgpu host server and the network.
        std::size_t ulMaxActiveChunks = 2;
        std::int64_t i64StartTimestamp =
            m_activeChunksQueue.back()->m_i64timestamp + m_iTimestampStep * m_iHeapsPerFenginePerChunk;

        // 4.2.1 If the next chunk is not the next sequential one, then the pipeline is flushed. This is done by setting
        // max_chunks to zero.
        if (i64Timestamp >=
            i64StartTimestamp +
                std::int64_t(m_iTimestampStep *
                             m_iHeapsPerFenginePerChunk)) // True if the next chunk is not the next sequential one
        {
            spead2::log_info("The next chunk is not the next sequential one. SPEAD RX pipeline is being flushed");
            // 4.2.2 The i64StartTimestamp of the next chunk needs to be aligned correctly to multiples of
            // m_iTimestampStep * m_iHeapsPerFenginePerChunk. The step variable below is calculated using integer
            // division to determine how many multiples the new chunk is off from the old.
            int64_t step = (i64Timestamp - i64StartTimestamp) / (m_iTimestampStep * m_iHeapsPerFenginePerChunk);
            i64StartTimestamp += (step * m_iTimestampStep * m_iHeapsPerFenginePerChunk);
            ulMaxActiveChunks = 0;
        }

        // 4.2.2 Flush the active chunks queue. Only one chunk is flushed if its a continious stream of data - this is
        // the most likley case. Otherwise all chunks are flushed - this will mostly likely occur if a stream is
        // interrupted or stopped and a new stream is started.
        while (m_activeChunksQueue.size() > ulMaxActiveChunks)
        {
            // spead2::log_info("Flushing chunks %1% %2%",m_activeChunksQueue.size(),ulMaxActiveChunks);
            if (!flush_chunk())
            {
                // 4.2.3. This should not really happen - flush chunk should normally return true.
                spead2::log_info("No chunk to flush");
                // ringbuffer was stopped, so no point in continuing.
                return std::make_tuple(nullptr, nullptr, 0);
            }
        }

        // 4.2.3 Grab a new chunk from the free_chunk stack and add it to the active chunks queue. The received heap is
        // assigned to this chunk.
        grab_chunk(i64StartTimestamp);
        return calculate_packet_destination(i64Timestamp, i64FengineID, *m_activeChunksQueue.back());
    }
}

void *stream::allocate(std::size_t ulHeapSize_bytes, spead2::recv::packet_header &receivedPacket)
{
    // 1. Perform some basic checks on the received packet of the new heap to confirm that it is what we expect.
    if (ulHeapSize_bytes != m_ulPacketSize_bytes * m_iNumChannels)
    {
        spead2::log_info("Allocating incorrect size");
        return nullptr;
    }
    std::int64_t i64Timestamp = -1;
    std::int64_t i64FengineID = -1;
    spead2::recv::pointer_decoder decoder(receivedPacket.heap_address_bits);

    // 2. Extract the timestamp and F-Engine ID from the packet
    for (int i = 0; i < receivedPacket.n_items; i++)
    {

        spead2::item_pointer_t itemPointer;
        itemPointer = spead2::load_be<spead2::item_pointer_t>(receivedPacket.pointers + i * sizeof(itemPointer));
        if (decoder.is_immediate(itemPointer) && decoder.get_id(itemPointer) == TIMESTAMP_ID)
        {
            i64Timestamp = decoder.get_immediate(itemPointer);
        }

        if (decoder.is_immediate(itemPointer) && decoder.get_id(itemPointer) == FENGINE_ID)
        {
            i64FengineID = decoder.get_immediate(itemPointer);
        }
    }
    // This should not happen - if either is zero, it means some expected fields are missing from this packet.
    if (i64Timestamp == -1 || i64FengineID == -1)
        return nullptr;

    // 3. Use the fengine_id and timestamp to decide which chunk the packet must be allocated to.
    return std::get<0>(calculate_packet_destination(i64Timestamp, i64FengineID));
}

void stream::heap_ready(spead2::recv::live_heap &&live_heap)
{
    // 1. Check that all required packets in a heap have been received.
    if (!live_heap.is_complete())
    {
        spead2::log_info("Heap not complete. Received Length %1% Expected Length %2%.", live_heap.get_received_length(),
                         live_heap.get_heap_length());
        return;
    }

    // 2. Get the required heap details
    spead2::recv::heap heap(std::move(live_heap));
    std::int64_t i64Timestamp = -1;
    std::int64_t i64FengineID = -1;
    void *pActualDataPtr = nullptr;
    std::size_t ulHeapSize_bytes = 0;

    void *pExpectedDataPtr;
    chunk *pChunk;
    std::size_t ulHeapIndex;

    for (const auto &item : heap.get_items())
    {
        if (item.id == TIMESTAMP_ID && item.is_immediate)
            i64Timestamp = item.immediate_value;
        if (item.id == FENGINE_ID && item.is_immediate)
            i64FengineID = item.immediate_value;
        else if (item.id == DATA_ID)
        {
            pActualDataPtr = item.ptr;
            ulHeapSize_bytes = item.length;
        }
    }

    // 3. Perform some basic checks on the heap.
    if (i64Timestamp == -1 || ulHeapSize_bytes == 0)
    {
        spead2::log_info("It's not a valid digitiser packet");
        return;
    }
    if (ulHeapSize_bytes != m_ulPacketSize_bytes * m_iNumChannels)
    {
        spead2::log_info("Heap size incorrect. Received (%1%) bytes, expected (%2%) bytes", ulHeapSize_bytes,
                         m_ulPacketSize_bytes * m_iNumChannels);
        return;
    }

    // 4. Run the calculate_packet_destination() function again (It was last called in the allocater when the first
    // packet was received.). This function should ideally return a pointer equal to the current heap's pointer. This
    // would not be equal if for some reason the chunk this heap belongs to has been moved off of the active pile - this
    // means a stale heap was received. If this occurs frequently, the number of allowed active chunks is probably not
    // high enough (set it higher in the calculate_packet_destination(...) function.).
    std::tie(pExpectedDataPtr, pChunk, ulHeapIndex) = calculate_packet_destination(i64Timestamp, i64FengineID);
    if (pExpectedDataPtr != pActualDataPtr)
    {
        spead2::log_info("This should only happen if we receive data that is too old (%1%)", m_iTimestampStep);
        // TODO: Figure out what to do in this case.
        return;
    }

    // 5. Mark the heap location in the chunk as filled.
    pChunk->m_vbPacketPresent[ulHeapIndex] = true;
}

void stream::stop_received()
{
    while (!m_activeChunksQueue.empty())
        if (!flush_chunk())
            break;
    m_completedChunksRingbuffer.stop();
    spead2::recv::stream::stop_received();
}

void stream::add_buffer_reader(pybind11::buffer buffer)
{
    // This view object needs to be held as long as the receiver is making use of the buffer. If it is released, Python
    // will release the buffer back to the OS causing segfaults when C++ tries to access the buffer. Took me a while to
    // figure this out - dont make my mistakes.
    m_BufferView = katxgpu::request_buffer_info(buffer, PyBUF_C_CONTIGUOUS);
    // In normal SPEAD2, a buffer_reader wraps a mem reader and handles all the casting seen in the line below. In the
    // katxgpu case, I just copied the logic of the buffer_reader without creating the class.
    emplace_reader<spead2::recv::mem_reader>(reinterpret_cast<const std::uint8_t *>(m_BufferView.ptr),
                                             m_BufferView.itemsize * m_BufferView.size);
}

void stream::add_udp_pcap_file_reader(const std::string &strFilename)
{
    emplace_reader<spead2::recv::udp_pcap_file_reader>(strFilename);
}

void stream::add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &vEndpoints,
                                const std::string &strInterfaceAddress, std::size_t ulBufferSize_bytes, int iCompVector,
                                int iMaxPoll)
{
    spead2::recv::udp_ibv_config streamConfig;
    for (const auto &ep : vEndpoints)
        streamConfig.add_endpoint(
            boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(ep.first), ep.second));
    streamConfig.set_interface_address(boost::asio::ip::address::from_string(strInterfaceAddress));
    streamConfig.set_max_size(m_ulPacketSize_bytes + 96); // Header is 12 fields of 8 bytes each: So 96 bytes of header
    streamConfig.set_buffer_size(ulBufferSize_bytes);
    streamConfig.set_comp_vector(iCompVector);
    streamConfig.set_max_poll(iMaxPoll);
    emplace_reader<spead2::recv::udp_ibv_reader>(streamConfig);
}

std::size_t stream::get_chunk_bytes() const
{
    return m_ulChunkSize_bytes;
}

stream::ringbuffer_t &stream::get_ringbuffer()
{
    return m_completedChunksRingbuffer;
}

const stream::ringbuffer_t &stream::get_ringbuffer() const
{
    return m_completedChunksRingbuffer;
}

void stream::stop()
{
    m_completedChunksRingbuffer.stop();
    spead2::recv::stream::stop();
}

} // namespace katxgpu::recv
