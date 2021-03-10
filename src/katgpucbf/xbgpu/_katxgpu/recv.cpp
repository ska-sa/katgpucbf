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

allocator::allocator(stream &recv) : m_recv(recv)
{
}

auto allocator::allocate(std::size_t size, void *hint) -> pointer
{
    if (hint)
    {
        void *ptr = m_recv.allocate(size, *reinterpret_cast<spead2::recv::packet_header *>(hint));
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

stream::stream(int n_ants, int n_channels, int n_samples_per_channel, int n_pols, int sample_bits, int timestamp_step,
               size_t heaps_per_fengine_per_chunk, ringbuffer_t &ringbuffer, int thread_affinity, bool use_gdrcopy)
    : spead2::thread_pool(1, thread_affinity < 0 ? std::vector<int>{} : std::vector<int>{thread_affinity}),
      spead2::recv::stream(*static_cast<thread_pool *>(this),
                           spead2::recv::stream_config()
                               .set_max_heaps(n_ants * heaps_per_fengine_per_chunk * 10)
                               .set_allow_unsized_heaps(false)
                               .set_memory_allocator(std::make_shared<katxgpu::recv::allocator>(*this))
                               .set_memcpy(use_gdrcopy ? spead2::MEMCPY_NONTEMPORAL : spead2::MEMCPY_STD)
                               .set_allow_out_of_order(true)),
      n_ants(n_ants), n_channels(n_channels), n_samples_per_channel(n_samples_per_channel), n_pols(n_pols),
      m_sample_bits(sample_bits), m_timestamp_step(timestamp_step),
      m_heaps_per_fengine_per_chunk(heaps_per_fengine_per_chunk),
      m_packet_bytes(n_samples_per_channel * n_pols * m_complexity / 8 * sample_bits),
      m_chunk_bytes(m_packet_bytes * n_channels * n_ants * heaps_per_fengine_per_chunk), m_ringbuffer(ringbuffer)
{
    if (sample_bits != 8)
        throw std::invalid_argument("sample_bits must equal 8 - logic for other sample sizes has not been tested.");
    if (n_pols != 2)
        throw std::invalid_argument(
            "n_pols must equal 8 - logic for other types of polarisations has not been added yet.");
    if (n_ants <= 0)
        throw std::invalid_argument("n_ants must be greater than 0");
    if (n_channels <= 0)
        throw std::invalid_argument("n_channels must be greater than 0");
    if (n_samples_per_channel <= 0)
        throw std::invalid_argument("n_samples_per_channel must be greater than 0");
    if (m_packet_bytes <= 0)
        throw std::invalid_argument("m_packet_bytes must be greater than 0");
}

stream::~stream()
{
    this->stop();
}

void stream::add_chunk(std::unique_ptr<chunk> &&c)
{
    if (buffer_size(c->storage) != m_chunk_bytes)
        throw std::invalid_argument("Chunk has incorrect size");

    c->present.clear();
    c->present.resize(n_ants * m_heaps_per_fengine_per_chunk);
    {
        std::lock_guard<std::mutex> lock(m_free_chunks_lock);
        m_free_chunks_stack.push(std::move(c));
    }
    m_free_chunks_sem.put();
}

void stream::grab_chunk(std::int64_t timestamp)
{
    pre_wait_chunk();
    semaphore_get(m_free_chunks_sem);
    post_wait_chunk();
    {
        std::lock_guard<std::mutex> lock(m_free_chunks_lock);
        assert(!m_free_chunks_stack.empty());
        m_active_chunks_queue.push_back(std::move(m_free_chunks_stack.top()));
        m_free_chunks_stack.pop();
    }
    m_active_chunks_queue.back()->timestamp = timestamp;
}

bool stream::flush_chunk()
{
    assert(!m_active_chunks_queue.empty());
    try
    {
        pre_ringbuffer_push();
        // Take the oldest active chunk and add it to the ringbuffer.
        m_ringbuffer.push(std::move(m_active_chunks_queue[0]));
        post_ringbuffer_push();
        // Once the chunk has been put on the ringbuffer, remove it from the active chunks queue.
        m_active_chunks_queue.pop_front();
        return true;
    }
    catch (spead2::ringbuffer_stopped &e)
    {
        // TODO: add to counter
        return false;
    }
}

std::tuple<void *, chunk *, std::size_t> stream::calculate_packet_destination(std::int64_t timestamp,
                                                                              std::int64_t fengine_id, chunk &c)
{
    // The timestamp index is a number of  m_timestamp_steps from the chunks base timestamp.
    std::size_t timestamp_idx = (timestamp - c.timestamp) / (m_timestamp_step);

    // The fengine index is very simple to calculate/
    std::size_t fengine_idx = fengine_id;

    // The heap index in the chunk is a combination of the heap timestamp and f-engine id.
    std::size_t heap_idx = timestamp_idx * n_ants + fengine_idx;

    // The buffer offset is the number of bytes the current heap is offset from the chunk buffer's base address. It is
    // basically the heap index multiplied by the heap size (n_channels * m_packet_bytes).
    std::size_t buffer_offset = heap_idx * n_channels * m_packet_bytes;

    // Use the buffer offset to calculate the actual pointer to allocate the heap to.
    void *ptr = boost::asio::buffer_cast<std::uint8_t *>(c.storage) + buffer_offset;

    // Return all pertinent values in a tuple.
    return std::make_tuple(ptr, &c, heap_idx);
}

std::tuple<void *, chunk *, std::size_t> stream::calculate_packet_destination(std::int64_t timestamp,
                                                                              std::int64_t fengine_id)
{
    // 1. The very first heap is used to populate the first timestamp.
    if (m_first_timestamp == -1)
    {
        m_first_timestamp = timestamp;
        grab_chunk(m_first_timestamp);
    }
    // 2. We check that the heap timestamp is a multiple of m_timestamp_step from the first timestamp. If not, then
    // something has gone very wrong.
    if ((timestamp - m_first_timestamp) % m_timestamp_step != 0)
    {
        spead2::log_info("Timestamp is broken");
        return std::make_tuple(nullptr, nullptr, 0);
    }

    // 3. We search through all active chunks in the m_active_chunks_queue queue to see if the new heap falls within one
    // of these chunks.
    std::int64_t base = m_active_chunks_queue[0]->timestamp;
    for (const auto &c : m_active_chunks_queue)
    {
        if (timestamp >= c->timestamp && timestamp < c->timestamp + m_timestamp_step * m_heaps_per_fengine_per_chunk)
        {
            // 3.1. If the heap falls within a specific active chunk we use the timestamp and the fengine_id to
            // determine the exact destination in the timestamp by calling the overloaded calculate_packet_destination()
            // function.
            return calculate_packet_destination(timestamp, fengine_id, *c);
        }
    }

    // 4. If the heap does not fall within the active chunks, then we need to decide what to do with it.
    // 4.1 If the heap is older than the oldest active heap then we drops it.
    if (timestamp < base)
    {
        spead2::log_info("Seem to have gone backwards in time");
        return std::make_tuple(nullptr, nullptr, 0);
    }
    else
    {
        // 4.2. If the heap is newer in time than the last active heap, then we need to add a new chunk to the active
        // chunks list. We can only have max_active numbers of chunks on the queue at any one time. So we need to make
        // room if needed. Usually the next chunk will be the next sequential chunk. If not, we flush all chunks rather
        // than leaving m_active_chunks_queue discontiguous.
        //
        // This discontiunity will happen very rarely as data is being received from multiple senders and the chance of
        // all senders being down is negligible. The most likely cause of this issue would be an interruption in the
        // link between the katxgpu host server and the network.
        std::size_t max_active = 2;
        std::int64_t start = m_active_chunks_queue.back()->timestamp + m_timestamp_step * m_heaps_per_fengine_per_chunk;

        // 4.2.1 If the next chunk is not the next sequential one, then the pipeline is flushed. This is done by setting
        // max_chunks to zero.
        if (timestamp >=
            start +
                std::int64_t(m_timestamp_step *
                             m_heaps_per_fengine_per_chunk)) // True if the next chunk is not the next sequential one
        {
            spead2::log_info("The next chunk is not the next sequential one. SPEAD RX pipeline is being flushed");
            // 4.2.2 The start timestamp of the next chunk needs to be aligned correctly to multiples of
            // m_timestamp_step * m_heaps_per_fengine_per_chunk. The step variable below is calculated using integer
            // division to determine how many multiples the new chunk is off from the old.
            int64_t step = (timestamp - start) / (m_timestamp_step * m_heaps_per_fengine_per_chunk);
            start += (step * m_timestamp_step * m_heaps_per_fengine_per_chunk);
            max_active = 0;
        }

        // 4.2.2 Flush the active chunks queue. Only one chunk is flushed if its a continious stream of data - this is
        // the most likley case. Otherwise all chunks are flushed - this will mostly likely occur if a stream is
        // interrupted or stopped and a new stream is started.
        while (m_active_chunks_queue.size() > max_active)
        {
            // spead2::log_info("Flushing chunks %1% %2%",m_active_chunks_queue.size(),max_active);
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
        grab_chunk(start);
        return calculate_packet_destination(timestamp, fengine_id, *m_active_chunks_queue.back());
    }
}

void *stream::allocate(std::size_t size, spead2::recv::packet_header &packet)
{
    // 1. Perform some basic checks on the received packet of the new heap to confirm that it is what we expect.
    if (size != m_packet_bytes * n_channels)
    {
        spead2::log_info("Allocating incorrect size");
        return nullptr;
    }
    std::int64_t timestamp = -1;
    std::int64_t fengine_id = -1;
    spead2::recv::pointer_decoder decoder(packet.heap_address_bits);

    // 2. Extract the timestamp and F-Engine ID from the packet
    for (int i = 0; i < packet.n_items; i++)
    {

        spead2::item_pointer_t pointer;
        pointer = spead2::load_be<spead2::item_pointer_t>(packet.pointers + i * sizeof(pointer));
        if (decoder.is_immediate(pointer) && decoder.get_id(pointer) == TIMESTAMP_ID)
        {
            timestamp = decoder.get_immediate(pointer);
        }

        if (decoder.is_immediate(pointer) && decoder.get_id(pointer) == FENGINE_ID)
        {
            fengine_id = decoder.get_immediate(pointer);
        }
    }
    // This should not happen - if either is zero, it means some expected fields are missing from this packet.
    if (timestamp == -1 || fengine_id == -1)
        return nullptr;

    // 3. Use the fengine_id and timestamp to decide which chunk the packet must be allocated to.
    return std::get<0>(calculate_packet_destination(timestamp, fengine_id));
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
    std::int64_t timestamp = -1;
    std::int64_t fengine_id = -1;
    void *actual_ptr = nullptr;
    std::size_t length = 0;

    void *expected_ptr;
    chunk *c;
    std::size_t heap_idx;

    for (const auto &item : heap.get_items())
    {
        if (item.id == TIMESTAMP_ID && item.is_immediate)
            timestamp = item.immediate_value;
        if (item.id == FENGINE_ID && item.is_immediate)
            fengine_id = item.immediate_value;
        else if (item.id == DATA_ID)
        {
            actual_ptr = item.ptr;
            length = item.length;
        }
    }

    // 3. Perform some basic checks on the heap.
    if (timestamp == -1 || length == 0)
    {
        spead2::log_info("It's not a valid digitiser packet");
        return;
    }
    if (length != m_packet_bytes * n_channels)
    {
        spead2::log_info("Heap size incorrect. Received (%1%) bytes, expected (%2%) bytes", length,
                         m_packet_bytes * n_channels);
        return;
    }

    // 4. Run the calculate_packet_destination() function again (It was last called in the allocater when the first
    // packet was received.). This function should ideally return a pointer equal to the current heap's pointer. This
    // would not be equal if for some reason the chunk this heap belongs to has been moved off of the active pile - this
    // means a stale heap was received. If this occurs frequently, the number of allowed active chunks is probably not
    // high enough (set it higher in the calculate_packet_destination(...) function.).
    std::tie(expected_ptr, c, heap_idx) = calculate_packet_destination(timestamp, fengine_id);
    if (expected_ptr != actual_ptr)
    {
        spead2::log_info("This should only happen if we receive data that is too old (%1%)", m_timestamp_step);
        // TODO: Figure out what to do in this case.
        return;
    }

    // 5. Mark the heap location in the chunk as filled.
    c->present[heap_idx] = true;
}

void stream::stop_received()
{
    while (!m_active_chunks_queue.empty())
        if (!flush_chunk())
            break;
    m_ringbuffer.stop();
    spead2::recv::stream::stop_received();
}

void stream::add_buffer_reader(pybind11::buffer buffer)
{
    // This view object needs to be held as long as the receiver is making use of the buffer. If it is released, Python
    // will release the buffer back to the OS causing segfaults when C++ tries to access the buffer. Took me a while to
    // figure this out - dont make my mistakes.
    m_view = katxgpu::request_buffer_info(buffer, PyBUF_C_CONTIGUOUS);
    // In normal SPEAD2, a buffer_reader wraps a mem reader and handles all the casting seen in the line below. In the
    // katxgpu case, I just copied the logic of the buffer_reader without creating the class.
    emplace_reader<spead2::recv::mem_reader>(reinterpret_cast<const std::uint8_t *>(m_view.ptr),
                                             m_view.itemsize * m_view.size);
}

void stream::add_udp_pcap_file_reader(const std::string &filename)
{
    emplace_reader<spead2::recv::udp_pcap_file_reader>(filename);
}

void stream::add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                                const std::string &interface_address, std::size_t buffer_size, int comp_vector,
                                int max_poll)
{
    spead2::recv::udp_ibv_config config;
    for (const auto &ep : endpoints)
        config.add_endpoint(boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(ep.first), ep.second));
    config.set_interface_address(boost::asio::ip::address::from_string(interface_address));
    config.set_max_size(m_packet_bytes + 96); // Header is 12 fields of 8 bytes each: So 96 bytes of header
    config.set_buffer_size(buffer_size);
    config.set_comp_vector(comp_vector);
    config.set_max_poll(max_poll);
    emplace_reader<spead2::recv::udp_ibv_reader>(config);
}

std::size_t stream::get_chunk_bytes() const
{
    return m_chunk_bytes;
}

stream::ringbuffer_t &stream::get_ringbuffer()
{
    return m_ringbuffer;
}

const stream::ringbuffer_t &stream::get_ringbuffer() const
{
    return m_ringbuffer;
}

void stream::stop()
{
    m_ringbuffer.stop();
    spead2::recv::stream::stop();
}

} // namespace katxgpu::recv
