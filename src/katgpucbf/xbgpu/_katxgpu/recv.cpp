#include "recv.h"
#include <cassert>
#include <iostream> // TODO: for debugging
#include <map>      // TODO: workaround for it missing in recv_heap.h
#include <spead2/common_endian.h>
#include <spead2/common_logging.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_udp_pcap.h>
#include <stdexcept>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

namespace katxgpu::recv
{

static constexpr int HEAP_OFFSET_ID = 0x0003;
static constexpr int TIMESTAMP_ID = 0x1600;
static constexpr int FENGINE_ID = 0x4101;
static constexpr int DATA_ID = 0x4300;

allocator::allocator(stream &recv) : recv(recv)
{
}

auto allocator::allocate(std::size_t size, void *hint) -> pointer
{
    // spead2::log_info("Shared allocator");
    if (hint)
    {
        void *ptr = recv.allocate(size, *reinterpret_cast<spead2::recv::packet_header *>(hint));
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
      spead2::recv::stream(
          *static_cast<thread_pool *>(this),
          spead2::recv::stream_config()
              .set_max_heaps(n_ants *
                             heaps_per_fengine_per_chunk) // Set max heaps needs to be large enough to accomodate
                                                          // packets interleaved from multiple F-Engines.
              .set_allow_unsized_heaps(false)
              .set_memory_allocator(std::make_shared<katxgpu::recv::allocator>(*this))
              .set_memcpy(use_gdrcopy ? spead2::MEMCPY_NONTEMPORAL : spead2::MEMCPY_STD)
              .set_allow_out_of_order(true)),
      n_ants(n_ants), n_channels(n_channels), n_samples_per_channel(n_samples_per_channel), n_pols(n_pols),
      sample_bits(sample_bits), timestamp_step(timestamp_step),
      heaps_per_fengine_per_chunk(heaps_per_fengine_per_chunk),
      packet_bytes(n_samples_per_channel * n_pols * complexity / 8 * sample_bits),
      chunk_packets(n_channels * n_ants * heaps_per_fengine_per_chunk), chunk_bytes(packet_bytes * chunk_packets),
      ringbuffer(ringbuffer)
{
    // py::print("Stream Created");

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
    if (packet_bytes <= 0)
        throw std::invalid_argument("packet_bytes must be greater than 0");
    if (chunk_packets <= 0)
        throw std::invalid_argument("n_channels * n_ants * heaps_per_fengine_per_chunk must be greater than 0");

    // spead2::log_info("a: %1% c: %2% t: %3% p: %4% packet bytes %5% chunk bytes: %6%", n_ants, n_channels,
    //                n_samples_per_channel, n_pols, packet_bytes, chunk_bytes);
}

stream::~stream()
{
    stop();
}

void stream::add_chunk(std::unique_ptr<chunk> &&c)
{
    if (buffer_size(c->storage) != chunk_bytes)
        throw std::invalid_argument("Chunk has incorrect size");

    c->present.clear();
    c->present.resize(n_ants * heaps_per_fengine_per_chunk);
    {
        std::lock_guard<std::mutex> lock(free_chunks_lock);
        free_chunks.push(std::move(c));
    }
    free_chunks_sem.put();
}

void stream::grab_chunk(std::int64_t timestamp)
{
    pre_wait_chunk();
    semaphore_get(free_chunks_sem);
    post_wait_chunk();
    {
        std::lock_guard<std::mutex> lock(free_chunks_lock);
        assert(!free_chunks.empty());
        active_chunks.push_back(std::move(free_chunks.top()));
        free_chunks.pop();
    }
    active_chunks.back()->timestamp = timestamp;
}

bool stream::flush_chunk()
{
    assert(!active_chunks.empty());
    try
    {
        pre_ringbuffer_push();
        ringbuffer.push(std::move(active_chunks[0]));
        post_ringbuffer_push();
        active_chunks.pop_front();
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
    // spead2::log_info("Determine packet position in chunk specific %1% %2% %3%",timestamp,c.timestamp);
    std::size_t timestamp_idx = (timestamp - c.timestamp) / (timestamp_step);
    std::size_t fengine_idx = fengine_id;
    std::size_t heap_idx = timestamp_idx * n_ants + fengine_idx;
    std::size_t byte_idx = heap_idx * n_channels * packet_bytes;
    // spead2::log_info("\tTimestamp Index (%1%), Fengine Index (%2%), HeapIdx (%3%), Byte Index (%4%)", timestamp_idx,
    //                 fengine_idx, heap_idx, byte_idx);
    void *ptr = boost::asio::buffer_cast<std::uint8_t *>(c.storage) + byte_idx;
    return std::make_tuple(ptr, &c, heap_idx);
}

std::tuple<void *, chunk *, std::size_t> stream::calculate_packet_destination(std::int64_t timestamp,
                                                                              std::int64_t fengine_id)
{
    //  spead2::log_info("Determine packet position in chunk broad");
    if (first_timestamp == -1)
    {
        first_timestamp = timestamp;
        grab_chunk(first_timestamp);
    }
    if ((timestamp - first_timestamp) % timestamp_step != 0)
    {
        // TODO: log/count. The timestamp is broken.
        spead2::log_info("Timestamp is broken");
        return std::make_tuple(nullptr, nullptr, 0);
    }
    std::int64_t base = active_chunks[0]->timestamp;
    // for (const auto &c : active_chunks)
    // {
    //     spead2::log_info("\t Timestamp %1%, Timestamp Low %2%, Timestamp High %3%", timestamp, c->timestamp,
    //                      c->timestamp + timestamp_step * heaps_per_fengine_per_chunk);
    // }

    for (const auto &c : active_chunks)
    {
        // spead2::log_info("\t\tTimestamp %1%, Timestamp Low %2%, Timestamp High %3%", timestamp, c->timestamp,
        //                  c->timestamp + timestamp_step * heaps_per_fengine_per_chunk);
        if (timestamp >= c->timestamp && timestamp < c->timestamp + timestamp_step * heaps_per_fengine_per_chunk)
        {
            // spead2::log_info("\t\tIn above ^");
            return calculate_packet_destination(timestamp, fengine_id, *c);
        }
    }
    if (timestamp < base)
    {
        spead2::log_info("Seem to have gone backwards in time");
        return std::make_tuple(nullptr, nullptr, 0);
    }
    else
    {
        std::size_t max_active = 1;
        /* We've gone forward beyond the last active chunk. Make room to add
         * a new one. Usually this will be the next sequential chunk. If not,
         * we flush all chunks rather than leaving active_chunks discontiguous.
         */
        std::int64_t start = active_chunks.back()->timestamp + timestamp_step * heaps_per_fengine_per_chunk;
        if (timestamp >= start + std::int64_t(timestamp_step)) // True if the next chunk is not the next sequential one
        {
            // I have not actually seen this line in action yet - it could produce an error.
            start += (timestamp - start);
            max_active = 0;
        }
        while (active_chunks.size() > max_active)
            if (!flush_chunk())
            {
                // ringbuffer was stopped, so no point in continuing.
                return std::make_tuple(nullptr, nullptr, 0);
            }
        grab_chunk(start);
        return calculate_packet_destination(timestamp, fengine_id, *active_chunks.back());
    }
}

void *stream::allocate(std::size_t size, spead2::recv::packet_header &packet)
{
    //    spead2::log_info("Receiver allocator 0");
    //    spead2::log_info("Receiver allocator 1 %1% %2% %3%", size, packet_bytes, packet.n_items);
    if (size != packet_bytes * n_channels)
    {
        spead2::log_info("Allocating incorrect size");
        return nullptr;
    }
    //    spead2::log_info("Receiver allocator 2");
    std::int64_t timestamp = -1;
    std::int64_t fengine_id = -1;
    spead2::recv::pointer_decoder decoder(packet.heap_address_bits);
    // Extract timestamp
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
    //    spead2::log_info("Packet Information Timestamp: %1% fengine_id %2%", timestamp, fengine_id);
    if (timestamp == -1 || fengine_id == -1)
        return nullptr;

    return std::get<0>(calculate_packet_destination(timestamp, fengine_id));
}

void stream::heap_ready(spead2::recv::live_heap &&live_heap)
{
    //    spead2::log_info("Heap Ready Called");
    if (!live_heap.is_complete())
    {
        spead2::log_info("Heap not complete. Received Length %1% Expected Length %2%.", live_heap.get_received_length(),
                         live_heap.get_heap_length());
        return; // should never happen: digitiser heaps are single packet
    }
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
    if (timestamp == -1 || length == 0)
    {
        spead2::log_info("It's not a valid digitiser packet");
        // TODO: log. It's not a valid digitiser packet.
        return;
    }
    if (length != packet_bytes * n_channels)
    {
        // TODO: log. It's the wrong size.
        spead2::log_info("Heap size incorrect. Received (%1%), expected (%2%)", length, packet_bytes * n_channels);
        return;
    }

    std::tie(expected_ptr, c, heap_idx) = calculate_packet_destination(timestamp, fengine_id);
    if (expected_ptr != actual_ptr)
    {
        spead2::log_info("This should only happen if we receive data that is too old (%1%)", timestamp_step);
        // TODO: log. This should only happen if we receive data that is too old.
        return;
    }
    c->present[heap_idx] = true;
    //    spead2::log_info("Heap Ready Finished");
}

void stream::stop_received()
{
    while (!active_chunks.empty())
        if (!flush_chunk())
            break;
    ringbuffer.stop();
    spead2::recv::stream::stop_received();
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
    config.set_max_size(packet_bytes + 96); // Header is 12 fields of 8 bytes each: So 96 bytes of header
    config.set_buffer_size(buffer_size);
    config.set_comp_vector(comp_vector);
    config.set_max_poll(max_poll);
    emplace_reader<spead2::recv::udp_ibv_reader>(config);
}

stream::ringbuffer_t &stream::get_ringbuffer()
{
    return ringbuffer;
}

int stream::get_sample_bits() const
{
    return sample_bits;
}

std::size_t stream::get_chunk_packets() const
{
    return chunk_packets;
}

std::size_t stream::get_chunk_bytes() const
{
    return chunk_bytes;
}

const stream::ringbuffer_t &stream::get_ringbuffer() const
{
    return ringbuffer;
}

void stream::stop()
{
    ringbuffer.stop();
    spead2::recv::stream::stop();
}

} // namespace katxgpu::recv
