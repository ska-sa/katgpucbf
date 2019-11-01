#include <utility>
#include <cassert>
#include <map>     // TODO: workaround for it missing in recv_heap.h
#include <iostream>  // TODO: for debugging
#include <stdexcept>
#include <spead2/recv_heap.h>
#include <spead2/recv_udp_pcap.h>
#include <spead2/common_endian.h>
#include "recv.h"

namespace katfgpu::recv
{

static constexpr int TIMESTAMP_ID = 0x1600;
static constexpr int DATA_ID = 0x3300;

allocator::allocator(stream &recv) : recv(recv) {}

auto allocator::allocate(std::size_t size, void *hint) -> pointer
{
    if (hint)
    {
        void *ptr = recv.allocate(size, *reinterpret_cast<spead2::recv::packet_header *>(hint));
        if (ptr)
            return pointer(reinterpret_cast<std::uint8_t *>(ptr),
                           deleter(shared_from_this(), (void *) std::uintptr_t(true)));
    }
    return spead2::memory_allocator::allocate(size, hint);
}

void allocator::free(std::uint8_t *ptr, void *user)
{
    if (!user)
        delete[] ptr;
}


stream::stream(int pol, int sample_bits, std::size_t packet_samples,
                   std::size_t chunk_samples, ringbuffer_t &ringbuffer, int thread_affinity)
    : spead2::thread_pool(
        1, thread_affinity < 0 ? std::vector<int>{} : std::vector<int>{thread_affinity}),
    spead2::recv::stream(*static_cast<thread_pool *>(this), 0, 1),
    pol(pol),
    sample_bits(sample_bits),
    packet_samples(packet_samples),
    chunk_samples(chunk_samples),
    chunk_packets(chunk_samples / packet_samples),
    packet_bytes(packet_samples / 8 * sample_bits),
    chunk_bytes(chunk_samples / 8 * sample_bits),
    ringbuffer(ringbuffer)
{
    if (sample_bits <= 0)
        throw std::invalid_argument("sample_bits must be greater than 0");
    if (packet_samples <= 0)
        throw std::invalid_argument("packet_samples must be greater than 0");
    if (chunk_samples <= 0)
        throw std::invalid_argument("chunk_samples must be greater than 0");
    if (packet_samples % 8 != 0)
        throw std::invalid_argument("packet_samples must be a multiple of 8");
    if (chunk_samples % packet_samples != 0)
        throw std::invalid_argument("chunk_samples must be a multiple of packet_samples");
    set_allow_unsized_heaps(false);
    set_memory_allocator(std::make_shared<katfgpu::recv::allocator>(*this));
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
    c->present.resize(chunk_packets);
    c->pol = pol;
    {
        std::lock_guard<std::mutex> lock(free_chunks_lock);
        free_chunks.push(std::move(c));
    }
    free_chunks_sem.put();
}

void stream::grab_chunk(std::int64_t timestamp)
{
    semaphore_get(free_chunks_sem);
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
        ringbuffer.push(std::move(active_chunks[0]));
        active_chunks.pop_front();
        return true;
    }
    catch (spead2::ringbuffer_stopped &e)
    {
        // TODO: add to counter
        return false;
    }
}

std::tuple<void *, chunk *, std::size_t>
stream::decode_timestamp(std::int64_t timestamp, chunk &c)
{
    std::size_t sample_idx = timestamp - c.timestamp;
    std::size_t packet_idx = sample_idx / packet_samples;
    std::size_t byte_idx = sample_idx / 8 * sample_bits;
    void *ptr = boost::asio::buffer_cast<std::uint8_t *>(c.storage) + byte_idx;
    return std::make_tuple(ptr, &c, packet_idx);
}

std::tuple<void *, chunk *, std::size_t>
stream::decode_timestamp(std::int64_t timestamp)
{
    if (first_timestamp == -1)
    {
        first_timestamp = timestamp / chunk_samples * chunk_samples;
        grab_chunk(first_timestamp);
    }
    if ((timestamp - first_timestamp) % packet_samples != 0)
    {
        // TODO: log/count. The timestamp is broken.
        return std::make_tuple(nullptr, nullptr, 0);
    }
    std::int64_t base = active_chunks[0]->timestamp;
    for (const auto &c : active_chunks)
    {
        if (timestamp >= c->timestamp
            && timestamp < c->timestamp + std::int64_t(chunk_samples))
            return decode_timestamp(timestamp, *c);
    }
    if (timestamp < base)
    {
        // TODO: log/count. Have gone backwards in time.
        return std::make_tuple(nullptr, nullptr, 0);
    }
    else
    {
        std::size_t max_active = 1;
        /* We've gone forward beyond the last active chunk. Make room to add
         * a new one. Usually this will be the next sequential chunk. If not,
         * we flush all chunks rather than leaving active_chunks discontiguous.
         */
        std::int64_t start = active_chunks.back()->timestamp + chunk_samples;
        if (timestamp >= start + std::int64_t(chunk_samples))
        {
            // TODO: log/count.
            start += (timestamp - start) / chunk_samples * chunk_samples;
            max_active = 0;
        }
        while (active_chunks.size() > max_active)
            if (!flush_chunk())
            {
                // ringbuffer was stopped, so no point in continuing.
                return std::make_tuple(nullptr, nullptr, 0);
            }
        grab_chunk(start);
        return decode_timestamp(timestamp, *active_chunks.back());
    }
}

void *stream::allocate(std::size_t size, spead2::recv::packet_header &packet)
{
    if (size != packet_bytes)
        return nullptr;
    std::int64_t timestamp = -1;
    spead2::recv::pointer_decoder decoder(packet.heap_address_bits);
    // Extract timestamp
    for (int i = 0; i < packet.n_items; i++)
    {
        spead2::item_pointer_t pointer;
        pointer = spead2::load_be<spead2::item_pointer_t>(packet.pointers + i * sizeof(pointer));
        if (decoder.is_immediate(pointer) && decoder.get_id(pointer) == TIMESTAMP_ID)
        {
            timestamp = decoder.get_immediate(pointer);
            break;
        }
    }
    if (timestamp == -1)
        return nullptr;

    return std::get<0>(decode_timestamp(timestamp));
}

void stream::heap_ready(spead2::recv::live_heap &&live_heap)
{
    if (!live_heap.is_complete())
        return;      // should never happen: digitiser heaps are single packet
    spead2::recv::heap heap(std::move(live_heap));
    std::int64_t timestamp = -1;
    void *actual_ptr = nullptr;
    std::size_t length = 0;

    void *expected_ptr;
    chunk *c;
    std::size_t packet_idx;

    for (const auto &item : heap.get_items())
    {
        if (item.id == TIMESTAMP_ID && item.is_immediate)
            timestamp = item.immediate_value;
        else if (item.id == DATA_ID)
        {
            actual_ptr = item.ptr;
            length = item.length;
        }
    }
    if (timestamp == -1 || length == 0)
    {
        // TODO: log. It's not a valid digitiser packet.
        return;
    }
    if (length != packet_bytes)
    {
        // TODO: log. It's the wrong size.
        return;
    }

    std::tie(expected_ptr, c, packet_idx) = decode_timestamp(timestamp);
    if (expected_ptr != actual_ptr)
    {
        // TODO: log. This should only happen if we receive data that is too old.
        return;
    }
    c->present[packet_idx] = true;
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
                                  const std::string &interface_address,
                                  std::size_t buffer_size, int comp_vector, int max_poll)
{
    std::vector<boost::asio::ip::udp::endpoint> endpoints2;
    for (const auto &ep : endpoints)
        endpoints2.emplace_back(boost::asio::ip::address::from_string(ep.first), ep.second);
    auto interface_address2 = boost::asio::ip::address::from_string(interface_address);
    emplace_reader<spead2::recv::udp_ibv_reader>(
        endpoints2, interface_address2, chunk_bytes + 128,
        buffer_size, comp_vector, max_poll);
}

stream::ringbuffer_t &stream::get_ringbuffer()
{
    return ringbuffer;
}

int stream::get_pol() const
{
    return pol;
}

int stream::get_sample_bits() const
{
    return sample_bits;
}

std::size_t stream::get_packet_samples() const
{
    return packet_samples;
}

std::size_t stream::get_chunk_samples() const
{
    return chunk_samples;
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

} // namespace katfgpu::recv
