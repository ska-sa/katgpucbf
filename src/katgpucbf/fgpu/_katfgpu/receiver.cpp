#include <utility>
#include <cassert>
#include <map>     // TODO: workaround for it missing in recv_heap.h
#include <spead2/recv_heap.h>
#include "receiver.h"

static constexpr int TIMESTAMP_ID = 0x1600;
static constexpr int DATA_ID = 0x3300;
static constexpr int SAMPLE_BITS = 10;

stream::stream(spead2::io_service_ref io_service, receiver &parent, int pol)
    : spead2::recv::stream(io_service, 0, 1),
    parent(parent),
    pol(pol)
{
    // TODO: set allow_unsized_heaps etc
}

void stream::heap_ready(spead2::recv::live_heap &&heap)
{
    parent.heap_ready(pol, std::move(heap));
}

void stream::stop_received()
{
    spead2::recv::stream::stop_received();
    parent.stop_received(pol);
}


receiver::receiver(spead2::io_service_ref io_service,
                   std::size_t packet_samples, std::size_t chunk_samples,
                   chunk_func ready_callback)
    : packet_samples(packet_samples),
    chunk_samples(chunk_samples),
    chunk_packets(chunk_samples / packet_samples),
    ready_callback(ready_callback)
{
    assert(chunk_samples % packet_samples == 0);
    for (int i = 0; i < N_POL; i++)
    {
        streams[i] = std::make_unique<stream>(io_service, *this, i);
        // TODO: set allocator
    }
}

void receiver::add_chunk(std::unique_ptr<in_chunk> &&chunk)
{
    for (int i = 0; i < N_POL; i++)
    {
        assert(buffer_size(chunk->storage[i]) == SAMPLE_BITS * chunk_samples / 8);
        chunk->present[i].clear();
        chunk->present[i].resize(chunk_packets);
    }
    {
        std::lock_guard<std::mutex> lock(free_chunks_lock);
        free_chunks.push(std::move(chunk));
    }
    free_chunks_sem.put();
}

void receiver::grab_chunk(std::int64_t timestamp)
{
    std::unique_ptr<in_chunk> chunk;

    semaphore_get(free_chunks_sem);
    {
        std::lock_guard<std::mutex> lock(free_chunks_lock);
        assert(!free_chunks.empty());
        chunk = std::move(free_chunks.top());
        free_chunks.pop();
    }
    chunk->timestamp = timestamp;
    active_chunks.push_back(std::move(chunk));
}

void receiver::flush_chunk()
{
    assert(!active_chunks.empty());
    auto chunk = std::move(active_chunks[0]);
    active_chunks.pop_front();
    ready_callback(std::move(chunk));
}

std::tuple<void *, in_chunk *, std::size_t>
receiver::decode_timestamp(int pol, std::int64_t timestamp, in_chunk &chunk)
{
    std::size_t sample_idx = timestamp - chunk.timestamp;
    std::size_t packet_idx = sample_idx / packet_samples;
    std::size_t byte_idx = sample_idx * SAMPLE_BITS / 8;
    void *ptr = boost::asio::buffer_cast<std::uint8_t *>(chunk.storage[pol]) + byte_idx;
    return std::make_tuple(ptr, &chunk, packet_idx);
}

std::tuple<void *, in_chunk *, std::size_t>
receiver::decode_timestamp(int pol, std::int64_t timestamp)
{
    if (first_timestamp == -1)
    {
        first_timestamp = timestamp;
        grab_chunk(timestamp);
    }
    if ((timestamp - first_timestamp) % packet_samples != 0)
    {
        // TODO: log/count. The timestamp is broken.
        return std::make_tuple(nullptr, nullptr, 0);
    }
    std::int64_t base = active_chunks[0]->timestamp;
    for (const auto &chunk : active_chunks)
    {
        if (timestamp >= chunk->timestamp
            && timestamp < chunk->timestamp + chunk_samples)
            return decode_timestamp(pol, timestamp, *chunk);
    }
    if (timestamp < base)
    {
        // TODO: log/count. Have gone backwards in time.
        return std::make_tuple(nullptr, nullptr, 0);
    }
    else
    {
        // We've gone forward beyond the last active chunk. Make room to add
        // a new one.
        while (active_chunks.size() >= 2)
            flush_chunk();
        // Usually this will be the next sequential chunk. If not, we flush
        // all chunks rather than leaving active_chunks discontiguous.
        std::int64_t start = active_chunks.back()->timestamp + chunk_samples;
        if (timestamp >= start + chunk_samples)
        {
            // TODO: log/count.
            start += (timestamp - start) / chunk_samples * chunk_samples;
            while (!active_chunks.empty())
                flush_chunk();
        }
        grab_chunk(start);
        return decode_timestamp(pol, timestamp, *active_chunks.back());
    }
}

void receiver::heap_ready(int pol, spead2::recv::live_heap &&live_heap)
{
    if (!live_heap.is_complete())
        return;      // should never happen: digitiser heaps are single packet
    spead2::recv::heap heap(std::move(live_heap));
    std::int64_t timestamp = -1;
    void *actual_ptr = nullptr;
    std::size_t length = 0;

    void *expected_ptr;
    in_chunk *chunk;
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
    if (length != SAMPLE_BITS * packet_samples / 8)
    {
        // TODO: log. It's the wrong size.
        return;
    }
    std::tie(expected_ptr, chunk, packet_idx) = decode_timestamp(pol, timestamp);
    if (expected_ptr != actual_ptr)
    {
        // TODO: log. This should only happen if we receive data that is too old.
        return;
    }
    chunk->present[pol][packet_idx] = true;
}

void receiver::stop_received(int pol)
{
    // TODO: wait for both pols!
    while (!active_chunks.empty())
        flush_chunk();
}

stream &receiver::get_stream(int pol)
{
    assert(0 <= pol && pol < N_POL);
    return *streams[pol];
}

const stream &receiver::get_stream(int pol) const
{
    assert(0 <= pol && pol < N_POL);
    return *streams[pol];
}
