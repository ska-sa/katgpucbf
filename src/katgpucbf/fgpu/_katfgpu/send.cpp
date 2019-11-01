#include <iostream>      // TODO: debugging
#include <stdexcept>
#include <spead2/send_udp.h>
#include <spead2/send_udp_ibv.h>
#include "send.h"

namespace katfgpu::send
{

static constexpr int TIMESTAMP_ID = 0x1600;
static constexpr int FENG_ID_ID = 0x4101;
static constexpr int FREQUENCY_ID = 0x4103;
static constexpr int FENG_RAW_ID = 0x4300;

struct context
{
    ringbuffer_t &free_ring;
    std::unique_ptr<chunk> c;
    std::vector<spead2::send::heap> heaps;
    std::size_t remaining;

    context(ringbuffer_t &free_ring, std::unique_ptr<chunk> &&c, std::size_t n_heaps)
        : free_ring(free_ring), c(std::move(c)), remaining(n_heaps)
    {
        heaps.reserve(n_heaps);
    }
};

sender::sender(int streams, int free_ring_space, int thread_affinity)
    : worker(1, thread_affinity < 0 ? std::vector<int>{} : std::vector<int>{thread_affinity}),
    free_ring(free_ring_space)
{
    if (streams <= 0)
        throw std::invalid_argument("streams must be positive");
    this->streams.reserve(streams);
}

sender::~sender()
{
    stop();
}

template<typename Stream, typename... Args>
void sender::emplace_stream(Args&&... args)
{
    if (streams.size() == streams.capacity())
        throw std::length_error("too many streams");
    streams.push_back(std::make_unique<Stream>(worker.get_io_service(),
                                               std::forward<Args>(args)...));
    streams.back()->set_cnt_sequence(streams.size(), streams.capacity());
}

void sender::add_udp_stream(const std::string &address, std::uint16_t port,
                            int ttl, const std::string &interface_address, bool ibv,
                            std::size_t max_packet_size, double rate, std::size_t max_heaps)
{
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address::from_string(address), port);
    boost::asio::ip::address interface = boost::asio::ip::address::from_string(interface_address);
    spead2::send::stream_config config;
    config.set_max_packet_size(max_packet_size);
    config.set_rate(rate);
    config.set_max_heaps(max_heaps);  // TODO: get sender to compute it, given shape of chunks?
    // TODO: allow comp_vector to be set too
    if (ibv)
    {
        emplace_stream<spead2::send::udp_ibv_stream>(
            endpoint, config, interface,
            spead2::send::udp_ibv_stream::default_buffer_size, ttl);
    }
    else
    {
        emplace_stream<spead2::send::udp_stream>(
            endpoint, config, spead2::send::udp_stream::default_buffer_size,
            ttl, interface);
    }
}

void sender::stop()
{
    free_ring.stop();
    for (auto &stream : streams)
        stream->flush();
    worker.stop();
}

void sender::send_chunk(std::unique_ptr<chunk> &&c)
{
    if (streams.size() != streams.capacity())
        throw std::invalid_argument("cannot use send_chunk until streams have been added");
    if (c->channels % streams.size() != 0)
        throw std::invalid_argument("channels must be divisible by the number of streams");

    const spead2::flavour flavour(spead2::maximum_version, 64, 48);
    const std::size_t n_heaps = streams.size() * c->frames;
    const int channels_per_stream = c->channels / streams.size();
    const std::size_t frame_bytes = sizeof(real_t) * c->channels * c->acc_len * c->pols * 2;
    const std::size_t heap_bytes = frame_bytes / streams.size();
    const std::int64_t timestamp_step = c->acc_len * c->channels * 2;
    if (boost::asio::buffer_size(c->storage) < c->frames * frame_bytes)
        throw std::invalid_argument("send_chunk storage is too small");

    c->error = boost::system::error_code();
    if (c->frames <= 0 || c->channels <= 0 || c->acc_len <= 0)
    {
        // Chunk contains no data, so send it directly to the free ring
        free_ring.push(std::move(c));
        return;
    }

    auto ctx = std::make_shared<context>(free_ring, std::move(c), n_heaps);
    auto callback = [ctx] (const boost::system::error_code &ec, spead2::item_pointer_t bytes_transferred)
    {
        if (ec)
        {
            ctx->c->error = ec;
            std::cout << "Error in send: " << ec << '\n';
        }
        if (--ctx->remaining == 0)
            ctx->free_ring.push(std::move(ctx->c));
    };

    std::cout << "About to send: frames=" << ctx->c->frames << " channels=" << ctx->c->channels
        << " acc_len=" << ctx->c->acc_len << " pols=" << ctx->c->pols << '\n';
    std::cout << "heap_bytes = " << heap_bytes << '\n';
    for (int i = 0; i < ctx->c->frames; i++)
        for (std::size_t j = 0; j < streams.size(); j++)
        {
            auto heap_data = boost::asio::buffer(ctx->c->storage + i * frame_bytes + j * heap_bytes,
                                                 heap_bytes);
            ctx->heaps.emplace_back(flavour);
            spead2::send::heap &heap = ctx->heaps.back();
            // TODO: Consider pre-creating the heaps and recycling
            heap.set_repeat_pointers(true);
            heap.add_item(TIMESTAMP_ID, ctx->c->timestamp + i * timestamp_step);
            heap.add_item(FENG_ID_ID, 0);    // TODO: take feng_id in constructor
            heap.add_item(FREQUENCY_ID, j * channels_per_stream);
            heap.add_item(FENG_RAW_ID,
                          boost::asio::buffer_cast<const void *>(heap_data),
                          boost::asio::buffer_size(heap_data),
                          false);
            streams[j]->async_send_heap(heap, callback);
        }
    std::cout << "Sent " << ctx->c->frames * streams.size() << " heaps\n";
}

ringbuffer_t &sender::get_free_ring()
{
    return free_ring;
}

const ringbuffer_t &sender::get_free_ring() const
{
    return free_ring;
}

} // namespace katfgpu::send
