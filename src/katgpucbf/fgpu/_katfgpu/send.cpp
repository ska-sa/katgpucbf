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
    send::sender &sender;
    std::unique_ptr<chunk> c;
    std::vector<spead2::send::heap> heaps;
    std::size_t remaining;

    context(send::sender &sender, std::unique_ptr<chunk> &&c, std::size_t n_heaps)
        : sender(sender), c(std::move(c)), remaining(n_heaps)
    {
        heaps.reserve(n_heaps);
    }

    ~context()
    {
        sender.push_free_ring(std::move(c));
    }
};

sender::sender(std::vector<std::unique_ptr<chunk>> &&initial_chunks,
               int thread_affinity, int comp_vector,
               const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
               int ttl, const std::string &interface_address, bool ibv,
               std::size_t max_packet_size, double rate, std::size_t max_heaps)
    : thread_pool(1, thread_affinity >= 0 ? std::vector<int>{thread_affinity} : std::vector<int>{}),
    free_ring(initial_chunks.size())
{
    if (endpoints.empty())
        throw std::invalid_argument("must have at least one endpoint");
    std::vector<boost::asio::ip::udp::endpoint> ep;
    for (const auto &e : endpoints)
        ep.emplace_back(boost::asio::ip::address::from_string(e.first), e.second);
    boost::asio::ip::address interface = boost::asio::ip::address::from_string(interface_address);
    spead2::send::stream_config config;
    config.set_max_packet_size(max_packet_size);
    config.set_rate(rate);
    config.set_max_heaps(max_heaps);  // TODO: get sender to compute it, given shape of chunks?
    if (ibv)
    {
        spead2::send::udp_ibv_config ibv_config;
        ibv_config.set_endpoints(ep);
        ibv_config.set_interface_address(interface);
        ibv_config.set_ttl(ttl);
        ibv_config.set_comp_vector(comp_vector);
        for (const auto &c : initial_chunks)
        {
            const void *ptr = boost::asio::buffer_cast<const void *>(c->storage);
            std::size_t length = boost::asio::buffer_size(c->storage);
            ibv_config.add_memory_region(ptr, length);
        }
        stream = std::make_unique<spead2::send::udp_ibv_stream>(thread_pool, config, ibv_config);
    }
    else
    {
        stream = std::make_unique<spead2::send::udp_stream>(
            thread_pool, ep, config, spead2::send::udp_stream::default_buffer_size,
            ttl, interface);
    }

    for (auto &c : initial_chunks)
        push_free_ring(std::move(c));
}

sender::~sender()
{
    stop();
}

void sender::push_free_ring(std::unique_ptr<chunk> &&c)
{
    pre_push_free_ring();
    free_ring.push(std::move(c));
    post_push_free_ring();
}

void sender::stop()
{
    free_ring.stop();
    stream->flush();
    thread_pool.stop();
}

void sender::send_chunk(std::unique_ptr<chunk> &&c)
{
    std::size_t n_substreams = stream->get_num_substreams();
    if (c->channels % n_substreams != 0)
        throw std::invalid_argument("channels must be divisible by the number of substreams");

    const spead2::flavour flavour(spead2::maximum_version, 64, 48);
    const std::size_t n_heaps = n_substreams * c->frames;
    const int channels_per_substream = c->channels / n_substreams;
    const std::size_t frame_bytes = sizeof(real_t) * c->channels * c->acc_len * c->pols * 2;
    const std::size_t heap_bytes = frame_bytes / n_substreams;
    const std::int64_t timestamp_step = c->acc_len * c->channels * 2;
    if (boost::asio::buffer_size(c->storage) < c->frames * frame_bytes)
        throw std::invalid_argument("send_chunk storage is too small");

    c->error = boost::system::error_code();
    if (c->frames <= 0 || c->channels <= 0 || c->acc_len <= 0)
    {
        // Chunk contains no data, so send it directly to the free ring
        push_free_ring(std::move(c));
        return;
    }

    auto ctx = std::make_shared<context>(*this, std::move(c), n_heaps);
    auto callback = [ctx] (const boost::system::error_code &ec, spead2::item_pointer_t bytes_transferred)
    {
        if (ec)
        {
            ctx->c->error = ec;
            std::cout << "Error in send: " << ec << '\n';
        }
    };

    int frames = ctx->c->frames;
    for (int i = 0; i < frames; i++)
        for (std::size_t j = 0; j < n_substreams; j++)
        {
            auto heap_data = boost::asio::buffer(ctx->c->storage + i * frame_bytes + j * heap_bytes,
                                                 heap_bytes);
            ctx->heaps.emplace_back(flavour);
            spead2::send::heap &heap = ctx->heaps.back();
            // TODO: Consider pre-creating the heaps and recycling
            heap.set_repeat_pointers(true);
            heap.add_item(TIMESTAMP_ID, ctx->c->timestamp + i * timestamp_step);
            heap.add_item(FENG_ID_ID, 0);    // TODO: take feng_id in constructor
            heap.add_item(FREQUENCY_ID, j * channels_per_substream);
            heap.add_item(FENG_RAW_ID,
                          boost::asio::buffer_cast<const void *>(heap_data),
                          boost::asio::buffer_size(heap_data),
                          false);
            for (int pad = 0; pad < 3; pad++)
                heap.add_item(0, 0);
            stream->async_send_heap(heap, callback, j);
        }
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
