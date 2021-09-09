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

/* context struct holds sender, chunk, a vector of heaps, and a count of how much
 * remaining stuff to transmit there is.
 * Constructor reserves space in memory for the heaps,
 * destructor returns the chunk to the free ring.
 */
struct context
{
    send::sender &sender;
    std::unique_ptr<chunk> c;
    std::vector<spead2::send::heap> heaps;

    context(send::sender &sender, std::unique_ptr<chunk> &&c, std::size_t n_heaps)
        : sender(sender), c(std::move(c))
    {
        heaps.reserve(n_heaps);
    }

    ~context()
    {
        sender.push_free_ring(std::move(c));
    }
};

sender::sender(std::size_t free_ring_capacity,
               const std::vector<std::pair<const void *, std::size_t>> &memory_regions,
               int thread_affinity, int comp_vector, int feng_id, int num_ants,
               const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
               int ttl, const std::string &interface_address, bool ibv,
               std::size_t max_packet_size, double rate, std::size_t max_heaps)
    : thread_pool(1, thread_affinity >= 0 ? std::vector<int>{thread_affinity} : std::vector<int>{}),
    free_ring(free_ring_capacity),
    feng_id(feng_id)
{
    // Convert list of string & int endpoints to more useful boost::asio types.
    if (endpoints.empty())
        throw std::invalid_argument("must have at least one endpoint");
    std::vector<boost::asio::ip::udp::endpoint> ep;
    for (const auto &e : endpoints)
        ep.emplace_back(boost::asio::ip::address::from_string(e.first), e.second);

    // Ihis is the IP address of the interface we will be using for sending. ibverbs
    // needs this information.
    boost::asio::ip::address interface = boost::asio::ip::address::from_string(interface_address);

    // Configure and create the spead2 stream.
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
        for (const auto &c : memory_regions)
            ibv_config.add_memory_region(c.first, c.second);
        stream = std::make_unique<spead2::send::udp_ibv_stream>(thread_pool, config, ibv_config);
    }
    else
    {
        stream = std::make_unique<spead2::send::udp_stream>(
            thread_pool, ep, config, spead2::send::udp_stream::default_buffer_size,
            ttl, interface);
    }
    stream->set_cnt_sequence(this->feng_id, num_ants);
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
    // TODO: It makes more sense to do this check in the constructor, but it doesn't
    // currently have access to the number of channels.
    std::size_t n_substreams = stream->get_num_substreams();
    if (c->channels % n_substreams != 0)
        throw std::invalid_argument("channels must be divisible by the number of substreams");

    const spead2::flavour flavour(spead2::maximum_version, 64, 48);
    const std::size_t n_heaps = n_substreams * c->frames;
    const int channels_per_substream = c->channels / n_substreams;
    const std::size_t frame_bytes = sizeof(real_t) * c->channels * c->spectra_per_heap_out * c->pols * 2; // TODO either a constexpr or #define to make 2 explicitly "complexity"
    const std::size_t heap_bytes = frame_bytes / n_substreams;
    const std::int64_t timestamp_step = c->spectra_per_heap_out * c->channels * 2;

    if (boost::asio::buffer_size(c->storage) < c->frames * frame_bytes)
        throw std::invalid_argument("send_chunk storage is too small");

    c->error = boost::system::error_code();
    if (c->frames <= 0 || c->channels <= 0 || c->spectra_per_heap_out <= 0)
    {
        // Chunk contains no data, so send it directly to the free ring
        push_free_ring(std::move(c));
        return;
    }

    auto ctx = std::make_shared<context>(*this, std::move(c), n_heaps);

    // Lambda as a callback to handle potential errors in transmission.
    // Records the error code in the chunk being transmitted, and emits a debug message.
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
            // Buffer "pointing" to the sub-region in the chunk which has the
            // data to be sent next
            auto heap_data = boost::asio::buffer(ctx->c->storage + i * frame_bytes + j * heap_bytes,
                                                 heap_bytes);

            // Add a new heap to the queue with the appropriate fields.
            ctx->heaps.emplace_back(flavour);
            spead2::send::heap &heap = ctx->heaps.back();
            // TODO: Consider pre-creating the heaps and recycling
            heap.set_repeat_pointers(true);
            heap.add_item(TIMESTAMP_ID, ctx->c->timestamp + i * timestamp_step);
            heap.add_item(FENG_ID_ID, feng_id);
            heap.add_item(FREQUENCY_ID, j * channels_per_substream);
            heap.add_item(FENG_RAW_ID,
                          boost::asio::buffer_cast<const void *>(heap_data),
                          boost::asio::buffer_size(heap_data),
                          false);
            // Padding for compatibility with MeerKAT's F-engine.
            for (int pad = 0; pad < 3; pad++)
                heap.add_item(0, 0);

            // Mark the heap for asynchronous sending.
            stream->async_send_heap(heap, callback, -1, j);
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

std::size_t sender::get_num_substreams() const
{
    return stream->get_num_substreams();
}

} // namespace katfgpu::send
