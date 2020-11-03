#ifndef KATFGPU_SEND_H
#define KATFGPU_SEND_H

#include <vector>
#include <memory>
#include <spead2/send_stream.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_thread_pool.h>

namespace katfgpu::send
{

typedef std::int8_t real_t;

/**
 * Collection of packet data for transmission. It is split into frames,
 * each of shape channels x acc-len x pols x 2 (the last axis being
 * real/imag).
 *
 * This class does not own its storage, but it is polymorphic so subclasses
 * can be owners.
 *
 * TODO: add flag bits to indicate missing data, or possibly
 * timestamp per frame.
 */
struct chunk
{
    std::int64_t timestamp;              ///< Timestamp of first frame
    int channels;
    int acc_len;
    int pols;
    int frames;
    boost::asio::const_buffer storage;   ///< Storage for data
    boost::system::error_code error;     ///< First error from sending the data

    virtual ~chunk() = default;   // make it polymorphic
};

using ringbuffer_t = spead2::ringbuffer<std::unique_ptr<chunk>, spead2::semaphore_fd>;

class sender
{
private:
    spead2::thread_pool thread_pool;
    std::unique_ptr<spead2::send::stream> stream;
    ringbuffer_t free_ring;

public:
    sender(std::vector<std::unique_ptr<chunk>> &&initial_chunks,
           int thread_affinity, int comp_vector,
           const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
           int ttl, const std::string &interface_address, bool ibv,
           std::size_t max_packet_size, double rate, std::size_t max_heaps);
    ~sender();

    /// Send a chunk asynchronously, and put it onto the free ring when complete.
    void send_chunk(std::unique_ptr<chunk> &&c);

    void stop();

    const ringbuffer_t &get_free_ring() const;
    ringbuffer_t &get_free_ring();
};

} // namespace katfgpu::send

#endif // KATFGPU_SEND_H
