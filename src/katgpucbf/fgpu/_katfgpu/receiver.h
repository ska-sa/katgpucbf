#include <stack>
#include <deque>
#include <mutex>
#include <memory>
#include <tuple>
#include <bitset>
#include <cstdint>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_ringbuffer.h>
#include <boost/asio.hpp>

static constexpr int N_POL = 2;
static constexpr int SAMPLE_BITS = 10;

/**
 * Collection of contiguous 10-bit samples in memory, together with information
 * about which samples are present. It references the sample storage but does
 * not own it (although it is polymorphic, so subclasses can own storage).
 */
struct in_chunk
{
    std::int64_t timestamp;                       ///< Timestamp of first sample
    std::vector<bool> present[N_POL];             ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage[N_POL];   ///< Storage for each polarisation

    virtual ~in_chunk() = default;
};

class receiver;

class stream : public spead2::recv::stream
{
private:
    virtual void heap_ready(spead2::recv::live_heap &&heap) override;
    virtual void stop_received() override;

    receiver &parent;
    int pol;

public:
    stream(spead2::io_service_ref io_service, receiver &parent, int pol);
};

class allocator : public spead2::memory_allocator
{
private:
    receiver &recv;
    int pol;

public:
    allocator(receiver &recv, int pol);

    virtual pointer allocate(std::size_t size, void *hint) override;
    virtual void free(std::uint8_t *ptr, void *user) override;
};

class receiver
{
private:
    friend class stream;
    friend class allocator;

    void heap_ready(int pol, spead2::recv::live_heap &&heap);
    void stop_received(int pol);

    const std::size_t packet_samples;        ///< Number of samples in each packet
    const std::size_t chunk_samples;         ///< Number of samples in each chunk
    const std::size_t chunk_packets;         ///< Number of packets in each chunk

    std::int64_t first_timestamp = -1;       ///< Very first timestamp observed

    mutable std::mutex free_chunks_lock;     ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;       ///< Semaphore that is put whenever chunks are added
    std::stack<std::unique_ptr<in_chunk>> free_chunks;     ///< Chunks available for allocation
    std::deque<std::unique_ptr<in_chunk>> active_chunks;   ///< Chunks currently being filled

    std::bitset<N_POL> stream_stopped;
    std::unique_ptr<stream> streams[N_POL];

    /// Obtain a fresh chunk from the free pool (blocking if necessary)
    void grab_chunk(std::int64_t timestamp);

    /// Send the first active chunk to the callback
    void flush_chunk();

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * If the timestamp is beyond the last active chunk, old chunks may be
     * flushed and new chunks appended.
     */
    std::tuple<void *, in_chunk *, std::size_t>
    decode_timestamp(int pol, std::int64_t timestamp);

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * This overload operates on a specific chunk. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main
     * overload.
     */
    std::tuple<void *, in_chunk *, std::size_t>
    decode_timestamp(int pol, std::int64_t timestamp, in_chunk &chunk);

    void *allocate(int pol, std::size_t size, spead2::recv::packet_header &packet);

public:
    spead2::ringbuffer<std::unique_ptr<in_chunk>> ringbuffer;    ///< Chunks ready to be processed

    // NB: io_service must be running with only one thread!
    receiver(spead2::io_service_ref io_service,
             std::size_t packet_samples, std::size_t chunk_samples);

    /// Get one of the underlying streams (e.g. to add readers)
    stream &get_stream(int pol);
    const stream &get_stream(int pol) const;

    /// Add a chunk to the free pool
    void add_chunk(std::unique_ptr<in_chunk> &&chunk);
};
