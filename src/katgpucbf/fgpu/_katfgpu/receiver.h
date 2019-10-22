#include <stack>
#include <deque>
#include <mutex>
#include <memory>
#include <tuple>
#include <functional>
#include <cstdint>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_semaphore.h>
#include <boost/asio.hpp>

/**
 * Collection of contiguous 10-bit samples in memory, together with information
 * about which samples are present. It references the sample storage but does
 * not own it.
 */
struct in_chunk
{
    std::int64_t timestamp;       ///< Timestamp of first sample
    std::vector<bool> present;    ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage;
};

typedef std::function<void(std::unique_ptr<in_chunk> &&)> chunk_func;

class receiver : public spead2::recv::stream
{
private:
    virtual void heap_ready(spead2::recv::live_heap &&heap) override;
    virtual void stop_received() override;

    const std::size_t packet_samples;        ///< Number of samples in each packet
    const std::size_t chunk_samples;         ///< Number of samples in each chunk
    const std::size_t chunk_packets;         ///< Number of packets in each chunk
    const chunk_func ready_callback;         ///< Called when a chunk has been filled

    std::int64_t first_timestamp;            ///< Very first timestamp observed

    mutable std::mutex free_chunks_lock;     ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;       ///< Semaphore that is put whenever chunks are added
    std::stack<std::unique_ptr<in_chunk>> free_chunks;     ///< Chunks available for allocation
    std::deque<std::unique_ptr<in_chunk>> active_chunks;   ///< Chunks currently being filled

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
    std::tuple<void *, in_chunk *, std::size_t> decode_timestamp(std::int64_t timestamp);

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * This overload operates on a specific chunk. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main
     * overload.
     */
    std::tuple<void *, in_chunk *, std::size_t> decode_timestamp(std::int64_t timestamp,
                                                                 in_chunk &chunk);

public:
    receiver(spead2::io_service_ref io_service,
             std::size_t packet_samples, std::size_t chunk_samples,
             chunk_func ready_callback);

    /// Add a chunk to the free pool
    void add_chunk(std::unique_ptr<in_chunk> &&chunk);
};
