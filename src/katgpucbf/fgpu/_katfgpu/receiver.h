#include <stack>
#include <deque>
#include <mutex>
#include <memory>
#include <vector>
#include <tuple>
#include <bitset>
#include <utility>
#include <string>
#include <cstdint>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_ringbuffer.h>
#include <boost/asio.hpp>

static constexpr int SAMPLE_BITS = 10;

/**
 * Collection of contiguous 10-bit samples in memory, together with information
 * about which samples are present. It references the sample storage but does
 * not own it. However, it is polymorphic, so subclasses can own storage, and
 * it is guaranteed that it will not be destroyed from the worker thread.
 */
struct in_chunk
{
    int pol;                               ///< Polarisation index
    std::int64_t timestamp;                ///< Timestamp of first sample
    std::vector<bool> present;             ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage;   ///< Storage for samples

    virtual ~in_chunk() = default; // makes it polymorphic
};

class receiver;

class sample_stream : public spead2::recv::stream
{
private:
    virtual void heap_ready(spead2::recv::live_heap &&heap) override;
    virtual void stop_received() override;

    receiver &parent;

public:
    sample_stream(spead2::io_service_ref io_service, receiver &parent);
};

class allocator : public spead2::memory_allocator
{
private:
    receiver &recv;

public:
    explicit allocator(receiver &recv);

    virtual pointer allocate(std::size_t size, void *hint) override;
    virtual void free(std::uint8_t *ptr, void *user) override;
};

class receiver
{
public:
    using ringbuffer_t = spead2::ringbuffer<std::unique_ptr<in_chunk>,
                                            spead2::semaphore_fd>;

private:
    friend class sample_stream;
    friend class allocator;

    void heap_ready(spead2::recv::live_heap &&heap);
    void stop_received();

    const int pol;                           ///< Polarisation index
    const std::size_t packet_samples;        ///< Number of samples in each packet
    const std::size_t chunk_samples;         ///< Number of samples in each chunk
    const std::size_t chunk_packets;         ///< Number of packets in each chunk

    std::int64_t first_timestamp = -1;       ///< Very first timestamp observed

    mutable std::mutex free_chunks_lock;     ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;       ///< Semaphore that is put whenever chunks are added
    std::stack<std::unique_ptr<in_chunk>> free_chunks;     ///< Chunks available for allocation
    std::deque<std::unique_ptr<in_chunk>> active_chunks;   ///< Chunks currently being filled

    spead2::thread_pool thread_pool;         ///< Single-threaded pool servicing the stream
    sample_stream stream;
    ringbuffer_t &ringbuffer;  ///< Chunks ready to be processed

    /// Obtain a fresh chunk from the free pool (blocking if necessary)
    void grab_chunk(std::int64_t timestamp);

    /**
     * Send the first active chunk to the ringbuffer.
     * 
     * @retval true on success
     * @retval false if the ringbuffer has already been stopped
     */
    bool flush_chunk();

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * If the timestamp is beyond the last active chunk, old chunks may be
     * flushed and new chunks appended.
     */
    std::tuple<void *, in_chunk *, std::size_t>
    decode_timestamp(std::int64_t timestamp);

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * This overload operates on a specific chunk. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main
     * overload.
     */
    std::tuple<void *, in_chunk *, std::size_t>
    decode_timestamp(std::int64_t timestamp, in_chunk &chunk);

    void *allocate(std::size_t size, spead2::recv::packet_header &packet);

public:
    receiver(int pol, std::size_t packet_samples, std::size_t chunk_samples,
             ringbuffer_t &ringbuffer, int thread_affinity = -1);
    ~receiver();

    void add_udp_pcap_file_reader(const std::string &filename);

    void add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                            const std::string &interface_address,
                            std::size_t buffer_size, int comp_vector = 0,
                            int max_poll = spead2::recv::udp_ibv_reader::default_max_poll);

    /// Get the underlying stream (e.g. to add readers)
    sample_stream &get_stream();
    const sample_stream &get_stream() const;

    /// Get the referenced ringbuffer
    ringbuffer_t &get_ringbuffer();
    const ringbuffer_t &get_ringbuffer() const;

    /// Add a chunk to the free pool
    void add_chunk(std::unique_ptr<in_chunk> &&chunk);

    void stop();
};
