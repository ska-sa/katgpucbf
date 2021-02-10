#ifndef KATXGPU_RECV_H
#define KATXGPU_RECV_H

#include <bitset>
#include <boost/asio.hpp>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <pybind11/pybind11.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace katxgpu::recv
{

/**
 * Collection of contiguous 10-bit samples in memory, together with information
 * about which samples are present. It references the sample storage but does
 * not own it. However, it is polymorphic, so subclasses can own storage, and
 * it is guaranteed that it will not be destroyed from the worker thread.
 */
struct chunk
{
    std::int64_t timestamp;              ///< Timestamp of first sample
    std::vector<bool> present;           ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage; ///< Storage for samples

    virtual ~chunk() = default; // makes it polymorphic
};

class stream;

class allocator : public spead2::memory_allocator
{
  private:
    stream &recv;

  public:
    explicit allocator(stream &recv);

    virtual pointer allocate(std::size_t size, void *hint) override;
    virtual void free(std::uint8_t *ptr, void *user) override;
};

class stream : private spead2::thread_pool, public spead2::recv::stream
{
  public:
    using ringbuffer_t = spead2::ringbuffer<std::unique_ptr<chunk>, spead2::semaphore_fd>;

  private:
    friend class allocator;

    virtual void heap_ready(spead2::recv::live_heap &&heap) override final;
    virtual void stop_received() override final;

    // Profiling hooks
    virtual void pre_wait_chunk()
    {
    }
    virtual void post_wait_chunk()
    {
    }
    virtual void pre_ringbuffer_push()
    {
    }
    virtual void post_ringbuffer_push()
    {
    }

    const int sample_bits;                 ///< Number of bits per sample
    const int n_ants;                      ///< Number of antennas in the array
    const int n_channels;                  ///< Number of channels in each packet
    const int n_samples_per_channel;       ///< Number of samples stored in a single channel
    const int n_pols;                      ///< Number of polarisations in each sample
    const int complexity = 2;              ///< Indicates two values per sample - one real and one imaginary.
    const int heaps_per_fengine_per_chunk; ///< A chunk has this many heaps per F-Engine.
    const int timestamp_step;              ///< Increase in timestamp between successive heaps from the same F-Engine.
    const std::size_t packet_bytes;        ///< Number of payload bytes in each packet
    const std::size_t chunk_packets;       ///< Number of packets in each chunk
    const std::size_t chunk_bytes;         ///< Number of payload bytes in each chunk

    std::int64_t first_timestamp = -1; ///< Very first timestamp observed

    mutable std::mutex free_chunks_lock;              ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;                ///< Semaphore that is put whenever chunks are added
    std::stack<std::unique_ptr<chunk>> free_chunks;   ///< Chunks available for allocation
    std::deque<std::unique_ptr<chunk>> active_chunks; ///< Chunks currently being filled

    ringbuffer_t &ringbuffer; ///< When a chunk has been fully assembled it is put on this ringbuffer.

    /* This view is only used during unit testing. It stores the python view of the buffer containing simulated packets.
     * More detail in "add_buffer_reader()" function. 
     */
    pybind11::buffer_info view;

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
     * Determine data pointer, chunk and packet index from packet timestamp and
     * F-Engine ID.
     *
     * If the timestamp is beyond the last active chunk, old chunks may be
     * flushed and new chunks appended.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t timestamp,
                                                                          std::int64_t fengine_id);

    /**
     * Determine data pointer, chunk and packet index from packet timestamp and
     * F-Engine ID.
     *
     * This overload operates on a specific chunk. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main
     * overload.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t timestamp,
                                                                          std::int64_t fengine_id, chunk &c);

    void *allocate(std::size_t size, spead2::recv::packet_header &packet);

  public:
    stream(int n_ants, int n_channels, int n_samples_per_channel, int n_pols, int sample_bits, int timestamp_step,
           size_t heaps_per_fengine_per_chunk, ringbuffer_t &ringbuffer, int thread_affinity = -1,
           bool use_gdrcopy = false);
    ~stream();

    void add_udp_pcap_file_reader(const std::string &filename);

    void add_buffer_reader(pybind11::buffer buffer);

    void add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                            const std::string &interface_address, std::size_t buffer_size, int comp_vector = 0,
                            int max_poll = spead2::recv::udp_ibv_config::default_max_poll);

    /// Get the referenced ringbuffer
    ringbuffer_t &get_ringbuffer();
    const ringbuffer_t &get_ringbuffer() const;

    int get_sample_bits() const;
    std::size_t get_chunk_packets() const;
    std::size_t get_chunk_bytes() const;

    /// Add a chunk to the free pool
    void add_chunk(std::unique_ptr<chunk> &&c);

    virtual void stop() override;
};

} // namespace katxgpu::recv

#endif // KATXGPU_RECV_H
