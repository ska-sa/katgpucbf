// This file implements a chunking C++ SPEAD2 receiver for the F-Engine.

#ifndef KATFGPU_RECV_H
#define KATFGPU_RECV_H

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

namespace katfgpu::recv
{

/**
 * Collection of contiguous 10-bit samples in memory, together with information
 * about which samples are present. It references the sample storage but does
 * not own it. However, it is polymorphic, so subclasses can own storage, and
 * it is guaranteed that it will not be destroyed from the worker thread.
 */
struct chunk
{
    int pol;                               ///< Polarisation index
    std::int64_t timestamp;                ///< Timestamp of first sample
    std::vector<bool> present;             ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage;   ///< Storage for samples

    virtual ~chunk() = default; // makes it polymorphic
};

class stream;

/* When SPEAD2 receives the first packet in a new heap, it needs to allocate
 * space in the list of chunks where the completed heap will go. This function
 * is the allocator that does this. It is called once per heap and only when the
 * first packet in the heap is received.
 *
 * It examines the received packet header and uses the timestamp_id field to
 * determine where the packet must go.
 *
 * The allocator and the stream are tightly coupled. The stream holds an
 * (ownership) reference to the allocator, and the allocator points back at the
 * owning stream. Logically it would be simpler if they were just the same class
 * but spead2's API splits the stream from the allocator.
 *
 * The stream::allocate(...) function does most of the legwork here, and by
 * being part of the stream object it is able to access usfeul object variables
 * for determing allocation. This memory allocator object is the mechanism to
 * get the SPEAD2 library to call the stream::allocate(...) function.
 */
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
    using ringbuffer_t = spead2::ringbuffer<std::unique_ptr<chunk>,
                                            spead2::semaphore_fd>;

private:
    friend class allocator;

    /* When the last packet in a heap is received, this function is called by
     * the receiver (In contrast to the stream::allocate(...) function which is
     * called when the first packet in a heap is received.). This function does
     * final checks to ensure that the heap is not too old and then populates
     * the corresponding present field in the appropriate chunk.
     */
    virtual void heap_ready(spead2::recv::live_heap &&heap) override final;

    /* Stops the stream and blocks until all readers have wound up.
     */
    virtual void stop_received() override final;

    /* The four functions below are hooks that can be used in child classes for
     * metric tracking. These functions are empty in this class. The
     * *_wait_chunk() methods are called in the stream::grab_chunk() function -
     * one while trying to grab the chunk semaphore and the next once the
     * semaphore has been grabbed. The *_ringbuffer_push(...) methods are called
     * in the stream::flush_chunk() function. One before the active chunk is
     * pushed to the ringbuffer and one after.
     */
    virtual void pre_wait_chunk() {}
    virtual void post_wait_chunk() {}
    virtual void pre_ringbuffer_push() {}
    virtual void post_ringbuffer_push() {}

    const int pol;                           ///< Polarisation index
    const int sample_bits;                   ///< Number of bits per sample
    const std::size_t packet_samples;        ///< Number of samples in each packet
    const std::size_t chunk_samples;         ///< Number of samples in each chunk
    const std::size_t chunk_packets;         ///< Number of packets in each chunk
    const std::size_t packet_bytes;          ///< Number of payload bytes in each packet
    const std::size_t chunk_bytes;           ///< Number of payload bytes in each chunk
    const std::uint64_t timestamp_mask;      ///< Anded with incoming timestamp to clear low bits

    std::int64_t first_timestamp = -1;       ///< Very first timestamp observed

    mutable std::mutex free_chunks_lock;     ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;       ///< Semaphore that is put whenever chunks are added

    /**
     * @brief Chunks available for allocation
     *
     * When the user gives a new chunk (or a recycled old chunk) to the
     * receiver, it is added to free_chunks. The chunks on this stack are not in
     * use, but when they are required, they will be moved from this stack to
     * the active_chunks queue.
     */
    std::stack<std::unique_ptr<chunk>> free_chunks;

    /**
     * @brief Chunks currently being filled
     *
     * Chunks that are actively being assembled from multiple heaps are stored
     * in this queue. The receiver can be assembling multiple chunks at any one
     * time. Once a chunk is fully assembled, the receiver will move it to the
     * ringbuffer object.
     */
    std::deque<std::unique_ptr<chunk>> active_chunks;

    /**
     * @brief Chunks ready to be processed
     *
     * All chunks that have been assembled by the receiver and are ready to be
     * passed to the user will be pushed onto this ringbuffer.
     */
    ringbuffer_t &ringbuffer;

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
    std::tuple<void *, chunk *, std::size_t>
    decode_timestamp(std::int64_t timestamp);

    /**
     * Determine data pointer, chunk and packet index from packet timestamp.
     *
     * This overload operates on a specific chunk. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main
     * overload.
     */
    std::tuple<void *, chunk *, std::size_t>
    decode_timestamp(std::int64_t timestamp, chunk &c);

    void *allocate(std::size_t size, spead2::recv::packet_header &packet);

public:
    /* See pybind11::class_<py_stream>(m, "Stream", ...) documentation in
     * py_recv.cpp for a description of these member functions.
     */
    stream(int pol, int sample_bits, std::size_t packet_samples, std::size_t chunk_samples,
           ringbuffer_t &ringbuffer, int thread_affinity = -1, bool mask_timestamp = false,
           bool use_gdrcopy = false);
    ~stream();

    void add_udp_pcap_file_reader(const std::string &filename);

    void add_udp_reader(const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                        const std::string &interface_address,
                        std::size_t buffer_size, bool ibv, int comp_vector = 0,
                        int max_poll = spead2::recv::udp_ibv_config::default_max_poll);

    /// Get the referenced ringbuffer
    ringbuffer_t &get_ringbuffer();
    const ringbuffer_t &get_ringbuffer() const;

    int get_pol() const;
    int get_sample_bits() const;
    std::size_t get_packet_samples() const;
    std::size_t get_chunk_samples() const;
    std::size_t get_chunk_packets() const;
    std::size_t get_chunk_bytes() const;

    /// Add a chunk to the free pool
    void add_chunk(std::unique_ptr<chunk> &&c);

    /// Stop stream and block until all readers have wound up.
    virtual void stop() override;
};

} // namespace katfgpu::recv

#endif // KATFGPU_RECV_H
