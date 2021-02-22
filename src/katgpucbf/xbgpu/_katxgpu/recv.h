/**
 * TAlk a bit about chunk lifecycle - free pool, active pool, ringbuffer
 *
 * TODO: Move 'pybind11::buffer_info view' and associated functions to;
 */

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

/* Collection of contiguous samples in memory, together with information about which samples are present. It references
 * the sample storage but does not own it. However, it is polymorphic, so subclasses can own storage, and it is
 * guaranteed that it will not be destroyed from the worker thread.
 *
 * The samples are stored as a 1 dimensional array contiguous array. If the array were indexed  as a multidimensional
 * array, the indexing would be as follows:
 * [timestamp_index][f_engine_index][channel_index][spectrum_index][polarisation_index]. It is left to the user to
 * calculate the strides for multidimensional indexing in a 1 dimensional array.
 *
 * The timestamp_index and f_engine_index correspond to different packets while the remaining indexes correspond to how
 * data is arranged within a chunk.
 */
struct chunk
{
    std::int64_t timestamp;              ///< Timestamp of first collection of heaps
    std::vector<bool> present;           ///< Bitmask of packets that are present
    boost::asio::mutable_buffer storage; ///< Storage for samples

    virtual ~chunk() = default; // makes it polymorphic
};

class stream;

/* When SPEAD2 recieves the first packet in a new heap, it needs to allocate space in the list of chunks where the
 * completed heap will go. This function is the allocator that does this. It is called once per heap and only when the
 * first packet in the heap is recieved.
 *
 * It examines the recieved packet header and uses the timestamp_id and f_engine_id fields to determine where the packet
 * must go.
 *
 * There is some strange circular(ish) referencing in this allocator. This allocator is assigned to the receiver stream
 * object, but the allocator calls the stream::allocate(...) function requiring the stream object to be passed to the
 * allocator function. The stream::allocate(...) function does most of the legwork here, and by being part of the stream
 * object it is able to access usfeul object variables for determing allocation. This memory allocator object is the
 * mechanism to get the SPEAD2 library to call the stream::allocate(...) function.
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
    using ringbuffer_t = spead2::ringbuffer<std::unique_ptr<chunk>, spead2::semaphore_fd>;

  private:
    friend class allocator;

    /* When the last packet in a heap is received, this function is called by the receiver (In contrast to the
     * stream::allocate(...) function which is called when the first packet in a heap is received.). This function does
     * final checks to ensure that the heap is not too old and then populates the corresponding present field in the
     * appropriate chunk.
     */
    virtual void heap_ready(spead2::recv::live_heap &&heap) override final;

    /* Stops the stream and blocks until all readers have wound up.
     */
    virtual void stop_received() override final;

    /* The four functions below are hooks that can be used in child classes for metric tracking. These functions are
     * empty in this class. The *_wait_chunk() methods are called in the stream::grab_chunk() function - one while
     * trying to grab the chunk semaphor and the next once the semaphor has been grabbed. The *_ringbuffer_push(...)
     * methods are called in the stream::flush_chunk() function. One before the active chunk is pushed to the ringbuffer
     * and one after.
     */
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
    const std::size_t chunk_bytes;         ///< Number of payload bytes in each chunk

    std::int64_t first_timestamp = -1; ///< Very first timestamp observed. Populated when first packet is received.

    mutable std::mutex free_chunks_lock; ///< Protects access to @ref free_chunks
    spead2::semaphore free_chunks_sem;   ///< Semaphore that is put whenever chunks are added

    /* When the user gives a new chunk (or a recycled old chunk) to the receiver, it is added to the free_chunks stack.
     * The chunks on this stack are not in use, but when they are required, they will be moved from this stack to the
     * active chunks queue.
     */
    std::stack<std::unique_ptr<chunk>> free_chunks;

    /* Chunks that are activly being assembled from multiple heaps are stored in this queue. The receiver can be
     * assembling multiple chunks at any one time. Once a chunk is fully assembled, the receiver will move it to the
     * ringbuffer object.
     */
    std::deque<std::unique_ptr<chunk>> active_chunks; ///< Chunks currently being filled

    /* All chunks that have been assembled by the receiver and are ready to be passed to the user will be pushed onto
     * this ringbuffer.
     */
    ringbuffer_t &ringbuffer;

    /* This view is only used during unit testing. It stores the python view of the buffer containing simulated packets.
     * More detail in "add_buffer_reader()" function. I do not think that this file is the best place for this object.
     * Its a pybind11 object, so it should go under py_recv.h. The add_buffer_reader() function would need to be
     * modified too to accomodate this change. I would need to move a bunch of other functions around to make that
     * happen, so I will wait until I have a spare moment.
     */
    pybind11::buffer_info view;

    /// Obtain a fresh chunk from the free pool (blocking if necessary)
    void grab_chunk(std::int64_t timestamp);

    /**
     * Send the first active chunk in the queue to the ringbuffer.
     *
     * @retval true on success
     * @retval false if the ringbuffer has already been stopped
     */
    bool flush_chunk();

    /* This function is called by the stream::allocate() function. It takes in a timestamp and fengine_id and determines
     * which chunk the heap must be assigned to and the specific location in the chunk's buffer that the heap must be
     * copied to.
     *
     * This function also manages the active_chunk queue, flushing chunks when they are complete and adding new chunks
     * to the queue from the free queue when they are required.
     *
     * This function returns a tuple consisting of three values:
     * 1. void* - A pointer to the location in memory that the received heap data must be allocated to.
     * 2. chunk* - A pointer to the specific chunk that contains the buffer the heaps data is being allocated to.
     * 3. std::size_t - An index indicating which heap within the chunk that the heap belongs to.
     *
     * TODO: Clarify what the packet index means, mention what its called in katfgpu and say why the overloading is
     * used. Determine data pointer, chunk and packet index from packet timestamp and F-Engine ID.
     *
     * If the timestamp is beyond the last active chunk, old chunks may be flushed and new chunks appended.
     *
     * NOTE: This function was copied, renamed and reworked from the decode_timestamp() function in katfgpu but performs
     * essentially the same function.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t timestamp,
                                                                          std::int64_t fengine_id);

    /* This overload operates on a specific chunk, not the whole set of active chunks. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main overload.
     *
     * This function can be seen as a helper function to calculate_packet_destination(std::int64_t timestamp,
     * std::int64_t fengine_id) as it is only called by that function.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t timestamp,
                                                                          std::int64_t fengine_id, chunk &c);

    /* Function called by allocator::allocator function to determine which location in memory and chunk to assign a
     * heap to. This function will call the calculate_packet_destination(...) function to determine the actual location
     * to store this data in.
     *
     * See the allocator class and calculate_packet_destination() documentation above for more information.
     */
    void *allocate(std::size_t size, spead2::recv::packet_header &packet);

  public:
    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this constructor.
     */
    stream(int n_ants, int n_channels, int n_samples_per_channel, int n_pols, int sample_bits, int timestamp_step,
           size_t heaps_per_fengine_per_chunk, ringbuffer_t &ringbuffer, int thread_affinity = -1,
           bool use_gdrcopy = false);
    ~stream();

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_udp_pcap_file_reader(const std::string &filename);

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_buffer_reader(pybind11::buffer buffer);

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                            const std::string &interface_address, std::size_t buffer_size, int comp_vector = 0,
                            int max_poll = spead2::recv::udp_ibv_config::default_max_poll);

    /// Get the referenced ringbuffer
    ringbuffer_t &get_ringbuffer();
    const ringbuffer_t &get_ringbuffer() const;

    // Get the number of bytes in a specifc chunks buffer.
    std::size_t get_chunk_bytes() const;

    /// Add a chunk to the free pool.
    void add_chunk(std::unique_ptr<chunk> &&c);

    /// Stop stream and block until all readers have wound up.
    virtual void stop() override;
};

} // namespace katxgpu::recv

#endif // KATXGPU_RECV_H
