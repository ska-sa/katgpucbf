/* This file contains the collection of classes used to implement a C++ SPEAD2 receiver for a single X-Engine that
 * receives data from multiple F-Engines.
 *
 * The three classes that are defined here are:
 * 1. katxbgpu::recv::chunk - A chunk is a class containing a buffer and associated metadata where a number of
 * received heaps are stored in a single contigous manner.
 * 2. katxbgpu::recv::stream - A stream manages the process of receiving network packets and reassembling them
 * into SPEAD heaps. The stream then copies these heaps into the relevant chunk.
 * 3. katxgp::recv::allocator - This allocator is used for telling the SPEAD receiver where in a chunk a specific heap
 * must be copied. This allocator also determines the specific chunk that this heap belongs to. The allocator is very
 * lightweight with most of its logic being shifted to sub-functions within the katxbgpu::recv::stream class.
 *
 * This class is registered as a C++ module. SPEAD2 already has a python module. The reason a custom module is used over
 * the standard SPEAD2 module is because the data rates being dealt with are very high. The standard SPEAD2 python
 * library would be producing heaps too quickly for Python to keep up. This class combines a number of heaps into a
 * single chunk significantly reducing the amount of computation that is required to be done in Python.
 *
 * The functioning of this class is explained in more detail in katxbgpu/src/README.md
 *
 * TODO: Move 'pybind11::buffer_info m_BufferView' and associated functions to py_recv.h
 * TODO: Implement logging slightly differently - at the moment, the SPEAD2 logger is used. It may be worthwhile to have
 * a seperate katxbgpu logger to seperate a SPEAD2 log from a katxbgpu log.
 * TODO: Class member variables should start with an m_ for clarity.
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

namespace katxbgpu::recv
{

/* Collection of contiguous samples in memory, together with information about which samples are present. It references
 * the sample storage but does not own it. However, it is polymorphic, so subclasses can own storage, and it is
 * guaranteed that it will not be destroyed from the worker thread.
 *
 * The samples are stored as a 1 dimensional array contiguous array. If the array were indexed  as a multidimensional
 * array, the dimensions would be as follows:
 * [heaps_per_fengine_per_chunk][n_ants][n_channels_per_stream][n_samples_per_channel][n_pols]. It is left to the user
 * to calculate the strides for multidimensional indexing in a 1 dimensional array.
 *
 * The heaps_per_fengine_per_chunk and n_ants correspond to different packets while the remaining indexes correspond to
 * how data is arranged within a chunk.
 */
struct chunk
{
    std::int64_t m_i64timestamp;           ///< Timestamp of first collection of heaps
    std::vector<bool> m_vbPacketPresent;   ///< Bitmask of packets that are present
    boost::asio::mutable_buffer m_storage; ///< Storage for samples

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
    stream &m_receiverStream;

  public:
    explicit allocator(stream &receiverStream);

    virtual pointer allocate(std::size_t ulHeapSize_bytes, void *receivedPacket) override;
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

    // Array configuration parameters
    const int m_iSampleBits;           ///< Number of bits per sample
    const int m_iNumAnts;              ///< Number of antennas in the array
    const int m_iNumChannels;          ///< Number of channels in each packet
    const int m_iNumSamplesPerChannel; ///< Number of samples stored in a single channel
    const int m_iNumPols;              ///< Number of polarisations in each sample
    const int m_iComplexity = 2;       ///< Indicates two values per sample - one real and one imaginary.

    // Internal parameters
    const int64_t m_i64HeapsPerFenginePerChunk;   ///< A chunk has this many heaps per F-Engine.
    const std::size_t m_ulMaxActiveChunks;  ///< Maximum number of chunks in m_activeChunksQueue
    const int64_t m_i64TimestampStep;             ///< Increase in timestamp between successive heaps from the same F-Engine.
    const std::size_t m_ulPacketSize_bytes; ///< Number of payload bytes in each packet
    const std::size_t m_ulChunkSize_bytes;  ///< Number of payload bytes in each chunk

    std::int64_t m_i64FirstTimestamp = -1; ///< Very first timestamp observed. Populated when first packet is received.

    mutable std::mutex m_freeChunksLock;     ///< Protects access to @ref m_freeChunksStack
    spead2::semaphore m_freeChunksSemaphore; ///< Semaphore that is put whenever chunks are added

    /* When the user gives a new chunk (or a recycled old chunk) to the receiver, it is added to the
     * m_freeChunksStack. The chunks on this stack are not in use, but when they are required, they will be moved from
     * this stack to the active chunks queue.
     */
    std::stack<std::unique_ptr<chunk>> m_freeChunksStack;

    /* Chunks that are actively being assembled from multiple heaps are stored in this queue. The receiver can be
     * assembling multiple chunks at any one time. Once a chunk is fully assembled, the receiver will move it to the
     * m_completedChunksRingbuffer object.
     */
    std::deque<std::unique_ptr<chunk>> m_activeChunksQueue; ///< Chunks currently being filled

    /* All chunks that have been assembled by the receiver and are ready to be passed to the user will be pushed onto
     * this ringbuffer.
     */
    ringbuffer_t &m_completedChunksRingbuffer;

    /* TODO: This m_BufferView is only used during unit testing. It stores the python view of the buffer containing
     * simulated packets. More detail in "add_buffer_reader()" function. I do not think that this file is the best place
     * for this object. Its a pybind11 object, so it should go under py_recv.h. The add_buffer_reader() function would
     * need to be modified too to accomodate this change. I would need to move a bunch of other functions around to make
     * that happen, so I will wait until I have a spare moment.
     */
    pybind11::buffer_info m_BufferView;

    /// Obtain a fresh chunk from the free pool (blocking if necessary)
    void grab_chunk(std::int64_t i64Timestamp);

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
     * If the timestamp is beyond the last active chunk, old chunks may be flushed and new chunks appended.
     *
     * NOTE: This function was copied, renamed and reworked from the decode_timestamp() function in katfgpu but performs
     * essentially the same function.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t i64Timestamp,
                                                                          std::int64_t i64FengineID);

    /* This overload operates on a specific chunk, not the whole set of active chunks. Returning the same chunk is
     * redundant, but allows this function to be tail-called from the main overload.
     *
     * This function can be seen as a helper function to calculate_packet_destination(std::int64_t i64Timestamp,
     * std::int64_t fengine_id) as it is only called by that function.
     */
    std::tuple<void *, chunk *, std::size_t> calculate_packet_destination(std::int64_t i64Timestamp,
                                                                          std::int64_t i64FengineID, chunk &chunk);

    /* Function called by allocator::allocator function to determine which location in memory and chunk to assign a
     * heap to. This function will call the calculate_packet_destination(...) function to determine the actual location
     * to store this data in.
     *
     * See the allocator class and calculate_packet_destination() comments above for more information.
     */
    void *allocate(std::size_t ulHeapSize, spead2::recv::packet_header &receivedPacket);

  public:
    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this constructor.
     */
    stream(int iNumAnts, int iNumChannels, int iNumSamplesPerChannel, int iNumPols, int iSampleBits, int iTimestampStep,
           size_t iHeapsPerFenginePerChunk, size_t ulMaxActiveChunks,
           ringbuffer_t &m_completedChunksRingbuffer, int iThreadAffinity = -1,
           bool bUseGDRCopy = false);
    ~stream();

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_udp_pcap_file_reader(const std::string &strFilename);

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_buffer_reader(pybind11::buffer buffer);

    /* See pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver") documentation in py_recv.cpp for a
     * descritpion of this function.
     */
    void add_udp_ibv_reader(const std::vector<std::pair<std::string, std::uint16_t>> &vEndpoints,
                            const std::string &strInterfaceAddress, std::size_t ulBufferSize_bytes, int iCompVector = 0,
                            int iMaxPoll = spead2::recv::udp_ibv_config::default_max_poll);

    /// Get the referenced ringbuffer
    ringbuffer_t &get_ringbuffer();
    const ringbuffer_t &get_ringbuffer() const;

    // Get the number of bytes in a specifc chunks buffer.
    std::size_t get_chunk_bytes() const;

    /// Add a chunk to the free pool.
    void add_chunk(std::unique_ptr<chunk> &&pChunk);

    /// Stop stream and block until all readers have wound up.
    virtual void stop() override;
};

} // namespace katxbgpu::recv

#endif // KATXGPU_RECV_H
