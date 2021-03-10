#include "py_common.h"
#include "recv.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace katxgpu::recv
{

/* This class wraps a chunk class so that it can be registered in a python module.
 *
 * More information on the methods exposed to python can be seen in register_module() below.
 */
class py_chunk : public chunk
{
  public:
    // Pointer to the buffer object the chunk wraps.
    pybind11::buffer base;

    // If GPUDirect is being used to copy, this function stores a pointer to the GPU object. See src/README/md for more
    // information.
    pybind11::object device;

    // Pointer to the python view of the buffer being used. Holding this view ensures that python does not garbage
    // collect the buffer while it is being used in C++.
    std::shared_ptr<pybind11::buffer_info> buffer_info;

    py_chunk(pybind11::buffer base, pybind11::object device)
        : base(std::move(base)), device(std::move(device)),
          buffer_info(std::make_shared<pybind11::buffer_info>(
              request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        storage = boost::asio::mutable_buffer(buffer_info->ptr, buffer_info->size * buffer_info->itemsize);
    }
};

/* This class wraps a stream class so that it can be registered in a python module.
 *
 * More information on the methods exposed to python can be seen in register_module() below.
 */
class py_stream : public stream
{
  private:
    // Profiling hooks used when using the python monitor class for metric tracking. The *_wait_chunk() methods are
    // called in the stream::grab_chunk() function - one while trying to grab the chunk semaphor and the next once the
    // semaphor has been grabbed. *_ringbuffer_push(...) is called in the stream::flush_chunk() function. One before the
    // active chunk is pushed to the ringbuffer and one after.
    virtual void pre_wait_chunk() override final;
    virtual void post_wait_chunk() override final;
    virtual void pre_ringbuffer_push() override final;
    virtual void post_ringbuffer_push() override final;

  public:
    pybind11::object monitor;

    py_stream(int n_ants, int n_channels, int n_samples_per_channel, int n_pols, int sample_bits, int timestamp_step,
              std::size_t heaps_per_fengine_per_chunk, ringbuffer_t &ringbuffer, int thread_affinity, bool use_gdrcopy,
              pybind11::object monitor)
        : stream(n_ants, n_channels, n_samples_per_channel, n_pols, sample_bits, timestamp_step,
                 heaps_per_fengine_per_chunk, ringbuffer, thread_affinity, use_gdrcopy),
          monitor(std::move(monitor))
    {
    }
};

void py_stream::pre_wait_chunk()
{
    pybind11::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "wait free_chunk");
}

void py_stream::post_wait_chunk()
{
    pybind11::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("free_chunks", -1);
    }
}

void py_stream::pre_ringbuffer_push()
{
    pybind11::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "push ringbuffer");
}

void py_stream::post_ringbuffer_push()
{
    pybind11::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("recv_ringbuffer", 1);
    }
}

pybind11::module register_module(pybind11::module &parent)
{
    using namespace pybind11::literals;

    pybind11::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katxgpu";

    pybind11::class_<py_chunk>(
        m, "Chunk",
        "Chunk of samples. The samples are stored as a 1-dimensional array contiguous array. If the array were indexed "
        "as a multidimensional array, the indexing would be as follows: chunk.base[timestamp_index][f_engine "
        "_index][channel_index][spectrum_index][polarisation_index]. It is left to the user to calculate the strides "
        "for multidimensional indexing in a 1 dimensional array.")
        .def(pybind11::init<pybind11::buffer, pybind11::object>(), "base"_a, "device"_a = pybind11::none(),
             "Initialises the chunk object.\n\n"
             "Parameters\n"
             "----------\n"
             "base:\n"
             "    Buffer object - needs to occupy a contiguous section of memory.\n"
             "device:\n"
             "    This parameter is only needs to be provided when using gpudirect.")
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readonly("present", &py_chunk::present)
        .def_readonly("base", &py_chunk::base)
        .def_readonly("device", &py_chunk::device);

    pybind11::class_<py_stream>(m, "Stream", "SPEAD stream receiver")
        .def(pybind11::init<int, int, int, int, int, int, std::size_t, stream::ringbuffer_t &, int, bool,
                            pybind11::object>(),
             "n_ants"_a, "n_channels"_a, "n_samples_per_channel"_a, "n_pols"_a, "sample_bits"_a, "timestamp_step"_a,
             "heaps_per_fengine_per_chunk"_a, "ringbuffer"_a, "thread_affinity"_a = -1, "use_gdrcopy"_a = false,
             "monitor"_a = pybind11::none(),
             pybind11::keep_alive<1, 9>(), // The keep_alive is used to tell python not to release
                                           // the ringbuffer until the pystream object is destroyed.
             "Initialises a custom high performance SPEAD2 receiver to receive F-Engine output data on a specific "
             "multicast stream at high data rates. This receiver stores received data in chunks and passes those "
             "chunks to the user.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "n_ants: int\n"
             "    The number of antennas that data will be received from.\n"
             "n_channels: int\n"
             "    The total number of frequency channels out of the F-Engine.\n"
             "n_channels_per_stream: int\n"
             "    The number of frequency channels contained in the stream.\n"
             "n_samples_per_channel: int\n"
             "    The number of time samples received per frequency channel.\n"
             "n_pols: int\n"
             "    The number of pols per antenna. Expected to always be 2 at the moment.\n"
             "sample_bits: int\n"
             "    The number of bits per sample. Only 8 bits is supported at the moment.\n"
             "timestamp_step: int\n"
             "    Each heap contains a timestamp. The timestamp between consecutive heaps changes depending on the FFT "
             "size and the number of time samples per channel. This parameter defines the difference in timestamp "
             "values between consecutive heaps. This parameter can be calculated from the array configuration "
             "parameters for power-of-two array sizes, but is configurable to allow for greater flexibility during "
             "testing.\n"
             "heaps_per_fengine_per_chunk: int\n"
             "    Each chunk out of the SPEAD2 receiver will contain multiple heaps from each antenna. This parameter "
             "specifies the number of heaps per antenna that each chunk will contain.\n"
             "ringbuffer: katxgpu._katxgpu.recv.Ringbuffer\n"
             "    All completed heaps will be queued on this ringbuffer object.\n"
             "thread_affinity: int\n"
             "    CPU Thread that this receiver will use for processing.\n"
             "use_gdrcopy: bool\n"
             "    Set to true when transferring data directly from the NIC to the GPU using gpudirect. See katxgpu "
             "documentation for more information on how to do this.\n"
             "monitor: katxgpu.monitor.Monitor\n"
             "    Monitor object for collecting metrics.\n"
             "\n")
        .def_property_readonly("ringbuffer",
                               [](py_stream &self) -> stream::ringbuffer_t & { return self.get_ringbuffer(); })
        .def_property_readonly("chunk_bytes", &py_stream::get_chunk_bytes)
        .def(
            "add_chunk",
            [](py_stream &self, const py_chunk &chunk) {
                if (!self.monitor.is_none())
                    self.monitor.attr("event_qsize_delta")("free_chunks", 1);
                self.add_chunk(std::make_unique<py_chunk>(std::move(chunk)));
            },
            "chunk"_a,
            "Give a free chunk to the reciever to be used to store received data.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "chunk: katxgpu._katxgpu.recv.Chunk\n"
            "    Empty chunk to be given to receiver.\n")
        .def("add_udp_pcap_file_reader", &py_stream::add_udp_pcap_file_reader, "filename"_a,
             "Add the pcap transport to the receiver. The receiver will read packet data from a pcap file.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "filename: string\n"
             "    Name of PCAP file to open.\n")
        .def(
            "add_buffer_reader", &py_stream::add_buffer_reader, "buffer"_a,
            "Add the pcap transport to the receiver. The receiver will read packet data python ByteArray generated by "
            "a spead2.send.BytesStream object.\n"
            "\n"
            "It is *NOT* safe to add more than one buffer reader to this class as there is no mechanism for storing "
            "multiple buffers and the old buffer could be released and garbage collected prematurely when a new one "
            "is added. No error will be thrown when this occurs, it is up to the user to ensure this does not happen.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "buffer: bytes\n"
            "    Buffer containing simulated packet data.\n")
        .def("add_udp_ibv_reader", &py_stream::add_udp_ibv_reader, "endpoints"_a, "interface_address"_a,
             "buffer_size"_a, "comp_vector"_a = 0, "max_poll"_a = spead2::recv::udp_ibv_config::default_max_poll,
             "Add the ibv_udp transport. The receiver will read udp packets off of the specified interface using the "
             "ibverbs library to offload processing from the CPU.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "endpoints: [(string, int)]\n"
             "    IP address and port of multicast stream to listen for traffic on. Needs to be stored as a list of "
             "one item for reasons.\n"
             "interface_address: string\n"
             "    IP address of interface to listen for data on.\n"
             "buffer_size: int\n"
             "    Set the size in bytes of buffer where packets are received bbefore being transferred to a chunk.\n"
             "comp_vector: int\n"
             "    Set the completion channel vector (interrupt) for asynchronous operation. Use a negative value to "
             "poll continuously. Polling should not be used if there are other users of the thread pool. If a "
             "non-negative value is provided, it is taken modulo the number of available completion vectors.\n"
             "max_poll: int\n"
             "    Set maximum number of times to poll in a row. If interrupts are enabled (default), it is the maximum "
             "number of times to poll before waiting for an interrupt; if they are disabled, it is the number of times "
             "to poll before letting other code run on the thread.\n")
        .def("stop", &py_stream::stop, "Stop stream and block until all readers have wound up.");

    register_ringbuffer<stream::ringbuffer_t, py_chunk>(
        m, "Ringbuffer",
        "Ringbuffer for received channelised data. Once the receiver assembles a chunk it places it on this "
        "ringbuffer. The user can then pop this completed chunk off of the ringbuffer.");

    return m;
}

} // namespace katxgpu::recv
