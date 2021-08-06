#include <memory>
#include <utility>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "recv.h"
#include "py_common.h"

namespace py = pybind11;

namespace katfgpu::recv
{

/* Wraps a chunk class so that it can be registered in a Python module.
 */
class py_chunk : public chunk
{
public:
    /// Pointer to the buffer object the chunk wraps.
    py::buffer base;

    /// If GPUDirect is being used to copy, this function stores a pointer to the GPU object.
    py::object device;

    /// Pointer to the Python view of the buffer being used, telling Python that it is in use.
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base, py::object device)
        : base(std::move(base)), device(std::move(device)),
        buffer_info(std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        storage = boost::asio::mutable_buffer(buffer_info->ptr,
                                              buffer_info->size * buffer_info->itemsize);
    }
};

/* This class wraps a stream class so that it can be registered in a python module.
 */
class py_stream : public stream
{
private:
    /* Profiling hooks used when using the Python `Monitor` class for
     * metric-tracking. The *_wait_chunk() methods are called in the
     * stream::grab_chunk() function - one while trying to grab the chunk
     * semaphore and the next once the semaphore has been grabbed.
     * *_ringbuffer_push(...) is called in the stream::flush_chunk() function.
     * One before the active chunk is pushed to the ringbuffer and one after.
     */
    virtual void pre_wait_chunk() override final;
    virtual void post_wait_chunk() override final;
    virtual void pre_ringbuffer_push() override final;
    virtual void post_ringbuffer_push() override final;

public:
    py::object monitor;

    py_stream(int pol, int sample_bits, std::size_t packet_samples, std::size_t chunk_samples,
              ringbuffer_t &ringbuffer, int thread_affinity, bool mask_timestamp, bool use_gdrcopy,
              py::object monitor)
        : stream(pol, sample_bits, packet_samples, chunk_samples, ringbuffer, thread_affinity,
                 mask_timestamp, use_gdrcopy),
        monitor(std::move(monitor))
    {
    }
};

void py_stream::pre_wait_chunk()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "wait free_chunk");
}

void py_stream::post_wait_chunk()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("free_chunks", -1);
    }
}

void py_stream::pre_ringbuffer_push()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("recv", "push ringbuffer");
}

void py_stream::post_ringbuffer_push()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("recv", "other");
        monitor.attr("event_qsize_delta")("recv_ringbuffer", 1);
    }
}

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katfgpu";

    py::class_<py_chunk>(m, "Chunk", R"pydocstring(
        Chunk of samples.

        The samples are stored as a 1-dimensional, contiguous array. The samples
        are 10-bit, and therefore can't be read out using normal indexing.
        Decoding of 10-bit samples is currently handled on the GPU.

        Parameters
        ----------
        base
            Buffer object - needs to occupy a contiguous section of memory.
        device
            This parameter only needs to be provided when using gpudirect.
        )pydocstring")
        .def(py::init<py::buffer, py::object>(), "base"_a, "device"_a = py::none())
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readwrite("pol", &py_chunk::pol)
        .def_readonly("present", &py_chunk::present)
        .def_property_readonly("n_present", [](const py_chunk &c) {
            return std::accumulate(c.present.begin(), c.present.end(), std::size_t(0));
        })
        .def_readonly("base", &py_chunk::base)
        .def_readonly("device", &py_chunk::device)
    ;

    py::class_<py_stream>(m, "Stream", R"pydocstring(
        Chunked Spead2 stream.

        A custom, high-performance SPEAD2 receiver to receive digitiser output
        data on a specific multicast stream at high data rates. This receiver
        stores received data in chunks and passes those chunks to the user.

        Parameters
        ----------
        pol
            Which pol this stream represents. Should be 0 or 1.
        sample_bits
            The number of bits per sample.
        packet_samples
            The number of samples per digitiser packet.
        chunk_samples
            The number of samples to gather up into a chunk.
        ringbuffer: recv.Ringbuffer
            All completed chunks will be queued on this ringbuffer object.
        thread_affinity
            CPU core that this receiver will use for processing, or -1 to not
            set an affinity.
        mask_timestamp
            Mask off bottom bits of timestamp (workaround for broken digitiser).
        use_gdrcopy
            Set to true when transferring data directly from the NIC to the GPU
            using GPUdirect.
        monitor: Optional[katgpucbf.fgpu.monitor.Monitor]
            Monitor object for collecting metrics.
        )pydocstring")
        /* TODO: These types are C++ types in many cases. Figure out how to get
         * them nicely into pybind11 types.
         * https://pybind11.readthedocs.io/en/stable/advanced/misc.html#avoiding-c-types-in-docstrings
         */
        .def(py::init<int, int, std::size_t, std::size_t, stream::ringbuffer_t &, int, bool, bool,
                      py::object>(),
             "pol"_a, "sample_bits"_a, "packet_samples"_a, "chunk_samples"_a,
             "ringbuffer"_a, "thread_affinity"_a = -1, "mask_timestamp"_a = false,
             "use_gdrcopy"_a = false, "monitor"_a = py::none(),
             py::keep_alive<1, 6>())
        .def_property_readonly("ringbuffer", [](py_stream &self) -> stream::ringbuffer_t &
        {
            return self.get_ringbuffer();
        })
        .def_property_readonly("pol", &py_stream::get_pol)
        .def_property_readonly("sample_bits", &py_stream::get_sample_bits)
        .def_property_readonly("packet_samples", &py_stream::get_packet_samples)
        .def_property_readonly("chunk_samples", &py_stream::get_chunk_samples)
        .def_property_readonly("chunk_packets", &py_stream::get_chunk_packets)
        .def_property_readonly("chunk_bytes", &py_stream::get_chunk_bytes)
        .def("add_chunk", [](py_stream &self, const py_chunk &chunk)
            {
                if (!self.monitor.is_none())
                    self.monitor.attr("event_qsize_delta")("free_chunks", 1);
                self.add_chunk(std::make_unique<py_chunk>(std::move(chunk)));
            }, "chunk"_a, R"pydocstring(
            Give a free chunk to the receiver to be used to store received data.

            Parameters
            ----------
            chunk
                Empty chunk to be given to receiver.
            )pydocstring")
        .def("add_udp_pcap_file_reader", &py_stream::add_udp_pcap_file_reader,
            "filename"_a, R"pydocstring(
            Add the pcap transport to the receiver. The receiver will read packet
            data from a pcap file.

            Parameters
            ----------
            filename
                Name of PCAP file to open.
            )pydocstring")
        .def("add_udp_reader", &py_stream::add_udp_reader,
            "endpoints"_a, "interface_address"_a, "buffer_size"_a, "ibv"_a,
            "comp_vector"_a = 0,
            "max_poll"_a = spead2::recv::udp_ibv_config::default_max_poll,
            R"pydocstring(
            Add the UDP transport.

            The receiver will read UDP packets off of the specified interface.

            Parameters
            ----------
            endpoints
                IP addresses and ports of multicast stream to listen for traffic
                on.
            interface_address
                IP address of interface to listen for data on.
            buffer_size
                The size in bytes of buffer where packets are received before
                being transferred to a chunk.
            ibv
                If true, use ibverbs to bypass the kernel for higher efficiency.
                If false, `comp_vector` and `max_poll` are ignored.
            comp_vector
                The completion channel vector (interrupt) for asynchronous
                operation. Use a negative value to poll continuously. If a
                non-negative value is provided, it is taken modulo the number of
                available completion vectors.
            max_poll
                The maximum number of times to poll in a row. If interrupts are
                enabled (default), it is the maximum number of times to poll
                before waiting for an interrupt; if they are disabled, it is the
                number of times to poll before letting other code run on the
                thread.
        )pydocstring")
        .def("stop", &py_stream::stop, "Stop stream and block until all readers have wound up.")
    ;

    register_ringbuffer<stream::ringbuffer_t, py_chunk>(m, "Ringbuffer", R"pydocstring(
        Ringbuffer of chunks for digitiser samples.

        Parameters
        ----------
        cap
            Capacity of the ringbuffer.
    )pydocstring");

    return m;
}

} // namespace katfgpu::recv
