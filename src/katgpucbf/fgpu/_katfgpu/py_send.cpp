#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "send.h"
#include "py_send.h"
#include "py_common.h"

namespace py = pybind11;

namespace katfgpu::send
{

class py_chunk : public chunk
{
public:
    py::buffer base;
    py::object device;
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base, py::object device)
        : base(std::move(base)), device(std::move(device)),
        buffer_info(std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS)))
    {
        storage = boost::asio::const_buffer(buffer_info->ptr,
                                            buffer_info->size * buffer_info->itemsize);
    }

    explicit py_chunk(py::buffer base) : py_chunk(base, py::none()) {}

    py::buffer get_base() const
    {
        return base;
    }

    void set_base(py::buffer base)
    {
        this->base = std::move(base);
        buffer_info = nullptr;
        buffer_info = std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS));
        storage = boost::asio::const_buffer(buffer_info->ptr,
                                            buffer_info->size * buffer_info->itemsize);
    }
};

/* For enrichment and education:
 * The C++ destructor for py::buffer_info calls PyBuffer_Release, which tells
 * Python that C/C++ is no longer using the buffer (and which has other
 * implications beyond just garbage collection). So this is to manage the
 * lifetime of the py::buffer_info rather than the buffer object itself.
 */
class memory_regions_holder
{
private:
    std::vector<py::buffer_info> buffer_infos;

public:
    explicit memory_regions_holder(std::vector<py::buffer> &buffers);
    std::vector<std::pair<const void *, std::size_t>> get_memory_regions() const;
};

memory_regions_holder::memory_regions_holder(std::vector<py::buffer> &buffers)
{
    buffer_infos.reserve(buffers.size());
    for (py::buffer &buffer : buffers)
        buffer_infos.push_back(request_buffer_info(buffer, PyBUF_C_CONTIGUOUS));
}

std::vector<std::pair<const void *, std::size_t>> memory_regions_holder::get_memory_regions() const
{
    std::vector<std::pair<const void *, std::size_t>> memory_regions;
    memory_regions.reserve(buffer_infos.size());
    for (const py::buffer_info &buffer_info : buffer_infos)
    {
        memory_regions.emplace_back(buffer_info.ptr,
                                    buffer_info.size * buffer_info.itemsize);
    }
    return memory_regions;
}

class py_sender : private memory_regions_holder, public sender
{
private:
    py::object monitor;

    virtual void pre_push_free_ring() override final;
    virtual void post_push_free_ring() override final;

public:
    py_sender(std::size_t free_ring_capacity,
              std::vector<py::buffer> &memory_regions,
              int thread_affinity, int comp_vector, int feng_id,
              const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
              int ttl, const std::string &interface_address, bool ibv,
              std::size_t max_packet_size, double rate, std::size_t max_heaps,
              py::object monitor)
        : memory_regions_holder(memory_regions),
        sender(free_ring_capacity, get_memory_regions(),
               thread_affinity, comp_vector, feng_id, endpoints,
               ttl, interface_address, ibv, max_packet_size, rate, max_heaps),
        monitor(std::move(monitor))
    {
    }
};

void py_sender::pre_push_free_ring()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
        monitor.attr("event_state")("send", "push free_ringbuffer");
}

void py_sender::post_push_free_ring()
{
    py::gil_scoped_acquire gil;
    if (!monitor.is_none())
    {
        monitor.attr("event_state")("send", "other");
        monitor.attr("event_qsize_delta")("send_free_ringbuffer", 1);
    }
}

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("send");
    m.doc() = "sender for katfgpu";

    // TODO: provide access to error information
    py::class_<py_chunk>(m, "Chunk", "Chunk of heaps")
        .def(py::init<py::buffer, py::object>(), "base"_a, "device"_a = py::none())
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readwrite("channels", &py_chunk::channels)
        .def_readwrite("acc_len", &py_chunk::acc_len)
        .def_readwrite("frames", &py_chunk::frames)
        .def_readwrite("pols", &py_chunk::pols)
        .def_property("base", &py_chunk::get_base, &py_chunk::set_base)
        .def_readwrite("device", &py_chunk::device)
    ;

    register_ringbuffer<ringbuffer_t, py_chunk>(m, "Ringbuffer", R"pydocstring(
        Ringbuffer of chunks for channelised output samples.

        Parameters
        ----------
        cap
            Capacity of the ringbuffer.
    )pydocstring");

    py::class_<py_sender>(m, "Sender", R"pydocstring(
        Converts Chunks to heaps and transmits them.

        Parameters
        ----------
        free_ring_capacity
            Capacity of the ringbuffer of free chunks.
        memory_regions: List[Chunk]
            A list of the chunks' bases that will be worked on.
        thread_affinity
            CPU core for output-handling thread.
        comp_vector
            Completion vector for transmission, or -1 for polling.
        feng_id
            ID of the F-engine indicating which one in the array this is.
        endpoints
            IP addresses and ports of multicast stream to listen for traffic on.
        ttl
            TTL for outgoing packets.
        interface_address: str
            IP address of the network device to use for output.
        ibv
            Use ibverbs for output.
        max_packet_size
            Size for output packets.
        rate: float
            Target network transmission rate.
        max_heaps
            Maximum number of in-flight heaps for the sender Stream.
        monitor: Optional[Monitor]
            Monitor to use for collecting metrics.
        )pydocstring")
        .def(py::init<
                 std::size_t,
                 std::vector<py::buffer> &,
                 int, int, int,
                 const std::vector<std::pair<std::string, std::uint16_t>> &,
                 int, const std::string &, bool,
                 std::size_t, double, std::size_t,
                 py::object>(),
            "free_ring_capacity"_a,
            "memory_regions"_a,
            "thread_affinity"_a,
            "comp_vector"_a,
            "feng_id"_a,
            "endpoints"_a,
            "ttl"_a,
            "interface_address"_a,
            "ibv"_a,
            "max_packet_size"_a,
            "rate"_a,
            "max_heaps"_a,
            "monitor"_a = py::none())
        .def("send_chunk", [](py_sender &self, py_chunk &chunk)
        {
            auto c = std::make_unique<py_chunk>(std::move(chunk));
            self.send_chunk(std::move(c));
        }, "chunk"_a,
        py::doc(R"pydocstring(
            Schedule transmission of the chunk.

            Loops through the "frames" within the chunk, and through the
            substreams (a few channels need to go to each substream), creating
            an appropriate number of heaps and scheduling them for sending by
            the Stream. The chunk is returned to the free ring once done.
        )pydocstring" ))
        .def("push_free_ring", [](py_sender &self, py_chunk &chunk)
        {
            auto c = std::make_unique<py_chunk>(std::move(chunk));
            self.push_free_ring(std::move(c));
        }, "chunk"_a,
        py::doc("Add a chunk to the ringbuffer, blocking if necessary.\n"))
        .def("stop", &py_sender::stop,
        py::doc(R"pydocstring(
            Cleanly shut down transmission.

            Stop the ringbuffer from accepting new chunks, block until all
            enqueued heaps have been sent, and shut down the sender's thread
            pool.
            )pydocstring"))
        .def_property_readonly("free_ring", [](py_sender &self) -> ringbuffer_t &
        {
            return self.get_free_ring();
        })
    ;

    return m;
}

} // namespace katfgpu::send
