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
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base)
        : base(std::move(base)),
        buffer_info(std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS)))
    {
        storage = boost::asio::const_buffer(buffer_info->ptr,
                                            buffer_info->size * buffer_info->itemsize);
    }
};

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("send");
    m.doc() = "sender for katfgpu";

    // TODO: provide access to error information
    py::class_<py_chunk>(m, "Chunk", "Chunk of heaps")
        .def(py::init<py::buffer>(), "base"_a)
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readwrite("channels", &py_chunk::channels)
        .def_readwrite("acc_len", &py_chunk::acc_len)
        .def_readwrite("frames", &py_chunk::frames)
        .def_readwrite("pols", &py_chunk::pols)
        .def_readonly("base", &py_chunk::base)
    ;

    register_ringbuffer<ringbuffer_t, py_chunk>(m, "Ringbuffer", "Ringbuffer of chunks");

    py::class_<sender>(m, "Sender", "Converts Chunks to heaps and transmit them")
        .def(py::init([](
                const std::vector<py::buffer> &initial_buffers,
                int thread_affinity, int comp_vector,
                const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
                int ttl, const std::string &interface_address, bool ibv,
                std::size_t max_packet_size, double rate, std::size_t max_heaps)
            {
                std::vector<std::unique_ptr<chunk>> initial_chunks;
                for (const auto &buffer : initial_buffers)
                    initial_chunks.push_back(std::make_unique<py_chunk>(buffer));
                return std::make_unique<sender>(
                    std::move(initial_chunks),
                    thread_affinity, comp_vector, endpoints, ttl, interface_address,
                    ibv, max_packet_size, rate, max_heaps);
            }),
            "initial_buffers"_a,
            "thread_affinity"_a,
            "comp_vector"_a,
            "endpoints"_a,
            "ttl"_a,
            "interface_address"_a,
            "ibv"_a,
            "max_packet_size"_a,
            "rate"_a,
            "max_heaps"_a)
        .def("send_chunk", [](sender &self, py_chunk &chunk)
        {
            auto c = std::make_unique<py_chunk>(std::move(chunk));
            self.send_chunk(std::move(c));
        }, "chunk"_a)
        .def("stop", &sender::stop)
        .def_property_readonly("free_ring", [](sender &self) -> ringbuffer_t &
        {
            return self.get_free_ring();
        })
    ;

    return m;
}

} // namespace katfgpu::send
