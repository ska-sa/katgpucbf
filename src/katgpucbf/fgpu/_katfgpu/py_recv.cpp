#include <memory>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "recv.h"
#include "py_common.h"

namespace py = pybind11;

namespace katfgpu::recv
{

class py_chunk : public chunk
{
public:
    py::buffer base;
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base)
        : base(std::move(base)),
        buffer_info(std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        storage = boost::asio::mutable_buffer(buffer_info->ptr,
                                              buffer_info->size * buffer_info->itemsize);
    }
};

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katfgpu";

    py::class_<py_chunk>(m, "Chunk", "Chunk of samples")
        .def(py::init<py::buffer>(), "base"_a)
        .def_readwrite("timestamp", &py_chunk::timestamp)
        .def_readwrite("pol", &py_chunk::pol)
        .def_readonly("present", &py_chunk::present)
        .def_readonly("base", &py_chunk::base)
    ;

    py::class_<stream>(m, "Stream", "SPEAD stream receiver")
        .def(py::init<int, int, std::size_t, std::size_t, stream::ringbuffer_t &, int, bool>(),
             "pol"_a, "sample_bits"_a, "packet_samples"_a, "chunk_samples"_a,
             "ringbuffer"_a, "thread_affinity"_a = -1, "mask_timestamp"_a = false,
             py::keep_alive<1, 6>())
        .def_property_readonly("ringbuffer", [](stream &self) -> stream::ringbuffer_t &
        {
            return self.get_ringbuffer();
        })
        .def_property_readonly("pol", &stream::get_pol)
        .def_property_readonly("sample_bits", &stream::get_sample_bits)
        .def_property_readonly("packet_samples", &stream::get_packet_samples)
        .def_property_readonly("chunk_samples", &stream::get_chunk_samples)
        .def_property_readonly("chunk_packets", &stream::get_chunk_packets)
        .def_property_readonly("chunk_bytes", &stream::get_chunk_bytes)
        .def("add_chunk", [](stream &self, const py_chunk &chunk)
        {
            self.add_chunk(std::make_unique<py_chunk>(std::move(chunk)));
        }, "chunk"_a)
        .def("add_udp_pcap_file_reader", &stream::add_udp_pcap_file_reader,
             "filename"_a)
        .def("add_udp_ibv_reader", &stream::add_udp_ibv_reader,
             "endpoints"_a, "interface_address"_a, "buffer_size"_a,
             "comp_vector"_a = 0,
             "max_poll"_a = spead2::recv::udp_ibv_config::default_max_poll)
        .def("stop", &stream::stop)
    ;

    register_ringbuffer<stream::ringbuffer_t, py_chunk>(m, "Ringbuffer", "Ringbuffer for samples");

    return m;
}

} // namespace katfgpu::recv
