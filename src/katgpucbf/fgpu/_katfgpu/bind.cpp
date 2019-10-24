#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "receiver.h"

namespace py = pybind11;

// Copied from spead2
py::buffer_info request_buffer_info(py::buffer &buffer, int extra_flags)
{
    std::unique_ptr<Py_buffer> view(new Py_buffer);
    int flags = PyBUF_STRIDES | PyBUF_FORMAT | extra_flags;
    if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0)
        throw py::error_already_set();
    py::buffer_info info(view.get());
    view.release();
    return info;
}

class py_in_chunk : public in_chunk
{
public:
    py::buffer base;
    std::shared_ptr<py::buffer_info> buffer_info;

    py_in_chunk(py::buffer base)
        : base(std::move(base)),
        buffer_info(std::make_shared<py::buffer_info>(
            request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        storage = boost::asio::mutable_buffer(buffer_info->ptr,
                                              buffer_info->size * buffer_info->itemsize);
    }
};

PYBIND11_MODULE(_katfgpu, m)
{
    using namespace pybind11::literals;
    m.doc() = "C++ backend of fgpu";

    py::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    py::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    py::class_<py_in_chunk>(m, "InChunk", "Chunk of samples")
        .def(py::init<py::buffer>(), "base"_a)
        .def_readwrite("timestamp", &py_in_chunk::timestamp)
        .def_readwrite("pol", &py_in_chunk::pol)
        .def_readonly("present", &py_in_chunk::present)
        .def_readonly("base", &py_in_chunk::base)
    ;

    py::class_<receiver> receiver_class(m, "Receiver", "SPEAD stream receiver");
    receiver_class
        .def(py::init<int, int, std::size_t, std::size_t, receiver::ringbuffer_t &, int>(),
             "pol"_a, "sample_bits"_a, "packet_samples"_a, "chunk_samples"_a,
             "ringbuffer"_a, "thread_affinity"_a = -1, py::keep_alive<1, 6>())
        .def_property_readonly("ringbuffer", [](receiver &self) -> receiver::ringbuffer_t &
        {
            return self.get_ringbuffer();
        })
        .def_property_readonly("pol", &receiver::get_pol)
        .def_property_readonly("sample_bits", &receiver::get_sample_bits)
        .def_property_readonly("packet_samples", &receiver::get_packet_samples)
        .def_property_readonly("chunk_samples", &receiver::get_chunk_samples)
        .def_property_readonly("chunk_packets", &receiver::get_chunk_packets)
        .def_property_readonly("chunk_bytes", &receiver::get_chunk_bytes)
        .def("add_chunk", [](receiver &self, const py_in_chunk &chunk)
        {
            self.add_chunk(std::make_unique<py_in_chunk>(chunk));
        })
        .def("add_udp_pcap_file_reader", &receiver::add_udp_pcap_file_reader,
             "filename"_a)
        .def("add_udp_ibv_reader", &receiver::add_udp_ibv_reader,
             "endpoints"_a, "interface_address"_a, "buffer_size"_a,
             "comp_vector"_a = 0,
             "max_poll"_a = spead2::recv::udp_ibv_reader::default_max_poll)
        .def("stop", &receiver::stop)
    ;

    py::class_<receiver::ringbuffer_t>(receiver_class, "Ringbuffer", "Ringbuffer for samples")
        .def(py::init<int>())
        .def("pop", [](receiver::ringbuffer_t &self)
        {
            std::unique_ptr<in_chunk> chunk = self.pop();
            return std::unique_ptr<py_in_chunk>(&dynamic_cast<py_in_chunk &>(*chunk.release()));
        })
        .def_property_readonly("data_fd", [](receiver::ringbuffer_t &self)
        {
            return self.get_data_sem().get_fd();
        })
    ;
}
