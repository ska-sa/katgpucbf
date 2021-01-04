#include "py_recv.h"
#include "recv.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace katxgpu
{

//Copied straight from SPEAD2 
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

} // namespace katxgpu

namespace katxgpu::recv
{

class py_chunk : public chunk
{
  public:
    py::buffer base;
    std::shared_ptr<py::buffer_info> buffer_info;

    py_chunk(py::buffer base)
        : base(std::move(base)), buffer_info(std::make_shared<py::buffer_info>(
                                     request_buffer_info(this->base, PyBUF_C_CONTIGUOUS | PyBUF_WRITEABLE)))
    {
        uint8_t *ptr = (uint8_t *)buffer_info->ptr;
        py::print(ptr[0], ptr[1], ptr[2]);
        py::print(1, 2.0, "three");
        ptr[0] = 11;
        py::print(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11]);
        storage = boost::asio::mutable_buffer(buffer_info->ptr, buffer_info->size * buffer_info->itemsize);
    }
};

py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katxgpu";
    m.attr("the_answer") = 47;

    py::class_<py_chunk>(m, "Chunk", "Chunk of samples").def(py::init<py::buffer>(), "base"_a);
    // .def_readwrite("timestamp", &py_chunk::timestamp)
    // .def_readwrite("pol", &py_chunk::pol)
    // .def_readonly("present", &py_chunk::present)
    // .def_readonly("base", &py_chunk::base)
    // .def_readonly("device", &py_chunk::device);

    return m;
}

} // namespace katxgpu::recv