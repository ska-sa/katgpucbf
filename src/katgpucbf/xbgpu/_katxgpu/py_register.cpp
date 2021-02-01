#include <pybind11/pybind11.h>
#include <spead2/common_ringbuffer.h>
#include "py_recv.h"
//#include "py_send.h"

namespace py = pybind11;

PYBIND11_MODULE(_katxgpu, m)
{
    m.doc() = "C++ backend for katxgpu";
    py::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    py::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    katxgpu::recv::register_module(m);
    //katxgpu::send::register_module(m);
}
