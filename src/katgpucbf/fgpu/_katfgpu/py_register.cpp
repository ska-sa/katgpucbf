#include <pybind11/pybind11.h>
#include <spead2/common_ringbuffer.h>
#include "py_recv.h"
#include "py_send.h"

namespace py = pybind11;

PYBIND11_MODULE(_katfgpu, m)
{
    m.doc() = "C++ backend for katfgpu";
    py::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    py::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    katfgpu::recv::register_module(m);
    katfgpu::send::register_module(m);
}
