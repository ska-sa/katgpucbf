/* This top level file is called when the katgpucbf module is being installed
 * (i.e. When setup.py is run). This file starts the process of wrapping the C++
 * code into Python modules. This process is known as "registering".
 *
 * The pybind11 library is used to facilitate this process. The pybind11
 * documentation can be found here:
 * https://pybind11.readthedocs.io/en/stable/
 *
 * This module calls the register functions defined in various submodules.
 */

#include <pybind11/pybind11.h>
#include <spead2/common_ringbuffer.h>
#include "py_send.h"

namespace py = pybind11;

PYBIND11_MODULE(_katfgpu, m)
{
    m.doc() = "C++ backend for katgpucbf.fgpu";
    py::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    py::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    katfgpu::send::register_module(m);
}
