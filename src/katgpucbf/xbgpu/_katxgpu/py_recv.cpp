#include <pybind11/pybind11.h>
#include "py_recv.h"

namespace py = pybind11;

namespace katxgpu::recv
{

py::module register_module(py::module &parent)
{
    py::module m = parent.def_submodule("recv");
    m.doc() = "receiver for katxgpu";

    m.attr("the_answer") = 47;
    return m;
}

}