#ifndef KATXGPU_PY_RECV_H
#define KATXGPU_PY_RECV_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace katxgpu::recv
{

py::module register_module(py::module &parent);

} // namespace katxgpu::recv

#endif // KATXGPU_PY_RECV_H
