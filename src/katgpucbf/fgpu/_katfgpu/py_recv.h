#ifndef KATFGPU_PY_RECV_H
#define KATFGPU_PY_RECV_H

#include <pybind11/pybind11.h>

namespace katfgpu::recv
{

pybind11::module register_module(pybind11::module &parent);

}

#endif // KATFGPU_PY_RECV_H
