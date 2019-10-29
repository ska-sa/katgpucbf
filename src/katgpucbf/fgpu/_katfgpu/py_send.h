#ifndef KATFGPU_PY_SEND_H
#define KATFGPU_PY_SEND_H

#include <pybind11/pybind11.h>

namespace katfgpu::send
{

pybind11::module register_module(pybind11::module &parent);

}

#endif // KATFGPU_PY_SEND_H
