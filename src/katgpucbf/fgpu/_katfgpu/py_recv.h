/* This file, along with py_recv.cpp, registers the SPEAD2 C++ receiver module
 * in Python as katfgpu._katfgpu.recv.
 *
 * This module enables efficient ingest of digitiser output multicast data at
 * high data rates.
 *
 * This header file contains very little information as only the
 * register_module() function needs to be called outside of this file. For more
 * detailed information of the various wrapper classes please see the
 * py_recv.cpp file.
 */

#ifndef KATFGPU_PY_RECV_H
#define KATFGPU_PY_RECV_H

#include <pybind11/pybind11.h>

namespace katfgpu::recv
{

/* This function registers the SPEAD2 C++ katfgpu receiver module in Python.
 *
 * Lots of the syntax here is specific to pybind11 and should be looked at in
 * conjunction with pybind11's documentation if a deeper understanding is
 * required.
 */
pybind11::module register_module(pybind11::module &parent);

}

#endif // KATFGPU_PY_RECV_H
