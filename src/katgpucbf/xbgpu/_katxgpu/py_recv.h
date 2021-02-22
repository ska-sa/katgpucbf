/* This file along with the py_recv.cpp registers the SPEAD2 C++ receiver katxgpu._katxgpu.recv module in python.
 *
 * This module enables efficient, lightweight packet reception of MeerKAT F-Engine output multicast data at high data
 * rates.
 *
 * This header file contains very little information as only the register_module() function needs to be called outside
 * of this file. For more detailed information of the various wrapper classes please see the py_recv.cpp file.
 */

#ifndef KATXGPU_PY_RECV_H
#define KATXGPU_PY_RECV_H

#include <pybind11/pybind11.h>

namespace katxgpu::recv
{

/* This function registers the SPEAD2 C++ katxgpu receiver module in python. This module can be imported into python
 * using "import katxgpu._katxgpu.recv". This function exposes three main classes into python. Interacting with these
 * three classes gives the user all the functionality needed to control the receiver.
 *
 * The three classes of interest are:
 * 1. katxgpu._katxgpu.recv.Chunk - A chunk is a class containing a buffer and associated metadata where a number of
 * received heaps are stored.
 * 2. katxgpu._katxgpu.recv.Stream - A stream manages the process of receiving network packets and reassembling them
 * into SPEAD heaps. These heaps are then copied to the relevant chunk.
 * 3. katxgpu._katxgpu.recv.Ringbuffer - Once the receiver has copied all the required heaps into a chunk, the chunk is
 * moved to this ringbuffer. The user can then pop the chunks off of this ringbuffer for processing.
 *
 * These three classes are all described in more detail in py_recv.cpp. The main description can be found in the
 * register_module(...) definition so that the documentation is also available in python.
 *
 * Lots of the syntax here is specific to pybind11 and should be looked at in conjunction with pybind11s documentation
 * if a deeper understanding is required.
 */
pybind11::module register_module(pybind11::module &parent);

} // namespace katxgpu::recv

#endif // KATXGPU_PY_RECV_H
