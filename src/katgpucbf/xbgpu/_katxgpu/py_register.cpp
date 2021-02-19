
/* This top level file is called when the katxgpu module is being installed (i.e. When setup.py is run). This file
 * starts the process of wrapping the C++ code into python modules. This process is known as "registering".
 *
 * The pybind11 library is used to facilitate this process. The pybind11 documentation can be found here:
 * https://pybind11.readthedocs.io/en/stable/
 *
 * This module calls the register functions defined in various submodules.
 */

#include "py_recv.h"
#include <pybind11/pybind11.h>
#include <spead2/common_ringbuffer.h>
//#include "py_send.h" - This file still needs to be written.

/* This function registers the _katxgpu networking submodule in the katxgpu python project.
 *
 * This function does not do much. It basically calls the register functions for the different submodules within this
 * module.
 *
 * This module can be imported using "import katxgpu._katxgpu".
 */
PYBIND11_MODULE(_katxgpu, m)
{
    m.doc() = "C++ backend for katxgpu";

    // Not quite sure where these exceptions occur or what they do but they seemed important
    py::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    py::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    // Register various submodules
    katxgpu::recv::register_module(m);
    // katxgpu::send::register_module(m); 
}
