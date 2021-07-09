
/* This top level file is called when the katxbgpu module is being installed (i.e. When setup.py is run). This file
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

/* This function registers the _katxbgpu networking submodule in the katxbgpu python project.
 *
 * This function does not do much. It basically calls the register functions for the different submodules within this
 * module.
 *
 * This module can be imported using "import katgpucbf.xbgpu._katxbgpu".
 */
PYBIND11_MODULE(_katxbgpu, m)
{
    m.doc() = "C++ backend for katgpucbf.xbgpu";

    // Not quite sure where these exceptions occur or what they do but they seem important.
    pybind11::register_exception<spead2::ringbuffer_stopped>(m, "Stopped");
    pybind11::register_exception<spead2::ringbuffer_empty>(m, "Empty");

    // Register various submodules
    katxbgpu::recv::register_module(m);
    // katxbgpu::send::register_module(m); 
}
