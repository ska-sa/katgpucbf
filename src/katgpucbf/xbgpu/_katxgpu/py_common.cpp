#include <pybind11/pybind11.h>
#include "py_common.h"

namespace py = pybind11;

namespace katxgpu
{

py::buffer_info request_buffer_info(py::buffer &buffer, int extra_flags)
{
    std::unique_ptr<Py_buffer> view(new Py_buffer);
    int flags = PyBUF_STRIDES | PyBUF_FORMAT | extra_flags;
    if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0)
        throw py::error_already_set();
    py::buffer_info info(view.get());
    view.release();
    return info;
}

} // namespace katxgpu
