/* Common functions and classes used in multiple different pybind11 submodules.
 *
 * Currently this file provides a method for registering a ringbuffer object in
 * pybind11 submodules and the request_buffer_info() function for converting
 * Python buffer objects into a C++ memory object.
 */

#ifndef KATFGPU_PY_COMMON_H
#define KATFGPU_PY_COMMON_H

#include <pybind11/pybind11.h>

namespace katfgpu
{

/* Registers a SPEAD2 ringbuffer into python.
 *
 * Send and receive modules both require a ringbuffer. By calling this function
 * while registering a specific module, the ringbuffer is registered in the
 * correct submodule. A ringbuffer object can be registered to multiple modules
 * while the registration only has to be defined here.
 *
 * Basic docstrings are included here but the best place to look for a more
 * thorough understanding is spead2's source.
 */
template<typename R, typename T>
pybind11::class_<R> register_ringbuffer(pybind11::module &m, const char *name, const char *doc)
{
    using namespace pybind11::literals;

    return pybind11::class_<R>(m, name, doc)
        .def(pybind11::init<int>(), "cap"_a)
        .def("pop", [](R &self)
        {
            auto item = self.pop();
            return std::unique_ptr<T>(&dynamic_cast<T &>(*item.release()));
        }, pybind11::doc("Retrieve an item from the queue, blocking until there is one or until the queue is stopped."))
        .def("try_pop", [](R &self)
        {
            auto item = self.try_pop();
            return std::unique_ptr<T>(&dynamic_cast<T &>(*item.release()));
        }, pybind11::doc("Retrieve an item from the queue, if there is one."))
        .def("try_push", [](R &self, T &item)
        {
            std::unique_ptr<T> item2 = std::make_unique<T>(std::move(item));
            try
            {
                self.try_push(std::move(item2));
            }
            catch (std::exception &)
            {
                // We failed, so transfer the content back again.
                if (item2)
                    item = std::move(*item2);
                throw;
            }
        }, "item"_a, pybind11::doc("Append an item to the queue, if there is space."))
        .def_property_readonly("data_fd", [](R &self)
        {
            return self.get_data_sem().get_fd();
        }, pybind11::doc("Get access to the data semaphore's file descriptor."))
    ;
}


/* TODO: replace the `extra_flags` explanation with a reference to either
 * pybind11 or Python C API docs.
 */

/**
 * Take in a pybind11 buffer object from Python and return a buffer_info view.
 * This view can be used by a C++ function to access the memory in the buffer
 * directly. This view also marks the buffer as in-use. As long as this view is
 * held in C++, the Python garbage collector will not release the buffer.
 *
 * \param[in] buffer      Python object supporting the python buffer protocol.
 * \param[in] extra_flags Python buffers support a variety of different internal
 *                        structures. These flags specify the type of buffer the
 *                        C++ program is expecting and can handle. For this
 *                        program the most likely flag to be used is the
 *                        PyBUF_C_CONTIGUOUS flag which indicates that the
 *                        buffer is expected to occupy a contiguous section of
 *                        memory.
 *
 * \return A view to a Python buffer object to allow access to the underlying
 *         buffer memory.
 */
pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

}

#endif  // KATFGPU_PY_COMMON_H
