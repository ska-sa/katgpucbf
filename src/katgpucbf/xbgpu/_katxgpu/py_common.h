#ifndef KATXGPU_PY_COMMON_H
#define KATXGPU_PY_COMMON_H

#include <pybind11/pybind11.h>

namespace katxgpu
{

/* Registers a SPEAD2 ringbuffer into python.
 *
 * This function is kept in py_common.h as multiple different modules could require a ringbuffer. By calling this
 * function while registering a specific module, the ringbuffer is registered in the correct submodule. A ringbuffer
 * object can be registered to multiple modules while the registration only has to be defined here. See py_recv.cpp
 * for an example of how this has been done for the spead2.recv submodule.
 *
 * This code has been copied directly from katfgpu. Figuring out the details is an exercise left to the reader.
 */
template <typename R, typename T>
pybind11::class_<R> register_ringbuffer(pybind11::module &m, const char *name, const char *doc)
{
    using namespace pybind11::literals;

    return pybind11::class_<R>(m, name, doc)
        .def(pybind11::init<int>(), "cap"_a)
        .def(
            "pop",
            [](R &self) {
                auto item = self.pop();
                return std::unique_ptr<T>(&dynamic_cast<T &>(*item.release()));
            },
            "Wait until an item is available on the ringbuffer, removes it from the ringbuffer and returns it.")
        .def(
            "try_pop",
            [](R &self) {
                auto item = self.try_pop();
                return std::unique_ptr<T>(&dynamic_cast<T &>(*item.release()));
            },
            "Similar to try_pop(), but if no item is available, throws a 'ringbuffer_empty' error.")
        .def(
            "try_push",
            [](R &self, T &item) {
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
            },
            "item"_a, "Tries to add an item to the ringbuffer. Returns an exception if this fails.")
        .def_property_readonly(
            "data_fd", [](R &self) { return self.get_data_sem().get_fd(); },
            "Return the posix semaphor file descriptor for this ringbuffer.");
}

/* This function takes in a pybind11 buffer object form python and returns a buffer_info view. This view can be used by
 * a C++ function to access the memory in the buffer directly. This view also marks the buffer as in-use. As long as
 * this view is held in C++, the Python garbage collector will not release the buffer.
 *
 * This function has been copied directly from the SPEAD2 C++ source code. Not much effort has been put into
 * understanding the nuanced details.
 *
 * \param[in] buffer      Python object supporting the python buffer protocol.
 * \param[in] extra_flags Python buffers support a variety of different internal structures. These flags specify the
 * type of buffer the C++ program is expecting and can handle. For this program the most likely flag to be used is the
 * PyBUF_C_CONTIGUOUS flag which indicates that the buffer is expected to occupy a contiguous section of memory.
 *
 * \return Returns a view to a python buffer object to allow access to the underlying buffer memory.
 */
pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

} // namespace katxgpu

#endif // KATXGPU_PY_COMMON_H
