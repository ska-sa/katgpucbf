#ifndef KATFGPU_PY_COMMON_H
#define KATFGPU_PY_COMMON_H

#include <pybind11/pybind11.h>

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
        })
        .def("try_pop", [](R &self)
        {
            auto item = self.try_pop();
            return std::unique_ptr<T>(&dynamic_cast<T &>(*item.release()));
        })
        .def_property_readonly("data_fd", [](R &self)
        {
            return self.get_data_sem().get_fd();
        })
    ;
}

pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

#endif  // KATFGPU_PY_COMMON_H
