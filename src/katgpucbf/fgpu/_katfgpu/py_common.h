#ifndef KATFGPU_PY_COMMON_H
#define KATFGPU_PY_COMMON_H

#include <pybind11/pybind11.h>

namespace katfgpu
{

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
        }, "item"_a)
        .def_property_readonly("data_fd", [](R &self)
        {
            return self.get_data_sem().get_fd();
        })
    ;
}

pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

}

#endif  // KATFGPU_PY_COMMON_H
