#include <pybind11/pybind11.h>

#ifndef EXAMPLE_H
#define EXAMPLE_H

namespace py = pybind11;

int add(int i, int j);

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring


    m.def("add1", &add, "A function which adds two numbers");
    
    m.def("add2", &add, "A function which adds two numbers",
    py::arg("i"), py::arg("j"));

    using namespace pybind11::literals;

    m.def("add3", &add, "A function which adds two numbers",
    "i"_a, "j"_a);

    m.def("add4", &add, "A function which adds two numbers",
    "i"_a = 10, "j"_a = 11);

    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;    
}


#endif