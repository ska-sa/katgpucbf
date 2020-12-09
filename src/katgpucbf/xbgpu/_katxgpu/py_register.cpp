// Top level file for binding C++ functions to python
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j)
{
    return i + j;
}

/* This macro creates a function that will be called when calling "import katxgpu._katxgpu" in 
 * python. Code within this macro will bind C++ functions, classes and attributes to python.
 */
PYBIND11_MODULE(_katxgpu, m) {
    m.doc() = "C++ backend for katxgpu."; // optional module docstring


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