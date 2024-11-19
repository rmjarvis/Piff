#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//#include <pybind11/stl.h>
//#include <pybind11/eigen.h>

namespace py = pybind11;

void pyExportCppSolve(py::module&);

PYBIND11_MODULE(_piff, m) {
    pyExportCppSolve(m);
}
