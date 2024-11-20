#include <pybind11/pybind11.h>

namespace py = pybind11;

void pyExportCppSolve(py::module&);

PYBIND11_MODULE(_piff, m) {
    pyExportCppSolve(m);
}
