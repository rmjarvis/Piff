#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

// Useful shorthand
const int X = Eigen::Dynamic;

// AlignedDeleted and allocateAlignedMemory are borrowed from GalSim:
// https://github.com/GalSim-developers/GalSim/blob/releases/2.6/src/Image.cpp#L127
template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { delete [] ((char**)p)[-1]; }
};

template <typename T>
std::shared_ptr<T> allocateAlignedMemory(int n)
{
    char* mem = new char[n * sizeof(T) + sizeof(char*) + 15];
    T* data = reinterpret_cast<T*>( (uintptr_t)(mem + sizeof(char*) + 15) & ~(size_t) 0x0F );
    ((char**)data)[-1] = mem;
    std::shared_ptr<T> owner(data, AlignedDeleter<T>());
    return owner;
}

template<class T, ssize_t N>
Eigen::Vector<T, X> _solve_direct_cpp_impl(
        std::vector<py::array_t<T, py::array::f_style>> bList,
        std::vector<py::array_t<T, py::array::f_style>> AList,
        std::vector<py::array_t<T, py::array::f_style>> KList,
        ssize_t n)
{

    // get size info
    py::buffer_info b_buffer_info = bList[0].request();
    py::buffer_info A_buffer_info = AList[0].request();

    std::vector<ssize_t> b_shape = b_buffer_info.shape;
    std::vector<ssize_t> A_shape = A_buffer_info.shape;

    // Allocate space for ATA, making sure to use 128 bit aligned memory, since that
    // helps Eigen use faster SSE commands.
    std::shared_ptr<T> ATA_data = allocateAlignedMemory<T>(A_shape[1]*N * A_shape[1]*N);
    memset(ATA_data.get(), 0, A_shape[1]*n*A_shape[1]*n*sizeof(T));
    Eigen::Map< Eigen::Matrix<T, X, X, Eigen::ColMajor>> ATA(
        ATA_data.get(), A_shape[1]*n, A_shape[1]*n);

    // Allocate space for ATb
    std::shared_ptr<T> ATb_data = allocateAlignedMemory<T>(A_shape[1]*n);
    memset(ATb_data.get(), 0, A_shape[1]*n*sizeof(T));
    Eigen::Map<Eigen::Vector<T, X>> ATb(ATb_data.get(), A_shape[1]*n);

    // Allocate intermediate arrays that will be reused in the following loops
    Eigen::Matrix<T, X, X> ATA_small(A_shape[1],A_shape[1]);
    Eigen::Matrix<T, N, N> K_cross(n, n);
    Eigen::Vector<T, X> ATb_small(A_shape[1]);

    // loop over each Star
    for (size_t c=0; c < AList.size(); ++c) {
        // build b
        py::buffer_info b_buffer_info = bList[c].request();
        T __restrict *b_data = static_cast<T *>(b_buffer_info.ptr);

        // build A
        py::buffer_info A_buffer_info = AList[c].request();
        T __restrict *A_data = static_cast<T *>(A_buffer_info.ptr);

        //build K
        py::buffer_info K_buffer_info = KList[c].request();
        T __restrict *K_data = static_cast<T *>(K_buffer_info.ptr);

        // set some variables for sizes
        ssize_t A_shape_one = A_buffer_info.shape[1];
        ssize_t A_shape_zero = A_buffer_info.shape[0];

        // calculate the outer product of the K parameters so that it can be re-used in the
        // following loops
        Eigen::Map<const Eigen::Vector<T, N>> K_vec(K_data, n);
        K_cross = K_vec *K_vec.transpose();

        Eigen::Map<const Eigen::Matrix<T, X, X>> A_map(A_data, A_shape_zero, A_shape_one);
        // reset the ATA array to all zeros
        ATA_small.setZero();
        // since the array will be self-adjoint, use Eigen such that only half the number
        // of comutations need done.
        ATA_small.template selfadjointView<Eigen::Upper>().rankUpdate(A_map.transpose());

        Eigen::Map<const Eigen::Matrix<T, X, 1>> b_map(b_data, b_buffer_info.shape[0]);
        ATb_small = A_map.transpose() * b_map;
    
        // Calculate ATb contribution for this list entry
        for (ssize_t k = 0; k < A_shape_one; k ++) {
            ssize_t base_pos = k*n;
            for (ssize_t l = 0; l < n; l++) {
                ATb[base_pos+l] += K_data[l] * ATb_small(k);
            }
        }

        // Calculate ATA contribution for this list entry
        for (ssize_t j = 0; j < A_shape_one; j++) {
            ssize_t base_pos_j = j*n;
            for (ssize_t i = 0; i < j+1; i++) {
                ssize_t base_pos = i*n;
                ATA.template block<N, N>(base_pos, base_pos_j, n, n) += ATA_small(i,j)*K_cross;
            }
        }
    }

    // solve Ax=b. If the matrix can be decomposed using llt do that (faster) if it can't be
    // then use the slower ldlt method
    auto ATA_sym = ATA.template selfadjointView<Eigen::Upper>();
    Eigen::Vector<T, X> result;

    auto llt_decomp = ATA_sym.llt();
    if (llt_decomp.info() == Eigen::ComputationInfo::Success) {
        result = llt_decomp.solve(ATb);
    } else {
        auto ldlt_decomp = ATA_sym.ldlt();
        if (ldlt_decomp.info() == Eigen::ComputationInfo::Success) {
            result = ldlt_decomp.solve(ATb);
        } else {
            // Really slow, but back-up solution if everything up does not work.
            // There is something similar implemented in the python/scipy solutions.
            auto pqrp_decomp = ATA.fullPivHouseholderQr();
            result = pqrp_decomp.solve(ATb);
        }
    }
    return result;
}

template<class T>
Eigen::Vector<T, X> _solve_direct_cpp(
        std::vector<py::array_t<T, py::array::f_style>> bList,
        std::vector<py::array_t<T, py::array::f_style>> AList,
        std::vector<py::array_t<T, py::array::f_style>> KList)
{

    // check some properties of the inputs, more will be done later in the
    // implementation function
    if (bList.size() < 1) {
        throw std::runtime_error("Input data must have at least 1 element");
    }
    if (bList.size() != AList.size() || KList.size() != bList.size()) {
        throw std::runtime_error("All inputs must be the same length");
    }

    py::buffer_info K_buffer_info = KList[0].request();
    const int n = K_buffer_info.shape[0];

    // Switch on the length of the K parameter, this templated value
    // allows the compiler to make various optimizations
    switch (n) {
        case 1:
            return _solve_direct_cpp_impl<T, 1>(bList, AList, KList, n);
        case 2:
            return _solve_direct_cpp_impl<T, 2>(bList, AList, KList, n);
        case 3:
            return _solve_direct_cpp_impl<T, 3>(bList, AList, KList, n);
        case 4:
            return _solve_direct_cpp_impl<T, 4>(bList, AList, KList, n);
        case 5:
            return _solve_direct_cpp_impl<T, 5>(bList, AList, KList, n);
        case 6:
            return _solve_direct_cpp_impl<T, 6>(bList, AList, KList, n);
        case 7:
            return _solve_direct_cpp_impl<T, 7>(bList, AList, KList, n);
        case 8:
            return _solve_direct_cpp_impl<T, 8>(bList, AList, KList, n);
        case 9:
            return _solve_direct_cpp_impl<T, 9>(bList, AList, KList, n);
        case 10:
            return _solve_direct_cpp_impl<T, 10>(bList, AList, KList, n);
        case 11:
            return _solve_direct_cpp_impl<T, 11>(bList, AList, KList, n);
        case 12:
            return _solve_direct_cpp_impl<T, 12>(bList, AList, KList, n);
        case 13:
            return _solve_direct_cpp_impl<T, 13>(bList, AList, KList, n);
        case 14:
            return _solve_direct_cpp_impl<T, 14>(bList, AList, KList, n);
        case 15:
            return _solve_direct_cpp_impl<T, 15>(bList, AList, KList, n);
        case 16:
            return _solve_direct_cpp_impl<T, 16>(bList, AList, KList, n);
        case 17:
            return _solve_direct_cpp_impl<T, 17>(bList, AList, KList, n);
        case 18:
            return _solve_direct_cpp_impl<T, 18>(bList, AList, KList, n);
        case 19:
            return _solve_direct_cpp_impl<T, 19>(bList, AList, KList, n);
        case 20:
            return _solve_direct_cpp_impl<T, 20>(bList, AList, KList, n);
        case 21:
            return _solve_direct_cpp_impl<T, 21>(bList, AList, KList, n);
        case 22:
            return _solve_direct_cpp_impl<T, 22>(bList, AList, KList, n);
        case 23:
            return _solve_direct_cpp_impl<T, 23>(bList, AList, KList, n);
        case 24:
            return _solve_direct_cpp_impl<T, 24>(bList, AList, KList, n);
        case 25:
            return _solve_direct_cpp_impl<T, 25>(bList, AList, KList, n);
        case 26:
            return _solve_direct_cpp_impl<T, 26>(bList, AList, KList, n);
        case 27:
            return _solve_direct_cpp_impl<T, 27>(bList, AList, KList, n);
        case 28:
            return _solve_direct_cpp_impl<T, 28>(bList, AList, KList, n);
        case 29:
            return _solve_direct_cpp_impl<T, 29>(bList, AList, KList, n);
        case 30:
            return _solve_direct_cpp_impl<T, 30>(bList, AList, KList, n);
        default:
            return _solve_direct_cpp_impl<T, X>(bList, AList, KList, n);
    }
}

void pyExportCppSolve(py::module& m)
{
    m.def("_solve_direct_cpp", &_solve_direct_cpp<float>, py::return_value_policy::move,
          py::arg("bList"),py::arg("AList"),py::arg("KList") );
    m.def("_solve_direct_cpp", &_solve_direct_cpp<double>, py::return_value_policy::move,
          py::arg("bList"),py::arg("AList"),py::arg("KList") );
}
