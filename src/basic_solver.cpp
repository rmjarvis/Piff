#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

// A custom deleter that finds the original address of the memory allocation directly
// before the stored pointer and frees that using delete []
template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { delete [] ((char**)p)[-1]; }
};

template <typename T>
std::shared_ptr<T> allocateAlignedMemory(int n)
{
    // This bit is based on the answers here:
    // http://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library/227900
    // The point of this is to get the _data pointer aligned to a 16 byte (128 bit) boundary.
    // Arrays that are so aligned can use SSE operations and so can be much faster than
    // non-aligned memroy.  FFTW in particular is faster if it gets aligned data.
    char* mem = new char[n * sizeof(T) + sizeof(char*) + 15];
    T* data = reinterpret_cast<T*>( (uintptr_t)(mem + sizeof(char*) + 15) & ~(size_t) 0x0F );
    ((char**)data)[-1] = mem;
    std::shared_ptr<T> owner(data, AlignedDeleter<T>());
    return owner;
}


template<class T, ssize_t N>
auto _solve_direct_cpp_impl(
        std::vector<py::array_t<T, py::array::f_style>> bList,
        std::vector<py::array_t<T, py::array::f_style>> AList,
        std::vector<py::array_t<T, py::array::f_style>> KList,
        ssize_t n) {

    // get size info
    py::buffer_info b_buffer_info = bList[0].request();
    py::buffer_info A_buffer_info = AList[0].request();

    std::vector<ssize_t> b_shape = b_buffer_info.shape;
    std::vector<ssize_t> A_shape = A_buffer_info.shape;

    // Allocate and initialize container array for the A transpose A matrix.
    // For some reason it is faster to allocate a numpy array and then map
    // it to Eigen than allocating an Eigen array directly
    // py::array_t<T, py::array::f_style> ATA(std::vector<ssize_t>({A_shape[1]*n, A_shape[1]*n}));
    // py::buffer_info ATA_buffer = ATA.request();
    // T __restrict * ATA_data = static_cast<T *>(ATA_buffer.ptr);
    auto ATA_data = allocateAlignedMemory<T>(A_shape[1]*N * A_shape[1]*N);
    // set it all to zero
    memset(ATA_data, 0, A_shape[1]*n*A_shape[1]*n*sizeof(T));
    Eigen::Map< Eigen::Matrix<T,
                              Eigen::Dynamic,
                              Eigen::Dynamic,
                              Eigen::ColMajor> > ATA_eig(ATA_data, A_shape[1]*n, A_shape[1]*n);

    // Allocate an array for b, similar to ATA
    auto ATb = py::array_t<T>(A_shape[1]*n);
    py::buffer_info ATb_buffer = ATb.request();
    T __restrict * ATb_data = static_cast<T *>(ATb_buffer.ptr);
    memset(ATb_data, 0, A_shape[1]*n*sizeof(T));

    // Allocate intermediate arrays that will be reused in the following loops
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ATA_small(A_shape[1],A_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K_cross(n, n);
    Eigen::Vector<T, Eigen::Dynamic> ATb_small(A_shape[1]);

    // loop over each Star
    for (ssize_t c=0; c < AList.size(); ++c) {
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
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> K_vec(K_data, n);
        K_cross = K_vec *K_vec.transpose();

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> A_map(A_data, A_shape_zero, A_shape_one);
        // reset the ATA array to all zeros
        ATA_small.setZero();
        // since the array will be self-adjoint, use Eigen such that only half the number
        // of comutations need done.
        ATA_small.template selfadjointView<Eigen::Upper>().rankUpdate(A_map.transpose());

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_map(b_data, b_buffer_info.shape[0]);
        ATb_small = A_map.transpose() * b_map;
    
        // Calculate ATb contribution for this list entry
        for (ssize_t k = 0; k < A_shape_one; k ++) {
            ssize_t base_pos = k*n;
            for (ssize_t l = 0; l < n; l++) {
                ATb_data[base_pos+l] += K_data[l] * ATb_small(k);
            }
        }

        // Calculate ATA contribution for this list entry
        if (n<31) {
            for (ssize_t j = 0; j < A_shape_one; j++) {
                ssize_t base_pos_j = j*N;
                for (ssize_t i = 0; i < j+1; i++) {
                    ssize_t base_pos = i*N;
                    ATA_eig.template block<N, N>(base_pos, base_pos_j) += ATA_small(i,j)*K_cross;
                }
            }
        }
        else {
            for (ssize_t j = 0; j < A_shape_one; j++) {
                ssize_t base_pos_j = j*n;
                for (ssize_t i = 0; i < j+1; i++) {
                    ssize_t base_pos = i*n;
                    ATA_eig.block(base_pos, base_pos_j, n, n) += ATA_small(i,j)*K_cross;
                }
            }
        }
    }

    // solve Ax=b. If the matrix can be decomposed using llt do that (faster) if it cant be
    // then use the slower ldlt method
    Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> ATb_map(ATb_data, A_shape[1]*n);
    auto view = ATA_eig. template selfadjointView<Eigen::Upper>();
    auto lltDecomp = view.llt();
    Eigen::Vector<T, Eigen::Dynamic> result;

    if (lltDecomp.info() == Eigen::ComputationInfo::Success) {
        result = lltDecomp.solve(ATb_map);
    } else {
        auto decomp = view.ldlt();
        if (decomp.info() == Eigen::ComputationInfo::Success) {
            result = decomp.solve(ATb_map);
        } else {
            // Really slow, but back-up solution if everything up does not work.
            // There is something similar implemented in the python/scipy solutions.
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temporary;
            temporary = ATA_eig;
            auto pinv = temporary.completeOrthogonalDecomposition().pseudoInverse();
            result = pinv * ATb_map;
        }
    }
    return result;
}

template<class T>
auto _solve_direct_cpp(
        std::vector<py::array_t<T, py::array::f_style>> bList,
        std::vector<py::array_t<T, py::array::f_style>> AList,
        std::vector<py::array_t<T, py::array::f_style>> KList) {

    // check some properties of the inputs, more will be done later in the
    // implementation function
    if (bList.size() < 1) {
        throw std::runtime_error("Input data must have at least 1 element");
    }
    if (bList.size() != AList.size() || KList.size() != bList.size()) {
        throw std::runtime_error("All inputs must be the same length");
    }

    py::buffer_info K_buffer_info = KList[0].request();

    // Switch on the length of the K parameter, this templated value
    // allows the compiler to make various optimizations
    switch (K_buffer_info.shape[0]) {
        case 1:
            return _solve_direct_cpp_impl<T, 1>(bList, AList, KList, K_buffer_info.shape[0]);
        case 2:
            return _solve_direct_cpp_impl<T, 2>(bList, AList, KList, K_buffer_info.shape[0]);
        case 3:
            return _solve_direct_cpp_impl<T, 3>(bList, AList, KList, K_buffer_info.shape[0]);
        case 4:
            return _solve_direct_cpp_impl<T, 4>(bList, AList, KList, K_buffer_info.shape[0]);
        case 5:
            return _solve_direct_cpp_impl<T, 5>(bList, AList, KList, K_buffer_info.shape[0]);
        case 6:
            return _solve_direct_cpp_impl<T, 6>(bList, AList, KList, K_buffer_info.shape[0]);
        case 7:
            return _solve_direct_cpp_impl<T, 7>(bList, AList, KList, K_buffer_info.shape[0]);
        case 8:
            return _solve_direct_cpp_impl<T, 8>(bList, AList, KList, K_buffer_info.shape[0]);
        case 9:
            return _solve_direct_cpp_impl<T, 9>(bList, AList, KList, K_buffer_info.shape[0]);
        case 10:
            return _solve_direct_cpp_impl<T, 10>(bList, AList, KList, K_buffer_info.shape[0]);
        case 11:
            return _solve_direct_cpp_impl<T, 11>(bList, AList, KList, K_buffer_info.shape[0]);
        case 12:
            return _solve_direct_cpp_impl<T, 12>(bList, AList, KList, K_buffer_info.shape[0]);
        case 13:
            return _solve_direct_cpp_impl<T, 13>(bList, AList, KList, K_buffer_info.shape[0]);
        case 14:
            return _solve_direct_cpp_impl<T, 14>(bList, AList, KList, K_buffer_info.shape[0]);
        case 15:
            return _solve_direct_cpp_impl<T, 15>(bList, AList, KList, K_buffer_info.shape[0]);
        case 16:
            return _solve_direct_cpp_impl<T, 16>(bList, AList, KList, K_buffer_info.shape[0]);
        case 17:
            return _solve_direct_cpp_impl<T, 17>(bList, AList, KList, K_buffer_info.shape[0]);
        case 18:
            return _solve_direct_cpp_impl<T, 18>(bList, AList, KList, K_buffer_info.shape[0]);
        case 19:
            return _solve_direct_cpp_impl<T, 19>(bList, AList, KList, K_buffer_info.shape[0]);
        case 20:
            return _solve_direct_cpp_impl<T, 20>(bList, AList, KList, K_buffer_info.shape[0]);
        case 21:
            return _solve_direct_cpp_impl<T, 21>(bList, AList, KList, K_buffer_info.shape[0]);
        case 22:
            return _solve_direct_cpp_impl<T, 22>(bList, AList, KList, K_buffer_info.shape[0]);
        case 23:
            return _solve_direct_cpp_impl<T, 23>(bList, AList, KList, K_buffer_info.shape[0]);
        case 24:
            return _solve_direct_cpp_impl<T, 24>(bList, AList, KList, K_buffer_info.shape[0]);
        case 25:
            return _solve_direct_cpp_impl<T, 25>(bList, AList, KList, K_buffer_info.shape[0]);
        case 26:
            return _solve_direct_cpp_impl<T, 26>(bList, AList, KList, K_buffer_info.shape[0]);
        case 27:
            return _solve_direct_cpp_impl<T, 27>(bList, AList, KList, K_buffer_info.shape[0]);
        case 28:
            return _solve_direct_cpp_impl<T, 28>(bList, AList, KList, K_buffer_info.shape[0]);
        case 29:
            return _solve_direct_cpp_impl<T, 29>(bList, AList, KList, K_buffer_info.shape[0]);
        case 30:
            return _solve_direct_cpp_impl<T, 30>(bList, AList, KList, K_buffer_info.shape[0]);
        default:
            return _solve_direct_cpp_impl<T, 31>(bList, AList, KList, K_buffer_info.shape[0]);
    }
}



PYBIND11_MODULE(basic_solver, m) {
    m.def("_solve_direct_cpp", &_solve_direct_cpp<float>, py::return_value_policy::move,
        py::arg("bList"),py::arg("AList"),py::arg("KList") );
    m.def("_solve_direct_cpp", &_solve_direct_cpp<double>, py::return_value_policy::move,
        py::arg("bList"),py::arg("AList"),py::arg("KList") );
}
