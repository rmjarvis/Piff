#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Cholesky>
#include <iostream>

namespace py = pybind11;


template<class T, ssize_t N>
auto _solve_direct_cpp_impl(
        std::vector<py::array_t<T>> bList,
        std::vector<py::array_t<T>> AList,
        std::vector<py::array_t<T>> KList) {
    // get size info
    py::buffer_info b_buffer_info = bList[0].request();
    py::buffer_info A_buffer_info = AList[0].request();

    std::vector<ssize_t> b_shape = b_buffer_info.shape;
    std::vector<ssize_t> A_shape = A_buffer_info.shape;

    // Allocate and initialize container array for the A transpose A matrix.
    // For some reason it is faster to allocate a numpy array and then map
    // it to Eigen than allocating an Eigen array directly
    py::array_t<T, py::array::f_style> ATA(std::vector({A_shape[1]*N, A_shape[1]*N}));
    py::buffer_info ATA_buffer = ATA.request();
    T __restrict * ATA_data = static_cast<T *>(ATA_buffer.ptr);
    // set it all to zero
    memset(ATA_data, 0, A_shape[1]*N*A_shape[1]*N*sizeof(T));
    Eigen::Map< Eigen::Matrix<T,
                              Eigen::Dynamic,
                              Eigen::Dynamic,
                              Eigen::ColMajor> > ATA_eig(ATA_data, A_shape[1]*N, A_shape[1]*N);

    // Allocate an array for b, similar to ATA
    auto ATb = py::array_t<T>(A_shape[1]*N);
    py::buffer_info ATb_buffer = ATb.request();
    T __restrict * ATb_data = static_cast<T *>(ATb_buffer.ptr);
    memset(ATb_data, 0, A_shape[1]*N*sizeof(T));

    /*
    Eigen::MatrixXd ATA(M*N, M*N);
    std::memset(ATA.data(), 0, M*N*M*N*sizeof(T));

    Eigen::VectorXd ATb(M*N);
    std::memset(ATb.data(), 0, M*N*sizeof(T));
    */


    // Allocate intermediate arrays that will be reused in the following loops
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ATA_small(A_shape[1],A_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K_cross(N, N);
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

        // verify that sizes for the input arrays
        /*
        if (b_buffer_info.shape[0] != A_shape_zero) {
            throw pybind11::value_error("The shape of A must match the length of b along the 1st axis");
        }

        if (K_buffer_info.shape[0] != N) {
            throw std::runtime_error("The shape of K must remain the same for every input");
        }

        if (b_shape[0] != b_buffer_info.shape[0]) {
            throw std::runtime_error("The shape of b must remain the same for every input");
        }

        if (A_shape[0] != A_shape_zero || A_shape[1] != A_shape_one) {
            throw std::runtime_error("The shape of A must remain the same for every input");
        }
        */

        // calculate the outer product of the K parameters so that it can be re-used in the
        // following loops
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> K_vec(K_data, N);
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
        for (ssize_t o = 0; o < A_shape_one; o ++) {
            ssize_t base_pos = o*N;
            for (ssize_t l = 0; l < N; l++) {
                ATb_data[base_pos+l] += K_data[l] * ATb_small(o);
            }
        }

        // Calculate ATA contribution for this list entry
        for (ssize_t j = 0; j < A_shape_one; j++) {
            ssize_t base_pos_j = j*N;
            for (ssize_t i = 0; i < j+1; i++) {
                ssize_t base_pos = i*N;
                ATA_eig.template block<N, N>(base_pos, base_pos_j) += ATA_small(i,j)*K_cross;
            }
        }
    }

    // solve Ax=b. If the matrix can be decomposed using llt do that (faster) if it cant be
    // then use the slower ldlt method
    Eigen::Map<const Eigen::VectorXd> ATb_map(ATb_data, A_shape[1]*N);
    auto view = ATA_eig. template selfadjointView<Eigen::Upper>();
    auto lltDecomp = view.llt();
    Eigen::Matrix<T, Eigen::Dynamic, 1> result;
    if (lltDecomp.info() == Eigen::ComputationInfo::Success) {
        result = lltDecomp.solve(ATb_map);
    } else {
        auto decomp = view.ldlt();
        result = decomp.solve(ATb_map);
    }
    return result;
    // return std::tuple (result, ATA, ATb);
}

template<class T>
auto _solve_direct_cpp(
        std::vector<py::array_t<T>> bList,
        std::vector<py::array_t<T>> AList,
        std::vector<py::array_t<T>> KList) {

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
            return _solve_direct_cpp_impl<T, 1>(bList, AList, KList);
        case 2:
            return _solve_direct_cpp_impl<T, 2>(bList, AList, KList);
        case 3:
            return _solve_direct_cpp_impl<T, 3>(bList, AList, KList);
        case 4:
            return _solve_direct_cpp_impl<T, 4>(bList, AList, KList);
        case 5:
            return _solve_direct_cpp_impl<T, 5>(bList, AList, KList);
        case 6:
            return _solve_direct_cpp_impl<T, 6>(bList, AList, KList);
        case 7:
            return _solve_direct_cpp_impl<T, 7>(bList, AList, KList);
        case 8:
            return _solve_direct_cpp_impl<T, 8>(bList, AList, KList);
        case 9:
            return _solve_direct_cpp_impl<T, 9>(bList, AList, KList);
        case 10:
            return _solve_direct_cpp_impl<T, 10>(bList, AList, KList);
        default:
            throw pybind11::value_error("Can only handle K with a side of length 1 to 10");
    }
}



PYBIND11_MODULE(basic_solver, m) {
    //m.def("_solve_direct_cpp", &_solve_direct_cpp<float>, py::return_value_policy::move,
    //    py::arg("bList"),py::arg("AList"),py::arg("KList"), py::arg("K_size") );
    m.def("_solve_direct_cpp", &_solve_direct_cpp<double>, py::return_value_policy::move,
        py::arg("bList"),py::arg("AList"),py::arg("KList") );
}