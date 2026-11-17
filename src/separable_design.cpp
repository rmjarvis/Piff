// Separable (Kronecker) form of the PixelGrid design matrix.
//
// For a PixelGrid model, row j of the unweighted design matrix A0 (one row per image pixel,
// one column per grid pixel) is the outer product of two 1-d interpolation weight vectors,
//
//     A0[j, (kv, ku)] = vwt[j, kv] * uwt[j, ku],
//
// on a (2*xr) x (2*xr) window of the size x size grid, clipped at the grid edges.
// This class stores those weights and applies A0, A0^T and A0^T W A0 directly from them,
// so that the pcg solver never needs the dense A0 (npix x size^2) nor
// alpha0 = A0^T W A0 (size^2 x size^2).  Per pixel the work is a (2*xr)^2 bilinear form and
// a rank-1 scatter on a working set of a few hundred KB, i.e. compute bound, instead of
// streaming the 3 MB alpha0 of every star through memory at each CG iteration.
//
// Implementation notes: the parameter grid is copied into a zero-padded buffer so that every
// pixel uses the full window without edge tests (weights outside the grid are set to zero at
// construction), and the kernels are templated on the window size NK = 2*xr so the inner loops
// have compile-time trip counts (measured ~2x faster than runtime-sized loops).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class SeparableDesign {
public:
    typedef py::array_t<double, py::array::c_style | py::array::forcecast> darray;
    typedef py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> iarray;

    SeparableDesign(darray uwt, darray vwt, iarray ui, iarray vi, darray w,
                    int size, int origin) :
        _size(size), _origin(origin)
    {
        if (uwt.ndim() != 2 || vwt.ndim() != 2)
            throw std::invalid_argument("uwt, vwt must be 2-d arrays");
        _npix = (int) uwt.shape(0);
        _nk = (int) uwt.shape(1);
        if (vwt.shape(0) != _npix || vwt.shape(1) != _nk)
            throw std::invalid_argument("uwt and vwt must have the same shape");
        if (_nk <= 0 || _nk % 2 != 0)
            throw std::invalid_argument("uwt, vwt must have an even number of columns (2*xr)");
        if (ui.ndim() != 1 || vi.ndim() != 1 || w.ndim() != 1 ||
            ui.shape(0) != _npix || vi.shape(0) != _npix || w.shape(0) != _npix)
            throw std::invalid_argument("ui, vi, w must be 1-d arrays of length npix");
        if (size <= 0)
            throw std::invalid_argument("size must be positive");
        _nparams = size * size;
        _psize = size + 2 * _nk;
        const int xr = _nk / 2;

        _uwt.assign(uwt.data(), uwt.data() + (size_t) _npix * _nk);
        _vwt.assign(vwt.data(), vwt.data() + (size_t) _npix * _nk);
        _w.assign(w.data(), w.data() + _npix);
        _off.resize(_npix);
        _i1.resize(_npix); _p1.resize(_npix); _nv.resize(_npix);
        _j1.resize(_npix); _q1.resize(_npix); _nu.resize(_npix);

        // Same edge clipping as PixelGrid._unweighted_design.
        const std::int64_t* uip = ui.data();
        const std::int64_t* vip = vi.data();
        for (int j = 0; j < _npix; ++j) {
            const long i1raw = vip[j] - xr + origin + 1;
            const long j1raw = uip[j] - xr + origin + 1;
            long i1 = i1raw, i2 = i1raw + 2 * xr;
            long j1 = j1raw, j2 = j1raw + 2 * xr;
            int p1 = 0, p2 = 2 * xr, q1 = 0, q2 = 2 * xr;
            if (i1 < 0) { p1 = (int) -i1; i1 = 0; }
            if (i2 > size) { p2 -= (int) (i2 - size); i2 = size; }
            if (j1 < 0) { q1 = (int) -j1; j1 = 0; }
            if (j2 > size) { q2 -= (int) (j2 - size); j2 = size; }
            double* u = &_uwt[(size_t) j * _nk];
            double* v = &_vwt[(size_t) j * _nk];
            if (i2 <= i1 || j2 <= j1) {
                // Pixel outside the support of the grid: an all-zero row.
                _off[j] = -1;
                _i1[j] = _p1[j] = _nv[j] = _j1[j] = _q1[j] = _nu[j] = 0;
                std::fill(u, u + _nk, 0.);
                std::fill(v, v + _nk, 0.);
            } else {
                _i1[j] = (int) i1; _p1[j] = p1; _nv[j] = (int) (i2 - i1);
                _j1[j] = (int) j1; _q1[j] = q1; _nu[j] = (int) (j2 - j1);
                // Window origin in the padded buffer.  A window that overlaps the grid
                // always lies within the padded buffer.
                _off[j] = (int) ((i1raw + _nk) * _psize + (j1raw + _nk));
                // Zero the weights that fall outside the grid, so the padded kernels can use
                // the full window.
                for (int k = 0; k < p1; ++k) v[k] = 0.;
                for (int k = p2; k < _nk; ++k) v[k] = 0.;
                for (int k = 0; k < q1; ++k) u[k] = 0.;
                for (int k = q2; k < _nk; ++k) u[k] = 0.;
            }
        }
    }

    int npix() const { return _npix; }
    int nparams() const { return _nparams; }
    int nk() const { return _nk; }
    size_t nbytes() const
    {
        return (_uwt.size() + _vwt.size() + _w.size()) * sizeof(double) +
            7 * _off.size() * sizeof(int);
    }
    size_t scratch_size() const { return (size_t) _psize * _psize; }
    py::array_t<double> weights() const
    { return py::array_t<double>(_npix, _w.data()); }

    // trace(A0^T W A0) = sum_j w_j (sum_kv vwt^2) (sum_ku uwt^2)
    double trace() const
    {
        double t = 0.;
        for (int j = 0; j < _npix; ++j) {
            const double* u = &_uwt[(size_t) j * _nk];
            const double* v = &_vwt[(size_t) j * _nk];
            double su = 0., sv = 0.;
            for (int k = 0; k < _nk; ++k) { su += u[k] * u[k]; sv += v[k] * v[k]; }
            t += _w[j] * su * sv;
        }
        return t;
    }

    // The dense A0 (npix x nparams, Fortran order), bitwise equal to the Python version.
    py::array_t<double, py::array::f_style> design() const
    {
        py::array_t<double, py::array::f_style> A({_npix, _nparams});
        double* a = A.mutable_data();
        std::fill(a, a + (size_t) _npix * _nparams, 0.);
        for (int j = 0; j < _npix; ++j) {
            const double* u = &_uwt[(size_t) j * _nk + _q1[j]];
            const double* v = &_vwt[(size_t) j * _nk + _p1[j]];
            for (int kv = 0; kv < _nv[j]; ++kv) {
                const size_t col0 = (size_t) (_i1[j] + kv) * _size + _j1[j];
                for (int ku = 0; ku < _nu[j]; ++ku)
                    a[j + (col0 + ku) * (size_t) _npix] = u[ku] * v[kv];
            }
        }
        return A;
    }

    // The four operations, dispatched on the window size NK (compile-time trip counts).
    // Scratch buffers have scratch_size() entries each.
#define PIFF_SEP_DISPATCH(CALL)                                   \
    switch (_nk) {                                                \
        case 2: return CALL(2);   case 4: return CALL(4);         \
        case 6: return CALL(6);   case 8: return CALL(8);         \
        case 10: return CALL(10); case 12: return CALL(12);       \
        case 14: return CALL(14); case 16: return CALL(16);       \
        case 18: return CALL(18); case 20: return CALL(20);       \
        case 22: return CALL(22); case 24: return CALL(24);       \
        case 26: return CALL(26); case 28: return CALL(28);       \
        case 30: return CALL(30); case 32: return CALL(32);       \
        default: return CALL(0);  /* runtime window size (slower) */ \
    }

    // m = A0 p
    void forward(const double* p, double* m, double* P) const
    {
#define CALL(NK) forward_tpl<NK>(p, m, P)
        PIFF_SEP_DISPATCH(CALL)
#undef CALL
    }
    // out += A0^T r
    void adjoint_accumulate(const double* r, double* out, double* Z) const
    {
#define CALL(NK) adjoint_tpl<NK>(r, out, Z)
        PIFF_SEP_DISPATCH(CALL)
#undef CALL
    }
    // sum_j w_j (A0 p)_j^2  (= p^T alpha0 p)
    double quad(const double* p, double* P) const
    {
#define CALL(NK) quad_tpl<NK>(p, P)
        PIFF_SEP_DISPATCH(CALL)
#undef CALL
    }
    // z += scale * A0^T W A0 y, fused per pixel (one pass over the weights).
    void apply_alpha_accumulate(const double* y, double* z, double scale,
                                double* P, double* Z) const
    {
#define CALL(NK) apply_tpl<NK>(y, z, scale, P, Z)
        PIFF_SEP_DISPATCH(CALL)
#undef CALL
    }
#undef PIFF_SEP_DISPATCH

    // Python wrappers
    py::array_t<double> forward_py(darray p) const
    {
        check_params(p);
        py::array_t<double> m(_npix);
        std::vector<double> P(scratch_size());
        forward(p.data(), m.mutable_data(), P.data());
        return m;
    }
    py::array_t<double> adjoint_py(darray r) const
    {
        if (r.ndim() != 1 || r.shape(0) != _npix)
            throw std::invalid_argument("r must have length npix");
        py::array_t<double> out(_nparams);
        double* o = out.mutable_data();
        std::fill(o, o + _nparams, 0.);
        std::vector<double> Z(scratch_size());
        adjoint_accumulate(r.data(), o, Z.data());
        return out;
    }
    double quad_py(darray p) const
    {
        check_params(p);
        std::vector<double> P(scratch_size());
        return quad(p.data(), P.data());
    }
    py::array_t<double> apply_alpha_py(darray y, double scale) const
    {
        check_params(y);
        py::array_t<double> z(_nparams);
        double* zp = z.mutable_data();
        std::fill(zp, zp + _nparams, 0.);
        std::vector<double> P(scratch_size()), Z(scratch_size());
        apply_alpha_accumulate(y.data(), zp, scale, P.data(), Z.data());
        return z;
    }

private:
    int _size, _origin, _npix, _nk, _nparams, _psize;
    std::vector<double> _uwt, _vwt, _w;
    std::vector<int> _off;                          // window origin in the padded buffer
    std::vector<int> _i1, _p1, _nv, _j1, _q1, _nu;  // clipped ranges (for design())

    void check_params(const darray& p) const
    {
        if (p.ndim() != 1 || p.shape(0) != _nparams)
            throw std::invalid_argument("parameter vector must have length size*size");
    }

    // Copy the size x size grid into the zero-padded buffer.
    void pad_in(const double* p, double* P) const
    {
        std::fill(P, P + scratch_size(), 0.);
        for (int r = 0; r < _size; ++r)
            std::copy(p + (size_t) r * _size, p + (size_t) (r + 1) * _size,
                      P + (size_t) (r + _nk) * _psize + _nk);
    }
    // Add the interior of the padded buffer to the size x size grid.
    void pad_out(const double* Z, double* z) const
    {
        for (int r = 0; r < _size; ++r) {
            const double* zr = Z + (size_t) (r + _nk) * _psize + _nk;
            double* o = z + (size_t) r * _size;
            for (int c = 0; c < _size; ++c) o[c] += zr[c];
        }
    }

    // v^T P_window u  for one pixel.
    template <int NK>
    inline double bilinear(const double* u, const double* v, const double* prow) const
    {
        const int n = NK > 0 ? NK : _nk;
        double m = 0.;
        for (int kv = 0; kv < n; ++kv, prow += _psize) {
            if (v[kv] == 0.) continue;   // rows of the window outside the grid
            // Four partial sums so the inner loop vectorizes without reassociation flags.
            double t0 = 0., t1 = 0., t2 = 0., t3 = 0.;
            int ku = 0;
            for (; ku + 3 < n; ku += 4) {
                t0 += u[ku] * prow[ku];
                t1 += u[ku+1] * prow[ku+1];
                t2 += u[ku+2] * prow[ku+2];
                t3 += u[ku+3] * prow[ku+3];
            }
            for (; ku < n; ++ku) t0 += u[ku] * prow[ku];
            m += v[kv] * ((t0 + t1) + (t2 + t3));
        }
        return m;
    }

    // Z_window += f v u^T  for one pixel.
    template <int NK>
    inline void scatter(const double* u, const double* v, double f, double* zrow) const
    {
        const int n = NK > 0 ? NK : _nk;
        for (int kv = 0; kv < n; ++kv, zrow += _psize) {
            if (v[kv] == 0.) continue;   // rows of the window outside the grid
            const double fv = f * v[kv];
            for (int ku = 0; ku < n; ++ku) zrow[ku] += fv * u[ku];
        }
    }

    template <int NK>
    void forward_tpl(const double* p, double* m, double* P) const
    {
        pad_in(p, P);
        for (int j = 0; j < _npix; ++j) {
            m[j] = _off[j] < 0 ? 0. :
                bilinear<NK>(&_uwt[(size_t) j * _nk], &_vwt[(size_t) j * _nk], P + _off[j]);
        }
    }

    template <int NK>
    void adjoint_tpl(const double* r, double* out, double* Z) const
    {
        std::fill(Z, Z + scratch_size(), 0.);
        for (int j = 0; j < _npix; ++j) {
            if (_off[j] < 0) continue;
            scatter<NK>(&_uwt[(size_t) j * _nk], &_vwt[(size_t) j * _nk], r[j], Z + _off[j]);
        }
        pad_out(Z, out);
    }

    template <int NK>
    double quad_tpl(const double* p, double* P) const
    {
        pad_in(p, P);
        double s = 0.;
        for (int j = 0; j < _npix; ++j) {
            if (_off[j] < 0) continue;
            const double m = bilinear<NK>(&_uwt[(size_t) j * _nk], &_vwt[(size_t) j * _nk],
                                          P + _off[j]);
            s += _w[j] * m * m;
        }
        return s;
    }

    template <int NK>
    void apply_tpl(const double* y, double* z, double scale, double* P, double* Z) const
    {
        pad_in(y, P);
        std::fill(Z, Z + scratch_size(), 0.);
        for (int j = 0; j < _npix; ++j) {
            if (_off[j] < 0) continue;
            const double* u = &_uwt[(size_t) j * _nk];
            const double* v = &_vwt[(size_t) j * _nk];
            const double m = bilinear<NK>(u, v, P + _off[j]);
            scatter<NK>(u, v, scale * _w[j] * m, Z + _off[j]);
        }
        pad_out(Z, z);
    }
};

// ZT[i] = scales[i] * A0_i^T W_i A0_i YT[i] for a list of stars: one call per CG iteration.
py::array_t<double> apply_alpha_batch(
        const std::vector<std::shared_ptr<SeparableDesign>>& designs,
        py::array_t<double, py::array::c_style | py::array::forcecast> scales,
        py::array_t<double, py::array::c_style | py::array::forcecast> YT)
{
    const ssize_t n = (ssize_t) designs.size();
    if (YT.ndim() != 2 || YT.shape(0) != n)
        throw std::invalid_argument("YT must have shape (ndesigns, nparams)");
    if (scales.ndim() != 1 || scales.shape(0) != n)
        throw std::invalid_argument("scales must have length ndesigns");
    const ssize_t nparams = YT.shape(1);
    py::array_t<double> ZT({n, nparams});
    double* z = ZT.mutable_data();
    std::fill(z, z + (size_t) n * nparams, 0.);
    const double* y = YT.data();
    const double* s = scales.data();
    size_t scratch = 0;
    for (ssize_t i = 0; i < n; ++i) {
        if (designs[i]->nparams() != nparams)
            throw std::invalid_argument("design nparams does not match YT");
        scratch = std::max(scratch, designs[i]->scratch_size());
    }
    std::vector<double> P(scratch), Z(scratch);
    for (ssize_t i = 0; i < n; ++i)
        designs[i]->apply_alpha_accumulate(y + i * nparams, z + i * nparams, s[i],
                                           P.data(), Z.data());
    return ZT;
}

void pyExportSeparableDesign(py::module& m)
{
    py::class_<SeparableDesign, std::shared_ptr<SeparableDesign>>(m, "SeparableDesign")
        .def(py::init<SeparableDesign::darray, SeparableDesign::darray,
                      SeparableDesign::iarray, SeparableDesign::iarray,
                      SeparableDesign::darray, int, int>(),
             py::arg("uwt"), py::arg("vwt"), py::arg("ui"), py::arg("vi"), py::arg("w"),
             py::arg("size"), py::arg("origin"))
        .def_property_readonly("npix", &SeparableDesign::npix)
        .def_property_readonly("nparams", &SeparableDesign::nparams)
        .def_property_readonly("nk", &SeparableDesign::nk)
        .def_property_readonly("nbytes", &SeparableDesign::nbytes)
        .def_property_readonly("weights", &SeparableDesign::weights)
        .def("trace", &SeparableDesign::trace)
        .def("design", &SeparableDesign::design)
        .def("forward", &SeparableDesign::forward_py, py::arg("p"))
        .def("adjoint", &SeparableDesign::adjoint_py, py::arg("r"))
        .def("quad", &SeparableDesign::quad_py, py::arg("p"))
        .def("apply_alpha", &SeparableDesign::apply_alpha_py, py::arg("y"),
             py::arg("scale") = 1.);
    m.def("apply_alpha_batch", &apply_alpha_batch,
          py::arg("designs"), py::arg("scales"), py::arg("YT"));
}
