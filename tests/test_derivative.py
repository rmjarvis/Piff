# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""Tests of the analytic derivative of the PSF with respect to its position.

The derivative is needed e.g. by scene-modeling photometry codes (Astier et al. 2013), where
the position of a point source is a fit parameter.
"""

import galsim
import numpy as np
import piff
import os

from piff_test_helper import timer
from piff.pixelgrid import (_lanczos_xval, _cubic_xval, _quintic_xval,
                            _kernel_xval_deriv, _kernel_xval_deriv_fd)


def make_params(size, sigma=1.5, rng=None):
    """Make a plausible (but not perfectly symmetric) set of PixelGrid parameters."""
    r = np.arange(size) - size//2
    params = np.exp(-(r[:,None]**2 + r[None,:]**2) / (2.*sigma**2))
    if rng is not None:
        params += 0.05 * params.max() * rng.standard_normal(params.shape)
    params = params.ravel()
    return params / np.sum(params)


def make_grid_star(mod, params, wcs, x=100.3, y=200.7, stamp_size=48, center=(0.,0.), flux=1.):
    """Make a star with the given PixelGrid parameters set as its fit."""
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=stamp_size)
    fit = star.fit.newParams(params, num=mod._num)
    fit.center = center
    fit.flux = flux
    return piff.Star(star.data, fit)


def finite_diff_draw(mod, star, h=1.e-4):
    """Central finite differences of Model.draw with respect to star.fit.center."""
    u0, v0 = star.fit.center
    def draw(center):
        return mod.draw(piff.Star(star.data, star.fit.withNew(center=center))).image.array
    dI_du = (draw((u0+h, v0)) - draw((u0-h, v0))) / (2.*h)
    dI_dv = (draw((u0, v0+h)) - draw((u0, v0-h))) / (2.*h)
    return dI_du, dI_dv


def make_stars(wcs, positions, sigma_func, nside=32, flux=100., colors=None, chipnum=None):
    """Make noiseless Gaussian stars at the given image positions.

    :param wcs:         The wcs of the image.
    :param positions:   A list of (x,y) image positions.
    :param sigma_func:  A function of the star returning the sigma of the Gaussian (in world
                        units).
    :param nside:       The stamp size.
    :param flux:        The flux of the stars.
    :param colors:      Optional list of colors to set as a 'color' property.
    :param chipnum:     Optional chipnum property.
    """
    stars = []
    for i, (x,y) in enumerate(positions):
        props = {}
        if colors is not None:
            props['color'] = colors[i]
        if chipnum is not None:
            props['chipnum'] = chipnum
        weight = galsim.Image(nside, nside, init_value=1.e6, dtype=float)
        star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=nside, properties=props,
                                    weight=weight)
        prof = galsim.Gaussian(sigma=sigma_func(star), flux=flux)
        prof.drawImage(star.image, method='no_pixel', center=star.image_pos)
        stars.append(star)
    return stars


def fit_psf(psf, stars, wcs_dict, max_iter=3):
    psf.max_iter = max_iter
    psf.set_context(wcs_dict, None, None)
    psf.fit(stars)
    return psf


def check_psf_derivative(psf, x, y, chipnum=None, h=1.e-3, tol=1.e-4, **kwargs):
    """Compare psf.draw_derivative with finite differences of psf.draw (center shifted by h).

    Note: the tolerance (relative to the maximum of the derivative) is set by the accuracy of
    the finite differences, not of the analytic derivative.  For some geometries, a pixel
    lands exactly at the edge of the kernel support (e.g. kernel argument = 7 for Lanczos(7)),
    where K'' is discontinuous.  The analytic derivative is fine there (K' is continuous), but
    the central finite difference has an O(h) error at that pixel.  The accuracy of the
    analytic derivative itself is checked more precisely in test_pixelgrid_draw.
    """
    dI_dx, dI_dy = psf.draw_derivative(x, y, chipnum=chipnum, **kwargs)
    if 'image' in kwargs:
        # draw fills in the given image, so use a copy to leave the input image alone.
        im = psf.draw(x, y, chipnum=chipnum, **dict(kwargs, image=kwargs['image'].copy()))
    else:
        im = psf.draw(x, y, chipnum=chipnum, **kwargs)
    assert dI_dx.bounds == im.bounds
    assert dI_dy.bounds == im.bounds
    assert dI_dx.wcs == im.wcs

    # What center does draw actually use?  Replicate the logic to get the finite differences
    # around the right point.
    center = kwargs.pop('center', None)
    offset = kwargs.pop('offset', None)
    image = kwargs.get('image', None)
    if center is None:
        center = (x, y)
    elif center is True:
        center = im.true_center
        center = (center.x, center.y)
    if offset is not None:
        center = (center[0]+offset[0], center[1]+offset[1])
    def draw(c):
        if image is not None:
            kwargs['image'] = image.copy()
        return psf.draw(x, y, chipnum=chipnum, center=c, **kwargs).array
    fd_x = (draw((center[0]+h, center[1])) - draw((center[0]-h, center[1]))) / (2.*h)
    fd_y = (draw((center[0], center[1]+h)) - draw((center[0], center[1]-h))) / (2.*h)
    scale = max(np.max(np.abs(fd_x)), np.max(np.abs(fd_y)))
    print('    max |dI/dx| = %.3g, max diff = %.3g, %.3g'%(
          scale, np.max(np.abs(dI_dx.array-fd_x)), np.max(np.abs(dI_dy.array-fd_y))))
    np.testing.assert_allclose(dI_dx.array, fd_x, rtol=0, atol=tol*scale)
    np.testing.assert_allclose(dI_dy.array, fd_y, rtol=0, atol=tol*scale)
    return dI_dx, dI_dy


@timer
def test_kernel_xval():
    """Check the numpy reimplementations of the GalSim kernels against galsim's xval.
    """
    x = np.linspace(-8.5, 8.5, 20001)
    # Include some special values: exactly 0, in the Taylor branch, and at the edges.
    x = np.concatenate([x, [0., 1.e-6, -1.e-6, 5.e-5, 1.e-4, 2.e-4, 1., -2., 3., -3., 7., -7.]])

    for n in [3, 4, 5, 7]:
        for conserve_dc in [True, False]:
            interp = galsim.Lanczos(n, conserve_dc=conserve_dc)
            val = _lanczos_xval(x, n, conserve_dc)
            ref = interp.xval(x)
            print('Lanczos(%d, %s): max err = %g'%(n, conserve_dc, np.max(np.abs(val-ref))))
            np.testing.assert_allclose(val, ref, rtol=0, atol=1.e-12)
            # Outside the support, the kernel is 0.
            assert np.all(val[np.abs(x) >= n] == 0.)

    val = _cubic_xval(x)
    ref = galsim.Cubic().xval(x)
    print('Cubic: max err = %g'%np.max(np.abs(val-ref)))
    np.testing.assert_allclose(val, ref, rtol=0, atol=1.e-12)

    val = _quintic_xval(x)
    ref = galsim.Quintic().xval(x)
    print('Quintic: max err = %g'%np.max(np.abs(val-ref)))
    np.testing.assert_allclose(val, ref, rtol=0, atol=1.e-12)

    # Scalars and 2d arrays work too.
    np.testing.assert_allclose(_lanczos_xval(0.3, 5), galsim.Lanczos(5).xval(0.3))
    x2 = x[:20000].reshape(100,200)
    np.testing.assert_allclose(_lanczos_xval(x2, 5), galsim.Lanczos(5).xval(x[:20000]).reshape(100,200))


@timer
def test_kernel_derivative():
    """Check the analytic kernel derivatives against finite differences of galsim's xval.
    """
    rng = np.random.default_rng(1234)
    # Avoid the knots (integers), where the FD estimate is less accurate for Cubic/Quintic.
    x = rng.uniform(-8, 8, 20000)
    x = x[np.abs(x - np.round(x)) > 1.e-3]
    x = np.concatenate([x, [0., 1.e-6, -1.e-6, 5.e-5, 1.e-4, 2.e-4]])

    interps = [galsim.Lanczos(3), galsim.Lanczos(5), galsim.Lanczos(7),
               galsim.Lanczos(7, conserve_dc=False), galsim.Cubic(), galsim.Quintic()]
    for interp in interps:
        dval = _kernel_xval_deriv(interp, x)
        fd = _kernel_xval_deriv_fd(interp, x, h=1.e-5)
        print('%r: max |K\'| = %g, max err = %g'%(interp, np.max(np.abs(fd)), np.max(np.abs(dval-fd))))
        # Central FD with h=1e-5 has truncation error ~ h^2 K''' / 6 ~ 1e-9 and roundoff
        # error ~ eps / h ~ 1e-11.
        np.testing.assert_allclose(dval, fd, rtol=0, atol=1.e-7)
        # The derivative is odd.
        np.testing.assert_allclose(_kernel_xval_deriv(interp, -x), -dval, rtol=0, atol=1.e-14)
        # And zero at 0 and outside the support.
        assert _kernel_xval_deriv(interp, 0.) == 0.
        assert np.all(dval[np.abs(x) >= interp.xrange] == 0.)
        # The shape of the input is preserved.
        assert _kernel_xval_deriv(interp, x[:100].reshape(10,10)).shape == (10,10)

    # The FD fallback is used for interpolants without an analytic implementation.
    # Mock this with a duck-typed object that is not a galsim.Interpolant subclass.
    class FakeInterp:
        def __init__(self, interp): self.interp = interp
        def xval(self, x): return self.interp.xval(x)
    fake = FakeInterp(galsim.Lanczos(5))
    dval = _kernel_xval_deriv(fake, x)
    np.testing.assert_allclose(dval, _kernel_xval_deriv(galsim.Lanczos(5), x), rtol=0, atol=1.e-6)

    # Non-differentiable kernels (or infinite support) raise an error.
    for interp in [galsim.Nearest(), galsim.Linear(), galsim.Delta(), galsim.SincInterpolant()]:
        np.testing.assert_raises(ValueError, _kernel_xval_deriv, interp, x)


@timer
def test_pixelgrid_draw():
    """Check that the kernel sum used for the derivative reproduces Model.draw exactly, and that
    the derivative matches finite differences of Model.draw.
    """
    rng = np.random.default_rng(1234)

    # A non-linear wcs, to make sure the local Jacobian at the right position is used.
    uv_wcs = galsim.UVFunction(ufunc='0.26*x + 1.e-5*x*y', vfunc='0.25*y - 2.e-5*x*x',
                               xfunc=None, yfunc=None)

    # (scale, size, stamp_size, wcs): model finer, equal, coarser than the pixels; odd and even
    # grid and stamp sizes; different wcs types.
    cases = [
        (0.26, 17, 48, galsim.PixelScale(0.26)),
        (0.2, 16, 33, galsim.JacobianWCS(0.25, 0.03, -0.04, 0.23)),
        (0.4, 21, 40, galsim.PixelScale(0.26)),
        (0.13, 25, 24, galsim.PixelScale(0.26)),
        (0.26, 15, 31, uv_wcs),
    ]
    interps = [galsim.Lanczos(7), galsim.Lanczos(3), galsim.Cubic(), galsim.Quintic()]

    for scale, size, stamp_size, wcs in cases:
        for interp in interps:
            for fit_flux in [False, True]:
                print(scale, size, stamp_size, wcs, interp, fit_flux)
                mod = piff.PixelGrid(scale, size, interp, fit_flux=fit_flux)
                params = make_params(size, sigma=1.5 if scale < 0.3 else 1., rng=rng)
                if fit_flux:
                    params /= 1.3
                star = make_grid_star(mod, params, wcs, stamp_size=stamp_size,
                                      center=(0.07,-0.11), flux=123.4)

                # The kernel sum reproduces Model.draw
                im = mod.draw(star).image.array
                im2 = mod._kernel_sum(star, params)
                print('    draw: max |I| = %.3g, max diff = %.3g'%(
                      np.max(np.abs(im)), np.max(np.abs(im-im2))))
                np.testing.assert_allclose(im2, im, rtol=0, atol=1.e-10*np.max(np.abs(im)))

                # The derivative matches finite differences.
                dI_du, dI_dv = mod.drawDerivative(star)
                assert dI_du.bounds == star.image.bounds
                assert dI_dv.bounds == star.image.bounds
                assert dI_du.wcs == star.image.wcs
                fd_du, fd_dv = finite_diff_draw(mod, star)
                dscale = max(np.max(np.abs(fd_du)), np.max(np.abs(fd_dv)))
                print('    deriv: max |dI| = %.3g, max diff = %.3g, %.3g'%(
                      dscale, np.max(np.abs(dI_du.array-fd_du)), np.max(np.abs(dI_dv.array-fd_dv))))
                np.testing.assert_allclose(dI_du.array, fd_du, rtol=0, atol=2.e-6*dscale)
                np.testing.assert_allclose(dI_dv.array, fd_dv, rtol=0, atol=2.e-6*dscale)

                # The 'u' and 'v' options give the same thing individually.
                np.testing.assert_array_equal(mod._kernel_sum(star, params, 'u'), dI_du.array)
                np.testing.assert_array_equal(mod._kernel_sum(star, params, 'v'), dI_dv.array)

    np.testing.assert_raises(ValueError, mod._kernel_sum, star, params, 'w')

    # Check the symmetry for a symmetric model centered on a pixel: dI/du is antisymmetric in u,
    # symmetric in v, and sums to zero.
    mod = piff.PixelGrid(0.26, 17)
    params = make_params(17)
    star = make_grid_star(mod, params, galsim.PixelScale(0.26), x=16, y=16, stamp_size=33)
    dI_du, dI_dv = mod.drawDerivative(star)
    np.testing.assert_allclose(dI_du.array, -dI_du.array[:,::-1], rtol=0, atol=1.e-14)
    np.testing.assert_allclose(dI_du.array, dI_du.array[::-1,:], rtol=0, atol=1.e-14)
    np.testing.assert_allclose(dI_dv.array, dI_du.array.T, rtol=0, atol=1.e-14)
    assert abs(np.sum(dI_du.array)) < 1.e-12
    assert np.max(np.abs(dI_du.array)) > 0.01

    # Other models don't implement this.
    gauss = piff.Gaussian()
    gstar = piff.Star(star.data, piff.StarFit(np.array([1.,0.,0.])))
    with np.testing.assert_raises(NotImplementedError):
        gauss.drawDerivative(gstar)


@timer
def test_psf_draw_derivative():
    """Test PSF.draw_derivative and PSF.drawStarDerivative for a SimplePSF.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    wcs = galsim.JacobianWCS(0.25, 0.02, -0.03, 0.27)
    positions = [(50.3, 60.1), (950.7, 70.2), (60.9, 940.4), (930.2, 960.8), (500.5, 500.5)]
    def sigma_func(star):
        return 0.8 + 0.1 * star['u'] / 250. - 0.05 * star['v'] / 250.
    stars = make_stars(wcs, positions, sigma_func)

    mod = piff.PixelGrid(0.25, 17)
    interp = piff.BasisPolynomial(order=1)
    psf = fit_psf(piff.SimplePSF(mod, interp), stars, {0: wcs})

    x, y = 300.2, 400.7
    # Default center
    print('center=None')
    dI_dx, dI_dy = check_psf_derivative(psf, x, y)
    # Various stamp sizes
    print('odd stamp size')
    check_psf_derivative(psf, x, y, stamp_size=31)
    print('center=True')
    check_psf_derivative(psf, x, y, center=True)
    print('center=tuple')
    check_psf_derivative(psf, x, y, center=(x+1.3, y-0.8))
    print('offset')
    check_psf_derivative(psf, x, y, offset=(0.4, -0.3))
    print('center=True, offset')
    check_psf_derivative(psf, x, y, center=True, offset=(0.4, -0.3), stamp_size=25)
    print('flux')
    dI_dx2, dI_dy2 = check_psf_derivative(psf, x, y, flux=17.)
    np.testing.assert_allclose(dI_dx2.array, 17.*dI_dx.array, rtol=1.e-12)
    np.testing.assert_allclose(dI_dy2.array, 17.*dI_dy.array, rtol=1.e-12)

    # Using an image with its own wcs (e.g. sky coordinates).
    print('image with own wcs')
    image = galsim.Image(40, 40, dtype=float)
    image.setCenter(int(x), int(y))
    image.wcs = galsim.PixelScale(0.2)
    dI_dx3, dI_dy3 = check_psf_derivative(psf, x, y, image=image)
    assert dI_dx3.wcs == galsim.PixelScale(0.2)
    assert dI_dx3.bounds == image.bounds
    # The input image is not modified.
    assert np.all(image.array == 0.)

    # Invalid center raises an error.
    with np.testing.assert_raises(ValueError):
        psf.draw_derivative(x, y, center=4)
    # chipnum errors
    with np.testing.assert_raises(ValueError):
        psf.draw_derivative(x, y, chipnum=1)
    # Unexpected kwargs raise an error
    with np.testing.assert_raises(TypeError):
        psf.draw_derivative(x, y, color=0.3)

    # Physics check: adding eps * dI/dx to the image moves the centroid by eps pixels.
    im = psf.draw(x, y, stamp_size=48)
    eps = 0.05
    im_x = galsim.Image(im.array + eps*dI_dx.array, wcs=im.wcs, bounds=im.bounds)
    im_y = galsim.Image(im.array + eps*dI_dy.array, wcs=im.wcs, bounds=im.bounds)
    mom = galsim.hsm.FindAdaptiveMom(im)
    mom_x = galsim.hsm.FindAdaptiveMom(im_x)
    mom_y = galsim.hsm.FindAdaptiveMom(im_y)
    print('centroid shift in x: ', mom_x.moments_centroid - mom.moments_centroid)
    print('centroid shift in y: ', mom_y.moments_centroid - mom.moments_centroid)
    np.testing.assert_allclose(mom_x.moments_centroid.x - mom.moments_centroid.x, eps, rtol=1.e-2)
    np.testing.assert_allclose(mom_x.moments_centroid.y - mom.moments_centroid.y, 0, atol=1.e-4)
    np.testing.assert_allclose(mom_y.moments_centroid.y - mom.moments_centroid.y, eps, rtol=1.e-2)
    np.testing.assert_allclose(mom_y.moments_centroid.x - mom.moments_centroid.x, 0, atol=1.e-4)
    # The total flux is (nearly) unchanged, so the derivative sums to (nearly) zero.
    # Note: the pixel sum of the (no_pixel) PixelGrid model is not exactly invariant under
    # sub-pixel shifts, since the model is not band-limited at the pixel scale, so this is
    # only approximately zero.
    assert abs(np.sum(dI_dx.array)) < 1.e-2 * np.sum(np.abs(dI_dx.array))
    assert abs(np.sum(dI_dy.array)) < 1.e-2 * np.sum(np.abs(dI_dy.array))

    # drawStarDerivative gives the derivative in world coordinates.  Compare to finite
    # differences of drawStar with respect to fit.center.
    target = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48)
    assert target.fit.params is None
    dI_du, dI_dv = psf.drawStarDerivative(target)   # interpolates the target first
    target = psf.interpolateStar(target)
    dI_du2, dI_dv2 = psf.drawStarDerivative(target)
    np.testing.assert_array_equal(dI_du.array, dI_du2.array)
    np.testing.assert_array_equal(dI_dv.array, dI_dv2.array)
    h = 1.e-4
    def draw(center):
        return psf.drawStar(piff.Star(target.data, target.fit.withNew(center=center))).image.array
    fd_du = (draw((h,0)) - draw((-h,0))) / (2*h)
    fd_dv = (draw((0,h)) - draw((0,-h))) / (2*h)
    dscale = np.max(np.abs(fd_du))
    np.testing.assert_allclose(dI_du.array, fd_du, rtol=0, atol=1.e-6*dscale)
    np.testing.assert_allclose(dI_dv.array, fd_dv, rtol=0, atol=1.e-6*dscale)
    # These are related to the image-coordinate derivatives by the Jacobian.
    jac = wcs.jacobian()
    np.testing.assert_allclose(dI_dx.array, dI_du.array*jac.dudx + dI_dv.array*jac.dvdx, rtol=1.e-12)
    np.testing.assert_allclose(dI_dy.array, dI_du.array*jac.dudy + dI_dv.array*jac.dvdy, rtol=1.e-12)

    # A non-uniform wcs.  The derivative uses the local Jacobian at the center position, held
    # fixed.  Finite differences of draw also pick up the (tiny) change of the local affine
    # approximation GalSim uses when the center moves, so use a realistically mild
    # non-linearity for this check.
    uv_wcs = galsim.UVFunction(ufunc='0.26*x + 1.e-6*x*y', vfunc='0.25*y - 2.e-6*x*x',
                               xfunc=None, yfunc=None)
    stars = make_stars(uv_wcs, positions, sigma_func)
    psf2 = fit_psf(piff.SimplePSF(piff.PixelGrid(0.25, 17), piff.BasisPolynomial(order=1)),
                   stars, {0: uv_wcs})
    print('non-uniform wcs')
    check_psf_derivative(psf2, x, y, h=1.e-4)
    print('non-uniform wcs, center=tuple')
    check_psf_derivative(psf2, x, y, center=(x+5., y-3.), h=1.e-4)

    # With a strongly non-linear wcs, check explicitly that the Jacobian at the center (not at
    # the nominal position (x,y)) is the one being used.
    uv_wcs = galsim.UVFunction(ufunc='0.26*x + 1.e-4*x*y', vfunc='0.25*y - 2.e-4*x*x',
                               xfunc=None, yfunc=None)
    stars = make_stars(uv_wcs, positions, sigma_func)
    psf3 = fit_psf(piff.SimplePSF(piff.PixelGrid(0.25, 17), piff.BasisPolynomial(order=1)),
                   stars, {0: uv_wcs})
    c = (x+20., y-15.)
    dI_dx, dI_dy = psf3.draw_derivative(x, y, center=c)
    # Build the equivalent by hand: interpolate at (x,y), draw the derivative at c.
    target = psf3.interpolateStar(piff.Star.makeTarget(x=x, y=y, wcs=uv_wcs, stamp_size=48))
    im = psf3.draw(x, y, center=c)
    data = piff.StarData(im, galsim.PositionD(*c))
    dI_du, dI_dv = psf3.drawStarDerivative(piff.Star(data, target.fit))
    jac = uv_wcs.local(galsim.PositionD(*c)).jacobian()
    np.testing.assert_allclose(dI_dx.array, dI_du.array*jac.dudx + dI_dv.array*jac.dvdx,
                               rtol=1.e-12, atol=1.e-15)
    np.testing.assert_allclose(dI_dy.array, dI_du.array*jac.dudy + dI_dv.array*jac.dvdy,
                               rtol=1.e-12, atol=1.e-15)
    # Using the Jacobian at (x,y) instead would be noticeably different.
    jac2 = uv_wcs.local(galsim.PositionD(x,y)).jacobian()
    assert not np.allclose(dI_dx.array, dI_du.array*jac2.dudx + dI_dv.array*jac2.dvdx,
                           rtol=1.e-4, atol=0)

    # Check write/read round trip preserves the ability to compute derivatives.
    psf_file = os.path.join('output', 'test_derivative_psf.fits')
    psf.write(psf_file)
    psf4 = piff.read(psf_file)
    dI_dx0, dI_dy0 = psf.draw_derivative(x, y)
    dI_dx4, dI_dy4 = psf4.draw_derivative(x, y)
    np.testing.assert_allclose(dI_dx4.array, dI_dx0.array, rtol=1.e-12, atol=1.e-15)
    np.testing.assert_allclose(dI_dy4.array, dI_dy0.array, rtol=1.e-12, atol=1.e-15)


@timer
def test_color():
    """Test draw_derivative when the interpolation uses an extra property (color).
    """
    wcs = galsim.PixelScale(0.26)
    rng = np.random.default_rng(42)
    positions = [(x, y) for x in [50.3, 500.5, 950.7] for y in [60.1, 500.5, 940.4]]
    colors = list(rng.uniform(-1, 1, len(positions)))
    def sigma_func(star):
        return 0.8 + 0.1 * star['color']
    stars = make_stars(wcs, positions, sigma_func, colors=colors)

    mod = piff.PixelGrid(0.26, 17)
    interp = piff.BasisPolynomial(order=[1,1,1], keys=['u','v','color'])
    psf = fit_psf(piff.SimplePSF(mod, interp), stars, {0: wcs})
    assert tuple(psf.interp_property_names) == ('u','v','color')

    x, y = 300.2, 400.7
    # color is required
    with np.testing.assert_raises(TypeError):
        psf.draw_derivative(x, y)
    dI_dx1, dI_dy1 = check_psf_derivative(psf, x, y, color=-0.5)
    dI_dx2, dI_dy2 = check_psf_derivative(psf, x, y, color=0.5)
    # Different colors give different derivatives
    assert np.max(np.abs(dI_dx1.array - dI_dx2.array)) > 1.e-3 * np.max(np.abs(dI_dx1.array))


@timer
def test_composite():
    """Test draw_derivative with SingleChip and Sum PSFs, and that Convolve raises.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    def sigma_func(star):
        return 0.8 + 0.1 * star['u'] / 250. - 0.05 * star['v'] / 250.
    positions = [(50.3, 60.1), (950.7, 70.2), (60.9, 940.4), (930.2, 960.8), (500.5, 500.5)]

    # SingleChip with two chips with different wcs
    wcs0 = galsim.PixelScale(0.26)
    wcs1 = galsim.JacobianWCS(0.25, 0.02, -0.03, 0.27)
    stars = (make_stars(wcs0, positions, sigma_func, chipnum=0) +
             make_stars(wcs1, positions, sigma_func, chipnum=1))
    single = piff.SimplePSF(piff.PixelGrid(0.25, 17), piff.BasisPolynomial(order=1))
    psf = piff.SingleChipPSF(single)
    psf = fit_psf(psf, stars, {0: wcs0, 1: wcs1})

    x, y = 300.2, 400.7
    print('SingleChip, chip 0')
    dI_dx0, dI_dy0 = check_psf_derivative(psf, x, y, chipnum=0)
    print('SingleChip, chip 1')
    dI_dx1, dI_dy1 = check_psf_derivative(psf, x, y, chipnum=1, center=(x+0.7, y-0.2))
    # The two chips have different solutions
    assert np.max(np.abs(dI_dx0.array - dI_dx1.array)) > 1.e-3 * np.max(np.abs(dI_dx0.array))
    # chipnum is required
    with np.testing.assert_raises(ValueError):
        psf.draw_derivative(x, y)
    # drawStarDerivative requires a chipnum property
    target = piff.Star.makeTarget(x=x, y=y, wcs=wcs1, stamp_size=48, chipnum=1)
    dI_du, dI_dv = psf.drawStarDerivative(target)
    jac = wcs1.jacobian()
    dI_dx2, dI_dy2 = psf.draw_derivative(x, y, chipnum=1)
    np.testing.assert_allclose(dI_dx2.array, dI_du.array*jac.dudx + dI_dv.array*jac.dvdx,
                               rtol=1.e-12)
    target = piff.Star.makeTarget(x=x, y=y, wcs=wcs1, stamp_size=48)
    target = piff.Star(target.data, psf.psf_by_chip[1].interpolateStar(
                       piff.Star.makeTarget(x=x, y=y, wcs=wcs1, stamp_size=48, chipnum=1)).fit)
    with np.testing.assert_raises(ValueError):
        psf.drawStarDerivative(target)

    # Sum of two PixelGrids with different scales.
    stars = make_stars(wcs0, positions, sigma_func)
    comp1 = piff.SimplePSF(piff.PixelGrid(0.26, 17), piff.BasisPolynomial(order=1))
    comp2 = piff.SimplePSF(piff.PixelGrid(0.52, 9, init='zero', fit_flux=True, centered=False),
                           piff.Mean())
    psf = piff.SumPSF([comp1, comp2])
    psf = fit_psf(psf, stars, {0: wcs0})
    print('Sum')
    dI_dx, dI_dy = check_psf_derivative(psf, x, y)
    # Explicitly check that this is the sum of the component derivatives.
    target = psf.interpolateStar(piff.Star.makeTarget(x=x, y=y, wcs=wcs0, stamp_size=48))
    dI_du, dI_dv = psf.drawStarDerivative(target)
    du1, dv1 = comp1.drawStarDerivative(target)
    du2, dv2 = comp2.drawStarDerivative(target)
    np.testing.assert_allclose(dI_du.array, du1.array + du2.array, rtol=1.e-12, atol=1.e-15)
    np.testing.assert_allclose(dI_dv.array, dv1.array + dv2.array, rtol=1.e-12, atol=1.e-15)
    assert np.max(np.abs(du2.array)) > 0.

    # A Sum with a component that doesn't implement the derivative raises.
    comp3 = piff.SimplePSF(piff.Gaussian(), piff.Mean())
    psf = piff.SumPSF([comp1, comp3])
    with np.testing.assert_raises(NotImplementedError):
        psf._drawStarDerivative(target)

    # Convolve raises.
    psf = piff.ConvolvePSF([comp1, comp3])
    with np.testing.assert_raises(NotImplementedError):
        psf._drawStarDerivative(target)

    # A real file with a GSObject model (Moffat) raises through draw_derivative.
    psf = piff.read('input/test_single_py27.piff')
    assert isinstance(psf.psf_by_chip[1].model, piff.Moffat)
    with np.testing.assert_raises(NotImplementedError):
        psf.draw_derivative(x, y, chipnum=1)


@timer
def test_timing():
    """Check that the derivative is not much slower than drawing the model.
    """
    import time
    mod = piff.PixelGrid(0.26, 17)
    params = make_params(17)
    star = make_grid_star(mod, params, galsim.PixelScale(0.26), stamp_size=48)
    mod.draw(star)
    mod.drawDerivative(star)
    n = 20
    t0 = time.time()
    for i in range(n):
        mod.draw(star)
    t1 = time.time()
    for i in range(n):
        mod.drawDerivative(star)
    t2 = time.time()
    print('Time for draw = %.4f s, drawDerivative = %.4f s'%((t1-t0)/n, (t2-t1)/n))


if __name__ == '__main__':
    test_kernel_xval()
    test_kernel_derivative()
    test_pixelgrid_draw()
    test_psf_draw_derivative()
    test_color()
    test_composite()
    test_timing()
