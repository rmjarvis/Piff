import pickle
import os
from functools import partial

import galsim
import numpy as np
import piff

from piff_test_helper import timer


def gaussian(covariance, flux=1.):
    """Independent positive-definite reference using an affine-transformed Gaussian."""
    values, vectors = np.linalg.eigh(covariance)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    return galsim.Gaussian(sigma=1.).transform(*root.ravel()).withFlux(flux)

@timer
def test_gaussian_blur():
    """Basic tests of GaussianBlur usage.
    """
    # Test a variety of blur covariance matrices, including singular and zero.
    for moments in [[0, 0, 0], [0.01, 0, 0], [0.01, 0.0025, 0.005],
                    [0.01, 1.e-14, 0], [0.004, 0.002, -0.001]]:
        model = piff.Model.process({'type': 'GaussianBlur', 'init': moments})
        star = piff.Star.makeTarget(x=25.2, y=35.3, scale=0.11, stamp_size=31)
        star = model.initialize(star)
        np.testing.assert_array_equal(star.fit.params, moments)
        profile = model.getProfile(star.fit.params)

        # Alway unit-flux, and centered
        assert profile.flux == 1.
        assert profile.centroid == galsim.PositionD(0, 0)

        # Test the profile's values in Fourier space
        ixx, iyy, ixy = moments
        covariance = np.array([[ixx, ixy], [ixy, iyy]])
        for kx, ky in [(0., 0.), (5., -3.), (20., 10.)]:
            k = np.array([kx,ky])
            kval = np.exp(-0.5 * k @ covariance @ k)
            np.testing.assert_allclose(profile.kValue(kx,ky), kval, rtol=1.e-13)

        # Basic properties of all GSObjects.
        assert eval(repr(profile)) == profile
        assert pickle.loads(pickle.dumps(profile)) == profile
        assert hash(pickle.loads(pickle.dumps(profile))) == hash(profile)
        assert profile != galsim.Gaussian(sigma=0.1)
        assert profile != piff.FourierGaussian([0.8, 0.9, 0.10])

        # Test drawing with a simple pixel scale and a distorted WCS.
        for wcs in [galsim.PixelScale(0.11), galsim.JacobianWCS(0.10, 0.02, -0.01, 0.12)]:
            kernel = galsim.Gaussian(sigma=0.12)
            reference = gaussian(covariance + np.eye(2)*0.12**2)
            image = galsim.ImageD(31, 31, wcs=wcs)
            actual = galsim.Convolve(kernel, profile).drawImage(
                image=image.copy(), offset=(0.2, -0.3))
            expected = reference.drawImage(image=image.copy(), offset=(0.2, -0.3))
            np.testing.assert_allclose(actual.array, expected.array, atol=1.e-5)

        # Check invalid params.
        assert model.is_valid_params(moments)
        assert not model.is_valid_params([-0.01, 0.01, 0])

    # Rendering projects a covariance with eigenvalues -0.01 and 0.03 onto rank-one blur.
    profile = model.getProfile([0.01, 0.01, 0.02])
    for kx, ky in [(1., 0.), (2., -2.), (3., 4.)]:
        np.testing.assert_allclose(profile.kValue(kx, ky),
                                   np.exp(-0.5*0.015*(kx+ky)**2), rtol=1.e-13)

@timer
def test_stepk_maxk():
    """Check the formulas for stepk and maxk."""
    # Test both circular and elliptical kernels, including their moment derivatives.
    for moments in [[0.04, 0.04, 0.], [0.04, 0.01, 0.012]]:
        for threshold in [0.01, 1.e-4, 1.e-8]:
            gsparams = galsim.GSParams(maxk_threshold=threshold, folding_threshold=threshold,
                                       stepk_minimum_hlr=0.)
            for derivative in [None, 0, 1, 2]:
                profile = piff.gaussian_blur.FourierGaussian(
                    moments, derivative=derivative, gsparams=gsparams)

                # The stepk value is based on gsparams.folding_threshold. It is supposed
                # to make sure when GalSim draws an image using a default image size that
                # at most folding_threshold * flux is lost off the edges of the stamp.
                # Derivatives have zero signed flux, so compare absolute flux with a larger
                # reference image.
                image = profile.drawImage(dtype=float, method='no_pixel')
                if derivative is None:
                    flux = image.array.sum()
                    real_flux = profile.flux
                else:
                    reference = profile.drawImage(
                        nx=2*image.array.shape[1], ny=2*image.array.shape[0],
                        scale=image.scale, dtype=float, method='no_pixel')
                    flux = np.abs(image.array).sum()
                    real_flux = np.abs(reference.array).sum()
                assert flux >= (1-threshold)*real_flux

                # The choice of maxk should be such that all Fourier space values at
                # larger k than this have a value < maxk_threshold * peak.
                # Usually peak is the (0,0) location, but since derivatives have 0 flux,
                # we use the peak absolute value in the image.
                # Use an image that goes from -2maxk .. 2maxk so we have lots of values
                # at |k| > maxk to test.
                dk = profile.maxk / 100
                kimage = profile.drawKImage(nx=400, ny=400, scale=dk)
                kx, ky = np.meshgrid(
                    np.arange(kimage.bounds.xmin, kimage.bounds.xmax+1),
                    np.arange(kimage.bounds.ymin, kimage.bounds.ymax+1))
                outside = np.hypot(kx, ky)*kimage.scale > profile.maxk
                amplitude = np.abs(kimage.array)
                assert np.max(amplitude[outside]) < threshold*np.max(amplitude)

@timer
def test_initialization():
    """Check various init options.
    """
    # Default init is zero blur, equivalent to a delta-function.
    model = piff.GaussianBlur()
    star = piff.Star.makeTarget(x=20, y=20, scale=0.1, stamp_size=25)
    np.testing.assert_array_equal(model.initialize(star).fit.params, [0., 0., 0.])
    for wcs in [
        galsim.PixelScale(0.2),
        galsim.JacobianWCS(0.1, 0.03, -0.02, 0.12),
        galsim.JacobianWCS(-0.1, 0.03, 0.02, 0.12),
    ]:
        image = galsim.ImageD(25, 25, wcs=wcs)
        target = piff.Star(piff.StarData(image, image.true_center), piff.StarFit(None))
        # Sum's 'zero' amplitude hint is ignored: this kernel always has unit flux.
        for default_init in [None, 'delta', 'zero']:
            initialized = model.initialize(target, default_init=default_init)
            np.testing.assert_array_equal(initialized.fit.params, [0., 0., 0.])
            profile = model.getProfile(initialized.fit.params)
            assert profile.flux == 1.
            assert profile.kValue(1., 2.) == 1.

    # An explicit list means [ixx, iyy, ixy].
    resolved = piff.GaussianBlur(init=[0.0025, 0.0025, 0.])
    initialized = resolved.initialize(star)
    rendered = resolved.draw(initialized)
    measured = resolved.chisq(piff.Star(rendered.data, initialized.fit))
    np.testing.assert_allclose(measured.fit.b, 0., atol=1.e-7)

    # A scalar is the trace ixx + iyy.
    scalar = piff.Model.process({'type': 'GaussianBlur', 'init': 0.006})
    np.testing.assert_allclose(scalar.initialize(star).fit.params, [0.003, 0.003, 0])

    # Explicit initialization overrides a composite's delta-function default.
    np.testing.assert_allclose(scalar.initialize(star, default_init='delta').fit.params,
                               [0.003, 0.003, 0.])
    zero = piff.GaussianBlur(init=0.)
    np.testing.assert_array_equal(zero.initialize(star).fit.params, [0, 0, 0])

    # Malformed, nonfinite, and nonphysical initial covariances are rejected.
    for init in [[1, 2], [1, 2, 3], [np.nan, 1, 0], [1, np.inf, 0], [1, -1, 0],
                 -0.1, np.nan]:
        with np.testing.assert_raises(ValueError):
            piff.GaussianBlur(init=init)

    # Projection removes negative eigenvalues, including a fully negative covariance.
    for proposal, expected in [([1., 1., 2.], [1.5, 1.5, 1.5]),
                               ([-1., -2., 0.], [0., 0., 0.])]:
        projected = model.constrain_params(proposal)
        np.testing.assert_allclose(projected, expected, atol=1.e-15)
        assert model.is_valid_params(projected)

@timer
def test_chisq():
    """Test Fourier derivatives and fitted chi-squared with masks, WCS shear, and offsets."""
    # Test physical covariance derivatives at boundary and interior points, and the
    # right-hand-side correction for a raw covariance with a negative x variance.
    for moments in [[0, 0, 0], [0.01, 0, 0], [0.004, 0.002, -0.001], [-0.001, 0.002, 0.]]:
        for wcs, center in [
            (galsim.PixelScale(0.1), (0., 0.)),
            (galsim.JacobianWCS(0.13, 0.02, -0.01, 0.09), (0.012, -0.023)),
        ]:
            physical = np.array(moments)
            physical[0] = max(physical[0], 0.)
            model = piff.GaussianBlur(init=physical)
            star = piff.Star.makeTarget(x=20.2, y=30.3, wcs=wcs, stamp_size=31)
            star = model.initialize(star.withFlux(100., center=center))
            star = piff.Star(star.data, star.fit.newParams(np.array(moments)))
            # Covariance derivatives multiply the Fourier tail by k squared, so
            # tighten the companion's cutoff before comparing small derivative values.
            gsparams = galsim.GSParams(maxk_threshold=1.e-12, kvalue_accuracy=1.e-12)
            kernel = galsim.Gaussian(sigma=0.2, gsparams=gsparams)
            ixx, iyy, ixy = physical
            covariance = np.array([[ixx, ixy], [ixy, iyy]]) + np.eye(2)*0.2**2
            inverse = np.linalg.inv(covariance)
            image = gaussian(covariance, flux=100.).shift(center).drawImage(
                image=galsim.ImageD(star.image.bounds, wcs=wcs),
                method='no_pixel', center=star.image_pos)
            weight = star.weight.copy()
            weight.fill(4.)
            weight.array[0, :2] = 0
            star = piff.Star(star.data.withNew(image=image+0.01, weight=weight), star.fit)
            measured = model.chisq(star, convert_func=partial(galsim.Convolve, kernel,
                                                             gsparams=gsparams),
                                   draw_method='no_pixel')
            x, y = np.meshgrid(
                np.arange(image.bounds.xmin, image.bounds.xmax+1)-star.x,
                np.arange(image.bounds.ymin, image.bounds.ymax+1)-star.y)
            uv = wcs.jacobian().getMatrix() @ np.array([x.ravel(), y.ravel()])
            uv -= np.array(center)[:, None]
            a, b = inverse @ uv
            signal = (100*wcs.pixelArea()/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
                      * np.exp(-0.5*np.sum(uv*(inverse @ uv), axis=0)))
            mask = weight.array.ravel() != 0
            derivatives = [signal*0.5*(a*a-inverse[0, 0]),
                           signal*0.5*(b*b-inverse[1, 1]),
                           signal*(a*b-inverse[0, 1])]
            expected = np.array([d[mask]*2 for d in derivatives]).T
            np.testing.assert_allclose(measured.fit.A, expected, atol=2.e-6, rtol=2.e-6)
            expected_b = 0.02 + expected @ (physical - moments)
            np.testing.assert_allclose(measured.fit.b, expected_b, atol=1.e-7)
            np.testing.assert_allclose(measured.fit.chisq, mask.sum()*0.02**2, rtol=1.e-6)
            assert measured.fit.dof == mask.sum()
            np.testing.assert_array_equal(measured.fit.params, moments)
            assert measured.fit.flux == 100.
            assert measured.fit.center == center

            # Check the final chisq, dof after fitting.
            initial_fit = star.fit.newParams(np.array(moments)+[0.001, 0.001, 0.],
                                             chisq=-1., dof=-1)
            initial = piff.Star(star.data, initial_fit)
            fitted = model.fit(
                initial, convert_func=partial(galsim.Convolve, kernel, gsparams=gsparams),
                draw_method='no_pixel')
            assert np.any(fitted.fit.params != initial.fit.params)
            profile = model.getProfile(fitted.fit.params).shift(center)*fitted.fit.flux
            rendered = galsim.Convolve(profile, kernel, gsparams=gsparams).drawImage(
                image=galsim.ImageD(star.image.bounds, wcs=wcs),
                center=star.image_pos, method='no_pixel')
            expected_chisq = np.sum(weight.array*(rendered.array-star.image.array)**2)
            np.testing.assert_allclose(fitted.fit.chisq, expected_chisq, rtol=1.e-10)
            assert fitted.fit.dof == mask.sum()

@timer
def test_fit():
    """Test fitting constant GaussianBlur components with a variety of interpolators, including
    both default and zero initializations
    """
    # Recover the same nearly one-dimensional blur with shared basis fitting and
    # independent per-star fitting, from default and zero starting covariances.
    for interp, initial in [('BasisPolynomial', None), ('BasisPolynomial', 0.0),
                            ('Mean', None), ('Mean', 0.0), ('Fixed', None)]:
        true = [0.003, 1.e-6, -0.00004]
        config = {
            'type': 'Convolve',
            'min_iter': 12,
            'max_iter': 30,
            'chisq_thresh': 1.e-8,
            'components': [
                {
                    'type': 'Simple',
                    'model': {'type': 'GaussianBlur', 'init': [0.0144, 0.0144, 0]},
                    'interp': {'type': 'Fixed'},
                },
                {
                    'type': 'Simple',
                    'model': {'type': 'GaussianBlur',
                              'init': true if interp == 'Fixed' else initial},
                    'interp': {'type': interp},
                },
            ],
        }
        if interp == 'BasisPolynomial':
            config['components'][1]['interp']['order'] = 0
        psf = piff.PSF.process(config)
        covariance = np.array([[true[0], true[2]], [true[2], true[1]]]) + np.eye(2)*0.12**2
        stars = []
        for x, y in [(30.2, 40.3), (50.4, 70.1), (80.1, 30.6)]:
            star = piff.Star.makeTarget(x=x, y=y, scale=0.11, stamp_size=25,
                                        properties={'chipnum': 0}).withFlux(1000.)
            gaussian(covariance, flux=1000.).drawImage(image=star.image, center=star.image_pos)
            stars.append(star)
        psf.fit(stars, wcs={0: galsim.PixelScale(0.11)})
        for star in psf.stars:
            assert not star.is_flagged
            np.testing.assert_allclose(star.fit.get_params(1), true, atol=2.e-6)
            # The separate Fourier kernels use GalSim's default max-k truncation;
            # compare image errors relative to the peak as well as recovered moments.
            np.testing.assert_allclose(psf.drawStar(star).image.array, star.image.array,
                                       atol=1.e-4*np.max(star.image.array))

        # Check serialization round trip.
        path = os.path.join('output', 'blur.piff')
        psf.write(path)
        restored = piff.read(path)
        kwargs = dict(x=40.2, y=50.4, chipnum=0, flux=1000., stamp_size=25)
        np.testing.assert_allclose(restored.draw(**kwargs).array, psf.draw(**kwargs).array,
                                   atol=1.e-12)

@timer
def test_no_improving_step():
    """Test where the blur cannot improve the fit

    If the data would want a sharper profile than the model already has, then no amount
    of blurring will improve the model. The fit stays at 0 blur.
    """
    # Initialize on the zero-blur boundary before testing a sharpening proposal.
    model = piff.GaussianBlur(init=0.)
    star = piff.Star.makeTarget(x=20.2, y=30.3, scale=0.1, stamp_size=31).withFlux(100.)
    star = model.initialize(star)
    # These data require sharpening the assumed kernel along x. The proposal step in the
    # fit is in the direction of y broadening, but that only increases chisq, so the
    # final fraction is 0, keeping the current model params, which are all 0.
    gaussian(np.diag([0.035, 0.041]), flux=100.).drawImage(
        image=star.image, center=star.image_pos)
    star = piff.Star(star.data, star.fit.withNew(chisq=-1., dof=-1))
    fitted = model.fit(star, convert_func=partial(galsim.Convolve, galsim.Gaussian(sigma=0.2)))
    np.testing.assert_array_equal(fitted.fit.params, [0, 0, 0])

    # Even though this component didn't change, the chisq and dof are updated correctly.
    profile = galsim.Convolve(model.getProfile(fitted.fit.params), galsim.Gaussian(sigma=0.2))
    rendered = profile.withFlux(fitted.fit.flux).drawImage(
        image=galsim.ImageD(star.image.bounds, wcs=star.image.wcs), center=star.image_pos)
    expected_chisq = np.sum(star.weight.array*(rendered.array-star.image.array)**2)
    np.testing.assert_allclose(fitted.fit.chisq, expected_chisq, rtol=1.e-10)
    assert fitted.fit.dof == np.count_nonzero(star.weight.array)

@timer
def test_varying_blur():
    """Recover linear and quadratic covariance fields with BasisPolynomial interpolation.
    """
    # Recover spatially varying blur from zero while retaining a fixed circular companion.
    for order in [1, 2]:
        psf = piff.PSF.process({
            'type': 'Convolve',
            'min_iter': 12,
            'max_iter': 30,
            'chisq_thresh': 1.e-8,
            'components': [
                {
                    'type': 'Simple',
                    'model': {'type': 'GaussianBlur', 'init': [0.0144, 0.0144, 0]},
                    'interp': {'type': 'Fixed'},
                },
                {
                    'type': 'Simple',
                    'model': {'type': 'GaussianBlur'},
                    'interp': {'type': 'BasisPolynomial', 'order': order},
                },
            ],
        })

        def moments(x, y):
            if order == 1:
                params = np.array([0.002+1.e-5*x, 0.0002+1.e-6*y, 0.0001+1.e-6*x])
            else: # order = 2
                # We want the best fit quadratic to cross into unphysical parameter vectors
                # to test that that works correctly.
                # The physical parameters bottom out at iyy=0 near y=60, but the best fit
                # quadratic should find the parabola that goes into iyy<0.
                # The model will automatically project these to have iyy=0.
                params = np.array([0.002+1.e-5*x+1.e-7*x*y,
                                   max(2.e-7*(y-60)**2-0.00002, 0.), 0.])
            return params

        stars = []
        for x in [30., 50., 70., 90.]:
            for y in [30., 50., 70., 90.]:
                ixx, iyy, ixy = moments(x, y)
                star = piff.Star.makeTarget(
                    x=x, y=y, scale=0.11, stamp_size=25,
                    properties={'chipnum': 0, 'is_reserve': False}).withFlux(1000.)
                covariance = np.array([[ixx, ixy], [ixy, iyy]]) + np.eye(2)*0.12**2
                gaussian(covariance, flux=1000.).drawImage(
                    image=star.image, center=star.image_pos)
                stars.append(star)

        # Make sure flagged and reserve images don't influence the fit.
        stars[0] = stars[0].withProperties(is_reserve=True)
        stars[1] = stars[1].flag_if(True)
        stars[0].image.array[:] *= 10
        stars[1].image.array[:] *= 20
        psf.fit(stars, wcs={0: galsim.PixelScale(0.11)})
        assert psf.stars[0].is_reserve
        assert psf.stars[1].is_flagged
        for star in psf.stars:
            physical = psf.components[1].model.constrain_params(star.fit.get_params(1))
            np.testing.assert_allclose(physical, moments(star.x, star.y),
                                       atol=2.e-6)

        # At new positions, the model projects the raw interpolation for rendering,
        # including where the quadratic minor-axis variance is negative.
        target = piff.Star.makeTarget(x=60., y=60., scale=0.11, stamp_size=25,
                                      properties={'chipnum': 0}).withFlux(1000.)
        interpolated = psf.interpolateStar(target)
        physical = psf.components[1].model.constrain_params(interpolated.fit.get_params(1))
        np.testing.assert_allclose(physical, moments(60., 60.), atol=2.e-6)

        # Rendering and file round trips use the same projected covariance field.
        ixx, iyy, ixy = moments(60., 60.)
        covariance = np.array([[ixx, ixy], [ixy, iyy]]) + np.eye(2)*0.12**2
        expected = gaussian(covariance, flux=1000.).drawImage(
            image=target.image.copy(), center=target.image_pos)
        rendered = psf.drawStar(interpolated).image
        np.testing.assert_allclose(rendered.array, expected.array,
                                   atol=1.e-4*expected.array.max())
        path = os.path.join('output', 'varying_blur.piff')
        psf.write(path)
        restored = piff.read(path)
        np.testing.assert_allclose(restored.drawStar(target).image.array, rendered.array,
                                   atol=1.e-12)


if __name__ == '__main__':
    test_gaussian_blur()
    test_stepk_maxk()
    test_initialization()
    test_chisq()
    test_fit()
    test_no_improving_step()
    test_varying_blur()
