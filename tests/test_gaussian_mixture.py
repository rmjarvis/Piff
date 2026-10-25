import os

import galsim
import numpy as np
import piff

from piff.readers import FitsReader
from piff.writers import FitsWriter
from piff_test_helper import timer


@timer
def test_mixture_basic():
    """Test basic use of GaussianMixture class with round, centered components.
    """
    pixel_scale = 0.2
    stamp_size = 41
    star = piff.Star.makeTarget(x=20, y=20, scale=pixel_scale, stamp_size=stamp_size)
    for weights, sigmas in [
        ([1.], [0.4]),                      # Single Gaussian
        ([0.3, 0.7], [0.2, 0.5]),           # Two components, already normalized.
        ([2., 5., 3.], [0.15, 0.3, 0.6]),   # Three components, needs normalization.
        ([0., 1.], [0.2, 0.4])              # Degenerate component (0 weight)
    ]:
        model = piff.GaussianMixture(weights=weights, sigmas=sigmas)
        normalized = np.array(weights) / np.sum(weights)
        params = np.concatenate([normalized, sigmas, np.zeros(4 * len(weights))])

        # The default_init is ignored, since we provide explicit values.
        for init in [None, 'delta', 'zero']:
            initialized = model.initialize(star, default_init=init)
            np.testing.assert_array_equal(initialized.fit.params, params)

        # Check that profile is the expected sum of Gaussians.
        profile = model.getProfile(params)
        np.testing.assert_allclose(profile.flux, 1., atol=1.e-15)
        assert profile.centroid == galsim.PositionD(0, 0)
        gauss_norm = (2. * np.pi * np.array(sigmas)**2)**-1
        for x, y in [(0., 0.), (1., 0.), (3., 4.), (8., 2.)]:
            expected = np.sum(normalized * gauss_norm *
                              np.exp(-0.5 * (x**2 + y**2) / np.array(sigmas)**2))
            np.testing.assert_allclose(profile.xValue(x, y), expected, rtol=1.e-13)

        # Draw the profile as a GalSim Sum of Gaussians.
        truth = galsim.Sum([galsim.Gaussian(sigma=s, flux=w)
                            for w, s in zip(normalized, sigmas)])
        initialized = initialized.withFlux(120., center=(0.03, -0.04))
        rendered = model.draw(initialized)
        expected = truth.withFlux(120.).shift(0.03, -0.04).drawImage(
            nx=stamp_size, ny=stamp_size, scale=pixel_scale)
        np.testing.assert_allclose(rendered.image.array, expected.array, atol=1.e-7)

        # Check that using the config interface does the same thing.
        config_model = piff.Model.process(
            {
                'type': 'GaussianMixture',
                'weights': weights,
                'sigmas': sigmas,
            })
        config_star = config_model.initialize(star)
        np.testing.assert_array_equal(config_star.fit.params, params)
        assert config_model.getProfile(params) == profile

        # Check serialization round trip
        file_name = os.path.join('output', 'gaussian_mixture_model_basic.fits')
        with FitsWriter.open(file_name) as writer:
            model.write(writer, 'model')
        with FitsReader.open(file_name) as reader:
            restored = piff.Model.read(reader, 'model')
        new = restored.initialize(star)
        np.testing.assert_array_equal(new.fit.params, initialized.fit.params)
        assert restored.getProfile(new.fit.params) == profile

    # GaussianMixture doesn't implement fit yet.
    with np.testing.assert_raises(NotImplementedError):
        model.fit(model.initialize(star))

    # Must provide some weight, sigma
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[], sigmas=[])
    # weight, sigma must be the same length array
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[1.0], sigmas=[0.3, 0.5])
    # arrays must be one-dimensional
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[[0.3, 0.7]], sigmas=[[0.3, 0.5]])
    # total weight cannot be 0.
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.0, 0.0], sigmas=[0.3, 0.5])
    # cannot use negative weights
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[-0.1, 1.], sigmas=[0.3, 0.5])
    # cannot use sigma <= 0
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, -0.5])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[-0.3, 0.5])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0., 0.1])

@timer
def test_mixture_elliptical():
    """Test GaussianMixture with at lest some components using g1,g2.
    """
    pixel_scale = 0.2
    stamp_size = 41
    star = piff.Star.makeTarget(x=20, y=20, scale=pixel_scale, stamp_size=stamp_size)
    weights = [0.3, 0.7]
    sigmas = [0.2, 0.5]
    for g1, g2 in [
        ([0.2, -0.1], None),
        (None, [0.1, 0.3]),
        ([0.2, -0.1], [0.1, 0.3])
    ]:
        model = piff.GaussianMixture(weights=weights, sigmas=sigmas, g1=g1, g2=g2)
        initialized = model.initialize(star)
        xg1 = np.zeros_like(weights) if g1 is None else g1
        xg2 = np.zeros_like(weights) if g2 is None else g2
        normalized = np.array(weights) / np.sum(weights)
        params= np.concatenate([weights, sigmas, xg1, xg2, np.zeros(4)])
        np.testing.assert_array_equal(initialized.fit.params, params)

        # Check flux and centroid
        profile = model.getProfile(params)
        np.testing.assert_allclose(profile.flux, 1., atol=1.e-15)
        assert profile.centroid == galsim.PositionD(0, 0)

        # Draw the profile as a GalSim Sum of Gaussians.
        truth = galsim.Sum([
            galsim.Gaussian(sigma=s, flux=w).shear(g1=_g1, g2=_g2)
            for w, s, _g1, _g2 in zip(normalized, sigmas, xg1, xg2)
        ])
        initialized = initialized.withFlux(120., center=(0.03, -0.04))
        rendered = model.draw(initialized)
        expected = truth.withFlux(120.).shift(0.03, -0.04).drawImage(
            nx=stamp_size, ny=stamp_size, scale=pixel_scale)
        np.testing.assert_allclose(rendered.image.array, expected.array, atol=1.e-7)

        # Check that using the config interface does the same thing.
        config = {
            'type': 'GaussianMixture',
            'weights': weights,
            'sigmas': sigmas,
        }
        if g1 is not None: config['g1'] = g1
        if g2 is not None: config['g2'] = g2
        config_model = piff.Model.process(config)
        config_star = config_model.initialize(star)
        np.testing.assert_array_equal(config_star.fit.params, params)
        assert config_model.getProfile(params) == profile

        # Check serialization round trip
        file_name = os.path.join('output', 'gaussian_mixture_model_shear.fits')
        with FitsWriter.open(file_name) as writer:
            model.write(writer, 'model')
        with FitsReader.open(file_name) as reader:
            restored = piff.Model.read(reader, 'model')
        new = restored.initialize(star)
        np.testing.assert_array_equal(new.fit.params, initialized.fit.params)
        assert restored.getProfile(new.fit.params) == profile

    # Shear cannot be a different length from weight, sigma
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[0.2], g2=[0.1, 0.0])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g2=[0.1])
    # g1,g2 must be one-dimensional
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[[0.1, 0.0]])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g2=[[0.0, 0.03]])
    # |g| must be < 1
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[1, 0], g2=[0, 0])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[0.6, 0], g2=[0.8, 0])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[0.9, 0], g2=[0.8, 0])

    # g1,g2, must be arrays.
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=0.1, g2=[0.1,-0.2])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], g1=[0.1,-0.2], g2=0)

@timer
def test_mixture_shift():
    """Test GaussianMixture with at least some components using shifts."""
    pixel_scale = 0.2
    stamp_size = 41
    star = piff.Star.makeTarget(x=20, y=20, scale=pixel_scale, stamp_size=stamp_size)
    weights = [0.3, 0.7]
    sigmas = [0.2, 0.5]
    for du, dv, g1, g2 in [
        ([0.2, -0.1], None, None, None),
        (None, [0.1, 0.3], None, None),
        ([0.2, -0.1], [0.1, 0.3], None, None),
        ([0.2, -0.1], [0.1, 0.3], [0.1, -0.03], [0.05, 0.])
    ]:
        model = piff.GaussianMixture(weights=weights, sigmas=sigmas, g1=g1, g2=g2, du=du, dv=dv)
        initialized = model.initialize(star)
        xdu = np.zeros_like(weights) if du is None else du
        xdv = np.zeros_like(weights) if dv is None else dv
        xg1 = np.zeros_like(weights) if g1 is None else g1
        xg2 = np.zeros_like(weights) if g2 is None else g2
        normalized = np.array(weights) / np.sum(weights)
        params= np.concatenate([weights, sigmas, xg1, xg2, xdu, xdv])
        np.testing.assert_array_equal(initialized.fit.params, params)

        # Check flux and centroid
        profile = model.getProfile(params)
        np.testing.assert_allclose(profile.flux, 1., atol=1.e-15)
        np.testing.assert_allclose([profile.centroid.x, profile.centroid.y],
                                   [np.dot(weights, xdu), np.dot(weights, xdv)], atol=1.e-15)

        # Draw the profile as a GalSim Sum of Gaussians.
        truth = galsim.Sum([
            galsim.Gaussian(sigma=s, flux=w).shear(g1=_g1,g2=_g2).shift(_du, _dv)
            for w, s, _g1, _g2, _du, _dv in zip(normalized, sigmas, xg1, xg2, xdu, xdv)
        ])
        initialized = initialized.withFlux(120., center=(0.03, -0.04))
        rendered = model.draw(initialized)
        expected = truth.withFlux(120.).shift(0.03, -0.04).drawImage(
            nx=stamp_size, ny=stamp_size, scale=pixel_scale)
        np.testing.assert_allclose(rendered.image.array, expected.array, atol=1.e-7)

        # Check that using the config interface does the same thing.
        config = {
            'type': 'GaussianMixture',
            'weights': weights,
            'sigmas': sigmas,
        }
        if g1 is not None: config['g1'] = g1
        if g2 is not None: config['g2'] = g2
        if du is not None: config['du'] = du
        if dv is not None: config['dv'] = dv
        config_model = piff.Model.process(config)
        config_star = config_model.initialize(star)
        np.testing.assert_array_equal(config_star.fit.params, params)
        assert config_model.getProfile(params) == profile

        # Check serialization round trip
        file_name = os.path.join('output', 'gaussian_mixture_model_shift.fits')
        with FitsWriter.open(file_name) as writer:
            model.write(writer, 'model')
        with FitsReader.open(file_name) as reader:
            restored = piff.Model.read(reader, 'model')
        new = restored.initialize(star)
        np.testing.assert_array_equal(new.fit.params, initialized.fit.params)
        assert restored.getProfile(new.fit.params) == profile

    # Shift cannot be a different length from weight, sigma
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], du=[0.2], dv=[0.1, 0.0])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], dv=[0.1])

    # du,dv must be one-dimensional
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], du=[[0.1, 0.0]])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], dv=[[0.0, 0.03]])

    # du,dv, must be arrays.
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], du=0.1, dv=[0.1,-0.2])
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5], du=[0.1,-0.2], dv=0)
 
    # Check a set that has the right total number of parameters, so could have been
    # parsed without error if each component not checked separately.
    with np.testing.assert_raises(ValueError):
        piff.GaussianMixture(weights=[0.3, 0.7], sigmas=[0.3, 0.5],
                             du=[0.1,-0.2,0.2], dv=[0])

@timer
def test_fixed():
    """Test the basic functionality of the Fixed interp type.
    """
    pixel_scale = 0.2

    # An arbitrary parameter array that the interp applies to all stars.
    nominal = np.array([-0.2, 1.5, 3.])

    # num is used with composite objects. Check None (for non-composite) and 1 to exercise
    # the pathwats that apply this only to the second component.
    for num in [None, 1]:
        interp = piff.Interp.process({'type': 'Fixed'})
        interp.set_num(num)
        assert interp.property_names == ()
        assert not interp.fit_parameters
        assert not interp.degenerate_points
        stars = []
        for x, y in [(10., 20.), (30., 40.)]:
            star = piff.Star.makeTarget(x=x, y=y, scale=pixel_scale)
            params = nominal if num is None else [np.array([7., 8.]), nominal]
            stars.append(piff.Star(star.data, piff.StarFit(params)))
        initialized = interp.initialize(stars)
        for star in initialized:
            np.testing.assert_array_equal(star.fit.get_params(num), nominal)

        # Solving from different fitted values returns to the original fixed values.
        changed = piff.Star(stars[1].data, stars[1].fit.newParams(nominal * 2, num=num))
        interp.solve([changed])
        predicted = interp.interpolate(changed)
        np.testing.assert_array_equal(predicted.fit.get_params(num), nominal)
        if num is not None:
            np.testing.assert_array_equal(predicted.fit.get_params(0), [7., 8.])

        # Check serialization round-trip.
        file_name = os.path.join('output', 'fixed_interp.fits')
        with FitsWriter.open(file_name) as writer:
            interp.write(writer, 'interp')
        with FitsReader.open(file_name) as reader:
            restored = piff.Interp.read(reader, 'interp')
        restored.set_num(num)
        assert isinstance(restored, piff.Fixed)
        assert not restored.fit_parameters
        np.testing.assert_array_equal(restored.interpolate(changed).fit.get_params(num),
                                      nominal)

        # It is an error to initialize stars with different initial parameters.
        inconsistent = piff.Star(changed.data, changed.fit.newParams(nominal + 1, num=num))
        with np.testing.assert_raises(ValueError):
            interp.initialize([changed, inconsistent])

@timer
def test_fixed_psf():
    """Test a PSF comprising a GaussianMixture model with a Fixed interp.
    """
    # Configure a fixed, asymmetric mixture with known component shapes and shifts.
    pixel_scale = 0.2
    stamp_size = 31
    weights = [0.3, 0.7]
    sigmas = [0.25, 0.5]
    g1 = [0.1, -0.2]
    g2 = [0.05, 0.1]
    du = [0.03, -0.02]
    dv = [-0.04, 0.01]
    config = {
        'type': 'Simple',
        'model': {
            'type': 'GaussianMixture',
            'weights': weights,
            'sigmas': sigmas,
            'g1': g1,
            'g2': g2,
            'du': du,
            'dv': dv,
        },
        'interp': {'type': 'Fixed'},
        'max_iter': 15,
        'chisq_thresh': 1.e-8,
    }
    psf = piff.PSF.process(config)

    # Generate independent GalSim images with different stellar fluxes and offsets.
    kernel = galsim.Sum(
        0.3 * galsim.Gaussian(sigma=0.25).shear(g1=0.1, g2=0.05).shift(0.03, -0.04),
        0.7 * galsim.Gaussian(sigma=0.5).shear(g1=-0.2, g2=0.1).shift(-0.02, 0.01))
    stars = []
    star_pos = [(30.1, 40.3), (60.4, 80.2), (90.2, 50.6)]
    truths = [(100., (0.03, -0.04)), (250., (-0.05, 0.02)), (80., (0.01, 0.06))]
    for (x, y), (flux, center) in zip(star_pos, truths):
        star = piff.Star.makeTarget(x=x, y=y, scale=pixel_scale, stamp_size=stamp_size,
                                    properties={'chipnum': 0})
        kernel.withFlux(flux).shift(center).drawImage(image=star.image, center=star.image_pos)
        stars.append(star)

    # The Fixed interp doesn't call the model fit (which would raise an exception for
    # GaussianMixture). Instead it just preserves the initial parameters.
    psf.fit(stars, wcs={0: galsim.PixelScale(pixel_scale)})
    expected = np.concatenate([weights, sigmas, g1, g2, du, dv])
    for star in psf.stars:
        assert not star.is_flagged
        np.testing.assert_array_equal(star.fit.params, expected)

    # With no noise here, reflux should keep the flux and center where it is.
    for star, (flux, center) in zip(psf.stars, truths):
        np.testing.assert_allclose(star.fit.flux, flux, rtol=1.e-5)
        np.testing.assert_allclose(star.fit.center, center, atol=1.e-5)

    # Drawing the profile should match the data image.
    for star in psf.stars:
        np.testing.assert_allclose(psf.drawStar(star).image.array, star.image.array, atol=1.e-12)

    # Check serialization round-trip.
    file_name = os.path.join('output', 'gaussian_mixture_psf.piff')
    psf.write(file_name)
    restored = piff.read(file_name)
    kwargs = dict(x=70.2, y=65.3, chipnum=0, flux=140., stamp_size=stamp_size)
    np.testing.assert_allclose(restored.draw(**kwargs).array, psf.draw(**kwargs).array, atol=1.e-12)

if __name__ == '__main__':
    test_mixture_basic()
    test_mixture_elliptical()
    test_mixture_shift()
    test_fixed()
    test_fixed_psf()
