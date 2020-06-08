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

from __future__ import print_function
import numpy as np
import piff
import os
import fitsio
import galsim

from piff_test_helper import timer


@timer
def test_init():
    # make sure we can init with defaults
    model = piff.Optical(template='des')
    prof = model.getProfile([])
    print('prof = ',prof)
    assert isinstance(prof, galsim.Convolution)

    # make a simple optical-only model
    model = piff.Optical(diam=4, lam=700)
    prof = model.getProfile([])
    print('prof = ',prof)
    assert isinstance(prof, galsim.OpticalPSF)


@timer
def test_optical(model=None):
    params = np.array([0] * (11 - 4 + 1))
    # add defocus
    params[0] = 0.5
    star = make_empty_star(params=params)
    if not model:
        model = piff.Optical(template='des')
    # given zernikes, make sure we can:
    star = model.draw(star)
    star_fitted = model.fit(star)

    np.testing.assert_almost_equal(star_fitted.fit.chisq, 0)
    np.testing.assert_almost_equal(star_fitted.fit.flux, star.fit.flux)
    np.testing.assert_almost_equal(star_fitted.fit.params, star.fit.params)

    # Reflux actually updates these to something reasonable.
    logger = piff.config.setup_logger(verbose=2)
    star_wrong = star.withFlux(2, (0.1,0.2))
    star_reflux = model.reflux(star_wrong, logger=logger)
    # There is no noise, but this is still not completely perfect.
    assert star_reflux.fit.chisq < 1.e-2
    np.testing.assert_allclose(star_reflux.fit.flux, 1.0, rtol=0.1)

    # test copy_image
    star_copy = model.draw(star, copy_image=True)
    star_nocopy = model.draw(star, copy_image=False)
    star.image.array[0,0] = 132435
    assert star_nocopy.image.array[0,0] == star.image.array[0,0]
    assert star_copy.image.array[0,0] != star.image.array[0,0]
    assert star_copy.image.array[1,1] == star.image.array[1,1]

    with np.testing.assert_raises(TypeError):
        piff.Optical(template='des', invalid=True)
    with np.testing.assert_raises(TypeError):
        piff.Optical(lam=700)  # missing diam
    with np.testing.assert_raises(TypeError):
        piff.Optical(diam=4)  # missing lam
    with np.testing.assert_raises(ValueError):
        piff.Optical(template='invalid')

@timer
def test_pupil_im(pupil_plane_file='input/DECam_pupil_128.fits'):
    import galsim
    print('test pupil im: ', pupil_plane_file)
    # make sure we can load up a pupil image
    model = piff.Optical(diam=4.274419, lam=500., r0=0.1, pupil_plane_im=pupil_plane_file)
    test_optical(model)
    # make sure we really loaded it
    pupil_plane_im = galsim.fits.read(pupil_plane_file)
    # Check the scale (and fix if necessary)
    print('pupil_plane_im.scale = ',pupil_plane_im.scale)
    ref_psf = galsim.OpticalPSF(lam=500., diam=4.274419)
    print('scale should be ',ref_psf._psf.aper.pupil_plane_scale)
    if pupil_plane_im.scale != ref_psf._psf.aper.pupil_plane_scale:
        print('fixing scale')
        pupil_plane_im.scale = ref_psf._psf.aper.pupil_plane_scale
        pupil_plane_im.write(pupil_plane_file)

    model_pupil_plane_im = model.opt_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)

    # test passing a different optical template that includes diam
    piff.optical_model.optical_templates['test'] = {'diam': 2, 'lam':500, 'r0':0.1}
    model = piff.Optical(pupil_plane_im=pupil_plane_im, template='test')
    model_pupil_plane_im = model.opt_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)


@timer
def test_kolmogorov():
    print('test kolmogorov')
    # make sure if we put in different kolmogorov things that things change
    star = make_empty_star()

    model = piff.Optical(r0=0.1, template='des')
    star = model.draw(star)

    model2 = piff.Optical(r0=0.2, template='des')
    star2 = model2.draw(star)

    chi2 = np.std((star.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'


@timer
def test_vonkarman():
    # Like above, but using L0.
    star = make_empty_star()

    model = piff.Optical(r0=0.1, L0=20, template='des')
    star = model.draw(star)

    model2 = piff.Optical(r0=0.1, L0=40, template='des')
    star2 = model2.draw(star)

    chi2 = np.std((star.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'


@timer
def test_shearing():
    print('test shearing')
    # make sure if we put in common mode ellipticities that things change
    star = make_empty_star()
    g1 = 0
    g2 = 0.05
    model = piff.Optical(r0=0.1, g1=g1, g2=g2, template='des')
    star = model.draw(star)
    gaussian = piff.Gaussian(include_pixel=False)
    star_gaussian = gaussian.fit(star)
    np.testing.assert_almost_equal(star_gaussian.fit.params[1], g1, 5)
    np.testing.assert_almost_equal(star_gaussian.fit.params[2], g2, 5)


@timer
def test_gaussian():
    gaussian = piff.Gaussian(include_pixel=False)
    print('test gaussian')
    star = make_empty_star()
    # test gaussian alone
    sigma = 1.7
    g1 = -0.1
    g2 = 0.05
    model = piff.Optical(r0=0, sigma=sigma, template='des')
    star = model.draw(star)
    # insert assert statement about sigma
    # We don't expect sigma to match.  But it should be the biggest component, so close-ish
    np.testing.assert_allclose(gaussian.fit(star).fit.params[0], sigma, rtol=1.e-2)

    # omitting atm params is equivalent
    kwargs = piff.optical_model.optical_templates['des'].copy()
    del kwargs['r0']
    model2 = piff.Optical(sigma=sigma, **kwargs)
    star2 = model2.draw(star)
    assert star.image == star2.image

    # gaussian and shear
    model = piff.Optical(r0=0, sigma=sigma, g1=g1, g2=g2, template='des')
    star = model.draw(star)
    params = gaussian.fit(star).fit.params
    np.testing.assert_allclose(gaussian.fit(star).fit.params[0], sigma, rtol=1.e-2)
    # Shears are much closer, since the optical part is round.
    np.testing.assert_allclose(params[1], g1, rtol=1.e-4)
    np.testing.assert_allclose(params[2], g2, rtol=1.e-4)

    # now gaussian, shear, aberration, r0
    star = make_empty_star(params=[0.5, 0.8, -0.7, 0.5, -0.2, 0.9, -1, 2.0])
    model = piff.Optical(r0=0.1, sigma=sigma, g1=g1, g2=g2, template='des')
    star = model.draw(star)


@timer
def test_disk():
    print('test read/write')
    # save and load
    r0 = 0.1
    sigma = 1.2
    g1 = -0.1
    g2 = 0.05
    model = piff.Optical(r0=r0, sigma=sigma, g1=g1, g2=g2, lam=700.0, template='des')
    model_file = os.path.join('output','optics.fits')
    with fitsio.FITS(model_file, 'rw', clobber=True) as f:
        model.write(f, 'optics')
        model2 = piff.Optical.read(f, 'optics')

    for key in model.kwargs:
        assert key in model2.kwargs, 'key %r missing from model2 kwargs'%key
        assert model.kwargs[key] == model2.kwargs[key], 'key %r mismatch'%key
    for key in model.opt_kwargs:
        assert key in model2.opt_kwargs, 'key %r missing from model2 opt_kwargs'%key
        assert model.opt_kwargs[key] == model2.opt_kwargs[key], 'key %r mismatch'%key
    for key in model.atm_kwargs:
        assert key in model2.atm_kwargs, 'key %r missing from model2 atm_kwargs'%key
        assert model.atm_kwargs[key] == model2.atm_kwargs[key], 'key %r mismatch'%key
    assert model.sigma == model2.sigma,'sigma mismatch'
    assert model.g1 == model2.g1,'g1 mismatch'
    assert model.g2 == model2.g2,'g2 mismatch'

#####
# convenience functions
#####

def plot_star(star):
    # convenience function
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.data.data)
    plt.colorbar()
    plt.show()

def plot_param(r0=0.1, params=[], g1=0, g2=0, sigma=0):
    # convenience function
    model = piff.Optical(r0=r0, g1=g1, g2=g2, sigma=sigma, template='des')
    star = make_empty_star(params=params)
    star = model.draw(star)
    plot_star(star)
    return star.data.data

def make_empty_star(icen=500, jcen=700, ccdnum=28, params=None,
                    properties={},
                    fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    star = piff.Star.makeTarget(x=icen, y=jcen, properties=properties,
                                scale=0.263)

    if params is None:
        starfit = None
    else:
        starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(star.data, starfit)

    return star

if __name__ == '__main__':
    test_init()
    test_optical()
    test_pupil_im(pupil_plane_file='input/DECam_pupil_128.fits')
    test_pupil_im(pupil_plane_file='input/DECam_pupil_512.fits')
    test_kolmogorov()
    test_vonkarman()
    test_shearing()
    test_gaussian()
    test_disk()
