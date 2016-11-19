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

def test_init():
    print('test init')
    # make sure we can init with defaults
    model = piff.Optical(template='des')
    return model

def test_optical(model=None):
    params = np.array([0] * (11 - 4 + 1))
    # add defocus
    params[0] = 0.5
    star = make_empty_star(params=params)
    if not model:
        print('test optical')
        model = piff.Optical(template='des')
    # given zernikes, make sure we can:
    star = model.draw(star)
    star_fitted = model.fit(star)

    np.testing.assert_almost_equal(star_fitted.fit.chisq, 0)
    np.testing.assert_almost_equal(star_fitted.fit.flux, star.fit.flux)
    np.testing.assert_almost_equal(star_fitted.fit.params, star.fit.params)

def test_pupil_im(pupil_plane_im='optics_test/DECam_pupil_128.fits'):
    import galsim
    print('test pupil im: ', pupil_plane_im)
    # make sure we can load up a pupil image
    model = piff.Optical(diam=4.274419, lam=500., r0=0.1, pupil_plane_im=pupil_plane_im)
    test_optical(model)
    # make sure we really loaded it
    pupil_plane_im = galsim.Image(fitsio.read(pupil_plane_im))
    model_pupil_plane_im = model.optical_psf_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)

    # test passing a different optical template that includes diam
    piff.optical_model.optical_templates['test'] = {'diam': 2, 'lam':500, 'r0':0.1}
    model = piff.Optical(pupil_plane_im=pupil_plane_im, template='test')
    model_pupil_plane_im = model.optical_psf_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)

def test_kolmogorov():
    print('test kolmogorov')
    # make sure if we put in different kolmogorov things that things change
    star = make_empty_star(params=[])

    model = piff.Optical(r0=0.1, template='des')
    star = model.draw(star)

    model2 = piff.Optical(r0=0.2, template='des')
    star2 = model2.draw(star)

    chi2 = np.std((star.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'

def test_shearing():
    print('test shearing')
    # make sure if we put in common mode ellipticities that things change
    star = make_empty_star(params=[])
    g1 = 0
    g2 = 0.05
    model = piff.Optical(r0=0.1, g1=g1, g2=g2, template='des')
    star = model.draw(star)
    gaussian = piff.Gaussian()
    star_gaussian = gaussian.fit(star)
    np.testing.assert_almost_equal(star_gaussian.fit.params[1], g1, 5)
    np.testing.assert_almost_equal(star_gaussian.fit.params[2], g2, 5)

def test_gaussian():
    gaussian = piff.Gaussian()
    print('test gaussian')
    star = make_empty_star(params=[])
    # test gaussian alone
    sigma = 1
    g1 = -0.1
    g2 = 0.05
    model = piff.Optical(r0=0, sigma=sigma, template='des')
    star = model.draw(star)
    # insert assert statement about sigma
    np.testing.assert_almost_equal(gaussian.fit(star).fit.params[0], sigma, 5)

    # gaussian and shear
    model = piff.Optical(r0=0, sigma=sigma, g1=g1, g2=g2, template='des')
    star = model.draw(star)
    params = gaussian.fit(star).fit.params
    np.testing.assert_almost_equal(params[0], sigma, 5)
    np.testing.assert_almost_equal(params[1], g1, 5)
    np.testing.assert_almost_equal(params[2], g2, 5)

    # now gaussian, shear, aberration, r0
    star = make_empty_star(params=[0.5, 0.8, -0.7, 0.5, -0.2, 0.9, -1, 2.0])
    model = piff.Optical(r0=0.1, sigma=sigma, g1=g1, g2=g2, template='des')
    star = model.draw(star)

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
    for key in model.optical_psf_kwargs:
        assert key in model2.optical_psf_kwargs, 'key %r missing from model2 optical_psf_kwargs'%key
        assert model.optical_psf_kwargs[key] == model2.optical_psf_kwargs[key], 'key %r mismatch'%key
    for key in model.kolmogorov_kwargs:
        assert key in model2.kolmogorov_kwargs, 'key %r missing from model2 kolmogorov_kwargs'%key
        assert model.kolmogorov_kwargs[key] == model2.kolmogorov_kwargs[key], 'key %r mismatch'%key
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

    if np.shape(params) == ():
        starfit = None
    else:
        starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(star.data, starfit)

    return star

if __name__ == '__main__':
    test_init()
    test_optical()
    test_pupil_im(pupil_plane_im='optics_test/DECam_pupil_128.fits')
    test_pupil_im(pupil_plane_im='optics_test/DECam_pupil_512.fits')
    test_kolmogorov()
    test_shearing()
    test_gaussian()
    test_disk()
