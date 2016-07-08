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
    model = piff.Optical()
    return model

def test_optical(model=None):
    params = np.array([0] * (11 - 4 + 1))
    # add defocus
    params[0] = 0.5
    star = make_empty_star(params=params)
    if not model:
        print('test optical')
        model = piff.Optical()
    # given zernikes, make sure we can:
    star = model.draw(star)
    star_fitted = model.fit(star)

    np.testing.assert_almost_equal(star_fitted.fit.chisq, 0)
    np.testing.assert_almost_equal(star_fitted.fit.flux, star.fit.flux)
    np.testing.assert_almost_equal(star_fitted.fit.params, star.fit.params)

def test_pupil_im(pupil_path='optics_test/DECam_pupil_128.fits'):
    print('test pupil im: ', pupil_path)
    # make sure we can load up a pupil image
    model = piff.Optical(load_pupil=pupil_path)
    test_optical(model)

def test_kolmogorov():
    print('test kolmogorov')
    # make sure if we put in different kolmogorov things that things change
    star = make_empty_star(params=[])

    model = piff.Optical(rzero=0.1)
    star = model.draw(star)

    model2 = piff.Optical(rzero=0.2)
    star2 = model2.draw(star)

    chi2 = np.std((star.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'

def test_shearing():
    print('test shearing')
    # make sure if we put in common mode ellipticities that things change
    star = make_empty_star(params=[])
    g1 = 0
    g2 = 0.05
    model = piff.Optical(rzero=0.1, g1=g1, g2=g2)
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
    model = piff.Optical(rzero=0, sigma=sigma)
    star = model.draw(star)
    # insert assert statement about sigma
    np.testing.assert_almost_equal(gaussian.fit(star).fit.params[0], sigma, 5)

    # gaussian and shear
    model = piff.Optical(rzero=0, sigma=sigma, g1=g1, g2=g2)
    star = model.draw(star)
    params = gaussian.fit(star).fit.params
    np.testing.assert_almost_equal(params[0], sigma, 5)
    np.testing.assert_almost_equal(params[1], g1, 5)
    np.testing.assert_almost_equal(params[2], g2, 5)

    # now gaussian, shear, aberration, rzero
    star = make_empty_star(params=[0.5, 0.8, -0.7, 0.5, -0.2, 0.9, -1, 2.0])
    model = piff.Optical(rzero=0.1, sigma=sigma, g1=g1, g2=g2)
    star = model.draw(star)

def test_disk():
    print('test read/write')
    # save and load
    rzero = 0.1
    sigma = 1.2
    g1 = -0.1
    g2 = 0.05
    model = piff.Optical(rzero=rzero, sigma=sigma, g1=g1, g2=g2, lam=700.0)
    model_file = os.path.join('output','optics.fits')
    with fitsio.FITS(model_file, 'rw', clobber=True) as f:
        model.write(f, 'optics')
        model2 = piff.Optical.read(f, 'optics')

    assert model.kwargs['lam'] == model2.kwargs['lam'],'lam mismatch'
    assert model.optical_psf_kwargs['lam'] == model2.optical_psf_kwargs['lam'],'optical_psf_kwargs lam mismatch'
    assert model.optical_psf_kwargs['diam'] == model2.optical_psf_kwargs['diam'],'diam mismatch'
    assert model.lam_over_r0 == model2.lam_over_r0,'lam_over_r0 mismatch'
    assert model.kwargs['rzero'] == model2.kwargs['rzero'],'rzero mismatch'
    assert model.kwargs['sigma'] == model2.kwargs['sigma'],'sigma mismatch'
    assert model.kwargs['g1'] == model2.kwargs['g1'],'g1 mismatch'
    assert model.kwargs['g2'] == model2.kwargs['g2'],'g2 mismatch'

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

def plot_param(rzero=0.1, params=[], g1=0, g2=0, sigma=0):
    # convenience function
    model = piff.Optical(rzero=rzero, g1=g1, g2=g2, sigma=sigma)
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
    test_pupil_im(pupil_path='optics_test/DECam_pupil_128.fits')
    test_pupil_im(pupil_path='optics_test/DECam_pupil_512.fits')
    test_kolmogorov()
    test_shearing()
    test_gaussian()
    test_disk()
