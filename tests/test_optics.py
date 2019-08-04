#This whole unit test file tests many methods of the Optical class, which have been replaced by equivalent ones in the OptAtmoPSF class; therefore, this unit test file serves no purpose. TODO: make an equivalent unit test file that tests the equivalent methods in the OptAtmoPSF class.

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

from piff_test_helper import timer


@timer
def test_init():
    print('test init')
    # make sure we can init with defaults
    model = piff.Optical(template='des')
    return model


@timer
def test_optical(model=None):
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_optical: Started')
    params = np.array([0] * (11 - 4 + 1 + 3))
    params[0] = 1.
    # add defocus
    params[3] = 0.5
    star = make_empty_star(params=params)
    if not model:
        model = piff.Optical(template='des')
    # given zernikes, make sure we can:
    star = model.draw(star)

    # test copy_image
    star_copy = model.draw(star, copy_image=True)
    star_nocopy = model.draw(star, copy_image=False)
    star.image.array[0,0] = 132435
    assert star_nocopy.image.array[0,0] == star.image.array[0,0]
    assert star_copy.image.array[0,0] != star.image.array[0,0]
    assert star_copy.image.array[1,1] == star.image.array[1,1]

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

    model_pupil_plane_im = model.optical_psf_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)

    # test passing a different optical template that includes diam
    piff.optical_model.optical_templates['test'] = {'diam': 2, 'lam':500, 'r0':0.1}
    model = piff.Optical(pupil_plane_im=pupil_plane_im, template='test')
    model_pupil_plane_im = model.optical_psf_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)


@timer
def test_kolmogorov():
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_kolmogorov')
    # make sure if we put in different kolmogorov things that things change
    star = make_empty_star()

    model = piff.Optical(r0=0.1, template='des')
    star = model.draw(star)

    model2 = piff.Optical(r0=0.2, template='des')
    star2 = model2.draw(star)

    chi2 = np.std((star.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'


@timer
def test_shearing():
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_shearing')
    # make sure if we put in common mode ellipticities that things change
    g1 = -0.015
    g2 = 0.02
    star = make_empty_star(params=np.array([1.0, g1, g2]))
    model = piff.Optical(r0=0, sigma=1.0, template='des', logger=logger)
    star = model.draw(star)
    gaussian = piff.Gaussian()
    star_gaussian = gaussian.fit(star)
    np.testing.assert_almost_equal(star_gaussian.fit.params[1], g1, 5)
    np.testing.assert_almost_equal(star_gaussian.fit.params[2], g2, 5)


@timer
def test_gaussian():
    sigma = 0.7
    r0 = 0.14
    size = 0.7
    g1 = -0.2
    g2 = 0.24
    import galsim
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
        r0s = [0, r0, 0, r0]
        fastfits = [True, True, False, False]
    else:
        logger = piff.config.setup_logger(verbose=1)
        r0s = [0, r0]
        fastfits = [True, False]
    logger.info('test_gaussian')
    gaussian = piff.Gaussian()
    # test gaussian alone
    fit_kwargs = {'flux': 123, 'center': (-0.1, 0.3)}
    star = make_empty_star(params=np.array([1, g1, g2]), fit_kwargs=fit_kwargs)
    model = piff.Optical(r0=0, sigma=sigma, template='des', logger=logger)
    star = model.draw(star)
    # insert assert statement about sigma
    np.testing.assert_almost_equal(gaussian.fit(star).fit.params[0], sigma, 5)

    # gaussian and shear
    model = piff.Optical(r0=0, sigma=sigma, template='des')
    star = model.draw(star)
    params = gaussian.fit(star).fit.params
    np.testing.assert_almost_equal(params[0], sigma, 5)
    np.testing.assert_almost_equal(params[1], g1, 5)
    np.testing.assert_almost_equal(params[2], g2, 5)

    # now gaussian, shear, aberration, r0
    fit_kwargs = {'flux': 123, 'center': (0, 0)}
    star = make_empty_star(params=np.array([0.7, 0.1, -0.1, 0.8, 0.3, 0.4, 0.6, -0.6, 0.4, 0.8, 1.0]), fit_kwargs=fit_kwargs)
    model_slow = piff.Optical(r0=r0, template='des', fastfit=False)
    model_r0 = piff.Optical(r0=r0, template='des')
    model_sigma = piff.Optical(r0=0, sigma=sigma, template='des')

    # check that model_r0.atmo is Kolmogorov, and model_sigma.atmo is Gaussian
    assert type(model_r0.atmo) is galsim.Kolmogorov
    assert type(model_sigma.atmo) is galsim.Gaussian

    for r0, fastfit in zip(r0s, fastfits):
        # atmo and optics terms are highly degenerate so don't bother testing together
        for vary_atmo, vary_optics in zip([True, False, False], [False, True, False]):
            logger.info('r0, fastfit, vary_atmo, vary_optics')
            logger.info('{0}, {1}, {2}, {3}'.format(r0, fastfit, vary_atmo, vary_optics))
            model = piff.Optical(r0=r0, sigma=sigma, template='des', vary_atmosphere=vary_atmo, vary_optics=vary_optics, fastfit=fastfit)
            drawn_star = model.draw(star)
            # create stripped star
            empty_star = piff.Star(drawn_star.data, None)

            params0 = star.fit.params.copy()
            if vary_atmo:
                params0[:3] = np.array([1, 0, 0])
            if vary_optics:
                # seed with some starting z4
                params0[3] = 0.2
                params0[4:] = 0

            # run fit
            fit_star = model.fit(empty_star, params0=params0, logger=logger)
            # plot_stars(drawn_star, model.draw(fit_star))

            # check the vary params pieces
            if not vary_atmo:
                np.testing.assert_array_equal(fit_star.fit.params[:3], star.fit.params[:3])
            if not vary_optics:
                np.testing.assert_array_equal(fit_star.fit.params[3:], star.fit.params[3:])

            # assert fit parameters work
            assert len(fit_star.fit.params) == len(star.fit.params)
            np.testing.assert_almost_equal(fit_star.fit.flux, star.fit.flux, decimal=0)
            # np.testing.assert_almost_equal(fit_star.fit.center[0], star.fit.center[0], decimal=2)
            # np.testing.assert_almost_equal(fit_star.fit.center[1], star.fit.center[1], decimal=2)
            try:
                np.testing.assert_allclose(fit_star.fit.params[:len(star.fit.params)], star.fit.params, rtol=1e-3, atol=1e-1)
            except AssertionError:
                # there is a degeneracy in defocus and astigmatism and spherical in that minus sign to all of them yields the same image
                params = fit_star.fit.params
                params[3] *= -1
                params[4] *= -1
                params[5] *= -1
                params[10] *= -1
                np.testing.assert_allclose(params[:len(star.fit.params)], star.fit.params, rtol=1e-3, atol=1e-1)


@timer
def test_disk():
    print('test read/write')
    # save and load
    r0 = 0.1
    model = piff.Optical(r0=r0, lam=700.0, template='des')
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
    assert model.vary_atmosphere == model2.vary_atmosphere,'vary_atmosphere mismatch'
    assert model.vary_optics == model2.vary_optics,'vary_optics mismatch'

#####
# convenience functions
#####

def plot_star(star):
    # convenience function
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.image.array)
    plt.colorbar()
    plt.show()
def plot_stars(star1, star2):
    # convenience function
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(figsize=(6*3, 4), ncols=3)

    for ax, arr in zip(axs, [star1.image.array, star2.image.array, star1.image.array - star2.image.array]):
        im = ax.imshow(arr)
        fig.colorbar(im, ax=ax)
    plt.show()

def plot_param(r0=0.1, params=[], sigma=0):
    # convenience function
    model = piff.Optical(r0=r0, sigma=sigma, template='des')
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
                                scale=0.263, stamp_size=24)

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
    test_shearing()
    test_gaussian()
    test_disk()
