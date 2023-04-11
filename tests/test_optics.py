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
from numpy.random import default_rng
import piff
import os
import fitsio
import galsim

from piff_test_helper import timer


@timer
def test_init():
    # make sure we can init with defaults
    model = piff.Optical(template='des_simple')
    prof = model.getProfile(zernike_coeff=[0,0,0,0,0.0],r0=0.15,L0=10.,g1=0.,g2=0.)
    print('prof = ',prof)
    assert isinstance(prof, galsim.Convolution)

    # make a simple optical model
    model = piff.Optical(diam=4, lam=700, atmo_type='Kolmogorov')
    prof = model.getProfile(zernike_coeff=[0,0,0,0,0.0],r0=0.15,g1=0.,g2=0.,L0=0.0)
    print('prof = ',prof)
    assert isinstance(prof, galsim.Convolution)


@timer
def test_optical(model=None):

    if not model:
        model = piff.Optical(template='des',atmo_type='Kolmogorov')

    # fill model's parameters for star's Fit with defocus and r0
    params = model.kwargs_toparams(zernike_coeff=[0,0,0,0,0.5],r0=0.15)
    star = make_empty_star(params=params)

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
    np.testing.assert_allclose(star_reflux.fit.flux, 1.0, rtol=0.15)

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
def test_pupil_im(pupil_plane_file='input/DECam_pupil_512uv.fits'):
    import galsim
    print('test pupil im: ', pupil_plane_file)
    # make sure we can load up a pupil image
    model = piff.Optical(diam=4.010, lam=500., pupil_plane_im=pupil_plane_file,atmo_type='Kolmogorov')
    test_optical(model)
    # make sure we really loaded it
    pupil_plane_im = galsim.fits.read(pupil_plane_file)
    # Check the scale (and fix if necessary)
    print('pupil_plane_im.scale = ',pupil_plane_im.scale)
    ref_psf = galsim.OpticalPSF(lam=500., diam=4.020)
    print('scale should be ',ref_psf._psf.aper.pupil_plane_scale)
    if pupil_plane_im.scale != ref_psf._psf.aper.pupil_plane_scale:
        print('fixing scale')
        pupil_plane_im.scale = ref_psf._psf.aper.pupil_plane_scale
        pupil_plane_im.write(pupil_plane_file)

    model_pupil_plane_im = model.opt_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im)

    # test passing a different optical template that includes diam
    piff.optical_model.optical_templates['test'] = {'diam': 2, 'lam':500}
    model = piff.Optical(pupil_plane_im=pupil_plane_im, template='test')
    model_pupil_plane_im = model.opt_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im)


@timer
def test_kolmogorov():
    print('test kolmogorov')

    # make sure if we put in different kolmogorov things that things change
    model = piff.Optical(template='des',atmo_type='Kolmogorov')
    params = model.kwargs_toparams(r0=0.1)
    star = make_empty_star(params=params)
    star1 = model.draw(star)

    params = model.kwargs_toparams(r0=0.2)
    star = make_empty_star(params=params)
    star2 = model.draw(star)

    chi2 = np.std((star1.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'


@timer
def test_vonkarman():
    print('test VonKarman')

    # Like above, but using L0.
    model = piff.Optical(template='des',atmo_type='VonKarman')
    params = model.kwargs_toparams(r0=0.1,L0=10.)
    star = make_empty_star(params=params)
    star1 = model.draw(star)

    params = model.kwargs_toparams(r0=0.1,L0=20.)
    star = make_empty_star(params=params)
    star2 = model.draw(star)

    chi2 = np.std((star1.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'


@timer
def test_shearing():
    print('test shearing')
    # make sure if we put in common mode ellipticities that things change
    g1 = -0.075
    g2 = 0.05
    model = piff.Optical(template='des',atmo_type='Kolmogorov',sigma=0.0)
    params = model.kwargs_toparams(r0=0.1,g1=g1,g2=g2)
    star = make_empty_star(params=params)
    star = model.draw(star)

    # gaussian = piff.Gaussian(include_pixel=False)
    # star_gaussian = gaussian.fit(star)
    # This doesn't work because Optical stores different quantities in the fit than a GSOject model

    flux, cenu, cenv, size, hsm_g1, hsm_g2, flag = star.hsm

    np.testing.assert_allclose(hsm_g1, g1, rtol=0.2)
    np.testing.assert_allclose(hsm_g2, g2, rtol=0.2)

@timer
def test_gaussian():
    print('test size and shape')

    # test gaussian alone
    sigma = 1.7
    fwhm = 2.355 * sigma
    r0 = 0.15/fwhm
    g1 = -0.1
    g2 = 0.05

    # gaussian and shear
    model = piff.Optical(template='des',atmo_type='Kolmogorov')
    params = model.kwargs_toparams(r0=r0,g1=g1,g2=g2)
    star = make_empty_star(params=params)
    star = model.draw(star)

    flux, cenu, cenv, size, hsm_g1, hsm_g2, flag = star.hsm

    np.testing.assert_allclose(size, sigma, rtol=0.2)
    np.testing.assert_allclose(hsm_g1, g1, rtol=0.2)
    np.testing.assert_allclose(hsm_g2, g2, rtol=0.2)


@timer
def test_disk():
    print('test read/write')
    # save and load
    model = piff.Optical(lam=700.0, template='des')
    model_file = os.path.join('output','optics.fits')
    with fitsio.FITS(model_file, 'rw', clobber=True) as f:
        model.write(f, 'optics')
        model2 = piff.Optical.read(f, 'optics')

    for key in model.kwargs:
        assert key in model2.kwargs, 'key %r missing from model2 kwargs'%key
        assert model.kwargs[key] == model2.kwargs[key], 'key %r mismatch'%key
    for key in model.opt_kwargs:
        assert key in model2.opt_kwargs, 'key %r missing from model2 opt_kwargs'%key
        if type(model.opt_kwargs[key])==np.ndarray:
            np.testing.assert_almost_equal(model.opt_kwargs[key],model2.opt_kwargs[key],err_msg='key %r mismatch' % key)
        else:
            assert model.opt_kwargs[key] == model2.opt_kwargs[key], 'key %r mismatch'%key
    for key in model.gsparams_kwargs:
        assert key in model2.gsparams_kwargs, 'key %r missing from model2 gsparams_kwargs'%key
        assert model.gsparams_kwargs[key] == model2.gsparams_kwargs[key], 'key %r mismatch'%key
    for key in model.other_kwargs:
        assert key in model2.other_kwargs, 'key %r missing from model2 other_kwargs'%key
        assert model.other_kwargs[key] == model2.other_kwargs[key], 'key %r mismatch'%key
    assert model.atmo_type == model2.atmo_type,'atmo_type mismatch'

@timer
def test_makestars(nstars=100,constant_atmoparams=True,template='des'):

    print("test_makestars, n=%d, constant_atmoparams=%d" % (nstars,constant_atmoparams))
    model = piff.Optical(template=template,atmo_type='VonKarman',gsparams='starby2')
    params = random_params(nstars,model,constant_atmoparams=constant_atmoparams)
    stars = make_stars(nstars,model,params)


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

def make_empty_star(icen=500, jcen=700, ccdnum=28, params=None, stamp_size=40,
                    properties={},
                    fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    star = piff.Star.makeTarget(x=icen, y=jcen, stamp_size=stamp_size, properties=properties,
                                scale=0.263)

    if params is None:
        starfit = None
    else:
        starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(star.data, starfit)

    return star

def random_params(nstars,model,seed=12345,constant_atmoparams=False):
    # make up some random star parameters
    rng = default_rng(seed)
    z4s = rng.uniform(-0.2,0.2,nstars)
    z5s = rng.uniform(-0.2,0.2,nstars)
    z6s = rng.uniform(-0.2,0.2,nstars)
    z7s = rng.uniform(-0.2,0.2,nstars)
    z8s = rng.uniform(-0.2,0.2,nstars)
    z9s = rng.uniform(-0.2,0.2,nstars)
    z10s = rng.uniform(-0.2,0.2,nstars)
    z11s = rng.uniform(-0.2,0.2,nstars)

    r0s = rng.uniform(0.13,0.18,nstars)
    g1s = rng.uniform(-0.05,0.05,nstars)
    g2s = rng.uniform(-0.05,0.05,nstars)
    L0s = rng.uniform(5.0,15.0,nstars)

    if constant_atmoparams:
        r0s = 0.15 * np.ones(nstars)
        L0s = 7.5 * np.ones(nstars)

    params = []
    for i in range(nstars):
        param = model.kwargs_toparams(zernike_coeff=[0.,0.,0.,0.,z4s[i],z5s[i],z6s[i],z7s[i],z8s[i],z9s[i],z10s[i],z11s[i]],
                                  r0=r0s[i],g1=g1s[i],g2=g2s[i],L0=L0s[i])
        params.append(param)
    return params

def make_stars(nstars,model,params,npixels=19):
    # make a list of stars
    rng = np.random.default_rng(123459)
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = rng.choice(chiplist,nstars)
    pixedge = 20
    icen = rng.uniform(1+pixedge,2048-pixedge,nstars)   # random pixel position inside CCD
    jcen = rng.uniform(1+pixedge,4096-pixedge,nstars)

    # fill stars
    stars = []

    for i in range(nstars):
        # make the shell of a Star object
        star = make_empty_star(icen[i],jcen[i],chipnum[i],params[i],stamp_size=npixels)

        # draw the star
        #star = model.draw(star)

        # draw using drawProfile instead of just draw
        prof = model.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        star = model.drawProfile(star, prof, params[i], use_fit=True, copy_image=True)

        stars.append(star)

    return stars



if __name__ == '__main__':
    test_init()
    test_optical()
    test_pupil_im(pupil_plane_file='input/DECam_pupil_512uv.fits')
    test_kolmogorov()
    test_vonkarman()
    test_shearing()
    test_gaussian()
    test_disk()
    print("des")
    test_makestars(nstars=100)
    test_makestars(nstars=100,constant_atmoparams=False)
    print("desparam")
    test_makestars(nstars=100,template='desparam')
    test_makestars(nstars=100,constant_atmoparams=False,template='desparam')
