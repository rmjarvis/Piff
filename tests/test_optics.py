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

    # strut_angle and pupil_angle can be either str or Angle
    model1 = piff.Optical(diam=4, lam=700, atmo_type='Kolmogorov',
                          pupil_angle=35*galsim.degrees, strut_angle=12*galsim.degrees)
    model2 = piff.Optical(diam=4, lam=700, atmo_type='Kolmogorov',
                          pupil_angle="35 degrees", strut_angle="12 degrees")
    prof1 = model1.getProfile(zernike_coeff=[0,0,0,0,0], r0=0.15)
    prof2 = model2.getProfile(zernike_coeff=[0,0,0,0,0], r0=0.15)
    assert prof1 == prof2


@timer
def test_optical(model=None):

    if not model:
        model = piff.Optical(template='des',atmo_type='Kolmogorov')

    psf = piff.SimplePSF(model, None)

    # fill model's parameters for star's Fit with defocus and r0
    params = model.kwargs_to_params(zernike_coeff=[0,0,0,0,0.5],r0=0.15)
    star = make_empty_star(params=params)

    # Check params_to_kwargs
    kwargs = model.params_to_kwargs(params)
    np.testing.assert_array_equal(kwargs['zernike_coeff'][:5], [0,0,0,0,0.5])
    np.testing.assert_array_equal(kwargs['zernike_coeff'][5:], 0.)
    assert kwargs['r0'] == 0.15
    assert kwargs['L0'] == 10
    assert kwargs['g1'] == 0.
    assert kwargs['g2'] == 0.

    # given zernikes, make sure we can:
    star = model.draw(star)
    star_fitted = model.fit(star)

    np.testing.assert_almost_equal(star_fitted.fit.chisq, 0)
    np.testing.assert_almost_equal(star_fitted.fit.flux, star.fit.flux)
    np.testing.assert_almost_equal(star_fitted.fit.params, star.fit.params)

    # Reflux actually updates these to something reasonable.
    logger = piff.config.setup_logger(verbose=2)
    star_wrong = star.withFlux(2, (0.1,0.2))
    star_reflux = psf.reflux(star_wrong, logger=logger)

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

    # test that errors are raised
    with np.testing.assert_raises(TypeError):
        piff.Optical(template='des', invalid=True)
    with np.testing.assert_raises(TypeError):
        piff.Optical(lam=700)  # missing diam
    with np.testing.assert_raises(TypeError):
        piff.Optical(diam=4)  # missing lam
    with np.testing.assert_raises(ValueError):
        piff.Optical(template='invalid')

    # test Zernike kwargs
    zernike_coeff_short = [0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.]
    nz = len(zernike_coeff_short)
    param_test = model.kwargs_to_params(zernike_coeff=zernike_coeff_short,r0=0.12,g1=-0.05,g2=0.03,L0=20.)
    for i in range(nz):
        assert zernike_coeff_short[i]==param_test[model.idx_z0+i]
    for i in range(nz+1,37+1):
        assert param_test[model.idx_z0+i]==0.0

    zernike_coeff_long = [0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,10.]
    nz = len(zernike_coeff_long)
    param_test = model.kwargs_to_params(zernike_coeff=zernike_coeff_long,r0=0.12,g1=-0.05,g2=0.03,L0=20.)
    for i in range(37+1):
        assert zernike_coeff_long[i]==param_test[model.idx_z0+i]
    assert param_test[model.idx_r0]==0.12

    # Test gsparams
    gsp_star = galsim.GSParams(minimum_fft_size=32, folding_threshold=0.02)
    gsp_donut = galsim.GSParams(minimum_fft_size=128, folding_threshold=0.005)
    model = piff.Optical(template='des', gsparams=gsp_star)
    assert model.gsparams == gsp_star
    model = piff.Optical(template='des', atmo_type='Kolmogorov', gsparams='star')
    assert model.gsparams == gsp_star
    model = piff.Optical(template='des_donut', gsparams='donut')
    assert model.gsparams == gsp_donut
    model = piff.Optical(template='des_donut', minimum_fft_size=128, folding_threshold=0.005)
    assert model.gsparams == gsp_donut
    with np.testing.assert_raises(ValueError):
        piff.Optical(gsparams='invalid')
    with np.testing.assert_raises(ValueError):
        piff.Optical(gsparams='donut', minimum_fft_size=128)

@timer
def test_draw():

    # setup model
    model = piff.Optical(template='des',atmo_type='Kolmogorov')

    # get a star to use for model star
    config = {
                'dir' : 'input',
                'image_file_name' : 'DECam_00241238_01.fits.fz',
                'cat_file_name' : 'DECam_00241238_01_cat.fits',
                'cat_hdu' : 2,
                'x_col' : 'XWIN_IMAGE',
                'y_col' : 'YWIN_IMAGE',
        }
    logger = piff.config.setup_logger(verbose=0)
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    astar = stars[10]

    # test draw
    astar.fit.params = model.kwargs_to_params(zernike_coeff=[0.,0.,0.,0.,0.2],r0=0.12,L0=10.,g1=0.01,g2=-0.02)
    astar1 = model.draw(astar)

@timer
def test_pupil_im(pupil_plane_file='input/DECam_pupil_512uv.fits'):
    print('test pupil im: ', pupil_plane_file)
    # make sure we can load up a pupil image
    model = piff.Optical(diam=4.010, lam=500., pupil_plane_im=pupil_plane_file,
                         atmo_type='Kolmogorov')
    test_optical(model)
    # make sure we really loaded it
    pupil_plane_im = galsim.fits.read(pupil_plane_file)
    # Check the scale (and fix if necessary)
    print('pupil_plane_im.scale = ',pupil_plane_im.scale)
    ref_aper = galsim.Aperture(diam=4.010, lam=500, pupil_plane_im=pupil_plane_im.array)
    ref_aper._load_pupil_plane()
    print('scale should be ',ref_aper.pupil_plane_scale)
    if pupil_plane_im.scale != ref_aper.pupil_plane_scale:
        print('fixing scale')
        pupil_plane_im.scale = ref_aper.pupil_plane_scale
        pupil_plane_im.write(pupil_plane_file)

    np.testing.assert_array_equal(pupil_plane_im.array, model.aperture._pupil_plane_im.array)

    print('Try diam=2 model')
    model = piff.Optical(pupil_plane_im=pupil_plane_im, diam=2, lam=500)
    model_pupil_plane_im = model.opt_kwargs['pupil_plane_im']
    np.testing.assert_array_equal(pupil_plane_im.array, model_pupil_plane_im.array)


@timer
def test_kolmogorov():
    print('test kolmogorov')

    # make sure if we put in different kolmogorov things that things change
    model = piff.Optical(template='des',atmo_type='Kolmogorov')
    params = model.kwargs_to_params(r0=0.1)
    star = make_empty_star(params=params)
    star1 = model.draw(star)

    params = model.kwargs_to_params(r0=0.2)
    star = make_empty_star(params=params)
    star2 = model.draw(star)

    chi2 = np.std((star1.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'

    # Usually mirror_figure_im is a file name, but you can also pass in an already read image.
    mirror_figure_image = galsim.fits.read('input/DECam_236392_finegrid512_nm_uv.fits')
    model2 = piff.Optical(template='des', atmo_type='Kolmogorov',
                          mirror_figure_im=mirror_figure_image)
    prof1 = model.getProfile(zernike_coeff=[0,0,0,0,0], r0=0.15)
    prof2 = model2.getProfile(zernike_coeff=[0,0,0,0,0], r0=0.15)
    assert prof1 == prof2


@timer
def test_vonkarman():
    print('test VonKarman')

    # Like above, but using L0.
    model = piff.Optical(template='des',atmo_type='VonKarman')
    params = model.kwargs_to_params(r0=0.1,L0=10.)
    star = make_empty_star(params=params)
    star1 = model.draw(star)

    params = model.kwargs_to_params(r0=0.1,L0=20.)
    star = make_empty_star(params=params)
    star2 = model.draw(star)

    chi2 = np.std((star1.image - star2.image).array)
    assert chi2 != 0,'chi2 is zero!?'

    model = piff.Optical(diam=4, lam=700, atmo_type='VonKarman')
    with np.testing.assert_raises(ValueError):
        # VonKarman requires L0
        model.getProfile(r0=0.15)
    with np.testing.assert_raises(ValueError):
        # VonKarman requires L0 != 0
        model.getProfile(r0=0.15, L0=0)

    atm = model.getAtmosphere(r0=0.15, L0=10)
    assert isinstance(atm, galsim.VonKarman)

    # Check other atmo_types here too.
    atm = piff.Optical(template='des', atmo_type=None).getAtmosphere(r0=0.15)
    assert isinstance(atm, galsim.DeltaFunction)
    atm = piff.Optical(template='des', atmo_type='None').getAtmosphere(r0=0.15)
    assert isinstance(atm, galsim.DeltaFunction)
    with np.testing.assert_raises(ValueError):
        piff.Optical(template='des', atmo_type='invalid').getAtmosphere(r0=0.15)


@timer
def test_shearing():
    print('test shearing')
    # make sure if we put in common mode ellipticities that things change
    g1 = -0.075
    g2 = 0.05
    model = piff.Optical(template='des',atmo_type='Kolmogorov',sigma=0.0)
    params = model.kwargs_to_params(r0=0.1,g1=g1,g2=g2)
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
    params = model.kwargs_to_params(r0=r0,g1=g1,g2=g2)
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
    assert model.gsparams == model2.gsparams
    for key in model.other_kwargs:
        assert key in model2.other_kwargs, 'key %r missing from model2 other_kwargs'%key
        assert model.other_kwargs[key] == model2.other_kwargs[key], 'key %r mismatch'%key
    assert model.atmo_type == model2.atmo_type,'atmo_type mismatch'

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


if __name__ == '__main__':
    test_init()
    test_optical()
    test_draw()
    test_pupil_im(pupil_plane_file='input/DECam_pupil_512uv.fits')
    test_pupil_im(pupil_plane_file='input/DECam_pupil_128uv.fits')
    test_kolmogorov()
    test_vonkarman()
    test_shearing()
    test_gaussian()
    test_disk()
