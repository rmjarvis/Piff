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
import galsim
import piff
import numpy as np
import os, sys
import fitsio
import copy
import unittest
import itertools

from piff_test_helper import timer

decaminfo = piff.des.DECamInfo()
def make_star(x, y, chipnum, properties={}, **kwargs):
    wcs = decaminfo.get_nominal_wcs(chipnum)
    properties_in = {'chipnum': chipnum}
    properties_in.update(properties)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=25, properties=properties_in,
                                **kwargs)
    return star

def plot_star(star, filename='test_optatmo.png', **kwargs):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.image.array, **kwargs)
    plt.colorbar()
    plt.savefig(filename)
    plt.close('all')
    plt.close('All')
    plt.close()

# default config
def return_config():
    return {
        'type': 'OptAtmo',
        'optical_psf_kwargs': {
            'template': 'des',
        },
        'reference_wavefront': {
            'type': 'DECamWavefront',
            'file_name': './input/Science-20121120s1-v20i2.fits',
            'extname': 1,
            'n_neighbors': 40,
            'weights': 'distance',
            'algorithm': 'auto',
            'p': 2,
        },
        'n_optfit_stars': 0,
        'fov_radius': 4500.,
        'jmax_pupil': 11,
        'jmax_focal': 11,
        'min_optfit_snr': 0,
        'higher_order_reference_wavefront_file':
            './input/decam_2012-nominalzernike-protocol2.pickle',
        'random_forest_shapes_model_pickles_location': './input',
        'optatmo_psf_kwargs': {
            'fix_zPupil011': True
        },
        'atmo_interp': {
            'type': 'Polynomial',
            'order': 2,
        },
        'reference_wavefront_zernikes_list': list(range(4,12)),
        'higher_order_reference_wavefront_zernikes_list': list(range(12,38)),
        'atmosphere_model': 'kolmogorov',
    }

# default config, big r0
def return_config_big_r0():
    return {
        'type': 'OptAtmo',
        'optical_psf_kwargs': {
            'template': 'des_big_r0',
        },
        'reference_wavefront': {
            'file_name': './input/Science-20121120s1-v20i2.fits',
            'extname': 1,
            'n_neighbors': 40,
            'weights': 'distance',
            'algorithm': 'auto',
            'p': 2,
            'type': 'DECamWavefront',
        },
        'n_optfit_stars': 0,
        'fov_radius': 4500.,
        'jmax_pupil': 11,
        'jmax_focal': 11,
        'min_optfit_snr': 0,
        'higher_order_reference_wavefront_file':
            './input/decam_2012-nominalzernike-protocol2.pickle',
        'init_with_rf' : (sys.version_info > (3,0)),
        'random_forest_shapes_model_pickles_location': './input',
        'optatmo_psf_kwargs': {
            'fix_zPupil011': True
        },
        'atmo_interp': {
            'type': 'Polynomial',
            'order': 2,
        },
        'reference_wavefront_zernikes_list': list(range(4,12)),
        'higher_order_reference_wavefront_zernikes_list': list(range(12,38)),
        'atmosphere_model': 'kolmogorov',
    }

@timer
def test_init():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_init: Started')
    config = return_config()
    psf = piff.PSF.process(config, logger=logger)

    # test init with different optical template
    config = return_config()
    optical_psf_kwargs = {
        'obscuration': 0,
        'pupil_angle': '45 * galsim.degrees',
        'lam': 700,
        'diam': 4,
        'r0': 0.1,
        'strut_angle': '10 * galsim.degrees',
        'pad_factor': 0.2,
        'oversampling': 1.0
    }
    config['optical_psf_kwargs'] = optical_psf_kwargs
    config['kolmogorov_kwargs'] = {'r0': 0}
    config['atmo_interp'] = 'None'
    config['reference_wavefront'] = 'none'
    config['higher_order_reference_wavefront_file'] = 'None'
    psf = piff.PSF.process(config, logger=logger)

    assert psf.atmo_interp == None
    assert psf.reference_wavefront == None
    assert psf.higher_order_reference_wavefront == None
    assert 'r0' not in psf.kolmogorov_kwargs
    assert psf.optical_psf_kwargs['pad_factor'] == optical_psf_kwargs['pad_factor']
    assert psf.optical_psf_kwargs['oversampling'] == optical_psf_kwargs['oversampling']

@timer
def test_aberrations():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_aberrations: Started')
    config = return_config()
    psf = piff.PSF.process(config)
    star = make_star(100, 100, 1)
    for j in range(4, config['jmax_pupil']):
        params = np.zeros(config['jmax_pupil'] + 4)
        params[0] = 1 #leave r0 at 0.1 here
        params[3] = -1 #leave L0 at -1 here
        params[4 - 1 + 4] = 1. #leave defocus at 1.0 here
        params[j - 1 + 4] = 1.
        prof = psf.getProfile(star,params)
        new_star = psf.drawProfile(star, prof, params)
        # check the new_star fit params
        np.testing.assert_array_equal(params, new_star.fit.params)
        # make sure the image arrays changed
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, star.image.array,
                                 new_star.image.array)

@timer
def test_reference_wavefront():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
        jmaxs = list(range(4, 14)) + [41]
    else:
        logger = piff.config.setup_logger(verbose=1)
        jmaxs = [4, 10, 11, 12, 41]
    logger.info('Entering test_reference_wavefront')

    star = make_star(0, 0, 1)

    # check that each gives aberrations, and that they give the same aberrations up to their max
    # order
    params_psfs = []
    for jmax_pupil in jmaxs:
        config = return_config()
        config['jmax_pupil'] = jmax_pupil
        psf = piff.PSF.process(config)
        params_psfs.append(psf.getParams(star))
    for i, jmax in enumerate(jmaxs[:-1]):
        np.testing.assert_equal(params_psfs[i], params_psfs[i + 1][:jmax + 4])
    # the reference wavefront does 7 to 40, or zernikes 4 to 37, so make sure that the beyond 40
    # there are no terms
    np.testing.assert_equal(0, params_psfs[-1][41:])
    # nor below 4, except for the optical size we put in (1.0 by default) and L0 (-1)
    np.testing.assert_equal(-1., params_psfs[-1][3])
    np.testing.assert_equal(1., params_psfs[-1][4])
    np.testing.assert_equal(0, params_psfs[-1][:3])
    np.testing.assert_equal(0, params_psfs[-1][5:7])

    # test that jmax_pupil < 4 throws error
    try:
        config = return_config()
        config['jmax_pupil'] = 3
        psf = piff.PSF.process(config)
        assert False
    except ValueError:
        assert True
    # test that jmax_focal < 1 throws error
    try:
        config = return_config()
        config['jmax_focal'] = 0
        psf = piff.PSF.process(config)
        assert False
    except ValueError:
        assert True

    # make sure we cannot update the psf in the disallowed ranges
    uvs = [3, 900]
    for uv in uvs:
        try:
            psf._update_optatmopsf({'zPupil{0:03d}_zFocal001'.format(uv): 1})
            assert False
        except ValueError:
            assert True
    xys = [0, 900]
    for xy in xys:
        try:
            psf._update_optatmopsf({'zPupil005_zFocal{0:03d}'.format(xy): 1})
            assert False
        except ValueError:
            assert True

@timer
def test_atmo_interp_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_atmo_interp_fit')

    np.random.seed(12345)
    jmax_pupil = 5
    config = return_config()
    config['jmax_focal'] = 1
    config['jmax_pupil'] = jmax_pupil

    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {
        'size': 1.3,
        'g1': 0.02, 'g2': -0.03,
        'zPupil004_zFocal001': -1.0,
    }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    # make star
    nstars = 117
    chipnums = np.random.choice(range(1,63), nstars)
    icens = np.random.randint(100, 1024, nstars)
    jcens = np.random.randint(100, 1024, nstars)
    stars_blank = [make_star(i, j, chip) for i, j, chip in zip(icens, jcens, chipnums)]
    wcs = {}
    for i in np.unique(chipnums):
        wcs[i] = decaminfo.get_nominal_wcs(i)

    # get params
    params = psf.getParamsList(stars_blank)
    # add atmosphere term to fit
    atmo_size = -0.2
    atmo_g1 = 0.04
    atmo_g2 = -0.03
    params[:, 0] = atmo_size
    params[:, 1] = atmo_g1
    params[:, 2] = atmo_g2
    # draw the stars
    stars = []
    for star, param in zip(stars_blank, params):
        prof = psf.getProfile(star,param)
        # draw the star
        star = psf.drawProfile(star, prof, param)
        stars.append(star)

    # fit model with atmo_interp
    psf.fit_atmosphere(stars, logger=logger)

    # compare inputs work
    np.testing.assert_allclose(atmo_size, psf.atmo_interp.coeffs[0][0,0], rtol=1.e-5)
    np.testing.assert_allclose(atmo_g1, psf.atmo_interp.coeffs[1][0,0], rtol=1.e-5)
    np.testing.assert_allclose(atmo_g2, psf.atmo_interp.coeffs[2][0,0], rtol=1.e-5)

    # check that the others are 0
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[0].flatten()[1:], atol=1e-16)
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[1].flatten()[1:], atol=1e-16)
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[2].flatten()[1:], atol=1e-16)

    # enable atmosphere
    psf._enable_atmosphere = True

    # test with drawStar
    star_index = 3
    star = stars[star_index]
    star_fit = psf.drawStar(star)
    np.testing.assert_allclose(star_fit.fit.params, params[star_index], rtol=1.e-5)
    np.testing.assert_allclose(star_fit.image.array, star.image.array, rtol=1.e-5)

@timer
def test_fit():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    np.random.seed(12345)
    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11
    config['higher_order_reference_wavefront_file'] = None

    optatmo_psf_kwargs = {
        'size': 1.0,
        'g1': 0, 'g2': 0,
        'fix_size': False,
        'fix_g1': True, 'fix_g2': True,
        'zPupil004_zFocal001': -0.15,
        'zPupil005_zFocal001': 0.1,
        'zPupil006_zFocal001': 0.25,
        'zPupil007_zFocal001': -0.1,
        'zPupil008_zFocal001': 0.1,
        'zPupil009_zFocal001': 0.3,
        'zPupil010_zFocal001': -0.4,
        'zPupil011_zFocal001': 0.2,
        'fix_zPupil011_zFocal001': True,
    }  # avoid the defocus,astig,spherical -> negatives degeneracy by fixing spherical
    config['optatmo_psf_kwargs'] = copy.deepcopy(optatmo_psf_kwargs)
    config_draw = copy.deepcopy(config)
    optatmo_psf_kwargs_values = {
        'size': 0.8,
        'zPupil004_zFocal001': 0.2,
        'zPupil005_zFocal001': 0.3,
        'zPupil006_zFocal001': -0.2,
        'zPupil007_zFocal001': 0.2,
        'zPupil008_zFocal001': 0.4,
        'zPupil009_zFocal001': -0.25,
        'zPupil010_zFocal001': 0.2
    }
    config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
    psf_draw = piff.PSF.process(config_draw)
    psf_train = piff.PSF.process(copy.deepcopy(config))

    # make stars
    logger.info('Making Stars')
    nstars = 117
    chipnums = np.random.choice(range(1,63), nstars)
    icens = np.random.randint(0, 1024, nstars)
    jcens = np.random.randint(0, 2048, nstars)
    stars_blank = [make_star(i, j, chip) for i, j, chip in zip(icens, jcens, chipnums)]
    wcs = {}
    for i in np.unique(chipnums):
        wcs[i] = decaminfo.get_nominal_wcs(i)
    pointing = None
    true_optatmo_psf_kwargs = copy.deepcopy(optatmo_psf_kwargs)
    params = psf_draw.getParamsList(stars_blank)

    stars = []
    for star, param in zip(stars_blank, params):
        prof = psf_draw.getProfile(star,param) * 1e6
        star = psf_draw.drawProfile(star, prof, param)
        stars.append(star)

    # do fit
    test_modes = ['shape', 'pixel']
    for fit_optics_mode in test_modes:

        if fit_optics_mode == 'shape':
            mask_frac = 1.e-3  # shape mode can't handle much masking.
            tol = 0.01 # Even with this mask level, only accurate to half a percent or so.
        else:
            mask_frac = 0.05  # Do this second, since modifying weight array.
            tol = 1.e-3  # The pixel mode is essentially perfect, even with 5% masked.
        for star in stars:
            # Mask mask_frac of the pixels.
            # 1-mask_frac: *= 1, mask_frac: *= 0
            star.weight.array[:,:] *= np.random.binomial(1,1-mask_frac, star.weight.array.shape)

        psf_train = piff.PSF.process(copy.deepcopy(config))
        logger.info('Test fitting optics mode {0}'.format(fit_optics_mode))
        psf_train.fit_optics_mode = fit_optics_mode

        test_fraction = config.get('test_fraction', 0.2)
        num_test = int(test_fraction * len(stars))
        test_indx = np.random.choice(len(stars), num_test, replace=False)
        test_stars = []
        train_stars = []
        for star_i, star in enumerate(stars):
            if star_i in test_indx:
                test_stars.append(star)
            else:
                train_stars.append(star)

        psf_train.fit(train_stars, test_stars, wcs, pointing, logger=logger)
        logger.info('Fit results for mode {0}'.format(fit_optics_mode))
        logger.info('Parameter: Input, Fit, Input - Fit')
        for key in sorted(optatmo_psf_kwargs_values):
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            logger.info('{0}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit))
        for key in sorted(optatmo_psf_kwargs_values):
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            assert np.isclose(fit, train, rtol=tol),\
                    'failed to fit {0} to tolerance {4}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(
                            key, train, fit, train - fit, tol)

@timer
def test_atmo_model_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_atmo_model_fit')
    config = return_config_big_r0()
    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {
        'size': 0.8,
        'g1': 0.01, 'g2': -0.01,
        'zPupil004_zFocal001': -1.0,
        'zPupil005_zFocal001': 1.0,
        'zPupil006_zFocal001': -1.0,
    }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)

    # make star
    star = make_star(0, 0, 1)

    # get params
    params = psf.getParams(star)

    # add additional flux, du, dv, scale, g1, g2 to profile
    atmo_flux = 1e1
    atmo_du = 0.3
    atmo_dv = -0.3
    atmo_size = -0.2
    atmo_g1 = 0.04
    atmo_g2 = 0.03
    params[0] = atmo_size
    params[1] = atmo_g1
    params[2] = atmo_g2

    prof = psf.getProfile(star,params).shift(atmo_du, atmo_dv) * atmo_flux

    # draw the star
    star = psf.drawProfile(star, prof, params)

    # fit star
    params_in = psf.getParams(star)
    star_fit = psf.fit_model(star, params=params_in, logger=logger)

    # check fitted params, centers, flux
    fit_flux = star_fit.fit.flux
    fit_du = star_fit.fit.center[0]
    fit_dv = star_fit.fit.center[1]
    arr_atmo = np.array([atmo_flux, atmo_du, atmo_dv, atmo_size, atmo_g1, atmo_g2])
    arr_fit = np.array([fit_flux, fit_du, fit_dv,
                        star_fit.fit.params[0], star_fit.fit.params[1], star_fit.fit.params[2]])
    np.testing.assert_allclose(arr_atmo, arr_fit, rtol=1e-5)
    assert len(star_fit.fit.params) == len(params_in)

    # also compare the shapes of the drawn fitted star and the original star
    shape, error = psf.measure_shape(star, logger=logger)
    # can't just draw with drawStar because we added the extra params outside of the getParams
    params_fit = psf.getParams(star_fit)
    params_fit[0] = star_fit.fit.params[0]
    params_fit[1] = star_fit.fit.params[1]
    params_fit[2] = star_fit.fit.params[2]
    prof_fit = psf.getProfile(star,params_fit)
    star_fit_drawn = psf.drawProfile(star_fit, prof_fit, params_fit)
    shape_drawn, error_drawn = psf.measure_shape(star_fit_drawn, logger=logger)
    np.testing.assert_allclose(shape, shape_drawn, rtol=1e-4)
    np.testing.assert_allclose(error, error_drawn, rtol=1e-4)

    # Check reflux
    star = make_star(0, 0, 1)
    star.fit.flux = atmo_flux
    star.fit.center = (atmo_du, atmo_dv)
    star = psf.drawStar(star)
    star_reflux = psf.reflux(star)
    np.testing.assert_allclose(star_reflux.fit.center, star.fit.center, rtol=1e-5)
    np.testing.assert_allclose(star_reflux.fit.flux, star.fit.flux, rtol=1e-3)

@timer
def test_jmaxs():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_jmaxs: Started')

    config = return_config()
    config['jmax_pupil'] = 21
    config['jmax_focal'] = 45
    psf = piff.PSF.process(config)
    # 0 and 1 share u, 0 and 2 share v
    stars = [make_star(100, 100, 1), make_star(100, 100, 60), make_star(100, 100, 3)]
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)

    assert psf.jmax_pupil == config['jmax_pupil']
    assert psf.jmax_focal == config['jmax_focal']
    assert psf._noll_coef_field.shape[2] == config['jmax_focal']
    assert psf._coef_arrays_field.shape[0] == config['jmax_pupil']
    assert np.shape(aberrations_pupil) == (len(stars), config['jmax_pupil'])

    # test that constant terms lead to all stars getting same aberrations from field
    optatmo_psf_kwargs = {'size': 0.2, 'g1': 0.3, 'zPupil004_zFocal001': -1.0}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    assert aberrations_pupil[0][3] == optatmo_psf_kwargs['zPupil004_zFocal001']
    assert aberrations_pupil[1][3] == aberrations_pupil[0][3]
    assert aberrations_pupil[2][3] == aberrations_pupil[0][3]
    # similarly with size
    assert aberrations_pupil[0][0] == optatmo_psf_kwargs['size']
    assert aberrations_pupil[1][0] == aberrations_pupil[0][0]
    assert aberrations_pupil[2][0] == aberrations_pupil[0][0]
    # similarly with g1
    assert aberrations_pupil[0][1] == optatmo_psf_kwargs['g1']
    assert aberrations_pupil[1][1] == aberrations_pupil[0][1]
    assert aberrations_pupil[2][1] == aberrations_pupil[0][1]

    # we can also test that the linear terms are proportional
    us = np.array([star.u for star in stars])
    vs = np.array([star.v for star in stars])
    optatmo_psf_kwargs = {'zPupil006_zFocal002': -2}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,5]
    # zFocal002 and the u coordinate are the same, so same u should lead to same z
    assert us[0] == us[1]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[1]
    assert us[0] != us[2]
    assert zs[0] != zs[2]
    optatmo_psf_kwargs = {'zPupil007_zFocal003': -2}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,6]
    # zFocal003 and the v coordinate are the same, so same u should lead to same z
    assert vs[0] == vs[2]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[2]
    assert vs[0] != vs[1]
    assert zs[0] != zs[1]

    # test that modifying a specific key ...actually modifies it
    optatmo_psf_kwargs = {
        'size': 0.2,
        'g1': 0.3, 'g2': 0.4,
        'zPupil021_zFocal045': 0.5,
        'zPupil004_zFocal001': 1.0
    }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    assert psf.aberrations_field[0, 0] == optatmo_psf_kwargs['size']
    assert psf.aberrations_field[1, 0] == optatmo_psf_kwargs['g1']
    assert psf.aberrations_field[2, 0] == optatmo_psf_kwargs['g2']
    assert psf.aberrations_field[21-1, 45-1] == optatmo_psf_kwargs['zPupil021_zFocal045']
    assert psf.aberrations_field[4-1, 1-1] == optatmo_psf_kwargs['zPupil004_zFocal001']

@timer
def test_snr_and_shapes():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_snr_and_shapes')

    np.random.seed(12345)
    config = return_config()
    config.pop('reference_wavefront')
    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {'size': 1.2, 'g1': 0.05, 'g2': -0.05}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    star = make_star(500, 500, 25)

    # draw stars, add noise, check shapes and errors
    Nsamples = 500
    # test for two levels of SNR
    for snr in [50]:
        flux = snr ** 2
        shapes = []
        errors = []
        snrs = []

        star_model = psf.drawStar(star)
        star_model.fit.flux = flux
        image = star_model.image.array * flux
        weight = 1. / image
        star.image.array[:] = image
        star.weight.array[:] = weight
        for i in range(Nsamples):
            noise = np.random.normal(size=image.shape, scale=np.sqrt(image))
            image_obs = image + noise
            star_model.image.array[:] = image_obs
            # any stars with -ve image are considered masked
            star_model.weight.array[:] = weight

            shape, error = psf.measure_shape(star_model, return_error=True, logger=logger)
            # make sure shape without error is the same value
            shape_no_error = psf.measure_shape(star_model, return_error=False, logger=logger)
            np.testing.assert_equal(shape, shape_no_error)
            shapes.append(shape)
            errors.append(error)

            snrs.append(piff.util.measure_snr(star_model))

        # not particularly concerned with flux, du, dv
        snrs = np.array(snrs)
        shapes = np.array(shapes)
        errors = np.array(errors)
        std_shapes = shapes.std(axis=0)
        mean_errors = errors.mean(axis=0)
        print('mean_shapes = ',np.mean(shapes, axis=0))
        print('std_shapes = ',std_shapes)
        print('mean_errors = ',mean_errors)
        print('ratio = ',mean_errors/std_shapes)
        # let's get the SNR back to within 10
        np.testing.assert_allclose(snrs, snr, atol=10)
        # Note: the above goal had to be loosened to 30 percent
        np.testing.assert_allclose(std_shapes, mean_errors, rtol=0.3)


@timer
def test_profile():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    config = return_config()
    psf = piff.PSF.process(config, logger=logger)
    star = make_star(500, 500, 25)
    # get getProfile vs getProfile(params)
    params = psf.getParams(star)
    prof = psf.getProfile(star,params)

    params_list_0 = psf.getParamsList([star])[0]
    prof_list = psf.getProfile(star,params_list_0)
    assert prof == prof_list

    star_drawstar = psf.drawStar(star)
    image = star_drawstar.image

    star_drawprofilestar = psf.drawProfile(star, prof, params)
    image_drawprofilestar = star_drawprofilestar.image
    assert image == image_drawprofilestar

    star_drawstarlist = psf.drawStarList([star])[0]
    image_drawstarlist = star_drawstarlist.image
    assert image == image_drawstarlist

    # NOTE: Not sure if I want to keep this functionality
    # also make sure that if the aberrations are all 0, then doesn't even bother convolving
    # opticalpsf


@timer
def test_random_forest():
    if sys.version_info < (3,0):
        print('Cannot use random forest on python2.7.  Skip this test')
        return

    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=0)
    logger.info('Testing random_forest')
    # fit the test ccd
    image_file = './input/DECam_00241238_01.fits.fz'
    cat_file = './input/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits'
    orig_image = galsim.fits.read(image_file)
    psf_file = os.path.join('output','optatmo_des_psf.fits')
    config_psf = return_config()
    config_psf['atmo_interp'] = 'none'
    config_psf['fit_optics_mode'] = 'random_forest'
    config_psf['n_fit_size_steps'] = 15
    config_psf['jmax_focal'] = 1
    config_psf['jmax_pupil'] = 4
    config = {
        'input': {
            'image_file_name' : image_file,
            'image_hdu' : 1,
            'weight_hdu' : 3,
            'badpix_hdu' : 2,
            'cat_file_name' : cat_file,
            'cat_hdu' : 2,
            'x_col' : 'XWIN_IMAGE',
            'y_col' : 'YWIN_IMAGE',
            'sky_col' : 'BACKGROUND',
            'stamp_size' : 19,
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            'nstars': 117,
            'min_snr': 40,
            'max_snr': 100,
        },
        'output': {'file_name': psf_file,},
        'psf': config_psf,
    }

    # run using piffify
    if os.path.exists(psf_file):
        os.remove(psf_file)
    logger.info('Running piffify')
    piff.piffify(copy.deepcopy(config), logger=logger)


@timer
def test_roundtrip():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=0)
    logger.info('Testing roundtrip')
    # fit the test ccd
    image_file = './input/DECam_00241238_01.fits.fz'
    cat_file = './input/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits'
    orig_image = galsim.fits.read(image_file)
    psf_file = os.path.join('output','optatmo_des_psf.fits')
    config_psf = return_config()
    config_psf['jmax_focal'] = 1
    config_psf['n_fit_size_steps'] = 15
    config_psf['jmax_pupil'] = 4
    config = {
        'input': {
            'image_file_name' : image_file,
            'image_hdu' : 1,
            'weight_hdu' : 3,
            'badpix_hdu' : 2,
            'cat_file_name' : cat_file,
            'cat_hdu' : 2,
            'x_col' : 'XWIN_IMAGE',
            'y_col' : 'YWIN_IMAGE',
            'sky_col' : 'BACKGROUND',
            'stamp_size' : 19,
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            'nstars': 117,
            'min_snr': 40,
            'max_snr': 100,
        },
        'output': {'file_name': psf_file,},
        'psf': config_psf,
    }

    # run using piffify
    if os.path.exists(psf_file):
        os.remove(psf_file)
    logger.info('Running piffify')
    piff.piffify(copy.deepcopy(config), logger=logger)

    # load results and compare with initial params
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf = piff.read(psf_file, logger=logger)

    # check that we enable atmosphere
    assert psf._enable_atmosphere
    # and that it isn't enabled if we haven't fit yet
    assert not psf_original._enable_atmosphere

    # check that the cache is gone
    assert not psf._caches
    assert not psf_original._caches

    # go through kwargs and check
    for key in psf_original.kwargs:
        #print("key: {0}".format(key))
        #print("psf_original.kwargs[key]: {0}".format(psf_original.kwargs[key]))
        #print("psf.kwargs[key]: {0}".format(psf.kwargs[key]))
        assert np.all(psf_original.kwargs[key] == psf.kwargs[key])

    # check the other named attributes
    np.testing.assert_allclose(psf_original._shape_weights, psf._shape_weights)
    np.testing.assert_allclose(psf_original._max_shapes, psf._max_shapes)
    assert psf_original.gsparams == psf.gsparams
    assert psf.kolmogorov_kwargs == psf_original.kolmogorov_kwargs
    assert psf.optical_psf_kwargs == psf_original.optical_psf_kwargs

    """
    # check analytic coefs
    for ac1, ac2 in zip(psf.analytic_coefs, psf_original.analytic_coefs):
        for c1, c2 in zip(ac1, ac2):
            np.testing.assert_allclose(c1, c2)
    """

    # copy these over to facilitate later writing for these tests
    stars = psf.stars
    wcs = psf.wcs
    pointing = psf.pointing
    atmo_interp = psf.atmo_interp

    if os.path.exists(psf_file):
        os.remove(psf_file)
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf_original.stars = stars
    psf_original.wcs = wcs
    psf_original.pointing = pointing
    psf_original.atmo_interp = atmo_interp
    psf_original.write(psf_file, logger=logger)
    psf = piff.read(psf_file, logger=logger)
    for key in psf_original.kwargs:
        assert np.all(psf_original.kwargs[key] == psf.kwargs[key])

    # test saving when there is no atmo_interp
    if os.path.exists(psf_file):
        os.remove(psf_file)
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf_original.atmo_interp = None
    psf_original.stars = stars
    psf_original.wcs = wcs
    psf_original.pointing = pointing
    psf_original.write(psf_file, logger=logger)
    psf = piff.read(psf_file, logger=logger)
    assert psf.atmo_interp == None
    assert psf._enable_atmosphere == False

    # test saving with no reference_wavefronts
    config_psf.pop('reference_wavefront')
    config_psf.pop('higher_order_reference_wavefront_file')
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf_original.stars = stars
    psf_original.wcs = wcs
    psf_original.pointing = pointing
    psf_original.atmo_interp = atmo_interp
    psf_original.write(psf_file, logger=logger)
    psf = piff.read(psf_file, logger=logger)
    for key in psf_original.kwargs:
        assert np.all(psf_original.kwargs[key] == psf.kwargs[key])
    assert psf_original.reference_wavefront == None
    assert psf_original.higher_order_reference_wavefront == None
    assert psf.reference_wavefront == psf_original.reference_wavefront
    assert psf.higher_order_reference_wavefront == psf_original.higher_order_reference_wavefront
    # check that this psf does nothing when we create cache
    psf._create_caches([], logger=logger)
    assert psf._caches == False
    assert psf._aberrations_reference_wavefronts == None

@timer
def test_moments():
    """Test piff.util_calculate_moments()
    """
    # First test simple sheared Gaussian.  Everything is easy to compute analytically.
    # Especially for no shear, so do that first.
    gal = galsim.Gaussian(sigma=1.7, flux=23.)
    image = galsim.Image(scale=0.12, ncol=256, nrow=256)
    gal.drawImage(image, offset=(-1.2, 0.8), method='no_pixel')
    weight = image.copy()
    weight.fill(0.15)
    star = piff.Star.makeTarget(x=123, y=234, image=image, weight=weight)
    data, weight, u, v = star.data.getDataVector(include_zero_weight=True)

    moments = star.calculate_moments()
    # M00, M10, M01, M11, M20, M02
    # aka flux, u0, v0, e0, e1, e2
    flux, u0, v0, e0, e1, e2 = moments

    # Note: offset does not affect any moments
    print('flux = ',flux, gal.flux / (4*np.pi*gal.sigma**2))
    assert np.isclose(flux, gal.flux / (4*np.pi*gal.sigma**2), rtol=1.e-5)
    assert np.isclose(u0, 0., atol=1.e-8)
    assert np.isclose(v0, 0., atol=1.e-8)
    assert np.isclose(e0, gal.sigma**2, rtol=1.e-5)
    assert np.isclose(e1, 0., atol=1.e-8)
    assert np.isclose(e2, 0., atol=1.e-8)

    moments = star.calculate_moments(third_order=True)
    np.testing.assert_allclose(moments[6:], 0., atol=1.e-8)

    moments = star.calculate_moments(fourth_order=True)
    assert np.isclose(moments[6], 2.*gal.sigma**4, rtol=1.e-5)
    np.testing.assert_allclose(moments[7:], 0., atol=1.e-6)

    moments = star.calculate_moments(radial=True)
    assert np.isclose(moments[6], 2, rtol=1.e-5)
    assert np.isclose(moments[7], 6, rtol=1.e-5)
    assert np.isclose(moments[8], 24., rtol=1.e-5)

    # Now sheared.
    # The math is a bit unwieldy for these.
    # I used Maple to calculate the expected values for a sheared Gaussian.
    shear = galsim.Shear(e1=0.29, e2=-0.14)
    gal.shear(shear).drawImage(image, offset=(-1.2, 0.8), method='no_pixel')
    moments = star.calculate_moments()
    flux, u0, v0, e0, e1, e2 = moments

    assert np.isclose(flux, gal.flux / (4*np.pi*gal.sigma**2), rtol=1.e-5)
    assert np.isclose(u0, 0., atol=1.e-8)
    assert np.isclose(v0, 0., atol=1.e-8)
    true_M11 = gal.sigma**2 * (1+shear.g**2)/(1-shear.g**2)
    assert np.isclose(e0, true_M11, rtol=1.e-5)
    assert np.isclose(e1/e0, shear.e1, rtol=1.e-5)
    assert np.isclose(e2/e0, shear.e2, rtol=1.e-5)

    moments = star.calculate_moments(third_order=True, fourth_order=True, radial=True)
    np.testing.assert_allclose(moments[6:10], 0., atol=1.e-6)

    true_M22 = 2.*gal.sigma**4*(1+4*shear.g**2+shear.g**4)/(1.-shear.g**2)**2
    assert np.isclose(moments[10], true_M22)
    true_M31 = 6.*gal.sigma**4*shear.g1*(1+shear.g**2)/(1.-shear.g**2)**2
    assert np.isclose(moments[11], true_M31)
    true_M13 = 6.*gal.sigma**4*shear.g2*(1+shear.g**2)/(1.-shear.g**2)**2
    assert np.isclose(moments[12], true_M13)
    true_M40 = 12.*gal.sigma**4*(shear.g1**2-shear.g2**2)/(1.-shear.g**2)**2
    assert np.isclose(moments[13], true_M40)
    true_M04 = 24.*gal.sigma**4*shear.g1*shear.g2/(1.-shear.g**2)**2
    assert np.isclose(moments[14], true_M04)

    assert np.isclose(moments[15], true_M22/true_M11**2, rtol=1.e-5)
    true_M33 = 6.*gal.sigma**6*(1+9*shear.g**2+9*shear.g**4+shear.g**6)/(1.-shear.g**2)**3
    assert np.isclose(moments[16], true_M33/true_M11**3, rtol=1.e-5)
    true_M44 = 24.*gal.sigma**8*(1+16*shear.g**2+36*shear.g**4+16*shear.g**6+shear.g**8)/(1.-shear.g**2)**4
    assert np.isclose(moments[17], true_M44/true_M11**4, rtol=1.e-5)

    # Check that we can get back to Ares's results.
    values = piff.util.calculate_moments(star, third_order=True, radial=True)
    shape = np.array(values)

    from old_moments import hsm_orthogonal
    values2 = hsm_orthogonal(star)
    shape2 = np.array(values2)

    hsm = piff.util.hsm(star)
    shape[0] *= hsm[0] * image.scale**2 * np.mean(weight)
    shape[1] += hsm[1]
    shape[2] += hsm[2]
    shape[3:] *= 2

    print(shape)
    print(shape2)
    print(shape/shape2)

    np.testing.assert_allclose(shape, shape2, rtol=1.e-5, atol=1.e-7)


@timer
def test_moments_errors():
    """Test piff.util_calculate_moments(errors=True)
    """
    # First test simple sheared Gaussian.  Everything is easy to compute analytically.
    # Especially for no shear, so do that first.
    gal = galsim.Gaussian(sigma=1.7, flux=39233.)
    image = galsim.Image(scale=0.32, ncol=128, nrow=128)
    shear = galsim.Shear()
    gal.shear(shear).drawImage(image, offset=(-1.2, 0.8), method='no_pixel')

    var = 2.9  # Variance per pixel, about 1/10 of maximum pixel value.
    weight = image.copy()
    weight.fill(1./var)
    noise = galsim.GaussianNoise(sigma=np.sqrt(var), rng=galsim.BaseDeviate(12345))
    noisy_image = image.copy()  # We will repeatedly add noise to this one.

    if __name__ == '__main__':
        num_iter = 10000
        rtol = 0.05
    else:
        num_iter = 100
        rtol = 0.2
    moments = np.empty((num_iter, 18))
    for i in range(num_iter):
        if i > 0 and i % 100 == 0: print(i)
        while True:
            try:
                noisy_image.copyFrom(image)
                noisy_image.addNoise(noise)
                star = piff.Star.makeTarget(x=123, y=234, image=noisy_image, weight=weight)
                moments[i,:] = star.calculate_moments(third_order=True, fourth_order=True,
                                                      radial=True)
            except RuntimeError as e:
                print('caught ',e)
                pass
            else:
                break

    # First check the hsm results.
    mean_shape = np.mean(moments, axis=0)
    meas_var = np.var(moments, axis=0)
    print('mean_shape = ',mean_shape)

    errors = star.calculate_moments(third_order=True, fourth_order=True, radial=True, errors=True)
    values = errors[:18]
    errors = errors[18:]

    # Check that we are currently matching the old hsm_* routines.
    # Have to do a bit of conversion to get to old definitions.
    hsm = piff.util.hsm(star)
    pix_area = star.data.image.wcs.local(star.data.image.center).pixelArea()
    alt_values = np.array(values)
    alt_values[0] *= hsm[0] * pix_area * star.data.weight.array.mean()
    alt_values[1] += hsm[1]
    alt_values[2] += hsm[2]
    alt_values[3:] *= 2
    alt_errors = np.array(errors)
    alt_errors[0] *= (hsm[0] * pix_area * star.data.weight.array.mean())**2
    alt_errors[3:] *= 4
    alt_errors = np.sqrt(alt_errors)

    # Check hsm_error: (Just first 6 moments)
    from old_moments import hsm_error
    old_errors = hsm_error(star)
    old_values = old_errors[:6]
    old_errors = old_errors[6:]
    np.testing.assert_allclose(alt_values[:6], old_values, rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[:6], old_errors, rtol=1.e-4, atol=1.e-7)

    # Check hsm_third_moments and hsm_error_third_moments:
    from old_moments import hsm_third_moments, hsm_error_third_moments
    old_values = hsm_third_moments(star)
    old_errors = hsm_error_third_moments(star)
    np.testing.assert_allclose(alt_values[:10], old_values, rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[:10], old_errors, rtol=1.e-4, atol=1.e-7)

    # Check hsm_fourth_moments and hsm_error_fourth_moments:
    from old_moments import hsm_fourth_moments, hsm_error_fourth_moments
    old_values = hsm_fourth_moments(star)
    old_errors = hsm_error_fourth_moments(star)
    np.testing.assert_allclose(alt_values[:15], old_values, rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[:15], old_errors, rtol=1.e-4, atol=1.e-7)

    # Check hsm_orthogonal and hsm_error_orthogonal
    from old_moments import hsm_orthogonal, hsm_error_orthogonal
    old_values = hsm_orthogonal(star)
    old_errors = hsm_error_orthogonal(star)
    np.testing.assert_allclose(alt_values[:10], old_values[:10], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_values[15], old_values[10], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_values[16], old_values[11], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_values[17], old_values[12], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[:10], old_errors[:10], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[15], old_errors[10], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[16], old_errors[11], rtol=1.e-4, atol=1.e-7)
    np.testing.assert_allclose(alt_errors[17], old_errors[12], rtol=1.e-4, atol=1.e-7)

    # The following tests are the ones we really do want to work, since they actually
    # check that the errors match the empirical values.
    # However, with the fudge factors from the old code, needed to make the above tests work,
    # the following tests fail.

    # The centroid errors are way over estimated of course, since they are essentially
    # set to zero by construction.  We don't try to get the error on this process correct.
    # So just check the other moments.
    print('estimated var = ',errors)
    print('measured var = ',meas_var)
    print('ratio = ',meas_var/errors)
    np.testing.assert_allclose(errors[0], meas_var[0], rtol=rtol)
    np.testing.assert_allclose(errors[3:], meas_var[3:], rtol=rtol)


if __name__ == '__main__':
    #import cProfile, pstats
    #pr = cProfile.Profile()
    #pr.enable()
    #test_init()
    #test_aberrations()
    #test_reference_wavefront()
    #test_jmaxs()
    #test_random_forest()
    #test_atmo_model_fit()
    #test_atmo_interp_fit()
    #test_profile()
    #test_snr_and_shapes()
    #test_roundtrip()
    #test_fit()
    test_moments()
    #test_moments_errors()
    #pr.disable()
    #pr.dump_stats('prof.out')
    #ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(20)
