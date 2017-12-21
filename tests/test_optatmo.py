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
import os
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
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=24, properties=properties_in, **kwargs)
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
    config = {  'optical_psf_kwargs':
                {
                    'template': 'des',
                },
            'reference_wavefront':
                {
                    'file_name': './input/Science-20121120s1-v20i2.fits',
                    'extname': 1,
                    'n_neighbors': 40,
                    'weights': 'distance',
                    'algorithm': 'auto',
                    'p': 2,
                    'type': 'DECamWavefront',
                },
            'shape_weights': [0.5, 1, 1],
            'shape_method': 'hsm',
            'shape_unnormalized': True,
            'fov_radius': 1. * 60 * 60,
            'jmax_pupil': 11,
            'jmax_focal': 11,
            'n_optfit_stars': 0,
            'min_optfit_snr': 0,
            'optfit_optimize': 'analytic',
            'optatmo_psf_kwargs':
                {
                },
            'analytic_coefs': './input/analytic_coefs_hsm.npy',
            'atmo_interp':
                {
                    'type': 'Polynomial',
                    'order': 2,
                },

            'type': 'OptAtmo',
         }
    return copy.deepcopy(config)

@timer
def test_init():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_init: Started')
    config = return_config()
    psf = piff.PSF.process(config, logger=logger)
    return psf

@timer
def test_aberrations():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_init: Started')
    config = return_config()
    psf = piff.PSF.process(config)
    star = make_star(100, 100, 1)
    for j in range(4, config['jmax_pupil']):
        params = np.zeros(config['jmax_pupil'])
        params[0] = 1
        params[4 - 1] = 1.
        params[j - 1] = 1.
        prof = psf._profile(params)
        new_star = psf.drawProfileStar(star, prof, params)
        # check the new_star fit params
        np.testing.assert_array_equal(params, new_star.fit.params)
        # make sure the image arrays changed
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, star.image.array, new_star.image.array)

@timer
def test_reference_wavefront():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_reference_wavefront')

    star = make_star(0, 0, 1)

    # check that each gives aberrations, and that they give the same aberrations up to their max order
    params_psfs = []
    jmaxs = [5, 11, 41]
    for jmax_pupil in jmaxs:
        config = return_config()
        config['jmax_pupil'] = jmax_pupil
        psf = piff.PSF.process(config)
        params_psfs.append(psf.getParams(star))
    np.testing.assert_equal(params_psfs[0], params_psfs[1][:jmaxs[0]])
    np.testing.assert_equal(params_psfs[1], params_psfs[2][:jmaxs[1]])
    # the reference wavefront does 4 to 11, so make sure that the beyond 11 there are no terms
    np.testing.assert_equal(0, params_psfs[2][11:])
    # nor below 4, except for the size we put in (1.0)
    np.testing.assert_equal(0, params_psfs[2][1:3])

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
            psf._update_optatmopsf({'zUV{0:03d}_zXY001'.format(uv): 1})
            assert False
        except ValueError:
            assert True
    xys = [0, 900]
    for xy in xys:
        try:
            psf._update_optatmopsf({'zUV005_zXY{0:03d}'.format(xy): 1})
            assert False
        except ValueError:
            assert True

@timer
def test_atmo_interp_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_atmo_interp_fit')

    jmax_pupil = 5
    config = return_config()
    config['jmax_focal'] = 1
    config['jmax_pupil'] = jmax_pupil

    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {'size': 1.3, 'g1': 0.02, 'g2': -0.03,
                        'zUV004_zXY001': -1.0,
                        }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    # make star
    nstars = 10
    chipnums = np.random.choice(range(1,63), nstars)
    icens = np.random.randint(100, 1024, nstars)
    jcens = np.random.randint(100, 1024, nstars)
    stars_blank = [make_star(i, j, chip) for i, j, chip in zip(icens, jcens, chipnums)]
    wcs = {}
    for i in np.unique(chipnums):
        wcs[i] = decaminfo.get_nominal_wcs(i)

    # get params
    params = psf.getParamsList(stars_blank)
    # add additional scale, g1, g2 to profile
    atmo_size = -0.2
    atmo_g1 = 0.04
    atmo_g2 = -0.03
    params[:, 0] += atmo_size
    params[:, 1] += atmo_g1
    params[:, 2] += atmo_g2
    # draw the stars
    stars_to_fit = []
    stars = []
    for star, param in zip(stars_blank, params):
        prof = psf._profile(param)
        # draw the star
        star = psf.drawProfileStar(star, prof, param)
        stars.append(star)
        star_to_fit = piff.Star(star.data, None)
        stars_to_fit.append(star_to_fit)

    # fit model with atmo_interp
    psf.fit_atmosphere(stars_to_fit, logger=logger)

    # compare inputs work
    np.testing.assert_allclose(atmo_size, psf.atmo_interp.coeffs[0][0,0])
    np.testing.assert_allclose(atmo_g1, psf.atmo_interp.coeffs[1][0,0])
    np.testing.assert_allclose(atmo_g2, psf.atmo_interp.coeffs[2][0,0])

    # check that the others are 0
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[0].flatten()[1:], atol=1e-16)
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[1].flatten()[1:], atol=1e-16)
    np.testing.assert_allclose(0, psf.atmo_interp.coeffs[2].flatten()[1:], atol=1e-16)

    # enable atmosphere
    psf._enable_atmosphere = True

    # test with drawStar
    star_index = 3
    star = stars_to_fit[star_index]
    star_fit = psf.drawStar(star)
    np.testing.assert_allclose(star_fit.fit.params, params[star_index])
    np.testing.assert_allclose(star_fit.image.array, star.image.array)

@timer
def test_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    jmax_pupil = 10
    config = return_config()
    config['jmax_focal'] = 1
    config['jmax_pupil'] = jmax_pupil

    psf = piff.PSF.process(copy.deepcopy(config))
    optatmo_psf_kwargs = {'size': 1., 'g1': 0, 'g2': 0,
                          'fix_size': True, 'fix_g1': True, 'fix_g2': True,
                          'zUV004_zXY001': 1.5,
                          'zUV005_zXY001': 0, 'fix_zUV005_zXY001': True,
                          'zUV006_zXY001': 0, 'fix_zUV006_zXY001': True,
                          'zUV007_zXY001': 0, 'fix_zUV007_zXY001': True,
                          'zUV008_zXY001': 0, 'fix_zUV008_zXY001': True,
                          'zUV009_zXY001': -1,
                          'zUV010_zXY001': 1,
                        }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    # make star
    nstars = 60
    chipnums = np.random.choice(range(1,63), nstars)
    icens = np.random.randint(0, 1024, nstars)
    jcens = np.random.randint(0, 2048, nstars)
    stars_blank = [make_star(i, j, chip) for i, j, chip in zip(icens, jcens, chipnums)]
    wcs = {}
    for i in np.unique(chipnums):
        wcs[i] = decaminfo.get_nominal_wcs(i)
    pointing = None

    # get params
    true_optatmo_psf_kwargs = copy.deepcopy(optatmo_psf_kwargs)
    params = psf.getParamsList(stars_blank)
    # add additional scale, g1, g2 to profile
    snr = 100
    atmo_flux = snr ** 2
    atmo_du = -0.1
    atmo_dv = 0.2
    # draw the stars
    logger.debug('Drawing test stars')
    stars_to_fit = []
    stars = []
    for star, param in zip(stars_blank, params):
        prof = psf._profile(param).shift(atmo_du, atmo_dv) * atmo_flux
        # draw the star
        star = psf.drawProfileStar(star, prof, param)
        # add noise
        image = star.image.array
        weight = 1. / image
        noise = np.random.normal(size=image.shape, scale=np.sqrt(image))
        star.image.array[:] = image + noise
        star.weight.array[:] = weight

        stars.append(star)
        star_to_fit = piff.Star(star.data, None)
        stars_to_fit.append(star_to_fit)

    if __name__ == '__main__':
        optfit_optimizers = ['moments', 'pixel', 'analytic']
        shape_methods = ['hsm', 'lmfit']
        shapes_unnormalized = [False, True]
    else:
        optfit_optimizers = ['moments']
        shape_methods = ['hsm']
        shapes_unnormalized = [True]

    for optfit_optimize, shape_method, shape_unnormalized in itertools.product(optfit_optimizers, shape_methods, shapes_unnormalized):
        logger.debug('test_fit: Fitting method {0}, shape method {1}, shape unnormalized {2}'.format(optfit_optimize, shape_method, shape_unnormalized))
        config['optfit_optimize'] = optfit_optimize
        config['shape_unnormalized'] = shape_unnormalized
        config['shape_method'] = shape_method
        if shape_method == 'lmfit' and optfit_optimize == 'analytic':
            # skip the lmfit analytic because we didn't copy those coefs over. The principal of the matter is tested with hsm analytic coefs
            continue
        elif optfit_optimize == 'analytic' and shape_method != 'hsm' and shape_unnormalized:
            continue
        if shape_unnormalized:
            config['analytic_coefs'] = './input/analytic_coefs_hsm.npy'
        else:
            config['analytic_coefs'] = './input/analytic_coefs_normalized_hsm.npy'

        config['n_optfit_stars'] = int(0.9 * nstars)
        psf_clean = piff.PSF.process(copy.deepcopy(config), logger=logger)

        # fix appropriate params
        for key in optatmo_psf_kwargs.keys():
            if 'fix_' in key:
                psf_clean.optatmo_psf_kwargs[key] = optatmo_psf_kwargs[key]

        psf_clean.fit(stars_to_fit, wcs, pointing, logger=logger)

        # plot star and true star
        for i, star in enumerate(psf_clean._model_fitted_stars[:5]):
            plot_star(star, 'test_star_{0}_truth.png'.format(i), vmin=0, vmax=150)
            star_model = psf_clean.drawStar(star, correct_flux_center=True)
            plot_star(star_model, 'test_star_{0}_drawn.png'.format(i), vmin=0, vmax=150)
            star_model.image.array[:] = star.image.array - star_model.image.array
            plot_star(star_model, 'test_star_{0}_zresidual.png'.format(i))



        g0_fit = psf_clean.aberrations_field[0,0] + psf_clean.atmo_interp.coeffs[0][0,0]
        g0_in = psf.aberrations_field[0,0]
        g1_fit = psf_clean.aberrations_field[1,0] + psf_clean.atmo_interp.coeffs[1][0,0]
        g1_in = psf.aberrations_field[1,0]
        g2_fit = psf_clean.aberrations_field[2,0] + psf_clean.atmo_interp.coeffs[2][0,0]
        g2_in = psf.aberrations_field[2,0]
        true_params = psf_clean._fit_optics_params(true_optatmo_psf_kwargs)
        true_chi_pixel = psf_clean._fit_optics_residual_pixel(true_params)
        true_chi_moments = psf_clean._fit_optics_residual_moments(true_params)
        true_chi_analytic = psf_clean._fit_optics_residual_analytic(true_params)
        fit_params = psf_clean._fit_optics_params(psf_clean.optatmo_psf_kwargs)
        fit_chi_pixel = psf_clean._fit_optics_residual_pixel(fit_params)
        fit_chi_moments = psf_clean._fit_optics_residual_moments(fit_params)
        fit_chi_analytic = psf_clean._fit_optics_residual_analytic(fit_params)
        chi2_ratio_pixel = np.sum(np.square(true_chi_pixel)) / np.sum(np.square(fit_chi_pixel))
        chi2_ratio_moments = np.sum(np.square(true_chi_moments)) / np.sum(np.square(fit_chi_moments))
        chi2_ratio_analytic = np.sum(np.square(true_chi_analytic)) / np.sum(np.square(fit_chi_analytic))

        # print(g0_fit, g0_in)
        # print(g1_fit, g1_in)
        # print(g2_fit, g2_in)
        # print(psf_clean.aberrations_field)
        # print(psf.aberrations_field)
        # print(chi2_ratio_pixel)
        # print(chi2_ratio_moments)
        # print(chi2_ratio_analytic)

        # import ipdb; ipdb.set_trace()

        # because of errors these may be slightly bigger than 1, but let's say no more than by 10%
        assert chi2_ratio_pixel < 1.1
        assert chi2_ratio_moments < 1.1
        # assert chi2_ratio_analytic < 1.1

        # check that n_optfit_stars restricted appropriately
        assert len(psf_clean._opt_stars) <= config['n_optfit_stars']

        # evaluate zernike terms
        np.testing.assert_allclose(psf.aberrations_field[3:,0], psf_clean.aberrations_field[3:,0], atol=1e-1)
        # spot check optatmo_psf_kwargs
        assert psf_clean.optatmo_psf_kwargs['size'] == psf_clean.aberrations_field[0, 0]
        assert psf_clean.optatmo_psf_kwargs['g2'] == psf_clean.aberrations_field[2, 0]
        assert psf_clean.optatmo_psf_kwargs['zUV004_zXY001'] == psf_clean.aberrations_field[3, 0]
        # note that when comparing the aberrations, because we put in a constant atmosphere, we actually expect the atmosphere interpolation to be zero, and the aberration field terms to be the sum of those pieces
        np.testing.assert_allclose(g0_fit, g0_in, atol=3e-2)
        np.testing.assert_allclose(g1_fit, g1_in, atol=3e-2)
        np.testing.assert_allclose(g2_fit, g2_in, atol=3e-2)

        if optfit_optimize in ['moments', 'analytic']:
            # check that changing the weights changes the chi2 of a given iteration
            pass

@timer
def test_atmo_model_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_atmo_model_fit')
    config = return_config()
    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {'size': 0.8, 'g1': 0.01, 'g2': -0.01,
                        'zUV004_zXY001': -1.0,
                        'zUV005_zXY001': 1.0,
                        'zUV006_zXY001': -1.0,
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
    params[0] += atmo_size
    params[1] += atmo_g1
    params[2] += atmo_g2

    prof = psf._profile(params).shift(atmo_du, atmo_dv) * atmo_flux

    # draw the star
    star = psf.drawProfileStar(star, prof, params)
    star_to_fit = piff.Star(star.data, None)

    # fit star
    params_in = psf.getParams(star)
    star_fit = psf._fit_model(star_to_fit, opt_params=params_in, vary_shape=True, logger=logger)

    # check fitted params, centers, flux
    fit_flux = star_fit.flux
    fit_du = star_fit.center[0]
    fit_dv = star_fit.center[1]
    arr_atmo = np.array([atmo_flux, atmo_du, atmo_dv, atmo_size, atmo_g1, atmo_g2])
    arr_fit = np.array([fit_flux, fit_du, fit_dv, star_fit.fit.params[0], star_fit.fit.params[1], star_fit.fit.params[2]])
    np.testing.assert_allclose(arr_atmo, arr_fit, rtol=1e-5)

    # also compare the shapes of the drawn fitted star and the original star
    shape, error = psf.measure_shape(star_to_fit, logger=logger)
    # can't just draw with drawStar because we added the extra params outside of the getParams
    params_fit = psf.getParams(star_fit)
    params_fit[0] += arr_fit[3]
    params_fit[1] += arr_fit[4]
    params_fit[2] += arr_fit[5]
    prof_fit = psf._profile(params_fit).shift(arr_fit[1], arr_fit[2]) * arr_fit[0]
    star_fit_drawn = psf.drawProfileStar(star_fit, prof_fit, params_fit)
    shape_drawn, error_drawn = psf.measure_shape(star_fit_drawn, logger=logger)
    np.testing.assert_allclose(shape, shape_drawn, rtol=1e-5)
    np.testing.assert_allclose(error, error_drawn, rtol=1e-5)

@timer
def test_jmaxs():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('test_jmaxs: Started')

    config = return_config()
    config['jmax_pupil'] = 21
    config['jmax_focal'] = 45
    psf = piff.PSF.process(config)
    stars = [make_star(100, 100, 1), make_star(100, 100, 60), make_star(100, 100, 3)]  # 0 and 1 share u, 0 and 2 share v
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)

    assert psf.jmax_pupil == config['jmax_pupil']
    assert psf.jmax_focal == config['jmax_focal']
    assert psf._noll_coef_field.shape[2] == config['jmax_focal']
    assert psf._coef_arrays_field.shape[0] == config['jmax_pupil']
    assert np.shape(aberrations_pupil) == (len(stars), config['jmax_pupil'])

    # test that constant terms lead to all stars getting same aberrations from field
    optatmo_psf_kwargs = {'size': 0.2, 'g1': 0.3, 'zUV004_zXY001': -1.0}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    assert aberrations_pupil[0][3] == optatmo_psf_kwargs['zUV004_zXY001']
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
    optatmo_psf_kwargs = {'zUV006_zXY002': -2}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,5]
    # zXY002 and the u coordinate are the same, so same u should lead to same z
    assert us[0] == us[1]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[1]
    assert us[0] != us[2]
    assert zs[0] != zs[2]
    optatmo_psf_kwargs = {'zUV007_zXY003': -2}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,6]
    # zXY003 and the v coordinate are the same, so same u should lead to same z
    assert vs[0] == vs[2]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[2]
    assert vs[0] != vs[1]
    assert zs[0] != zs[1]

    # test that modifying a specific key ...actually modifies it
    optatmo_psf_kwargs = {'size': 0.2, 'g1': 0.3, 'g2': 0.4, 'zUV021_zXY045': 0.5, 'zUV004_zXY001': 1.0}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    assert psf.aberrations_field[0, 0] == optatmo_psf_kwargs['size']
    assert psf.aberrations_field[1, 0] == optatmo_psf_kwargs['g1']
    assert psf.aberrations_field[2, 0] == optatmo_psf_kwargs['g2']
    assert psf.aberrations_field[21-1, 45-1] == optatmo_psf_kwargs['zUV021_zXY045']
    assert psf.aberrations_field[4-1, 1-1] == optatmo_psf_kwargs['zUV004_zXY001']

@timer
def test_analytic_coefs():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    config = return_config()
    psf = piff.PSF.process(config)
    # generate fake star params
    stars = [make_star(0, 0, i + 1) for i in range(1, 30)]
    params = psf.getParamsList(stars)
    # make sure the analytic shapes we loaded at least give something
    shapes = psf.analytic_shapes(params, psf.analytic_coefs)

    # put in fake params and coefs
    params = np.array([[1, 1, -1], [1, -1, 0], [-1, 0, 1]])
    indices = [np.array([[0, 0, 4,], [0, 1, 2]]),
               np.array([[0, 2, 3]])]
    coefs = [np.array([10, -10]), np.array([-10])]
    afterburner = np.array([[-1, 2], [1, 3]])
    analytic_coefs = [coefs, indices, afterburner]
    shapes = psf.analytic_shapes(params, analytic_coefs)
    # make sure I didn't break the cython code
    shapes_shouldbe = np.array([[-41., -29.], [-21., 31.], [-1., 1.]])
    np.testing.assert_equal(shapes, shapes_shouldbe)

    # remove afterburner
    shapes_shouldbe = np.array([[(xij - afterburner[j][0]) / afterburner[j][1] for j, xij in enumerate(xi)] for xi in shapes_shouldbe])
    afterburner = np.array([[0, 1], [0, 1]])
    analytic_coefs = [coefs, indices, afterburner]
    shapes = psf.analytic_shapes(params, analytic_coefs)
    np.testing.assert_equal(shapes, shapes_shouldbe)

@timer
def test_snr_and_shapes():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.info('Entering test_snr_and_shapes')
    config = return_config()
    config.pop('reference_wavefront')
    psf = piff.PSF.process(config)
    optatmo_psf_kwargs = {'size': 1.2, 'g1': 0.05, 'g2': -0.05}
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    star = make_star(500, 500, 25)

    # draw stars, add noise, check shapes and errors
    Nsamples = 300
    # test for two levels of SNR
    for snr in [60]:
        flux = snr ** 2
        shapes = [[[], []], [[], []]]
        errors = [[[], []], [[], []]]
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
            # star_model.weight.array[:] = np.where(image_obs < 0, 0, weight)
            star_model.weight.array[:] = weight

            # measure shape for various types
            for j, measure_shape in enumerate([psf.measure_shape_hsm, psf.measure_shape_lmfit]):
                for k, shape_unnormalized in enumerate([False, True]):
                    if j == 0 and k == 1:
                        logger_in = logger
                    else:
                        logger_in = None
                    logger_in = None
                    shape, error = measure_shape(star_model, shape_unnormalized=shape_unnormalized, return_error=True, logger=logger_in)
                    # make sure shape without error is close to the same value
                    shape_no_error = measure_shape(star_model, shape_unnormalized=shape_unnormalized, return_error=False, logger=logger_in)
                    np.testing.assert_equal(shape, shape_no_error)
                    shapes[j][k].append(shape)
                    errors[j][k].append(error)
            snrs.append(psf.measure_snr(star_model))

        # not particularly concerned with flux, du, dv
        shapes = np.array(shapes)
        errors = np.array(errors)
        std_shapes = np.array([[shapei.std(axis=0) for shapei in shape] for shape in shapes])
        mean_errors = np.array([[errori.mean(axis=0) for errori in error] for error in errors])
        snrs = np.array(snrs)
        # let's get the SNR back to within 10
        np.testing.assert_allclose(snrs, snr, atol=10)
        # let our errors be say within 20 percent
        # np.testing.assert_allclose(std_shapes, mean_errors, rtol=0.2)

    # note that we do not expect hsm and lmfit shapes to agree, because they are using different underlying shape models

    # make sure we can convert back and forth for the errors from lmfit
    x0 = shapes[1][0][0, 3:]
    x1 = shapes[1][1][0, 3:]
    std0 = errors[1][0][0, 3:]
    std1 = errors[1][1][0, 3:]

    std01 = np.array(psf.shape_convert_errors_to_unnormalized(std0[0], std0[1], std0[2], x0[0], x0[1], x0[1]))
    x01 = np.array(psf.shape_convert_to_unnormalized(x0[0], x0[1], x0[2]))
    x010 = np.array(psf.shape_convert_to_normalized(x01[0], x01[1], x01[2]))
    std010 = np.array(psf.shape_convert_errors_to_normalized(std01[0], std01[1], std01[2], x01[0], x01[1], x01[2]))
    np.testing.assert_allclose(x0, x010, atol=1e-5)
    np.testing.assert_allclose(std0, std010, atol=1e-3)
    np.testing.assert_allclose(x01, x1, atol=1e-5)
    np.testing.assert_allclose(std01, std1, atol=1e-3)


@timer
def test_profile():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    config = return_config()
    psf = piff.PSF.process(config, logger=logger)
    star = make_star(500, 500, 25)
    # get getProfile vs _profile(params)
    prof = psf.getProfile(star)
    params = psf.getParams(star)
    prof2 = psf._profile(params)
    assert prof == prof2
    params_list_0 = psf.getParamsList([star])[0]
    prof_list = psf._profile(params_list_0)
    assert prof == prof_list

    image = psf.drawProfile(star, prof)
    star_drawstar = psf.drawStar(star)
    image_drawstar = star_drawstar.image
    assert image == image_drawstar

    star_drawprofilestar = psf.drawProfileStar(star, prof, params)
    image_drawprofilestar = star_drawprofilestar.image
    assert image == image_drawprofilestar

    star_drawstarlist = psf.drawStarList([star])[0]
    image_drawstarlist = star_drawstarlist.image
    assert image == image_drawstarlist

    # NOTE: Not sure if I want to keep this functionality
    # # also make sure that if the aberrations are all 0, then doesn't even bother convolving opticalpsf
    # params_zeroed = params.copy()
    # params_zeroed[3:] = 0
    # prof_zero = psf._profile(params_zeroed)
    # prof = psf.atmo_model.dilate(params_zeroed[0]).shear(g1=params_zeroed[1], g2=params_zeroed[2])
    # assert prof == prof_zero

@timer
def test_roundtrip():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    # fit the test ccd
    image_file = './input/DECam_00241238_01.fits.fz'
    cat_file = './input/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits'
    orig_image = galsim.fits.read(image_file)
    psf_file = os.path.join('output','pixel_des_psf.fits')
    config_psf = return_config()
    # fit only defocus
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
            'stamp_size' : 24,
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            'nstars': 20,
            'min_snr': 40,
            'max_snr': 100,
            },
        'output': {'file_name': psf_file,},
        'psf': config_psf,
        }

    if __name__ == '__main__':
        fit_testers = ['pixel', 'moments', 'analytic']
    else:
        fit_testers = ['analytic']
    for fit_tester in fit_testers:
        logger.info('Testing roundtrip for fit method {0}'.format(fit_tester))
        config_psf['optfit_optimize'] = fit_tester
        # iirc this should also modify in config, but make sure
        config['psf']['optfit_optimize'] = fit_tester

        # run using piffify
        if os.path.exists(psf_file):
            os.remove(psf_file)
        logger.info('Running piffify')
        piff.piffify(copy.deepcopy(config), logger=logger)

        # load results and compare with initial params
        psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
        psf = piff.read(psf_file, logger=logger)

        # go through kwargs and check those
        for key in psf_original.kwargs:
            assert psf_original.kwargs[key] == psf.kwargs[key]

        # check the other named attributes
        np.testing.assert_allclose(psf_original.shape_weights, psf.shape_weights)
        assert psf_original.gsparams == psf.gsparams
        assert psf.kolmogorov_kwargs == psf_original.kolmogorov_kwargs
        assert psf.optical_psf_kwargs == psf_original.optical_psf_kwargs

        # check analytic coefs
        for ac1, ac2 in zip(psf.analytic_coefs, psf_original.analytic_coefs):
            for c1, c2 in zip(ac1, ac2):
                print(ac1)
                print(ac2)
                print(c1)
                print(c2)
                np.testing.assert_allclose(c1, c2)

    # copy these over to facilitate later writing for these tests
    stars = psf.stars
    wcs = psf.wcs
    pointing = psf.pointing
    atmo_interp = psf.atmo_interp

    # test saving when there is no atmo_interp
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
        assert psf_original.kwargs[key] == psf.kwargs[key]

    # test saving when analytic_coefs is None
    if os.path.exists(psf_file):
        os.remove(psf_file)
    config_psf.pop('analytic_coefs')
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf_original.stars = stars
    psf_original.wcs = wcs
    psf_original.pointing = pointing
    psf_original.atmo_interp = atmo_interp
    psf_original.write(psf_file, logger=logger)
    psf = piff.read(psf_file, logger=logger)
    for key in psf_original.kwargs:
        assert psf_original.kwargs[key] == psf.kwargs[key]
    assert psf_original.analytic_coefs == None
    assert psf.analytic_coefs == psf_original.analytic_coefs

    # test saving with no reference_wavefront
    config_psf.pop('reference_wavefront')
    psf_original = piff.PSF.process(copy.deepcopy(config_psf), logger=logger)
    psf_original.stars = stars
    psf_original.wcs = wcs
    psf_original.pointing = pointing
    psf_original.atmo_interp = atmo_interp
    psf_original.write(psf_file, logger=logger)
    psf = piff.read(psf_file, logger=logger)
    for key in psf_original.kwargs:
        assert psf_original.kwargs[key] == psf.kwargs[key]
    assert psf_original.reference_wavefront == None
    assert psf.reference_wavefront == psf_original.reference_wavefront

if __name__ == '__main__':
        test_init()
        test_aberrations()
        test_reference_wavefront()
        test_jmaxs()
        test_atmo_model_fit()
        test_atmo_interp_fit()
        test_profile()
        test_snr_and_shapes()
        test_analytic_coefs()
        test_roundtrip()

        # TODO: ones that are still tbd
        test_fit()
