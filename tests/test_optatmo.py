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

from piff_test_helper import timer

decaminfo = piff.des.DECamInfo()
def make_star(x, y, chipnum, properties={}, **kwargs):
    wcs = decaminfo.get_nominal_wcs(chipnum)
    properties_in = {'chipnum': chipnum}
    properties_in.update(properties)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=32, properties=properties_in, **kwargs)
    return star

# default config
def return_config():
    config = {  'optical_psf_kwargs':
                {
                    'template': 'des',
                },
            'reference_wavefront':
                {
                    'file_name': '/nfs/slac/g/ki/ki18/cpd/Projects/DES/Piff/tests/input/Science-20121120s1-v20i2.fits',
                    'extname': 1,
                    'n_neighbors': 40,
                    'weights': 'distance',
                    'algorithm': 'auto',
                    'p': 2,
                    'type': 'DECamWavefront',
                },
            'weights_moment_fit': [0.5, 1, 1],
            'fov_radius': 1.,  # TODO: figure this out!!
            'jmax_pupil': 11,
            'jmax_focal': 15,
            'n_optfit_stars': 0,
            'min_optfit_snr': 0,
            'optfit_optimize': 'analytic',
            'optatmo_psf_kwargs':
                {
                },
            'analytic_coefs': None,
            'atmo_interp':
                {
                    'type': 'Polynomial',
                    'order': 2,
                },

            'type': 'OptAtmo',
         }
    return config

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
    logger.info('test_init: Passed!')
    return psf

@timer
def test_reference_wavefront():
    # test that aberrations from wavefront appear in getParams

    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)

    star = make_star(0, 0, 1)

    # check that each gives aberrations, and that they give the same aberrations up to their max order
    params_psfs = []
    jmaxs = [5, 11, 41]
    for jmax_pupil in jmaxs:
        config = return_config()
        config['jmax_pupil'] = jmax_pupil
        psf = piff.PSF.process(config, logger=logger)
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
        psf = piff.PSF.process(config, logger=logger)
        assert False
    except ValueError:
        assert True
    # test that jmax_focal < 1 throws error
    try:
        config = return_config()
        config['jmax_focal'] = 0
        psf = piff.PSF.process(config, logger=logger)
        assert False
    except ValueError:
        assert True

    # make sure we cannot update the psf in the disallowed ranges
    uvs = [3, 900]
    for uv in uvs:
        try:
            psf._update_optatmopsf({'zUV{0:03d}_zXY001'.format(uv): 1}, logger=logger)
            assert False
        except ValueError:
            assert True
    xys = [0, 900]
    for xy in xys:
        try:
            psf._update_optatmopsf({'zUV005_zXY{0:03d}'.format(xy): 1}, logger=logger)
            assert False
        except ValueError:
            assert True

@timer
def test_weights():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_measure_shapes():
    # shape measurer, and errors. unnormalized_basis too
    pass

@timer
def test_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)

    jmax_pupil = 11
    config = return_config()
    config['jmax_focal'] = 1
    config['jmax_pupil'] = jmax_pupil

    psf = piff.PSF.process(config, logger=logger)
    optatmo_psf_kwargs = {'size': 1.3, 'g1': 0.02, 'g2': -0.03,
                        'zUV004_zXY001': -1.0,
                        'zUV005_zXY001': 0.5,
                        'zUV006_zXY001': -0.25,
                        'zUV007_zXY001': 0.5,
                        'zUV008_zXY001': -0.25,
                        'zUV009_zXY001': 0.5,
                        'zUV010_zXY001': -0.25,
                        }
    psf._update_optatmopsf(optatmo_psf_kwargs, logger=logger)
    # make star
    nstars = 300
    chipnums = np.random.choice(range(1,63), nstars)
    icens = np.random.randint(100, 1024, nstars)
    jcens = np.random.randint(100, 1024, nstars)
    stars_blank = [make_star(i, j, chip) for i, j, chip in zip(icens, jcens, chipnums)]
    wcs = {}
    for i in np.unique(chipnums):
        wcs[i] = decaminfo.get_nominal_wcs(i)
    pointing = None

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
        star = psf._drawProfile(star, prof, param)
        # stars definitely need some noise to help the fitting
        noise = 0.0001
        star.data.weight = star.image.copy()
        star.weight.fill(1./noise/noise)
        gn = galsim.GaussianNoise(sigma=noise)
        star.image.addNoise(gn)
        stars.append(star)
        star_to_fit = piff.Star(star.data, None)
        stars_to_fit.append(star_to_fit)

    # # fit model with atmo_interp
    # psf.fit_atmosphere(stars_to_fit, logger=logger)

    # # compare inputs work
    # np.testing.assert_allclose(atmo_size, psf.atmo_interp.coeffs[0][0,0])
    # np.testing.assert_allclose(atmo_g1, psf.atmo_interp.coeffs[1][0,0])
    # np.testing.assert_allclose(atmo_g2, psf.atmo_interp.coeffs[2][0,0])

    # # check that the others are 0
    # np.testing.assert_allclose(0, psf.atmo_interp.coeffs[0].flatten()[1:], atol=1e-16)
    # np.testing.assert_allclose(0, psf.atmo_interp.coeffs[1].flatten()[1:], atol=1e-16)
    # np.testing.assert_allclose(0, psf.atmo_interp.coeffs[2].flatten()[1:], atol=1e-16)

    # # enable atmosphere
    # psf._enable_atmosphere = True

    # # test with drawStar
    # star_index = 3
    # star = stars_to_fit[star_index]
    # star_fit = psf.drawStar(star)
    # np.testing.assert_allclose(star_fit.fit.params, params[star_index])
    # np.testing.assert_allclose(star_fit.image.array, star.image.array)

    # fit the above stuff for different optical models
    for optfit_optimize in ['pixel', 'moments', 'analytic']:
    # for optfit_optimize in ['moments']:
        config = return_config()
        config['optfit_optimize'] = optfit_optimize
        config['jmax_focal'] = 1
        config['jmax_pupil'] = jmax_pupil

        config['n_optfit_stars'] = int(0.3 * nstars)
        config['optatmo_psf_kwargs']['fix_zUV{0:03d}_zXY001'.format(jmax_pupil)] = True
        psf_clean = piff.PSF.process(config, logger=logger)
        psf_clean.fit(stars_to_fit, wcs, pointing, logger=logger)
        import ipdb; ipdb.set_trace()

        # check that the fixing actually fixed
        assert psf_clean.aberrations_field[jmax_pupil - 1, 0] == 0
        assert psf_clean.optatmo_psf_kwargs['fix_zUV011_zXY001']
        assert psf_clean.optatmo_psf_kwargs['zUV011_zXY001'] == 0

        # check that n_optfit_stars restricted appropriately
        assert len(psf_clean._opt_stars) <= config['n_optfit_stars']

        # evaluate zernike terms
        np.testing.assert_allclose(psf.aberrations_field[3:,0], psf_clean.aberrations_field[3:,0], atol=1e-6)
        # spot check optatmo_psf_kwargs
        assert psf_clean.optatmo_psf_kwargs['size'] == psf_clean.aberrations_field[0, 0]
        assert psf_clean.optatmo_psf_kwargs['g2'] == psf_clean.aberrations_field[2, 0]
        assert psf_clean.optatmo_psf_kwargs['zUV004_zXY001'] == psf_clean.aberrations_field[3, 0]
        # note that when comparing the aberrations, because we put in a constant atmosphere, we actually expect the atmosphere interpolation to be zero, and the aberration field terms to be the sum of those pieces
        np.testing.assert_allclose(psf.aberrations_field[0,0] + psf.atmo_interp.coeffs[0][0,0], psf_clean.aberrations_field[0,0], atol=1e-6)
        np.testing.assert_allclose(psf.aberrations_field[1,0] + psf.atmo_interp.coeffs[1][0,0], psf_clean.aberrations_field[1,0], atol=1e-6)
        np.testing.assert_allclose(psf.aberrations_field[2,0] + psf.atmo_interp.coeffs[2][0,0], psf_clean.aberrations_field[2,0], atol=1e-6)

@timer
def test_atmo_model_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    config = return_config()
    psf = piff.PSF.process(config, logger=logger)
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
    atmo_flux = 1e2
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
    star = psf._drawProfile(star, prof, params)
    star_to_fit = piff.Star(star.data, None)

    # fit star
    star_fit = psf._fit_atmosphere_model(star_to_fit, logger)

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
    star_fit_drawn = psf._drawProfile(star_fit, prof_fit, params_fit)
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
    psf = piff.PSF.process(config, logger=logger)
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

    logger.info('test_jmaxs: Passed!')

@timer
def test_atmosphere():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_analytic_coefs():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_drawstars():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_shape_modeller_and_snr():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_getProfile():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_roundtrip():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

if __name__ == '__main__':
        # test_init()
        test_fit()
        # test_reference_wavefront()
        # test_jmaxs()
        # test_atmo_model_fit()
