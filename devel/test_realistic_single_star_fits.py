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
import matplotlib
matplotlib.use('Agg')
import galsim
import piff
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import fitsio
import copy
import unittest
import itertools
import time

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
        'reference_wavefront_file': '../tests/input/Science-20121120s1-v20i2.fits',
        'reference_wavefront_type': 'DECamWavefront',
        'reference_wavefront_extname': 1,
        'reference_wavefront_n_neighbors': 40,
        'reference_wavefront_weights': 'distance',
        'reference_wavefront_algorithm': 'auto',
        'reference_wavefront_p': 2,
        'n_optfit_stars': 0,
        'fov_radius': 4500.,
        'jmax_pupil': 11,
        'jmax_focal': 11,
        'min_optfit_snr': 0,
        'higher_order_reference_wavefront_file':
            '../tests/input/decam_2012-nominalzernike-protocol2.pickle',
        'random_forest_shapes_model_pickles_location': '../tests/input',
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
        'init_with_rf': 'True',
    }

# default config, big r0
def return_config_big_r0():
    return {
        'type': 'OptAtmo',
        'optical_psf_kwargs': {
            'template': 'des_big_r0',
        },
        'reference_wavefront_file': '../tests/input/Science-20121120s1-v20i2.fits',
        'reference_wavefront_type': 'DECamWavefront',
        'reference_wavefront_extname': 1,
        'reference_wavefront_n_neighbors': 40,
        'reference_wavefront_weights': 'distance',
        'reference_wavefront_algorithm': 'auto',
        'reference_wavefront_p': 2,
        'n_optfit_stars': 0,
        'fov_radius': 4500.,
        'jmax_pupil': 11,
        'jmax_focal': 11,
        'min_optfit_snr': 0,
        'higher_order_reference_wavefront_file':
            '../tests/input/decam_2012-nominalzernike-protocol2.pickle',
        'init_with_rf' : (sys.version_info > (3,0)),
        'random_forest_shapes_model_pickles_location': '../tests/input',
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
        'init_with_rf': 'True',
    }



@timer
def test_fit_model_for_many_stars():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11

    optatmo_psf_kwargs = {'size': 1.0, 'g1': 0, 'g2': 0,
                          'fix_size': False, 'fix_g1': True, 'fix_g2': True,
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
    optatmo_psf_kwargs_values = {'size': 0.8,
            'zPupil004_zFocal001': 0.2,
            'zPupil005_zFocal001': 0.3,
            'zPupil006_zFocal001': -0.2,
            'zPupil007_zFocal001': 0.2,
            'zPupil008_zFocal001': 0.4,
            'zPupil009_zFocal001': -0.25,
            'zPupil010_zFocal001': 0.2}
    config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
    psf_draw = piff.PSF.process(config_draw)
    psf_train = piff.PSF.process(copy.deepcopy(config))

    # make stars
    logger.info('Making Stars')
    nstars = 117
    np.random.seed(12345)
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
    params_including_atmo_params = copy.deepcopy(params)
    draw_atmo_sizes = np.random.uniform(-0.1,0.1,params.shape[0])
    draw_atmo_g1s = np.random.uniform(-0.1,0.1,params.shape[0])
    draw_atmo_g2s = np.random.uniform(-0.1,0.1,params.shape[0])
    params_including_atmo_params[:,0] = draw_atmo_sizes
    params_including_atmo_params[:,1] = draw_atmo_g1s
    params_including_atmo_params[:,2] = draw_atmo_g2s

    delete_list = []
    stars = []
    fit_atmo_sizes = []
    fit_atmo_g1s = []
    fit_atmo_g2s = []
    number_of_failed_fits = 0
    for star, star_i, param, param_including_atmo_params in zip(stars_blank, list(range(0,len(stars_blank))), params, params_including_atmo_params):
        prof = psf_draw.getProfile(copy.deepcopy(star), param_including_atmo_params) * 1e6
        star = psf_draw.drawProfile(star, prof, param_including_atmo_params)
        stars.append(star)
        try:
            fitted_star = psf_draw.fit_model(piff.Star(star.data.copy(), None), params=params[star_i])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            delete_list.append(star_i)
            logger.warning('{0}'.format(str(e)))
            logger.warning('Warning! Failed to fit atmosphere model for star {0}. Removing star'.format(star_i))
            number_of_failed_fits = number_of_failed_fits + 1
            continue
        fit_atmo_size = fitted_star.fit.params[0]
        fit_atmo_sizes.append(fit_atmo_size)
        fit_atmo_g1 = fitted_star.fit.params[1]
        fit_atmo_g1s.append(fit_atmo_g1)
        fit_atmo_g2 = fitted_star.fit.params[2]
        fit_atmo_g2s.append(fit_atmo_g2)
    draw_atmo_sizes = np.delete(draw_atmo_sizes, delete_list).tolist()
    draw_atmo_g1s = np.delete(draw_atmo_g1s, delete_list).tolist()
    draw_atmo_g2s = np.delete(draw_atmo_g2s, delete_list).tolist()
    tol = 1e-6
    max_failed_fits = 3
    assert np.all(np.array(draw_atmo_sizes) - np.array(fit_atmo_sizes) <= tol),'failed to fit all atmo_sizes to tolerance {0}'.format(tol)
    assert np.all(np.array(draw_atmo_g1s) - np.array(fit_atmo_g1s) <= tol),'failed to fit all atmo_g1s to tolerance {0}'.format(tol)
    assert np.all(np.array(draw_atmo_g2s) - np.array(fit_atmo_g2s) <= tol),'failed to fit all atmo_g2s to tolerance {0}'.format(tol)
    assert number_of_failed_fits <= max_failed_fits,'number of failed fits is {0}, which exceeds the maximum for this unit test {1}'.format(number_of_failed_fits, max_failed_fits)



@timer
def test_fit_model_params_var_for_three_stars():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11

    optatmo_psf_kwargs = {'size': 1.0, 'g1': 0, 'g2': 0,
                          'fix_size': False, 'fix_g1': True, 'fix_g2': True,
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
    optatmo_psf_kwargs_values = {'size': 0.8,
            'zPupil004_zFocal001': 0.2,
            'zPupil005_zFocal001': 0.3,
            'zPupil006_zFocal001': -0.2,
            'zPupil007_zFocal001': 0.2,
            'zPupil008_zFocal001': 0.4,
            'zPupil009_zFocal001': -0.25,
            'zPupil010_zFocal001': 0.2}
    config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
    psf_draw = piff.PSF.process(config_draw)
    psf_train = piff.PSF.process(copy.deepcopy(config))

    # make stars
    logger.info('Making Stars')
    nstars = 3
    np.random.seed(12345)
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
    params_including_atmo_params = copy.deepcopy(params)
    draw_atmo_sizes = np.random.uniform(-0.1,0.1,params.shape[0])
    draw_atmo_g1s = np.random.uniform(-0.1,0.1,params.shape[0])
    draw_atmo_g2s = np.random.uniform(-0.1,0.1,params.shape[0])
    known_atmo_size_vars = np.array([9.428090381740227e-11, 1.2509087173271692e-10, 8.746287562949641e-11])
    known_atmo_g1_vars = np.array([1.2646342862727268e-10, 1.392361783212879e-10, 1.239358322159534e-10])
    known_atmo_g2_vars = np.array([1.2839876868239813e-10, 1.4015155202586417e-10, 1.2585684616270795e-10])
    params_including_atmo_params[:,0] = draw_atmo_sizes
    params_including_atmo_params[:,1] = draw_atmo_g1s
    params_including_atmo_params[:,2] = draw_atmo_g2s

    delete_list = []
    stars = []
    fit_atmo_sizes = []
    fit_atmo_g1s = []
    fit_atmo_g2s = []
    fit_atmo_size_vars = []
    fit_atmo_g1_vars = []
    fit_atmo_g2_vars = []
    number_of_failed_fits = 0
    for star, star_i, param, param_including_atmo_params in zip(stars_blank, list(range(0,len(stars_blank))), params, params_including_atmo_params):
        prof = psf_draw.getProfile(copy.deepcopy(star), param_including_atmo_params) * 1e6
        star = psf_draw.drawProfile(star, prof, param_including_atmo_params)
        stars.append(star)
        fitted_star = psf_draw.fit_model(piff.Star(star.data.copy(), None), params=params[star_i])
        try:
            fitted_star = psf_draw.fit_model(piff.Star(star.data.copy(), None), params=params[star_i])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            delete_list.append(star_i)
            logger.warning('{0}'.format(str(e)))
            logger.warning('Warning! Failed to fit atmosphere model for star {0}. Removing star'.format(star_i))
            number_of_failed_fits = number_of_failed_fits + 1
            continue
        fit_atmo_size = fitted_star.fit.params[0]
        fit_atmo_g1 = fitted_star.fit.params[1]
        fit_atmo_g2 = fitted_star.fit.params[2]
        fit_atmo_size_var = fitted_star.fit.params_var[0]
        fit_atmo_g1_var = fitted_star.fit.params_var[1]
        fit_atmo_g2_var = fitted_star.fit.params_var[2]
        fit_atmo_sizes.append(fit_atmo_size)
        fit_atmo_g1s.append(fit_atmo_g1)
        fit_atmo_g2s.append(fit_atmo_g2)
        fit_atmo_size_vars.append(fit_atmo_size_var)
        fit_atmo_g1_vars.append(fit_atmo_g1_var)
        fit_atmo_g2_vars.append(fit_atmo_g2_var)
    draw_atmo_sizes = np.delete(draw_atmo_sizes, delete_list).tolist()
    draw_atmo_g1s = np.delete(draw_atmo_g1s, delete_list).tolist()
    draw_atmo_g2s = np.delete(draw_atmo_g2s, delete_list).tolist()
    known_atmo_size_vars = np.delete(known_atmo_size_vars, delete_list).tolist()
    known_atmo_g1_vars = np.delete(known_atmo_g1_vars, delete_list).tolist()
    known_atmo_g2_vars = np.delete(known_atmo_g2_vars, delete_list).tolist()
    tol = 1e-6
    max_failed_fits = 1
    assert np.all(np.array(draw_atmo_sizes) - np.array(fit_atmo_sizes) <= tol),'failed to fit all atmo_sizes to tolerance {0}'.format(tol)
    assert np.all(np.array(draw_atmo_g1s) - np.array(fit_atmo_g1s) <= tol),'failed to fit all atmo_g1s to tolerance {0}'.format(tol)
    assert np.all(np.array(draw_atmo_g2s) - np.array(fit_atmo_g2s) <= tol),'failed to fit all atmo_g2s to tolerance {0}'.format(tol)
    assert np.all(np.array(known_atmo_size_vars) - np.array(fit_atmo_size_vars) <= tol),'failed to fit all atmo_sizes to tolerance {0}'.format(tol)
    assert np.all(np.array(known_atmo_g1_vars) - np.array(fit_atmo_g1_vars) <= tol),'failed to fit all atmo_g1s to tolerance {0}'.format(tol)
    assert np.all(np.array(known_atmo_g2_vars) - np.array(fit_atmo_g2_vars) <= tol),'failed to fit all atmo_g2s to tolerance {0}'.format(tol)
    assert number_of_failed_fits <= max_failed_fits,'number of failed fits is {0}, which exceeds the maximum for this unit test {1}'.format(number_of_failed_fits, max_failed_fits)



@timer
def test_star_residuals():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11

    optatmo_psf_kwargs = {'size': 1.0, 'g1': 0, 'g2': 0,
                          'fix_size': False, 'fix_g1': True, 'fix_g2': True,
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
    optatmo_psf_kwargs_values = {'size': 0.8,
            'zPupil004_zFocal001': 0.2,
            'zPupil005_zFocal001': 0.3,
            'zPupil006_zFocal001': -0.2,
            'zPupil007_zFocal001': 0.2,
            'zPupil008_zFocal001': 0.4,
            'zPupil009_zFocal001': -0.25,
            'zPupil010_zFocal001': 0.2}
    config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
    psf_draw = piff.PSF.process(config_draw)
    psf_train = piff.PSF.process(copy.deepcopy(config))

    # make stars
    logger.info('Making Stars')
    nstars = 117
    np.random.seed(12345)
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
    stars_to_fit = []
    for star, param in zip(stars_blank, params):
        prof = psf_draw.getProfile(copy.deepcopy(star), param) * 1e6
        star = psf_draw.drawProfile(star, prof, param)
        stars.append(star)
        stars_to_fit.append(piff.Star(star.data.copy(), None))

    # do fit
    for fit_optics_mode in ['shape']:
        psf_train = piff.PSF.process(copy.deepcopy(config))
        logger.info('Test fitting optics mode {0}'.format(fit_optics_mode))
        psf_train.fit_optics_mode = fit_optics_mode

        psf_train.fit(stars_to_fit, wcs, pointing, logger=logger)
        logger.info('Fit results for mode {0}'.format(fit_optics_mode))
        logger.info('Parameter: Input, Fit, Input - Fit')
        for key in sorted(optatmo_psf_kwargs_values):
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            logger.info('{0}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit))
        # I don't really trust these fits to better than 0.1
        for key in sorted(optatmo_psf_kwargs_values):
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            diff = np.abs(train - fit)
            if fit_optics_mode == 'random_forest':
                tol = 0.1  # lower expectations with the random_forest mode
            else:
                tol = 0.01
            assert diff <= tol,'failed to fit {0} to tolerance {4}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit, tol)

        key_test_star = stars_to_fit[0]
        key_params = psf_train.getParams(key_test_star)
        #pre_drawn_star = psf_train.fit_model(key_test_star, key_params, vary_shape=False)
        pre_drawn_star = psf_train.reflux(key_test_star, key_params)
        key_test_model_star = psf_train.drawStar(pre_drawn_star)
        key_test_star_image_array = key_test_star.image.array
        key_test_model_star_image_array = key_test_model_star.image.array
        key_test_difference_star_image_array = key_test_star_image_array - key_test_model_star_image_array
        tol = 50.0
        plt.figure()
        plt.imshow(key_test_star_image_array, vmin=np.percentile(key_test_star_image_array, q=2),vmax=np.percentile(key_test_star_image_array, q=98), cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title("Data for Key Star")
        plt.savefig("./key_star_data.png")
        plt.figure()
        plt.imshow(key_test_model_star_image_array, vmin=np.percentile(key_test_star_image_array, q=2),vmax=np.percentile(key_test_star_image_array, q=98), cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title("Model for Key Star")
        plt.savefig("./key_star_model.png")
        plt.figure()
        plt.imshow(key_test_difference_star_image_array, vmin=np.percentile(key_test_difference_star_image_array, q=2),vmax=np.percentile(key_test_difference_star_image_array, q=98), cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.title("Data - Model for Key Star")
        plt.savefig("./key_star_data_minus_model.png")
        assert np.all(np.abs(key_test_difference_star_image_array)<tol), 'failed to produce star residuals with all pixels below tol {0}'.format(tol)



@timer
def test_optics_and_test_fit_model():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11

    optatmo_psf_kwargs = {'size': 1.0, 'g1': 0, 'g2': 0,
                          'fix_size': False, 'fix_g1': False, 'fix_g2': False,
                          'zPupil004_zFocal001': 0.0,
                          'zPupil005_zFocal001': 0.0,
                          'zPupil006_zFocal001': 0.0,
                          'zPupil007_zFocal001': 0.0,
                          'zPupil008_zFocal001': 0.0,
                          'zPupil009_zFocal001': 0.0,
                          'zPupil010_zFocal001': 0.0,
                          'zPupil011_zFocal001': 0.0,
                          'fix_zPupil011_zFocal001': True,
                        }  # avoid the defocus,astig,spherical -> negatives degeneracy by fixing spherical
    success = { 'pixel' : set(), 'shape' : set() }
    fail = { 'pixel' : set(), 'shape' : set() }
    err = { 'pixel' : set(), 'shape' : set() }
    times = { 'pixel' : set(), 'shape' : set() }

    nstars = 500
    seed0 = 12345
    nseeds = 100
    for seed in range(seed0, seed0+nseeds):
        np.random.seed(seed)

        config['optatmo_psf_kwargs'] = copy.deepcopy(optatmo_psf_kwargs)
        config_draw = copy.deepcopy(config)
        if True:
            optatmo_psf_kwargs_values = {
                'size': np.random.uniform(0.7,0.8),  # shape fails when size > ~0.85
                'zPupil004_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil005_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil006_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil007_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil008_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil009_zFocal001': np.random.uniform(-0.4,0.4),
                'zPupil010_zFocal001': np.random.uniform(-0.4,0.4),
            }
        else:
            # Ares's original values
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
        print(optatmo_psf_kwargs_values)
        config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
        psf_draw = piff.PSF.process(config_draw)
        psf_train = piff.PSF.process(copy.deepcopy(config))

        # make stars
        logger.info('Making Stars')
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
        params_including_atmo_params = copy.deepcopy(params)
        draw_atmo_sizes = np.random.uniform(-0.02,0.02,params.shape[0])
        draw_atmo_g1s = np.random.uniform(-0.02,0.02,params.shape[0])
        draw_atmo_g2s = np.random.uniform(-0.02,0.02,params.shape[0])
        params_including_atmo_params[:,0] = draw_atmo_sizes
        params_including_atmo_params[:,1] = draw_atmo_g1s
        params_including_atmo_params[:,2] = draw_atmo_g2s

        stars = []
        stars_to_fit = []
        for star, star_i, param, param_including_atmo_params in zip(stars_blank, list(range(0,len(stars_blank))), params, params_including_atmo_params):
            prof = psf_draw.getProfile(copy.deepcopy(star), param_including_atmo_params) * 1e6
            star = psf_draw.drawProfile(star, prof, param_including_atmo_params)
            stars.append(star)
            stars_to_fit.append(piff.Star(star.data.copy(), None))
        key_test_star = copy.deepcopy(stars_to_fit[0])
        draw_atmo_size = params_including_atmo_params[0][0]
        draw_atmo_g1 = params_including_atmo_params[0][1]
        draw_atmo_g2 = params_including_atmo_params[0][2]

        # do fit
        for fit_optics_mode in ['shape', 'pixel']:
            t0 = time.time()
            config['init_with_rf'] = (fit_optics_mode == 'shape')
            psf_train = piff.PSF.process(copy.deepcopy(config))
            logger.info('Test fitting optics mode {0}'.format(fit_optics_mode))
            psf_train.fit_optics_mode = fit_optics_mode

            psf_train.fit(stars_to_fit, wcs, pointing, logger=logger)
            logger.info('Fit results for mode {0}'.format(fit_optics_mode))
            logger.info('Parameter: Input, Fit, Input - Fit')
            for key in sorted(optatmo_psf_kwargs_values):
                train = optatmo_psf_kwargs_values[key]
                fit = psf_train.optatmo_psf_kwargs[key]
                logger.info('{0}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit))
            # I don't really trust these fits to better than 0.1
            try:
                for key in sorted(optatmo_psf_kwargs_values):
                    train = optatmo_psf_kwargs_values[key]
                    fit = psf_train.optatmo_psf_kwargs[key]
                    diff = np.abs(train - fit)
                    if fit_optics_mode == 'random_forest':
                        tol = 0.5  # lower expectations with the random_forest mode
                    else:
                        tol = 0.05
                    assert diff <= tol,'failed to fit {0} to tolerance {4}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit, tol)

                key_params = psf_train.getParams(key_test_star)
                key_test_model_star = psf_train.fit_model(key_test_star, key_params)
                tol = 0.005
                #print("draw_atmo_size {0}".format(draw_atmo_size))
                #print("draw_atmo_g1 {0}".format(draw_atmo_g1))
                #print("draw_atmo_g2 {0}".format(draw_atmo_g2))
                #print("fit_atmo_size {0}".format(key_test_model_star.fit.params[0]))
                #print("fit_atmo_g1 {0}".format(key_test_model_star.fit.params[1]))
                #print("fit_atmo_g2 {0}".format(key_test_model_star.fit.params[2]))
                assert draw_atmo_size - key_test_model_star.fit.params[0] <= tol,'failed to fit atmo_size to tolerance {0}'.format(tol)
                assert draw_atmo_g1 - key_test_model_star.fit.params[1] <= tol,'failed to fit atmo_g1 to tolerance {0}'.format(tol)
                assert draw_atmo_g2 - key_test_model_star.fit.params[2] <= tol,'failed to fit atmo_g2 to tolerance {0}'.format(tol)

            except AssertionError:
                fail[fit_optics_mode].add(seed)
            except Exception:
                err[fit_optics_mode].add(seed)
            else:
                success[fit_optics_mode].add(seed)
            t1 = time.time()
            print('time = ',t1-t0)
            times[fit_optics_mode].add(t1-t0)
        print('After seed ',seed)
        print('N pixel success = ',len(success['pixel']))
        print('N pixel fail = ',len(fail['pixel']))
        print('N shape err = ',len(err['pixel']))
        print('N shape success = ',len(success['shape']))
        print('N shape fail = ',len(fail['shape']))
        print('N shape err = ',len(err['shape']))

    print('Done')
    print('pixel fail seeds = ',sorted(fail['pixel']))
    print('shape fail seeds = ',sorted(fail['shape']))
    print('Mean time for pixel = ',np.mean(list(times['pixel'])))
    print('Mean time for shape = ',np.mean(list(times['shape'])))

@timer
def test_size_fit():
    # setup logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=1)
    logger.debug('Entering test_fit')

    config = return_config()
    config['atmo_interp'] = 'None'
    config['jmax_focal'] = 1
    config['jmax_pupil'] = 11
    config['init_with_rf'] = False

    optatmo_psf_kwargs = {'size': 1.0, 'g1': 0, 'g2': 0,
                          'fix_size': False, 'fix_g1': True, 'fix_g2': True,
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
    optatmo_psf_kwargs_values = {'size': 0.8,
            'zPupil004_zFocal001': 0.2,
            'zPupil005_zFocal001': 0.3,
            'zPupil006_zFocal001': -0.2,
            'zPupil007_zFocal001': 0.2,
            'zPupil008_zFocal001': 0.4,
            'zPupil009_zFocal001': -0.25,
            'zPupil010_zFocal001': 0.2}
    config_draw['optatmo_psf_kwargs'].update(optatmo_psf_kwargs_values)
    psf_draw = piff.PSF.process(config_draw)
    psf_train = piff.PSF.process(copy.deepcopy(config))

    # make stars
    logger.info('Making Stars')
    nstars = 117
    np.random.seed(12345)
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
    stars_to_fit = []
    for star, param in zip(stars_blank, params):
        prof = psf_draw.getProfile(copy.deepcopy(star), param) * 1e6
        star = psf_draw.drawProfile(star, prof, param)
        stars.append(star)
        stars_to_fit.append(piff.Star(star.data.copy(), None))

    # do fit
    for fit_optics_mode in ['skip']:
        psf_train = piff.PSF.process(copy.deepcopy(config))
        logger.info('Test fitting optics mode {0}'.format(fit_optics_mode))
        psf_train.fit_optics_mode = fit_optics_mode

        psf_train.fit(stars_to_fit, wcs, pointing, logger=logger)
        logger.info('Fit results for fit_size() only')
        logger.info('Parameter: Input, Fit, Input - Fit')
        for key in sorted(optatmo_psf_kwargs_values):
            if key !='size':
                continue
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            logger.info('{0}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit))
        # I don't really trust these fits to better than 0.1
        for key in sorted(optatmo_psf_kwargs_values):
            if key !='size':
                continue
            train = optatmo_psf_kwargs_values[key]
            fit = psf_train.optatmo_psf_kwargs[key]
            diff = np.abs(train - fit)
            if fit_optics_mode == 'random_forest':
                tol = 0.2  # lower expectations with the random_forest mode
            else:
                tol = 0.02
            print("key: {0}".format(key))
            print("train: {0}".format(train))
            print("fit: {0}".format(fit))
            print("train - fit: {0}".format(train - fit))
            print("tol: {0}".format(tol))
            assert diff <= tol,'failed to fit {0} to tolerance {4}: {1:+.3f}, {2:+.3f}, {3:+.3f}'.format(key, train, fit, train - fit, tol)



if __name__ == '__main__':
    test_fit_model_params_var_for_three_stars()
    test_fit_model_for_many_stars()
    test_star_residuals()
    test_optics_and_test_fit_model()
    test_size_fit()
