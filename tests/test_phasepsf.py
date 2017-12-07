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

from piff_test_helper import timer

decaminfo = piff.des.DECamInfo()
def make_star_object(x, y, chipnum, **kwargs):
    wcs = decaminfo.get_nominal_wcs(chipnum)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=32, properties={'chipnum': chipnum})
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
            'phase_psf_kwargs':
                {
                },
            'analytic_coefs': None,
            'atmo_interp':
                {
                    'type': 'Polynomial',
                    'order': 2,
                },
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
    psf = piff.PhasePSF(logger=logger, **piff.PSF.parseKwargs(config, logger=logger))
    logger.info('test_init: Passed!')
    return psf

@timer
def test_reference_wavefront():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_weights():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_atmo_interp():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

@timer
def test_atmo_fit():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    pass

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
    psf = piff.PhasePSF(logger=logger, **piff.PSF.parseKwargs(config, logger=logger))
    stars = [make_star_object(100, 100, 1), make_star_object(100, 100, 60), make_star_object(100, 100, 3)]  # 0 and 1 share u, 0 and 2 share v
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)

    assert psf.jmax_pupil == config['jmax_pupil']
    assert psf.jmax_focal == config['jmax_focal']
    assert psf._noll_coef_field.shape[2] == config['jmax_focal']
    assert psf._coef_arrays_field.shape[0] == config['jmax_pupil']
    assert np.shape(aberrations_pupil) == (len(stars), config['jmax_pupil'])

    # test that constant terms lead to all stars getting same aberrations from field
    phase_psf_kwargs = {'r0': 0.2, 'g1': 0.3, 'zUV004_zXY001': -1.0}
    psf._update_optpsf(phase_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    assert aberrations_pupil[0][3] == phase_psf_kwargs['zUV004_zXY001']
    assert aberrations_pupil[1][3] == aberrations_pupil[0][3]
    assert aberrations_pupil[2][3] == aberrations_pupil[0][3]
    # similarly with r0
    assert aberrations_pupil[0][0] == phase_psf_kwargs['r0']
    assert aberrations_pupil[1][0] == aberrations_pupil[0][0]
    assert aberrations_pupil[2][0] == aberrations_pupil[0][0]
    # similarly with g1
    assert aberrations_pupil[0][1] == phase_psf_kwargs['g1']
    assert aberrations_pupil[1][1] == aberrations_pupil[0][1]
    assert aberrations_pupil[2][1] == aberrations_pupil[0][1]

    # we can also test that the linear terms are proportional
    us = np.array([star.u for star in stars])
    vs = np.array([star.v for star in stars])
    phase_psf_kwargs = {'zUV006_zXY002': -2}
    psf._update_optpsf(phase_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,5]
    # zXY002 and the u coordinate are the same, so same u should lead to same z
    assert us[0] == us[1]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[1]
    assert us[0] != us[2]
    assert zs[0] != zs[2]
    phase_psf_kwargs = {'zUV007_zXY003': -2}
    psf._update_optpsf(phase_psf_kwargs, logger=logger)
    aberrations_pupil = psf._getParamsList_aberrations_field(stars)
    zs = aberrations_pupil[:,6]
    # zXY003 and the v coordinate are the same, so same u should lead to same z
    assert vs[0] == vs[2]  # the next assert doesn't make sense to check if this isn't true
    assert zs[0] == zs[2]
    assert vs[0] != vs[1]
    assert zs[0] != zs[1]

    # test that modifying a specific key ...actually modifies it
    phase_psf_kwargs = {'r0': 0.2, 'g1': 0.3, 'g2': 0.4, 'zUV021_zXY045': 0.5, 'zUV004_zXY001': 1.0}
    psf._update_optpsf(phase_psf_kwargs, logger=logger)
    assert psf.aberrations_field[0, 0] == phase_psf_kwargs['r0']
    assert psf.aberrations_field[1, 0] == phase_psf_kwargs['g1']
    assert psf.aberrations_field[2, 0] == phase_psf_kwargs['g2']
    assert psf.aberrations_field[21-1, 45-1] == phase_psf_kwargs['zUV021_zXY045']
    assert psf.aberrations_field[4-1, 1-1] == phase_psf_kwargs['zUV004_zXY001']

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
def test_fit_optics():
    # set up logger
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    # make sure fixing params works
    # make sure weights work
    pass

@timer
def test_calculate_aberrations():
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
        test_init()
        test_jmaxs()
