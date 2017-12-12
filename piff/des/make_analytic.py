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

"""
.. module:: make_analytic
"""

from __future__ import print_function

import os
import piff
import galsim
import numpy as np
from sklearn.linear_model import LassoCV as lasso

def param_to_shape(params, star, psf):
    # profile
    prof = psf._profile(params)

    # draw onto image
    star = psf.drawProfileStar(star, prof, params)

    # measure shape 3 different ways
    shapes = []
    for measure_shape, shape_unnormalized, return_error in zip([psf.measure_shape_hsm, psf.measure_shape_hsm, psf.measure_shape_lmfit], [False, True, False], [False, True, False]):
        val = measure_shape(star, shape_unnormalized=shape_unnormalized, return_error=return_error)
        if return_error:
            shapes.append(val[0])
        else:
            shapes.append(val)

    return shapes

def make_sample(nsample, logger):

    # make PSF object
    config = {  'optical_psf_kwargs':
                {
                    'template': 'des',
                },
            'shape_weights': [0.5, 1, 1],
            'fov_radius': 1. * 60 * 60,  # TODO: figure this out!!
            'jmax_pupil': 11,
            'jmax_focal': 1,
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
    psf = piff.PSF.process(config, logger=logger)

    # make star image
    decaminfo = piff.des.DECamInfo()
    wcs = decaminfo.get_nominal_wcs(32)
    icen = 100
    jcen = 100
    flux = 1
    star = piff.Star.makeTarget(x=icen, y=jcen, wcs=wcs, stamp_size=32, flux=flux)

    shapes = []
    params = []
    for i in range(nsample):
        # draw g0
        g0 = np.random.random_sample() * (2.0 - 0.4) + 0.4
        g1 = np.random.random_sample() * (0.1 + 0.1) - 0.1
        g2 = np.random.random_sample() * (0.1 + 0.1) - 0.1
        z = np.random.random_sample(size=(8)) * 2 - 1
        param = np.hstack((g0, g1, g2, z))
        params.append(param)
        shape = param_to_shape(param, star, psf)
        shapes.append(shape)
        if i % 100000 == 0:
            cmd = logger.warn
        elif i % 1000 == 0:
            cmd = logger.info
        else:
            cmd = logger.debug
        cmd('Measured shape of star {0} / {1}'.format(i, nsample))
        param_str = '***Params: '
        for parami in param:
            param_str += '{0:+.2f}, '.format(parami)
        cmd(param_str)
        for j, shapei in enumerate(shape):
            shape_str = '***Shape {0}: '.format(j)
            for shapej in shapei:
                shape_str += '{0:+.2f}, '.format(shapej)
            cmd(shape_str)
    params = np.array(params)
    shapes = np.array(shapes)

    return params, shapes

def evaluate_sample(shapes, params, logger):
    # convert params to up to cube of g0 and square of rest
    # to do that, first put up front a set of ones
    params_onehot = np.vstack((np.ones(len(params)).T, params.T)).T
    ndim = params_onehot.shape[1]
    ppoly_full = (params_onehot[:, :, None, None] * params_onehot[:, None, :, None] * params_onehot[:, :2][:, None, None, :])
    ppoly = ppoly_full.reshape(len(shapes), 2 * ndim * ndim)

    # create pindx using python lists
    params_indx = ['1', 'g0', 'g1', 'g2'] + ['z{0:02d}'.format(i) for i in range(4, 12)]
    pindx = []
    for param in params_indx:
        for param_2 in params_indx:
            for param_3 in params_indx[:2]:
                pindx.append('{0} {1} {2}'.format(param, param_2, param_3))

    models = []
    for i in range(3):
        logger.info('Printing shape {0}'.format(i))
        y = shapes[:, i]
        alphas = np.logspace(0, -5, 100)
        model = lasso(alphas=alphas, max_iter=1000, fit_intercept=False, cv=5)
        model.fit(ppoly, y)
        models.append(model.coef_)
        conds = np.abs(model.coef_) > 0
        logger.info('\n"""')
        logger.info('{0} {1} coeffs,\nstd lasso {2:.02e}'.format(['T', 'g1', 'g2'][i], np.sum(conds),
            np.std(y - model.predict(ppoly))))
        logger.info('{0}:\t{1:+.02e}'.format('intercept_', model.intercept_))
        for ith in range(len(conds)):
            if conds[ith]:
                logger.info('{2} {0}:\t{1:+.02e}'.format(pindx[ith], model.coef_[ith], ith))
        logger.info('"""\n')
    models = np.array(models)

    return models

if __name__ == '__main__':
    # parse args
    import argparse
    description = "Build OptAtmoPSF Analytic relation"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-N', '--nsample', type=int, action='store', default=100)
    parser.add_argument('-v', '--verbose', type=int, action='store', default=1)
    parser.add_argument('-o', '--out', type=str, action='store', default='./optical')
    args = parser.parse_args()
    logger = piff.config.setup_logger(verbose=args.verbose)
    if os.path.exists('shapes.npy'):
        shapes = np.load('shapes.npy')
        params = np.load('params.npy')
    else:
        params, shapes = make_sample(args.nsample, logger=logger)
        # save params and shapes for now
        np.save('shapes.npy', shapes)
        np.save('params.npy', params)
    for i in range(3):
        shape = shapes[:, i, 3:]
        models = evaluate_sample(shape, params, logger=logger)
        np.save(args.out + '_' + str(i) + '.npy', models)
