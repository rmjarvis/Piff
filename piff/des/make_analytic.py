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
Example script that demonstrates how to make the analytic coefficients used in
the optatmo psf
"""

from __future__ import print_function

import piff
import numpy as np
import lmfit
from sklearn.linear_model import ElasticNet, LinearRegression

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
from piff.cypoly import cypoly_full, cypoly

def linear_resid(lmparams, x, y):
    m, b = lmparams.valuesdict().values()
    chi = m * x + b - y
    return chi

def make_opt_star(params, psf, decaminfo, du=0, dv=0, snr=0, stamp_size=24):
    chipnum = 1
    x, y = (0, 0)
    wcs = decaminfo.get_nominal_wcs(chipnum)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=stamp_size,
                                properties={'chipnum': chipnum})

    if snr == 0:
        flux = 1
    else:
        flux = snr ** 2

    prof = psf._profile(params)
    if du != 0 or dv != 0:
        prof = prof.shift(du, dv)
    prof = prof * flux
    star = psf.drawProfileStar(star, prof, params)

    if snr != 0:
        image = star.image.array
        weight = 1. / image
        star.weight.array[:] = weight
        star.image.array[:] = np.random.normal(size=image.shape, scale=np.sqrt(image)) + image
        star = star.withFlux(flux)

    return star

def make_sample(nsample=5000, snr=0, stamp_size=24, jmax=21,
                logger=None, **kwargs):

    # make PSF object
    config = {  'optical_psf_kwargs':
                {
                    'template': 'des',
                },
            'shape_weights': [0.5, 1, 1],
            'fov_radius': 1. * 60 * 60,  # TODO: figure this out!!
            'jmax_pupil': jmax,
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

    shapes_hsm = []
    shapes_lmfit = []
    shape_errors_hsm = []
    shape_errors_lmfit = []
    params_all = []

    for i in range(nsample):
        # draw g0
        g0 = np.random.random_sample() * (1.5 - 0.5) + 0.5
        g1 = np.random.random_sample() * (0.1 + 0.1) - 0.1
        g2 = np.random.random_sample() * (0.1 + 0.1) - 0.1
        z = np.random.random_sample(size=(jmax-3)) * (1 + 1) - 1
        params = np.hstack((g0, g1, g2, z))

        # make star
        star = make_opt_star(params, psf, decaminfo, du=0, dv=0, snr=snr, stamp_size=stamp_size)
        # measure shapes
        try:
            shape_hsm, error_hsm = psf.measure_shape_hsm(star, shape_unnormalized=True, return_error=True)
            shape_lmfit, error_lmfit = psf.measure_shape_lmfit(star, shape_unnormalized=True, return_error=True)
        except:
            continue

        shapes_hsm.append(shape_hsm)
        shape_errors_hsm.append(error_hsm)
        shapes_lmfit.append(shape_lmfit)
        shape_errors_lmfit.append(error_lmfit)
        params_all.append(params)

        # logger debug
        if i % 1000 == 0:
            cmd = logger.warn
        elif i % 100 == 0:
            cmd = logger.info
        else:
            cmd = logger.debug
        cmd('Measured shape of star {0} / {1}'.format(i, nsample))
        param_str = '***Params: '
        for parami in params:
            param_str += '{0:+.2f}, '.format(parami)
        cmd(param_str)
        shape_str = '***Shape: '
        for shapej in shape_lmfit:
            shape_str += '{0:+.2f}, '.format(shapej)
        cmd(shape_str)

    shapes_hsm = np.array(shapes_hsm)
    shape_errors_hsm = np.array(shape_errors_hsm)
    shapes_lmfit = np.array(shapes_lmfit)
    shape_errors_lmfit = np.array(shape_errors_lmfit)
    params = np.array(params_all)
    return shapes_hsm, shape_errors_hsm, shapes_lmfit, shape_errors_lmfit, params

def evaluate_sample(shapes, params, Nsample=400000, logger=None):
    if Nsample < len(params):
        # let's simplify this a bit by choosing a subsample
        choice = np.random.choice(len(params), Nsample, replace=False)
        params = params[choice]
        shapes = shapes[choice]

    params_onehot = np.vstack((np.ones(len(params)).T, np.ones(len(params)).T, params.T)).T # extra set of ones which we turn into 1/size
    params_onehot[:, 1] = 1. / params_onehot[:, 2]

    # create list of polynomials, including names
    nvar = params_onehot.shape[1]
    jmax = nvar - 2
    names = ['', '1/g0 ', 'g0 ', 'g1 ', 'g2 '] + ['z{0:02d} '.format(i) for i in range(4, jmax + 1)]
    name_indices = range(len(names))

    coefficient_names = []
    coef0 = []
    indices = []
    # now the complicated rules making
    # we allow g0**-2 through to g0 ** 2, and for that to be paired with double order g1,g2
    name_ell = ''
    ell = 0
    for i, name_i in zip(name_indices[:3], names[:3]):
        for j, name_j in zip(name_indices[:3], names[:3]):
            if i > j:
                continue
            for k, name_k in zip([0, 3, 4], ['', 'g1 ', 'g2 ']):
                if i == 1 and j == 2:
                    continue
                if i == 2 and j == 1:
                    continue
                name = '{0}{1}{2}{3}'.format(name_ell, name_i, name_j, name_k)[:-1]
                coefficient_names.append(name)
                indices.append([ell, i, j, k])
                coef0.append(1)
    # we allow pairs of zernikes to be mixed with g0, g0**2, g0**-1, g0**-2
    for ell, name_ell in zip(name_indices[:3], names[:3]):
        for i, name_i in zip(name_indices[:3], names[:3]):
            if ell > i:
                continue
            for j, name_j in zip(name_indices[3:], names[3:]):
                for k, name_k in zip(name_indices[3:], names[3:]):
                    if i == 1 and ell == 2:
                        continue
                    if i == 2 and ell == 1:
                        continue
                    if k < j:
                        continue
                    name = '{0}{1}{2}{3}'.format(name_ell, name_i, name_j, name_k)[:-1]
                    coefficient_names.append(name)
                    indices.append([ell, i, j, k])
                    coef0.append(1)

    # turn into 64 bit arrays
    indices = np.array(indices).astype(np.int64)
    coef0 = np.array(coef0).astype(np.float64)
    # create basic polynomial with cypoly
    Xpoly = cypoly_full(params_onehot, indices)

    # fit for the shapes
    models = []
    model_objs = []
    rmss = []
    for i in range(3, 6):
        logger.info('Fitting Shape {0}'.format(i))
        y = shapes[:, i]
        alphas = [5e-4]
        model_objs.append([])
        stdbest = 1e100
        best_i = 0
        for alpha_i, alpha in enumerate(alphas):
            model = ElasticNet(alpha=alpha, fit_intercept=False, selection='random', max_iter=100000, copy_X=True)
            # model = LinearRegression(fit_intercept=False, copy_X=True)
            # model.n_iter_ = 1
            model.fit(Xpoly, y)
            model_objs[-1].append(model)
            conds = np.abs(model.coef_) > 0
            yp = model.predict(Xpoly)
            rms = np.std(y - yp)# * np.sqrt(np.sum(conds))  # penalize by number of non-zero coefs, but be gentle about it
            rmss.append(rms)
            if rms < stdbest:
                best_i = alpha_i
                stdbest = rms


        # choose the best model to put into models
        coefs = model_objs[-1][best_i].coef_
        model = model_objs[-1][best_i]
        alpha = alphas[best_i]
        # print some info
        logger.info('best alpha is {0:2e} at rms of {1:2e}, with {2} non-zero coefficients'.format(alphas[best_i], stdbest, np.sum(np.abs(coefs) > 0)))
        logger.info('\n"""')
        logger.info('{0} {1} coeffs,\nstd linear_model {2:.02e}'.format(['T', 'g1', 'g2'][i - 3], np.sum(conds),
            rms))
        logger.info('{0}:\t{1:+.02e}'.format('intercept_', model.intercept_))
        logger.info('{0}:\t{1:+.02e}'.format('alpha_', alpha))
        logger.info('{0}:\t{1:+.02e}'.format('n_iter_', model.n_iter_))
        for ith in range(len(conds)):
            if conds[ith]:
                logger.info('{2} {0}:\t{1:+.02e}'.format(coefficient_names[ith], model.coef_[ith], ith))
        logger.info('"""\n')
        # intercept is dealt with in the afterburner
        models.append(coefs)
    models = np.array(models)

    # make sense of fitted parameters
    indices_final = []
    coefs = []
    for i in range(3):
        nonzero = np.abs(models[i]) > 0
        indices_final.append(indices[nonzero])
        coefs.append(models[i][nonzero])
    indices_final = np.array(indices_final)
    coefs = np.array(coefs)

    # fit a final overall correction to these based just on a linear model
    shapes_pred = np.array([cypoly(params_onehot, coef, index)
                           for coef, index in zip(coefs, indices_final)]).T
    after_burners = []
    for i in range(3):
        y = shapes[:, i + 3]
        ypred = shapes_pred[:, i]
        ysk = model_objs[i][0].predict(Xpoly)
        lmparams = lmfit.Parameters()
        lmparams.add('m', value=1, vary=True)
        lmparams.add('b', value=0, vary=True)
        results = lmfit.minimize(linear_resid, lmparams, args=(ypred, y))
        logger.info('After burner correction for shape {0}'.format(i))
        m, b = results.params.valuesdict().values()
        after_burners.append([b, m])
        logger.info(lmfit.fit_report(results))

    logger.info('Final RMSs: {0}'.format(str(rmss)))

    return coefs, indices_final, after_burners

if __name__ == '__main__':
    # parse args
    import argparse
    description = 'Build analytic relation between OptAtmoPSF coefs and shapes'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--stamp_size', type=int, default=24)
    parser.add_argument('--jmax', type=int, default=21)
    parser.add_argument('--nsample', type=int, default=10000)
    parser.add_argument('--snr', type=int, default=0)
    parser.add_argument('--make_shapes', action='store_true')
    parser.add_argument('--combine_shapes', action='store_true')
    parser.add_argument('--fit_shapes', action='store_true')
    parser.add_argument('--max_indices', type=int, default=1000, help='Number of different shapes to collate for the combine step.')
    parser.add_argument('--arr_indx', type=int, default=0)
    parser.add_argument('--out', default='.')
    parser.add_argument('-v', '--verbose', type=int, action='store', default=2)
    args = parser.parse_args()
    kwargs = vars(args)
    logger = piff.config.setup_logger(verbose=args.verbose)

    base_name = kwargs['out'] + '/analytic_monte_carlo__i_{0:03d}__snr_{1:03d}__nsample_{2:03d}__jmax_{3:02d}__stamp_size_{4:02d}'.format(kwargs['arr_indx'], kwargs['snr'], kwargs['nsample'], kwargs['jmax'], kwargs['stamp_size'])

    if kwargs['make_shapes'] > 0:
        # make samples
        shapes_hsm, shape_errors_hsm, shapes_lmfit, shape_errors_lmfit, params = make_sample(logger=logger, **kwargs)
        # save samples
        np.save(base_name + '__shapes_hsm.npy', shapes_hsm)
        np.save(base_name + '__shape_errors_hsm.npy', shape_errors_hsm)
        np.save(base_name + '__shapes_lmfit.npy', shapes_lmfit)
        np.save(base_name + '__shape_errors_lmfit.npy', shape_errors_lmfit)
        np.save(base_name + '__params.npy', params)

    # combine shapes if arr_indx == 0
    if kwargs['combine_shapes']:
        # I think there are 1k samples??
        first_done = False
        imax = kwargs['max_indices']
        idone = 0
        shapes_hsm = []
        shape_errors_hsm = []
        shapes_lmfit = []
        shape_errors_lmfit = []
        params = []
        for i in range(1, imax + 1):
            base_name_i = kwargs['out'] + '/analytic_monte_carlo__i_{0:03d}__snr_{1:03d}__nsample_{2:03d}__jmax_{3:02d}__stamp_size_{4:02d}'.format(i, kwargs['snr'], kwargs['nsample'], kwargs['jmax'], kwargs['stamp_size'])
            try:
                i_shapes_hsm = np.load(base_name_i + '__shapes_hsm.npy')
                i_shape_errors_hsm = np.load(base_name_i + '__shape_errors_hsm.npy')
                i_shapes_lmfit = np.load(base_name_i + '__shapes_lmfit.npy')
                i_shape_errors_lmfit = np.load(base_name_i + '__shape_errors_lmfit.npy')
                i_params = np.load(base_name_i + '__params.npy')
                idone += 1
            except IOError:
                continue

            shapes_hsm.append(i_shapes_hsm)
            shape_errors_hsm.append(i_shape_errors_hsm)
            shapes_lmfit.append(i_shapes_lmfit)
            shape_errors_lmfit.append(i_shape_errors_lmfit)
            params.append(i_params)

        shapes_hsm = np.vstack(shapes_hsm)
        shape_errors_hsm = np.vstack(shape_errors_hsm)
        shapes_lmfit = np.vstack(shapes_lmfit)
        shape_errors_lmfit = np.vstack(shape_errors_lmfit)
        params = np.vstack(params)

        logger.warn('Saving {0} out of {1} files. {2} shapes sampled'.format(idone, imax, len(params)))
        # save samples
        np.save(base_name + '__shapes_hsm.npy', shapes_hsm)
        np.save(base_name + '__shape_errors_hsm.npy', shape_errors_hsm)
        np.save(base_name + '__shapes_lmfit.npy', shapes_lmfit)
        np.save(base_name + '__shape_errors_lmfit.npy', shape_errors_lmfit)
        np.save(base_name + '__params.npy', params)

    # evaluate model if arr_indx == -1
    if kwargs['fit_shapes']:
        shapes_hsm = np.load(base_name + '__shapes_hsm.npy')
        shape_errors_hsm = np.load(base_name + '__shape_errors_hsm.npy')
        shapes_lmfit = np.load(base_name + '__shapes_lmfit.npy')
        shape_errors_lmfit = np.load(base_name + '__shape_errors_lmfit.npy')
        params = np.load(base_name + '__params.npy')

        if kwargs['normalized']:
            base_name += '__normalized'
            shapesnew = shapes_hsm.copy()
            for i, yj in enumerate(shapesnew):
                new_shape = np.array(piff.OptAtmoPSF.shape_convert_to_normalized(*yj[3:]))
                shapesnew[i, 3:] = new_shape
            shapes_hsm = shapesnew
            shapesnew = shapes_lmfit.copy()
            for i, yj in enumerate(shapesnew):
                new_shape = np.array(piff.OptAtmoPSF.shape_convert_to_normalized(*yj[3:]))
                shapesnew[i, 3:] = new_shape
            shapes_lmfit = shapesnew

        coefs_hsm, indices_hsm, after_burners_hsm = evaluate_sample(shapes_hsm, params, logger=logger)
        np.save(base_name + '__coefs_hsm.npy', [coefs_hsm, indices_hsm, after_burners_hsm])
        coefs_lmfit, indices_lmfit, after_burners_lmfit = evaluate_sample(shapes_lmfit, params, logger=logger)
        np.save(base_name + '__coefs_lmfit.npy', [coefs_lmfit, indices_lmfit, after_burners_lmfit])

