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

# Code by Ares to test the quality of errors reported by lmfit.

def test_lmfit_errors():
    import lmfit
    Nsamples = 300
    for force_model_center in [True]:
        for include_pixel in [True]:
            for model_name, model_init in zip(['Kolmogorov', 'Gaussian'], [piff.Kolmogorov, piff.Gaussian]):
                for noise in [1e-4, 1e-3]:
                    model = model_init(fastfit=False, force_model_center=force_model_center, include_pixel=include_pixel)

                    params = np.array([0.2, -0.3, 1.2, -0.05, 0.07])
                    if force_model_center:
                        params = params[2:]

                    params_out = []
                    errors_out = []
                    for i in range(Nsamples):
                        star = piff.Star(model.draw(piff.Star(piff.Star.makeTarget(x=0, y=0, scale=0.263, stamp_size=32).data, piff.StarFit(params))).data, None)
                        # add noise
                        star.weight.fill(1. / noise ** 2)
                        gn = galsim.GaussianNoise(sigma=noise, rng=None)
                        star.image.addNoise(gn)

                        # fit
                        star = model.initialize(star)
                        # TODO: with PFL's PR, the errors will be a property of the fit
                        lmparams = model._lmfit_params(star)
                        results = model._lmfit_minimize(lmparams, star)
                        flux, du, dv, scale, g1, g2 = results.params.valuesdict().values()
                        # only care about the shape terms, really
                        params_out.append(np.array([scale, g1, g2]))
                        error = np.sqrt(np.diag(results.covar)[3:])
                        errors_out.append(error)
                    params_out = np.array(params_out)
                    errors_out = np.array(errors_out)
                    # rough estimate of whether errors are working as designed
                    pull = ((params_out - params) / errors_out)
                    pull_mean = np.mean(pull, axis=0)
                    pull_std = np.std(pull, axis=0)

                    # rather roughly speaking, we expect pull to be gaussian about 0 with std of 1
                    np.testing.assert_allclose(pull_mean, np.zeros(3), atol=0.2)
                    np.testing.assert_allclose(pull_std, np.ones(3), atol=0.2)

test_lmfit_errors()
