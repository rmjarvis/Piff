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
import numpy
import piff

"""
Test hyperpsf with gaussian as model, polynomial as interp. Solve gaussian background
"""

def test_solve():
    true_background = 10.5
    init_background = 5
    # fit background level of model
    model = piff.Gaussian(background=init_background)
    interp = piff.Polynomial(order=1)  # should find that order=1 is better
    # create background model
    stars, true_model = generate_starlist(background=true_background)

    model_keys = ['background']
    model_init = [init_background]
    model_error = [1]
    model_limit = [(0, 100)]
    # interp_keys = []
    # interp_init = []
    # interp_error = []
    # interp_limit = []

    fit_kwargs = {'model_keys': model_keys,
                  'model_init': model_init,
                  'model_error': model_error,
                  'model_limit': model_limit,
                  }
    hyperpsf = piff.HyperPSF.build(stars, model, interp, print_level=4, **fit_kwargs)

    # this should be bang on
    numpy.testing.assert_almost_equal(hyperpsf._minuit.fitarg['p0'], true_background, decimal=5)

def test_model_comparer():
    # here we want ot use a gaussian to compare our fits
    true_background = 10.5
    init_background = true_background
    # fit background level of model
    model = piff.Gaussian(background=init_background)
    interp = piff.Polynomial(order=1)  # should find that order=1 is better
    # create background model
    stars, true_model = generate_starlist(background=true_background)

    model_keys = ['background']
    model_init = [init_background]
    model_error = [5]
    model_limit = [(0, 100)]

    # make sure model_comparerer_weights works
    # test by giving case where shear is way off but not fitted, but size can
    # be fitted. Should fit size and not shear
    model_comparer_weights = numpy.array([1.0, 1.0, 0.0])

    fit_kwargs = {'model_keys': model_keys,
                  'model_init': model_init,
                  'model_error': model_error,
                  'model_limit': model_limit,
                  'model_comparer_weights': model_comparer_weights,
                  'model_comparer': piff.Gaussian(),
                  }
    hyperpsf = piff.HyperPSF.build(stars, model, interp, print_level=4, **fit_kwargs)
    # TODO: This performs AWFULLY. Maybe second moments are not so sensitive to sky background?

def test_limits():
    true_background = 10.5
    init_background = 5
    upper_limit = 8
    # fit background level of model
    model = piff.Gaussian(background=init_background)
    interp = piff.Polynomial(order=1)  # should find that order=1 is better
    # create background model
    stars, true_model = generate_starlist(background=true_background)

    model_keys = ['background']
    model_init = [init_background]
    model_error = [1]
    model_limit = [(0, upper_limit)]
    # interp_keys = []
    # interp_init = []
    # interp_error = []
    # interp_limit = []

    fit_kwargs = {'model_keys': model_keys,
                  'model_init': model_init,
                  'model_error': model_error,
                  'model_limit': model_limit,
                  }
    hyperpsf = piff.HyperPSF.build(stars, model, interp, print_level=0, **fit_kwargs)

    # we should be really close to the upper_limit
    numpy.testing.assert_almost_equal(hyperpsf._minuit.fitarg['p0'], upper_limit, decimal=5)

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

def plot_diffstar(star, star2):
    # convenience function
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.data.data - star2.data.data)
    plt.colorbar()
    plt.show()

def make_star(icen=500, jcen=700, ccdnum=28,
              sigma=1, g1=0, g2=0,
              pixel_to_focal=False,
              properties={},
              fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    stardata = piff.StarData.makeTarget(x=icen, y=jcen, properties=properties,
                                        scale=0.263)
    # apply Gaussian sigma, g1, g2
    params = numpy.array([sigma, g1, g2])

    starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(stardata, starfit)

    return star

def generate_starlist(n_samples=1000, background=10.5):
    # create n_samples images from the 63 ccds and pixel coordinates
    icens = numpy.random.randint(100, 2048, n_samples)
    jcens = numpy.random.randint(100, 4096, n_samples)
    ccdnums = numpy.random.randint(1, 63, n_samples)
    jcenter = 2000
    icenter = 1000
    # throw in a 1d poly nomial function for sigma g1 and g2
    sigmas = icens * (3. - 1.) / (icens.max() - icens.min()) + 0.5
    g1s = (jcens - jcenter) * 0.1 / (jcens.max() - jcens.min())
    g2s = (icens - icenter) * 0.1 / (icens.max() - icens.min())

    # sigmas = numpy.random.random(0.5, 2, n_samples)
    # g1s = numpy.random.normal(0, 0.1, n_samples)
    # g2s = numpy.random.normal(0, 0.1, n_samples)
    star_list = [make_star(icen, jcen, ccdnum, sigma, g1, g2)
                 for icen, jcen, ccdnum, sigma, g1, g2
                 in zip(icens, jcens, ccdnums, sigmas, g1s, g2s)]

    # load up model and draw the stars
    model = piff.Gaussian(background=background)
    star_list = [model.draw(star) for star in star_list]

    return star_list, model

if __name__ == '__main__':
    test_solve()
    test_limits()
    test_model_comparer()
