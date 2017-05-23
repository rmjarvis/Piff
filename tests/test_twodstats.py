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
import piff
import os

from piff_test_helper import timer


@timer
def test_twodstats():
    """Make sure we can execute and print a readout of the plot
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
    else:
        logger = None

    model = piff.Gaussian(fastfit=True)
    interp = piff.Polynomial(order=1)  # should find that order=1 is better
    # create background model
    stars, true_model = generate_starlist(100)
    psf = piff.SimplePSF(model, interp)
    psf.fit(stars, None, None)

    # check the coeffs of sigma and g2, which are actually linear fits
    # skip g1 since it is actually a 2d parabola
    # factor of 0.263 is to account for going from pixel xy to wcs uv
    np.testing.assert_almost_equal(psf.interp.coeffs[0].flatten(), np.array([0.4, 0, 1. / (0.263 * 2048), 0]), decimal=4)
    np.testing.assert_almost_equal(psf.interp.coeffs[2].flatten(), np.array([-0.1 * 1000 / 2048, 0, 0.1 / (0.263 * 2048), 0]), decimal=4)

    stats = piff.TwoDHistStats(number_bins_u=5, number_bins_v=5, reducing_function='np.mean')
    stats.compute(psf, stars, logger=logger)
    # check the twodhists
    # get the average value in the bin
    u_i = 3
    v_i = 3
    icen = stats.twodhists['u'][v_i, u_i] / 0.263
    jcen = stats.twodhists['v'][v_i, u_i] / 0.263
    print('icen = ',icen)
    print('jcen = ',jcen)
    icenter = 1000
    jcenter = 2000
    # the average value in the bin should match up with the model for the average coordinates
    sigma, g1, g2 = psf_model(icen, jcen, icenter, jcenter)
    sigma_average = stats.twodhists['T'][v_i, u_i]
    g1_average = stats.twodhists['g1'][v_i, u_i]
    g2_average = stats.twodhists['g2'][v_i, u_i]
    # assert equal to 4th decimal
    print('sigma, g1, g2 = ',[sigma,g1,g2])
    print('av sigma, g1, g2 = ',[sigma_average,g1_average,g2_average])
    np.testing.assert_almost_equal([sigma, g1, g2], [sigma_average, g1_average, g2_average],
                                   decimal=2)

    # Test the plotting and writing
    twodstats_file = os.path.join('output','twodstats.pdf')
    stats.write(twodstats_file)

    # repeat for whisker
    stats = piff.WhiskerStats(number_bins_u=21, number_bins_v=21, reducing_function='np.mean')
    stats.compute(psf, stars)
    # Test the plotting and writing
    twodstats_file = os.path.join('output','whiskerstats.pdf')
    stats.write(twodstats_file)

def make_star(icen=500, jcen=700, ccdnum=28,
              sigma=1, g1=0, g2=0,
              pixel_to_focal=False,
              properties={},
              fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    stardata = piff.Star.makeTarget(x=icen, y=jcen, properties=properties,
                                    scale=0.263)
    # apply Gaussian sigma, g1, g2
    params = np.array([sigma, g1, g2])

    starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(stardata.data, starfit)

    return star

def psf_model(icens, jcens, icenter, jcenter):
    sigmas = icens * (2. - 1.) / 2048. + 0.4
    g1s = ((jcens - jcenter) / 4096.) ** 2 * -0.2
    g2s = (icens - icenter) * 0.1 / 2048.
    return sigmas, g1s, g2s

def generate_starlist(n_samples=500):
    # create n_samples images from the 63 ccds and pixel coordinates
    np_rng = np.random.RandomState(1234)
    icens = np_rng.randint(100, 2048, n_samples)
    jcens = np_rng.randint(100, 4096, n_samples)
    ccdnums = np_rng.randint(1, 63, n_samples)
    icenter = 1000
    jcenter = 2000

    # throw out any icens and jcens that are within 400 pixels of the center
    conds = (np.abs(icens - icenter) > 400) | (np.abs(jcens - jcenter) > 400)
    icens = icens[conds]
    jcens = jcens[conds]
    ccdnums = ccdnums[conds]

    sigmas, g1s, g2s = psf_model(icens, jcens, icenter, jcenter)

    # throw in a 2d polynomial function for sigma g1 and g2
    # all sigma > 0, all g1 < 0, and g2 straddles.

    star_list = [make_star(icen, jcen, ccdnum, sigma, g1, g2)
                 for icen, jcen, ccdnum, sigma, g1, g2
                 in zip(icens, jcens, ccdnums, sigmas, g1s, g2s)]

    # load up model and draw the stars
    model = piff.Gaussian(fastfit=True)
    star_list = [model.draw(star) for star in star_list]
    star_list = [model.initialize(star) for star in star_list]
    star_list = [model.fit(star) for star in star_list]

    return star_list, model

if __name__ == '__main__':
    test_twodstats()
    # yaml test is in test_simple
