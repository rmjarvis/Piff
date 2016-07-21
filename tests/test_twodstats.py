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

def test_twodstats():
    """Make sure we can execute and print a readout of the plot
    """

    model = piff.Gaussian()
    interp = piff.Polynomial(order=1)  # should find that order=1 is better
    # create background model
    stars, true_model = generate_starlist()
    psf = piff.SimplePSF(model, interp)
    psf.fit(stars, None, None)

    stats = piff.TwoDHistStats(number_bins_u=21, number_bins_v=21, reducing_function='np.mean')
    stats.compute(psf, stars)
    # Test the plotting and writing
    twodstats_file = os.path.join('output','twodstatsstats.png')
    stats.write(twodstats_file)
    fig, axs = stats.plot()
    import matplotlib.pyplot as plt
    plt.show()

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

def generate_starlist(n_samples=3000):
    # create n_samples images from the 63 ccds and pixel coordinates
    icens = np.random.randint(100, 2048, n_samples)
    jcens = np.random.randint(100, 4096, n_samples)
    ccdnums = np.random.randint(1, 63, n_samples)
    jcenter = 2000
    icenter = 1000
    # throw in a 2d polynomial function for sigma g1 and g2
    # all sigma > 0, all g1 < 0, and g2 straddles.
    sigmas = icens * (2. - 1.) / 2048. + 0.4
    g1s = ((jcens - jcenter) / 4096.) ** 2 * -0.2
    g2s = (icens - icenter) * 0.1 / 2048.

    star_list = [make_star(icen, jcen, ccdnum, sigma, g1, g2)
                 for icen, jcen, ccdnum, sigma, g1, g2
                 in zip(icens, jcens, ccdnums, sigmas, g1s, g2s)]

    # load up model and draw the stars
    model = piff.Gaussian()
    star_list = [model.draw(star) for star in star_list]
    star_list = [model.fit(star) for star in star_list]

    return star_list, model

if __name__ == '__main__':
    test_twodstats()
    # yaml test is in test_simple
