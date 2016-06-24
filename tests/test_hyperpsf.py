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
import numpy
import piff
import os
import yaml
import fitsio

def test_init():
    # should just be able to start a HyperPSF and build it
    pass

def test_model():
    # should only fit model, no interp
    pass

def test_interp():
    # should only fit interp, no model
    pass

def test_toomanyparameters():
    # should fail if we have >50 params
    pass

def test_limits():
    # model limits should work
    # test by giving a limit that doesn't include right answer. we should try to get close
    pass

def test_weights():
    # make sure model_comparerer_weights works
    # test by giving case where shear is way off but not fitted, but size can
    # be fitted. Should fit size and not shear
    pass


#####
# convenience functions
#####

# put in gaussian 

def plot_star(star):
    # convenience function
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(star.data.data)
    plt.colorbar()
    plt.show()

def make_empty_star(icen=500, jcen=700, ccdnum=28, params=None,
                    pixel_to_focal=False,
                    properties={},
                    fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    stardata = piff.StarData.makeTarget(x=icen, y=jcen, properties=properties,
                                        scale=0.263)

    if numpy.shape(params) == ():
        starfit = None
    else:
        starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(stardata, starfit)

    return star

def generate_starlist(n_samples=1000):
    # create n_samples images from the 63 ccds and pixel coordinates
    icens = numpy.random.randint(100, 2048, n_samples)
    jcens = numpy.random.randint(100, 4096, n_samples)
    ccdnums = numpy.random.randint(1, 63, n_samples)
    star_list = [make_empty_star(icen, jcen, ccdnum)
                 for icen, jcen, ccdnum in zip(icens, jcens, ccdnums)]

    # load up interpolator and interpolate
    star_list = interp.interpolateList(star_list)

    # load up model and draw the stars
    star_list = [model.draw(star) for star in star_list]

    return star_list

if __name__ == '__main__':
    test_init()
