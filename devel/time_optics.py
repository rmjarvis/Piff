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

import numpy as np
from numpy.random import default_rng
import piff
import time

def test_makestars(nstars=100,constant_atmoparams=True,template='des'):
    t0 = time.time()
    print("test_makestars, n=%d, constant_atmoparams=%d" % (nstars,constant_atmoparams))
    model = piff.Optical(template=template,atmo_type='VonKarman',gsparams='starby2')
    params = random_params(nstars,model,constant_atmoparams=constant_atmoparams)
    stars = make_stars(nstars,model,params)
    t1 = time.time()
    print('Time for test_makestars = ',t1-t0)


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

def plot_param(r0=0.1, params=[], g1=0, g2=0, sigma=0):
    # convenience function
    model = piff.Optical(r0=r0, g1=g1, g2=g2, sigma=sigma, template='des')
    star = make_empty_star(params=params)
    star = model.draw(star)
    plot_star(star)
    return star.data.data

def make_empty_star(icen=500, jcen=700, ccdnum=28, params=None, stamp_size=40,
                    properties={},
                    fit_kwargs={}):

    properties['ccdnum'] = ccdnum
    # setting scale is crucial
    star = piff.Star.makeTarget(x=icen, y=jcen, stamp_size=stamp_size, properties=properties,
                                scale=0.263)

    if params is None:
        starfit = None
    else:
        starfit = piff.StarFit(params, **fit_kwargs)

    star = piff.Star(star.data, starfit)

    return star

def random_params(nstars,model,seed=12345,constant_atmoparams=False):
    # make up some random star parameters
    rng = default_rng(seed)
    z4s = rng.uniform(-0.2,0.2,nstars)
    z5s = rng.uniform(-0.2,0.2,nstars)
    z6s = rng.uniform(-0.2,0.2,nstars)
    z7s = rng.uniform(-0.2,0.2,nstars)
    z8s = rng.uniform(-0.2,0.2,nstars)
    z9s = rng.uniform(-0.2,0.2,nstars)
    z10s = rng.uniform(-0.2,0.2,nstars)
    z11s = rng.uniform(-0.2,0.2,nstars)

    r0s = rng.uniform(0.13,0.18,nstars)
    g1s = rng.uniform(-0.05,0.05,nstars)
    g2s = rng.uniform(-0.05,0.05,nstars)
    L0s = rng.uniform(5.0,15.0,nstars)

    if constant_atmoparams:
        r0s = 0.15 * np.ones(nstars)
        L0s = 7.5 * np.ones(nstars)

    params = []
    for i in range(nstars):
        param = model.kwargs_to_params(
            zernike_coeff=[0.,0.,0.,0.,z4s[i],z5s[i],z6s[i],z7s[i],z8s[i],z9s[i],z10s[i],z11s[i]],
            r0=r0s[i], g1=g1s[i], g2=g2s[i], L0=L0s[i])
        params.append(param)
    return params

def make_stars(nstars,model,inparams,npixels=19):
    # make a list of stars
    rng = np.random.default_rng(123459)
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = rng.choice(chiplist,nstars)
    pixedge = 20
    icen = rng.uniform(1+pixedge,2048-pixedge,nstars)   # random pixel position inside CCD
    jcen = rng.uniform(1+pixedge,4096-pixedge,nstars)

    # fill stars
    stars = []

    for i in range(nstars):
        # make the shell of a Star object
        star = make_empty_star(icen[i], jcen[i], chipnum[i], inparams[i], stamp_size=npixels)

        # draw the star
        star = model.draw(star)
        stars.append(star)

    return stars



if __name__ == '__main__':
    print("des template")
    test_makestars(nstars=100)
    test_makestars(nstars=100,constant_atmoparams=False)
    print("des template 128x128 pupil")
    test_makestars(nstars=100,template='des_128')
    test_makestars(nstars=100,constant_atmoparams=False,template='des_128')
    print("desparam")
    test_makestars(nstars=100,template='des_param')
    test_makestars(nstars=100,constant_atmoparams=False,template='des_param')
