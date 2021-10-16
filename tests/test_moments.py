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
import piff
import galsim
import copy

from piff.util import calculate_moments
from piff_test_helper import timer

def makeStars(nstar, beta):

    rng = galsim.BaseDeviate(12345)
    np_rng = np.random.default_rng(12345)
    flux = 1.e6
    sky_level = 200.0  # For noise

    # Use a random DECam CCD for the wcs
    decaminfo = piff.des.DECamInfo()
    wcs = decaminfo.get_nominal_wcs(chipnum=10)

    # set the pixel index randomly x,y
    x = np_rng.uniform(0,2048,nstar)
    y = np_rng.uniform(0,4096,nstar)

    # pick random size, shape
    size = np_rng.uniform(0.7,1.2,nstar)
    g1 = np_rng.uniform(-0.1, 0.1, nstar)
    g2 = np_rng.uniform(-0.1, 0.1, nstar)

    noiseless_stars = []
    noisy_stars = []

    for i in range(nstar):

        # Use Moffat profile
        prof = galsim.Moffat(half_light_radius=size[i], beta=beta).shear(g1=g1[i], g2=g2[i])

        # make the star with no noise.
        noiseless_star = piff.Star.makeTarget(x=x[i], y=y[i], wcs=wcs, stamp_size=19)
        prof = prof.shift(noiseless_star.fit.center).withFlux(flux)
        im = noiseless_star.image
        prof.drawImage(image=im, center=noiseless_star.image_pos)
        noiseless_stars.append(noiseless_star)

        # Generate a Poisson noise model, with some foreground (assumes that this foreground
        # was already subtracted)
        poisson_noise = galsim.PoissonNoise(rng,sky_level=sky_level)
        im = im.copy()
        im.addNoise(poisson_noise)  # adds in place

        # get new weight in photo-electrons (not an array)
        inverse_weight = im + sky_level
        weight = 1.0/inverse_weight

        # make new noisy star by resetting data in the noiseless star
        noisy_star = copy.deepcopy(noiseless_star)
        noisy_star.data.image = im
        noisy_star.data.weight = weight
        noisy_stars.append(noisy_star)

    return noiseless_stars, noisy_stars

@timer
def test_moments_return():
    # Check that the same values are returned if errors=False
    noiseless_stars, _ = makeStars(nstar=3, beta=5.)

    for star in noiseless_stars:
        moments = calculate_moments(star, errors=True,
                                    third_order=True, fourth_order=True, radial=True)
        moments_check = calculate_moments(star, errors=False,
                                          third_order=True, fourth_order=True, radial=True)
        nval = len(moments_check)
        np.testing.assert_equal(np.array(moments)[0:nval], np.array(moments_check))

@timer
def test_moments_mask():
    # Check that the moments do change when the masking changes.
    _, noisy_stars = makeStars(nstar=20, beta=5.)

    for star in noisy_stars:
        moments_noise = calculate_moments(star, errors=True,
                                          third_order=True, fourth_order=True, radial=True)

        copy_star = copy.deepcopy(star)
        # mask out a pixel that is not too close to the center of the stamp.
        copy_star.data.weight[star.image_pos.x+5, star.image_pos.y+5] = 0
        test_moments = calculate_moments(copy_star, errors=True,
                                         third_order=True, fourth_order=True, radial=True)
        # make sure that they did actually change
        assert (np.array(moments_noise) != np.array(test_moments)).all()

@timer
def test_moments_options():
    # Check that when not returning some moments, the other ones are unchanged.
    noiseless_stars, _ = makeStars(nstar=2, beta=5.)

    for star in noiseless_stars:
        moments = calculate_moments(star, errors=True,
                                    third_order=True, fourth_order=True, radial=True)

        moments_34 = calculate_moments(star, errors=True,
                                       third_order=True, fourth_order=True, radial=False)
        # No radial moments, skip items 16-18 of moments and errors
        mask_34 = [True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, False, False, False,
                   True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, False, False, False]
        np.testing.assert_equal(np.array(moments)[mask_34], moments_34)

        moments_3r = calculate_moments(star, errors=True,
                                       third_order=True, fourth_order=False, radial=True)
        # No third order moments, skip items 11-15 of moments and errors
        mask_3r = [True, True, True, True, True, True, True, True, True, True,
                   False, False, False, False, False, True, True, True,
                   True, True, True, True, True, True, True, True, True, True,
                   False, False, False, False, False, True, True, True]
        np.testing.assert_equal(np.array(moments)[mask_3r], moments_3r)

        moments_4r = calculate_moments(star, errors=True,
                                       third_order=False, fourth_order=True, radial=True)
        # No third order moments, skip items 7-10 of moments and errors
        mask_4r = [True, True, True, True, True, True, False, False, False, False,
                   True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, False, False, False, False,
                   True, True, True, True, True, True, True, True]
        np.testing.assert_equal(np.array(moments)[mask_4r], moments_4r)

        moments_12 = calculate_moments(star, errors=True)
        # Default is only up to 2nd order, so 0-5
        mask_12 = [True, True, True, True, True, True, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   True, True, True, True, True, True, False, False, False, False,
                   False, False, False, False, False, False, False, False]
        np.testing.assert_equal(np.array(moments)[mask_12], moments_12)

        moments_def = calculate_moments(star)
        # Default is that but also without error terms.
        mask_def = [True, True, True, True, True, True, False, False, False, False,
                    False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False]
        np.testing.assert_equal(np.array(moments)[mask_def], moments_def)

@timer
def test_moments_fail():
    # If the weight is 0 everywhere, HSM will fail.
    _, noisy_stars = makeStars(nstar=1, beta=5.)

    star = noisy_stars[0]
    star.data.weight *= 0.

    with np.testing.assert_raises(galsim.GalSimHSMError):
        moments_noise = calculate_moments(star, errors=True,
                                          third_order=True, fourth_order=True, radial=True)

@timer
def test_moments_errors():

    betalist = [1.5, 2.5, 5.]

    tol_dict = dict(M00=2.00,
                    M10=0.10, M01=0.10,
                    M11=0.35, M20=0.15, M02=0.15,
                    M21=0.10, M12=0.10, M30=0.10, M03=0.10,
                    M22=0.50, M31=0.30, M13=0.30, M40=0.30, M04=0.30,
                    M22n=0.15, M33n=0.20, M44n=0.20)

    for beta in betalist:
        noiseless_stars, noisy_stars = makeStars(nstar=1000, beta=beta)

        noiseless_moments = []
        noisy_moments = []
        var_moments = []
        for noiseless_star, noisy_star in zip(noiseless_stars, noisy_stars):
            moments = calculate_moments(noiseless_star, errors=False,
                                        third_order=True, fourth_order=True, radial=True)
            moments_noise = calculate_moments(noisy_star, errors=True,
                                              third_order=True, fourth_order=True, radial=True)
            noiseless_moments.append(moments)
            noisy_moments.append(moments_noise[:18])
            var_moments.append(moments_noise[18:])

        # Transpose from arrays of moments by star to array of values by moment name
        noiseless_moments = np.column_stack(noiseless_moments)
        noisy_moments = np.column_stack(noisy_moments)
        var_moments = np.column_stack(var_moments)

        for k, name in enumerate(tol_dict.keys()):
            diff = noisy_moments[k] - noiseless_moments[k]
            pull = diff / np.sqrt(var_moments[k])
            print("beta = {}, {}: mean, rms pull = {:0.2f}, {:0.2f}".format(
                    beta, name, np.mean(pull), np.std(pull)))
            np.testing.assert_allclose(np.mean(pull), 0., atol=0.1)
            np.testing.assert_allclose(np.std(pull), 1., atol=tol_dict[name])


if __name__ == "__main__":
    test_moments_return()
    test_moments_mask()
    test_moments_options()
    test_moments_fail()
    test_moments_errors()
