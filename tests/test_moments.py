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
import os

from piff.util import calculate_moments
from piff_test_helper import timer

def makeStars(nstar, beta, size=None, g1=None, g2=None, seed=12345):

    rng = galsim.BaseDeviate(seed)
    np_rng = np.random.default_rng(seed)
    flux = 1.e6
    sky_level = 200.0  # For noise

    # Use a random DECam CCD for the wcs
    decaminfo = piff.des.DECamInfo()
    wcs = decaminfo.get_nominal_wcs(chipnum=10)

    # pick random size, shape if not given
    if size is None:
        size = np_rng.uniform(0.5, 0.9, nstar)
    else:
        size = np.array([size]*nstar)
    if g1 is None:
        g1 = np_rng.uniform(-0.1, 0.1, nstar)
    else:
        g1 = np.array([g1]*nstar)
    if g2 is None:
        g2 = np_rng.uniform(-0.1, 0.1, nstar)
    else:
        g2 = np.array([g2]*nstar)

    noiseless_stars = []
    noisy_stars = []

    for i in range(nstar):

        # Use Moffat profile
        prof = galsim.Moffat(half_light_radius=size[i], beta=beta).shear(g1=g1[i], g2=g2[i])

        # make the star with no noise.
        noiseless_star = piff.Star.makeTarget(x=0, y=0, wcs=wcs, stamp_size=19)
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
        copy_star.weight[5, 5] = 0
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
        calculate_moments(star)

    # If it's just really noisy, then HSM will return an error flag, which Piff turns into a
    # RuntimeError
    _, noisy_stars = makeStars(nstar=1, beta=5.)
    star = noisy_stars[0]
    rng = galsim.BaseDeviate(1234)
    star.data.image.addNoise(galsim.GaussianNoise(sigma=2.e7, rng=rng))
    with np.testing.assert_raises(RuntimeError):
        calculate_moments(star)

@timer
def test_moments_errors():

    ntests = 10
    np_rng = np.random.default_rng(1234)

    # Test N different stars with random choices for beta, g1, g2, size within reasonable ranges.
    betalist = np_rng.uniform(1.5, 6, size=ntests)
    g1list = np_rng.uniform(-0.05, 0.05, size=ntests)
    g2list = np_rng.uniform(-0.05, 0.05, size=ntests)
    sizelist = np_rng.uniform(0.5, 0.9, size=ntests)

    # The error estimates aren't super precise.
    # Here are the tolerances we need to get these tests to pass.  (To the nearest 0.05.)
    tol_dict = dict(M00=0.15,
                    M10=0.10, M01=0.10,
                    M11=0.15, M20=0.10, M02=0.10,
                    M21=0.20, M12=0.20, M30=0.15, M03=0.15,
                    M22=0.45, M31=0.40, M13=0.40, M40=0.35, M04=0.40,
                    M22n=0.35, M33n=0.35, M44n=0.40)

    for k in range(ntests):
        # Use a single value of beta, g1, g2, size for each run so empirical variance is accurate.
        beta = betalist[k]
        g1 = g1list[k]
        g2 = g2list[k]
        size = sizelist[k]
        print('beta, g1, g2, size = ',beta,g1,g2,size)

        # Calculate empirical variances from 1000 noise realizations.
        # These are saved in a file in the repo.  To regenerate, just delete the file.
        file_name = f'input/test_moments_errors_{beta:.2f}_{g1:.2f}_{g2:.2f}_{size:.2f}.npz'
        print(file_name)
        if not os.path.isfile(file_name):
            noiseless_moments = []
            noisy_moments = []
            var_moments = []
            noiseless_stars, noisy_stars = makeStars(nstar=1000, seed=31415,
                                                     beta=beta, g1=g1, g2=g2, size=size)
            for noiseless_star, noisy_star in zip(noiseless_stars, noisy_stars):
                moments = calculate_moments(noiseless_star, errors=False,
                                            third_order=True, fourth_order=True, radial=True)
                moments_noise = calculate_moments(noisy_star, errors=True,
                                                third_order=True, fourth_order=True, radial=True)
                noiseless_moments.append(moments)
                noisy_moments.append(moments_noise[:18])
                var_moments.append(moments_noise[18:])

            # Transpose from arrays of moments by star to array of values by moment nameI
            noiseless_moments = np.column_stack(noiseless_moments)
            noisy_moments = np.column_stack(noisy_moments)
            var_moments = np.column_stack(var_moments)

            np.savez(file_name,
                     noiseless_moments=noiseless_moments,
                     noisy_moments=noisy_moments,
                     var_moments=var_moments)

        data = np.load(file_name)
        noiseless_moments = data['noiseless_moments']
        noisy_moments = data['noisy_moments']
        var_moments = data['var_moments']

        # Check that the pulls seem reasonable.
        for k, name in enumerate(tol_dict.keys()):
            diff = noisy_moments[k] - noiseless_moments[k]
            pull = diff / np.sqrt(np.mean(var_moments[k]))
            print("{}: mean, rms pull = {:0.2f}, {:0.2f}".format(
                    name, np.mean(pull), np.std(pull)))
            np.testing.assert_allclose(np.mean(pull), 0., atol=0.2)
            np.testing.assert_allclose(np.std(pull), 1., atol=tol_dict[name])

        # Now test a single star to check that the error estimates match the empirical estimates.
        _, noisy_stars = makeStars(nstar=1, beta=beta, g1=g1, g2=g2, size=size)
        star = noisy_stars[0]
        moments = calculate_moments(star, errors=True,
                                    third_order=True, fourth_order=True, radial=True)
        for k, name in enumerate(tol_dict.keys()):
            mom = moments[k]
            var = moments[k+18]
            print("{}: mom, var = {:0.2f}, {:0.2e}  cf. {:0.2f}, {:0.2e}".format(
                    name, mom, var, np.mean(noisy_moments[k]), np.var(noisy_moments[k])))
            np.testing.assert_allclose(mom, np.mean(noisy_moments[k]), rtol=0.2, atol=0.01)
            # Note: 2x tol since var now rather than std.
            np.testing.assert_allclose(var, np.var(noisy_moments[k]), rtol=2*tol_dict[name])


if __name__ == "__main__":
    test_moments_return()
    test_moments_mask()
    test_moments_options()
    test_moments_fail()
    test_moments_errors()
