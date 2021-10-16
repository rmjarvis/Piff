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
        im = noiseless_star.image
        prof = prof.shift(noiseless_star.fit.center)
        prof.drawImage(image=im, center=noiseless_star.image_pos)
        noiseless_stars.append(noiseless_star)

        # Generate a Poisson noise model, with some foreground (assumes that this foreground
        # was already subtracted)
        poisson_noise = galsim.PoissonNoise(rng,sky_level=sky_level)
        im = noiseless_star.image * flux
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


def makepulldist(dft, beta, vname):

    name_noise = "%s_noise" % (vname)
    name_nonoise = "%s_nonoise" % (vname)
    name_sigma = "var%s_noise" % (vname)

    try:
        diff = dft[name_noise] - dft[name_nonoise]
    except KeyError:
        print (name_noise, name_nonoise, dft.keys())
    pull = diff/np.sqrt(dft[name_sigma])

    return pull

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
def test_moments():

    betalist = [1.5, 2.5, 5.]

    momentlist = ['M10','M01','M11','M20','M02','M21','M12','M30','M03',
                  'M31','M13','M40','M04','M22n','M33n','M44n']

    rmsval_dict = dict(M10=[1.024323, 0.996639,0.986769],
                       M01=[0.981495, 0.997452, 0.962892],
                       M11=[1.329671, 1.045360, 0.950834],
                       M20=[1.099344, 1.029830, 0.931873],
                       M02=[1.112284, 1.023089, 0.979789],
                       M21=[0.930467, 0.985090, 0.973814],
                       M12=[0.927560, 0.999851, 1.044756],
                       M30=[0.994757, 0.997164, 0.967364],
                       M03=[0.941321, 1.015403, 1.003081],
                       M31=[1.257617, 1.082664, 0.923452],
                       M13=[1.287733, 1.088511, 0.995732],
                       M40=[1.199421, 1.136400, 1.049415],
                       M04=[1.250599, 1.169380, 1.106795],
                       M22n=[0.879955, 0.985517, 1.017496],
                       M33n=[0.835669, 0.999379, 1.065365],
                       M44n=[0.809727, 1.021675, 1.119339])

    moment_str = ['M00','M10','M01','M11','M20','M02',
                  'M21', 'M12', 'M30', 'M03',
                  'M31','M13','M40','M04',
                  'M22dup', 'M22n','M33n','M44n',
                  'varM00','varM10','varM01','varM11','varM20','varM02',
                  'varM21', 'varM12', 'varM30', 'varM03',
                  'varM31','varM13','varM40','varM04',
                  'varM22dup','varM22n','varM33n','varM44n']

    # build names of columns
    moments_names = [s + "_nonoise" for s in moment_str]
    moments_noise_names = [s + "_noise" for s in moment_str]
    moffat_names = moments_names + moments_noise_names

    for i, beta in enumerate(betalist):
        noiseless_stars, noisy_stars = makeStars(nstar=1000, beta=beta)

        all_star_moments = []
        for noiseless_star, noisy_star in zip(noiseless_stars, noisy_stars):
            moments = calculate_moments(noiseless_star, errors=True,
                                        third_order=True, fourth_order=True, radial=True)
            moments_noise = calculate_moments(noisy_star, errors=True,
                                              third_order=True, fourth_order=True, radial=True)
            all_moments = moments + moments_noise
            all_star_moments.append(all_moments)

        # Transpose from array of moments by star to array of values by moment name
        all_moms = np.column_stack(all_star_moments)
        df = dict(zip(moffat_names, all_moms))

        stacked = np.vstack([ makepulldist(df, beta, amoment) for amoment in momentlist])
        testvals = np.array([ rmsval_dict[amoment][i] for amoment in momentlist])
        mean_pull = stacked.mean(axis=1)
        rms_pull = stacked.std(axis=1)
        failmask = np.fabs(rms_pull-testvals) > 0.1
        np.testing.assert_allclose(mean_pull, 0., atol=0.1)
        np.testing.assert_allclose(rms_pull, testvals, rtol=0.2)


if __name__ == "__main__":
    test_moments_return()
    test_moments_mask()
    test_moments_options()
    test_moments_fail()
    test_moments_moments()
