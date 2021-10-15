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

def makeStarsMoffat(*, nstar, beta, forcefail=False, test_return_error=False,
                    test_mask=False, test_options=False):

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

    noiseless_stars = []
    noisy_stars = []
    all_star_moments = []

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

        if forcefail:
            noisy_star.data.weight *= 0.

        noisy_stars.append(noisy_star)

        # moments
        moments = calculate_moments(star=noiseless_stars[i], errors=True,
                                    third_order=True, fourth_order=True, radial=True)
        moments_noise = calculate_moments(star=noisy_stars[i], errors=True,
                                          third_order=True, fourth_order=True, radial=True)

        if test_options:
            moments_34 = calculate_moments(star=noiseless_stars[i], errors=True,
                                           third_order=True, fourth_order=True, radial=False)
            # No radial moments, skip items 16-18 of moments and errors
            mask_34 = [True, True, True, True, True, True, True, True, True, True,
                       True, True, True, True, True, False, False, False,
                       True, True, True, True, True, True, True, True, True, True,
                       True, True, True, True, True, False, False, False]
            np.testing.assert_equal(np.array(moments)[mask_34], np.array(moments_34))

            moments_3r = calculate_moments(star=noiseless_stars[i], errors=True,
                                           third_order=True, fourth_order=False, radial=True)
            # No third order moments, skip items 11-15 of moments and errors
            mask_3r = [True, True, True, True, True, True, True, True, True, True,
                       False, False, False, False, False, True, True, True,
                       True, True, True, True, True, True, True, True, True, True,
                       False, False, False, False, False, True, True, True]
            np.testing.assert_equal(np.array(moments)[mask_3r], np.array(moments_3r))

            moments_4r = calculate_moments(star=noiseless_stars[i], errors=True,
                                           third_order=False, fourth_order=True, radial=True)
            # No third order moments, skip items 7-10 of moments and errors
            mask_4r = [True, True, True, True, True, True, False, False, False, False,
                       True, True, True, True, True, True, True, True,
                       True, True, True, True, True, True, False, False, False, False,
                       True, True, True, True, True, True, True, True]
            np.testing.assert_equal(np.array(moments)[mask_4r], np.array(moments_4r))

        if test_mask:
            copy_star = copy.deepcopy(noisy_star)
            # mask out a pixel that is not too close to the center of the stamp.
            copy_star.data.weight[x[i]+5, y[i]+5] = 0
            test_moments = calculate_moments(star=copy_star, errors=True,
                                             third_order=True, fourth_order=True, radial=True)
            # make sure that they did actually change
            assert (np.array(moments_noise) != np.array(test_moments)).all()

        if test_return_error:
            moments_check = calculate_moments(star=noiseless_stars[i], errors=False,
                                              third_order=True, fourth_order=True, radial=True)
            nval = len(moments_check)
            np.testing.assert_equal(np.array(moments)[0:nval], np.array(moments_check))


        all_moments = moments + moments_noise
        all_star_moments.append(all_moments)

    # Transpose from array of moments by star to array of values by moment name
    all_moms = np.column_stack(all_star_moments)
    df = dict(zip(moffat_names, all_moms))

    return df


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

    dft = makeStarsMoffat(nstar=1,beta=5.,test_return_error=True)

@timer
def test_moments_mask():

    dft = makeStarsMoffat(nstar=20,beta=5.,test_mask=True)

@timer
def test_moments_options():

    dft = makeStarsMoffat(nstar=2, beta=5.,test_options=True)

@timer
def test_moments_fail():

    with np.testing.assert_raises(galsim.GalSimHSMError):
        makeStarsMoffat(nstar=1,beta=5.,forcefail=True)

@timer
def test_moments():

    betalist = [1.5, 2.5, 5.]
    keylist = ["1p5", "2p5", "5"]
    dftlist = [makeStarsMoffat(nstar=1000,beta=betaval) for betaval in betalist]
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

    odict = {}
    for i, (dft, beta, key) in enumerate(zip(dftlist, betalist, keylist)):
        stacked = np.vstack([ makepulldist(dft, beta, amoment) for amoment in momentlist])
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
