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
import numpy as np
import piff
import os
import subprocess
import yaml
import fitsio

from piff_test_helper import get_script_name, timer

@timer
def test_compound():
    """This code tests the double PSF model. The wide-field piece is a
    polynomial gaussian, while the chip piece is a polynomial gaussian.
    """

    # true PSF params
    N_chips = 3
    n_per_row = 25
    N_stars_per_chip = n_per_row ** 2
    # set RNG
    np_rng = np.random.RandomState(1234)

    # generate grid of stars instead.
    ICENS = np.linspace(100, 2048 - 100, n_per_row)
    icens, jcens = np.meshgrid(ICENS, ICENS)
    icens = icens.flatten().tolist() * N_chips
    jcens = jcens.flatten().tolist() * N_chips
    ccdnums = []
    for i in range(5, 5 + N_chips):
        for j in range(n_per_row * n_per_row):
            ccdnums.append(i)

    # x0 gets shifted 2048 for each chip
    us = [icen + 2048 * ccdnum for (icen, ccdnum) in zip(icens, ccdnums)]
    centers_u = [2048 * (ccdnum + 0.5) for ccdnum in range(0, 5 + N_chips)]
    vs = jcens
    delta_u = N_chips * 2048
    center_u = (5 + (5 + N_chips)) * 2048 / 2.
    delta_v = 2048.
    center_v = 1024.
    # now generate the values for each star
    sigma_wide_vals = [0.5, 0.10, -0.10, 0, 0, 0]
    g1_wide_vals = [0, 0.10, -0.10, 0, 0, 0]
    g2_wide_vals = [0, 0.10, -0.10, 0, 0, 0]
    sigma_chip_vals = [0.5, -0.10, 0.10, 0.10, 0.10, 0.10]
    g1_chip_vals = [0, -0.10, 0.10, 0.10, 0.10, -0.10]
    g2_chip_vals = [0, -0.10, 0.10, 0.10, 0.10, -0.10]
    sigma_wide = [  sigma_wide_vals[0] +
                    sigma_wide_vals[1] * (u - center_u) / delta_u +
                    sigma_wide_vals[2] * (v - center_v) / delta_v +
                    sigma_wide_vals[3] * ((u - center_u) / delta_u) ** 2 +
                    sigma_wide_vals[4] * ((v - center_v) / delta_v) ** 2 +
                    sigma_wide_vals[5] * (u - center_u) / delta_u * (v - center_v) / delta_v
                    for (u, v) in zip(us, vs)]
    g1_wide = [ g1_wide_vals[0] +
                g1_wide_vals[1] * (u - center_u) / delta_u +
                g1_wide_vals[2] * (v - center_v) / delta_v +
                g1_wide_vals[3] * ((u - center_u) / delta_u) ** 2 +
                g1_wide_vals[4] * ((v - center_v) / delta_v) ** 2 +
                g1_wide_vals[5] * (u - center_u) / delta_u * (v - center_v) / delta_v
                for (u, v) in zip(us, vs)]
    g2_wide = [ g2_wide_vals[0] +
                g2_wide_vals[1] * (u - center_u) / delta_u +
                g2_wide_vals[2] * (v - center_v) / delta_v +
                g2_wide_vals[3] * ((u - center_u) / delta_u) ** 2 +
                g2_wide_vals[4] * ((v - center_v) / delta_v) ** 2 +
                g2_wide_vals[5] * (u - center_u) / delta_u * (v - center_v) / delta_v
                for (u, v) in zip(us, vs)]
    # now we do the chips. Because we only care about local chip coordinates,
    # which are all delta_v in size, we use those coordinates.
    sigma_chip = [  sigma_chip_vals[0] +
                    sigma_chip_vals[1] * (u - centers_u[ccdnum]) / delta_v +
                    sigma_chip_vals[2] * (v - center_v) / delta_v +
                    sigma_chip_vals[3] * ((u - centers_u[ccdnum]) / delta_v) ** 2 +
                    sigma_chip_vals[4] * ((v - center_v) / delta_v) ** 2 +
                    sigma_chip_vals[5] * (u - centers_u[ccdnum]) / delta_v * (v - center_v) / delta_v
                    for (u, v, ccdnum) in zip(us, vs, ccdnums)]
    g1_chip = [ g1_chip_vals[0] +
                g1_chip_vals[1] * (u - centers_u[ccdnum]) / delta_v +
                g1_chip_vals[2] * (v - center_v) / delta_v +
                g1_chip_vals[3] * ((u - centers_u[ccdnum]) / delta_v) ** 2 +
                g1_chip_vals[4] * ((v - center_v) / delta_v) ** 2 +
                g1_chip_vals[5] * (u - centers_u[ccdnum]) / delta_v * (v - center_v) / delta_v
                for (u, v, ccdnum) in zip(us, vs, ccdnums)]
    g2_chip = [ g2_chip_vals[0] +
                g2_chip_vals[1] * (u - centers_u[ccdnum]) / delta_v +
                g2_chip_vals[2] * (v - center_v) / delta_v +
                g2_chip_vals[3] * ((u - centers_u[ccdnum]) / delta_v) ** 2 +
                g2_chip_vals[4] * ((v - center_v) / delta_v) ** 2 +
                g2_chip_vals[5] * (u - centers_u[ccdnum]) / delta_v * (v - center_v) / delta_v
                for (u, v, ccdnum) in zip(us, vs, ccdnums)]

    true_params_all = np.vstack([sigma_wide, g1_wide, g2_wide, sigma_chip, g1_chip, g2_chip]).T

    # make star_list
    prof_list = []
    for indx in range(len(us)):
        sigma_w = sigma_wide[indx]
        g1_w = g1_wide[indx]
        g2_w = g2_wide[indx]
        prof_w = galsim.Gaussian(sigma=1.0).dilate(sigma_w).shear(g1=g1_w, g2=g2_w)
        sigma_c = sigma_chip[indx]
        g1_c = g1_chip[indx]
        g2_c = g2_chip[indx]
        prof_c = galsim.Gaussian(sigma=1.0).dilate(sigma_c).shear(g1=g1_c, g2=g2_c)
        prof_list.append(galsim.Convolve([prof_w, prof_c]))

    # draw images
    image_file = os.path.join('data','compound_image_{0:02d}.fits')
    cat_file = os.path.join('data','compound_cat_{0:02d}.fits')
    for ccdnum in np.unique(ccdnums):
        print('writing chip', ccdnum)
        # x0 gets shifted 2048 for each chip
        x0 = 2048 * ccdnum
        wcs = galsim.OffsetWCS(scale=0.26, origin=galsim.PositionD(-x0, 0))
        image = galsim.Image(2048, 2048, wcs=wcs)

        # select only stars with ccdnum property
        x_list = [icen for (icen, ccd) in zip(icens, ccdnums) if ccd == ccdnum]
        y_list = [jcen for (jcen, ccd) in zip(jcens, ccdnums) if ccd == ccdnum]
        prof_ccd = [prof for (prof, ccd) in zip(prof_list, ccdnums) if ccd == ccdnum]
        for x, y, prof in zip(x_list, y_list, prof_ccd):
            # write images
            bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
            offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
            prof.drawImage(image=image[bounds], method='no_pixel', offset=offset, add_to_image=True)

        image.write(image_file.format(ccdnum))

        # write tables
        dtype = [ ('x','f8'), ('y','f8')]
        data = np.empty(len(x_list), dtype=dtype)
        data['x'] = x_list
        data['y'] = y_list
        fitsio.write(cat_file.format(ccdnum), data, clobber=True)

    # Now test running it via the config parser
    psf_file = os.path.join('output','compound_psf.fits')
    config = {
        'input' : {
            'images' : os.path.join('data','compound_image_%02d.fits'),
            'cats' : os.path.join('data','compound_cat_%02d.fits'),
            'chipnums': "[ c for c in range(5, {0})]".format(5 + N_chips),
            # What hdu is everything in?
            'image_hdu' : 0,
            # 'badpix_hdu' : 2,
            # 'weight_hdu' : 3,
            'cat_hdu' : 1,

            # What columns in the catalog have things we need?
            'x_col' : 'x',
            'y_col' : 'y',

            # How large should the postage stamp cutouts of the stars be?
            'stamp_size' : 24,
        },
        'psf' : {
            'type': 'Compound',

            'psf_0': {
                'type': 'Simple',

                'model' : { 'type' : 'Gaussian',
                            'fastfit' : True},
                'interp' : { 'type' : 'Polynomial',
                            'order': 1,
                            },
            },

            'psf_1': {
                'type': 'SingleChip',

                'model' : { 'type' : 'Gaussian',
                            'fastfit' : True},
                'interp' : { 'type' : 'Polynomial',
                            'order': 2,
                            },
            }

        },
        'output' : { 'file_name' : psf_file,
            'stats': [
            {
                'type': 'ShapeHistograms',
                'file_name': os.path.join('output','compound_psf_shapestats.png')
            },
            {
                'type': 'Rho',
                'file_name': os.path.join('output','compound_psf_rhostats.png')
            },
            {
                'type': 'TwoDHist',
                'file_name': os.path.join('output', 'compound_psf_twodhiststats.png'),
                'number_bins_u': 30,
                'number_bins_v': 10
                }
            ]
    }
    }
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=0)

    with open('compound.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

    # Test using the piffify executable
    if os.path.exists(psf_file):
        os.remove(psf_file)
    config['verbose'] = 2
    with open('compound.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'compound.yaml'] )
    p.communicate()

    psf = piff.read(psf_file)
    # make sure that this psf is correct
    assert len(psf.psfs) == 2
    assert type(psf.psfs[0]) is piff.SimplePSF
    assert type(psf.psfs[0].model) is piff.Gaussian
    assert type(psf.psfs[0].interp) is piff.Polynomial
    assert type(psf.psfs[1]) is piff.SingleChipPSF
    assert len(psf.psfs[1].psf_by_chip) == N_chips
    assert type(psf.psfs[1].psf_by_chip[5]) is piff.SimplePSF
    assert type(psf.psfs[1].psf_by_chip[5].model) is piff.Gaussian
    assert type(psf.psfs[1].psf_by_chip[5].interp) is piff.Polynomial


    # check fit
    # do processing
    orig_stars, wcs, pointing = piff.Input.process(config['input'], logger)
    assert len(orig_stars) == N_stars_per_chip * N_chips
    assert orig_stars[0].image.array.shape == (config['input']['stamp_size'], config['input']['stamp_size'])

    # check that fit parameters match
    indx = np.random.choice(len(orig_stars))
    target = orig_stars[indx]
    true_params = true_params_all[indx]

    # test with target star
    test_star = psf.drawStar(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

@timer
def test_gsobject_convolve():
    from test_gsobject_model import make_data
    influx_1 = 1.
    scale_1 = 0.5
    u0_1, v0_1 = 0.01, 0.02
    g1_1, g2_1 = 0.03, -0.05
    influx_2 = 1.
    scale_2 = 0.7
    u0_2, v0_2 = -0.1, 0.05
    g1_2, g2_2 = 0.07, 0.13

    scale_12 = np.sqrt(scale_1 ** 2 + scale_2 ** 2)
    g1_12 = (scale_1 ** 2 * g1_1 + scale_2 ** 2 * g1_2) / scale_12 ** 2
    g2_12 = (scale_1 ** 2 * g2_1 + scale_2 ** 2 * g2_2) / scale_12 ** 2
    influx_12 = influx_1 * influx_2
    u0_12 = u0_1 + u0_2
    v0_12 = v0_1 + v0_2

    pix_scale=0.27
    nside=32
    sigma=1. / 32 * nside
    pix_scale=pix_scale * 32. / nside
    fiducial = galsim.Gaussian(sigma=sigma)
    for fastfit in [False, True]:
        for include_pixel in [False, True]:
            for force_model_center in [False, True]:
                model = piff.Gaussian(include_pixel=include_pixel, fastfit=fastfit, force_model_center=force_model_center)

                # create gaussian profile
                star_1 = make_data(fiducial, scale_1, g1_1, g2_1, u0_1, v0_1, influx_1, pix_scale=pix_scale, include_pixel=include_pixel, nside=nside)
                star_2 = make_data(fiducial, scale_2, g1_2, g2_2, u0_2, v0_2, influx_2,  pix_scale=pix_scale, include_pixel=include_pixel, nside=nside)
                star_12 = make_data(fiducial, scale_12, g1_12, g2_12, u0_12, v0_12, influx_12, pix_scale=pix_scale, include_pixel=include_pixel, nside=nside)
                star_1 = model.fit(model.initialize(star_1))
                star_2 = model.fit(model.initialize(star_2))
                star_12 = model.fit(model.initialize(star_12))
                star_model_12_conv = model.fit(model.draw(star_2, profile=model.getProfile(star_1.fit.params)))

                # these are some pretty loose checks on the tolerance...

                # make sure fit params make sense between 12 and 12_conv
                if force_model_center:
                    nstart = 0
                else:
                    np.testing.assert_allclose(star_12.fit.params[:2], star_model_12_conv.fit.params[:2], atol=1e-4)
                    nstart = 2
                np.testing.assert_allclose(star_12.fit.params[nstart + 0], star_model_12_conv.fit.params[nstart + 0], atol=1e-2)
                np.testing.assert_allclose(star_12.fit.params[nstart + 1:], star_model_12_conv.fit.params[nstart + 1:], atol=1e-2)
                # make sure image makes sense
                np.testing.assert_allclose(star_model_12_conv.image.array, star_12.image.array, atol=1e-3, rtol=1e-2)

if __name__ == '__main__':
    test_gsobject_convolve()
    test_compound()
