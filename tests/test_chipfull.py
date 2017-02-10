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

"""
Open questions:
    - parameter sharing between two PSFs (how do I check that the fit params work? create a partway interpolant?)
    - outlier removal -- shared or unshared?
"""

@timer
def test_chipfull():
    """This code tests the double PSF model. The wide-field piece is a
    polynomial gaussian, while the chip piece is a kolmogorov gsobject with
    gaussian process interpolation.
    """

    # true PSF params
    N_chips = 4
    N_stars_per_chip = 100
    chipnums = range(5, 5 + N_chips)

    # set RNG

    stars = []
    image_file = os.path.join('data','chipfull_image_%02d.fits')
    cat_file = os.path.join('data','chipfull_cat_%02d.fits')
    for chipnum in chipnums:
        # generate x, y coordinates for each chip

        # generate flags

        # generate use_list

        # generate widefield gaussian parameters

        # generate kolmogorov parameters

        # generate image

        # Write out the image to a file
        image.write(image_file.format(chipnum))

        # Write out the catalog to a file
        dtype = [ ('x','f8'), ('y','f8'), ('flag','i2'), ('use','i2') ]
        data = np.empty(len(x_list), dtype=dtype)
        data['x'] = x_list
        data['y'] = y_list
        data['flag'] = flag_list
        data['use'] = use_list

        fitsio.write(cat_file.format(chipnum), data, clobber=True)

        # create as stars

    # choose target star

    # get its true_params

    # test fitting each piece


    # Now test running it via the config parser
    psf_file = os.path.join('output','chipfull_psf.fits')
    # TODO: Fill out config with proper specs
    config = {
        'input' : {
            'images' : image_file,
            'cats' : cat_file,
            'chipnums': "[ c for c in range(5, {0})]".format(5 + N_chips),
            # What hdu is everything in?
            'image_hdu' : 1,
            # 'badpix_hdu' : 2,
            # 'weight_hdu' : 3,
            'cat_hdu' : 1,

            # What columns in the catalog have things we need?
            'x_col' : 'x',
            'y_col' : 'y',
            'flag_col' : 'flag',
            'use_col' : 'use',

            # How large should the postage stamp cutouts of the stars be?
            'stamp_size' : 48,
        },
        'psf' : {
            'type': 'ChipFullPSF',

            'chip': {
                'type': 'SimplePSF',

                'model' : { 'type' : 'GSObjectModel',
                            'fastfit' : True,
                            'gsobj' : 'galsim.Gaussian(sigma=1.0)' },
                'interp' : { 'type' : 'GPInterp',
                             'keys' : ['u', 'v'],
                             'kernel' : 'RBF(200.0)',
                             'optimize' : False,},
            },

            'full': {
                'type': 'SimplePSF',

                'model' : { 'type' : 'GSObjectModel',
                            'fastfit' : True,
                            'gsobj' : 'galsim.Gaussian(sigma=1.0)' },
                'interp' : { 'type' : 'Polynomial',
                            'order': 1,
                            },
            }

        },
        'output' : { 'file_name' : psf_file,
            'stats': {
            {
                'type': 'ShapeHistograms',
                'file_name': os.path.join('output','chipfull_psf_shapestats.pdf')
            },
            {
                'type': 'Rho',
                'file_name': os.path.join('output','chipfull_psf_rhostats.pdf')
            },
            {
                'type': 'TwoDHist',
                'file_name': os.path.join('output', 'chipfull_psf_twodhiststats.pdf'),
                'number_bins_u': 10,
                'number_bins_v': 10,
            },
            }
    }
    }
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(verbose=0)

    # do processing
    orig_stars, wcs, pointing = piff.Input.process(config['input'], logger)
    assert len(orig_stars) == N_stars_per_chip * N_chips
    assert orig_stars[0].image.array.shape == (config['input']['stamp_size'], config['input']['stamp_size'])

    # use ChipFullPSF to process the stars and fit
    psf = piff.ChipFullPSF.process(config['psf'], logger)

    psf.fit(orig_stars, wcs, pointing, logger=logger)

    # test with target star
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Round trip to a file
    psf.write(psf_file, logger)
    psf = piff.read(psf_file, logger)
    assert type(psf.model) is piff.Gaussian
    assert type(psf.interp) is piff.Mean
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Test using piffify function
    os.remove(psf_file)
    piff.piffify(config, logger)

    # check fit
    psf = piff.read(psf_file)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Test using the piffify executable
    os.remove(psf_file)
    config['verbose'] = 0
    with open('chipfull.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'chipfull.yaml'] )
    p.communicate()

    # check fit
    psf = piff.read(psf_file)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

if __name__ == '__main__':
    test_chipfull()
