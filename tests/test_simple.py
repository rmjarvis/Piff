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
def test_Gaussian():
    """This is about the simplest possible model I could think of.  It just uses the
    HSM adaptive moments routine to measure the moments, and then it models the
    PSF as a Gaussian.
    """

    # Here is the true PSF
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)
    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')

    # Make a StarData instance for this image
    stardata = piff.StarData(image, image.trueCenter())
    star = piff.Star(stardata, None)

    # Fit the model from the image
    model = piff.Gaussian(include_pixel=False)
    star = model.initialize(star)
    fit = model.fit(star).fit

    print('True sigma = ',sigma,', model sigma = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])

    # This test is pretty accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Gaussian.
    true_params = [ sigma, g1, g2 ]
    np.testing.assert_almost_equal(fit.params[0], sigma, decimal=7)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=7)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=7)
    np.testing.assert_almost_equal(fit.params, true_params, decimal=7)

    # Now test running it via the config parser
    config = {
        'model' : {
            'type' : 'Gaussian',
            'include_pixel': False
        }
    }
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_Gaussian.log')
    model = piff.Model.process(config['model'], logger)
    fit = model.fit(star).fit

    # Same tests.
    np.testing.assert_almost_equal(fit.params[0], sigma, decimal=7)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=7)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=7)
    np.testing.assert_almost_equal(fit.params, true_params, decimal=7)


@timer
def test_Mean():
    """For the interpolation, the simplest possible model is just a mean value, which barely
    even qualifies as doing any kind of interpolating.  But it tests the basic glue software.
    """
    # Make a list of parameter vectors to "interpolate"
    np_rng = np.random.RandomState(1234)
    nstars = 100
    vectors = [ np_rng.random_sample(10) for i in range(nstars) ]
    mean = np.mean(vectors, axis=0)
    print('mean = ',mean)

    # Make some dummy StarData objects to use.  The only thing we really need is the properties,
    # although for the Mean interpolator, even this is ignored.
    target_data = [
            piff.Star.makeTarget(x=np_rng.random_sample()*2048, y=np_rng.random_sample()*2048).data
            for i in range(nstars) ]
    fit = [ piff.StarFit(v) for v in vectors ]
    stars = [ piff.Star(d, f) for d,f in zip(target_data,fit) ]

    # Use the piff.Mean interpolator
    interp = piff.Mean()
    interp.solve(stars)

    print('True mean = ',mean)
    print('Interp mean = ',interp.mean)

    # This should be exactly equal, since we did the same calculation.  But use almost_equal
    # anyway, just in case we decide to do something slightly different, but equivalent.
    np.testing.assert_almost_equal(mean, interp.mean)

    # Now test running it via the config parser
    config = {
        'interp' : {
            'type' : 'Mean'
        }
    }
    logger = piff.config.setup_logger()
    interp = piff.Interp.process(config['interp'], logger)
    interp.solve(stars)
    np.testing.assert_almost_equal(mean, interp.mean)


@timer
def test_single_image():
    """Test the simple case of one image and one catalog.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_single_image.log')

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars.  Include some flagged and not used locations.
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]
    flag_list = [ 0, 0, 12, 0, 0, 1, 0, 0, 0, 0 ]
    use_list = [ 1, 1, 1, 1, 1, 0, 1, 1, 0, 1 ]

    # Draw a Gaussian PSF at each location on the image.
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    for x,y,flag,use in zip(x_list, y_list, flag_list, use_list):
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
        psf.drawImage(image=image[bounds], method='no_pixel', offset=offset)
        # corrupt the ones that are marked as flagged
        if flag:
            print('corrupting star at ',x,y)
            ar = image[bounds].array
            im_max = np.max(ar) * 0.2
            ar[ar > im_max] = im_max
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','simple_image.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8'), ('flag','i2'), ('use','i2') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    data['flag'] = flag_list
    data['use'] = use_list
    cat_file = os.path.join('output','simple_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # Use InputFiles to read these back in
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file }
    input = piff.InputFiles(config, logger=logger)
    assert input.image_file_name == [ image_file ]
    assert input.cat_file_name == [ cat_file ]

    # Check image
    assert len(input.images) == 1
    np.testing.assert_equal(input.images[0].array, image.array)

    # Check catalog
    assert len(input.image_pos) == 1
    np.testing.assert_equal([pos.x for pos in input.image_pos[0]], x_list)
    np.testing.assert_equal([pos.y for pos in input.image_pos[0]], y_list)

    # Repeat, using flag and use columns this time.
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file,
               'flag_col': 'flag',
               'use_col': 'use',
               'stamp_size': 48 }
    input = piff.InputFiles(config, logger=logger)
    print('pos = ',input.image_pos)
    print('pos[0] = ',input.image_pos[0])
    assert len(input.image_pos[0]) == 7

    # Make star data
    orig_stars = input.makeStars()
    assert len(orig_stars) == 7
    assert orig_stars[0].image.array.shape == (48,48)

    # Process the star data
    # can only compare to truth if include_pixel=False
    model = piff.Gaussian(fastfit=True, include_pixel=False)
    interp = piff.Mean()
    fitted_stars = [ model.fit(model.initialize(star)) for star in orig_stars ]
    interp.solve(fitted_stars)
    print('mean = ',interp.mean)

    # Check that the interpolation is what it should be
    target = piff.Star.makeTarget(x=1024, y=123) # Any position would work here.
    true_params = [ sigma, g1, g2 ]
    test_star = interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Now test running it via the config parser
    psf_file = os.path.join('output','simple_psf.fits')
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'flag_col' : 'flag',
            'use_col' : 'use',
            'stamp_size' : 48
        },
        'psf' : {
            'model' : { 'type' : 'Gaussian',
                        'fastfit': True,
                        'include_pixel': False},
            'interp' : { 'type' : 'Mean' },
        },
        'output' : { 'file_name' : psf_file },
    }
    orig_stars, wcs, pointing = piff.Input.process(config['input'], logger)

    # Use a SimplePSF to process the stars data this time.
    psf = piff.SimplePSF(model, interp)

    psf.fit(orig_stars, wcs, pointing, logger=logger)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Round trip to a file
    psf.write(psf_file, logger)
    psf = piff.read(psf_file, logger)
    assert type(psf.model) is piff.Gaussian
    assert type(psf.interp) is piff.Mean
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Do the whole thing with the config parser
    os.remove(psf_file)

    piff.piffify(config, logger)
    psf = piff.read(psf_file)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Test using the piffify executable
    os.remove(psf_file)
    # This would be simpler as a direct assignment, but this once, test the way you would set
    # this from the command line, which would call parse_variables.
    piff.config.parse_variables(config, ['verbose=0'], logger=logger)
    #config['verbose'] = 0
    with open('simple.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    config2 = piff.config.read_config('simple.yaml')
    assert config == config2
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'simple.yaml'] )
    p.communicate()
    psf = piff.read(psf_file)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Test that we can make rho statistics
    min_sep = 1
    max_sep = 100
    bin_size = 0.1
    stats = piff.RhoStats(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
    stats.compute(psf, orig_stars)

    rhos = [stats.rho1, stats.rho2, stats.rho3, stats.rho4, stats.rho5]
    for rho in rhos:
        # Test the range of separations
        radius = np.exp(rho.logr)
        # last bin can be one bigger than max_sep
        np.testing.assert_array_less(radius, np.exp(np.log(max_sep) + bin_size))
        np.testing.assert_array_less(min_sep, radius)
        np.testing.assert_array_almost_equal(np.diff(rho.logr), bin_size, decimal=5)

        # Test that the max absolute value of each rho isn't crazy
        np.testing.assert_array_less(np.abs(rho.xip), 1)

        # # Check that each rho isn't precisely zero. This means the sum of abs > 0
        np.testing.assert_array_less(0, np.sum(np.abs(rho.xip)))

    # Test the plotting and writing
    rho_psf_file = os.path.join('output','simple_psf_rhostats.pdf')
    stats.write(rho_psf_file)

    # Test that we can make summary shape statistics, using HSM
    shapeStats = piff.ShapeHistogramsStats()
    shapeStats.compute(psf, orig_stars)

    # test their characteristics
    np.testing.assert_array_almost_equal(sigma, shapeStats.T, decimal=4)
    np.testing.assert_array_almost_equal(sigma, shapeStats.T_model, decimal=3)
    np.testing.assert_array_almost_equal(g1, shapeStats.g1, decimal=4)
    np.testing.assert_array_almost_equal(g1, shapeStats.g1_model, decimal=3)
    np.testing.assert_array_almost_equal(g2, shapeStats.g2, decimal=4)
    np.testing.assert_array_almost_equal(g2, shapeStats.g2_model, decimal=3)

    shape_psf_file = os.path.join('output','simple_psf_shapestats.pdf')
    shapeStats.write(shape_psf_file)

    # Test that we can use the config parser for both RhoStats and ShapeHistogramsStats
    config['output']['stats'] = [
        {
            'type': 'ShapeHistograms',
            'file_name': shape_psf_file
        },
        {
            'type': 'Rho',
            'file_name': rho_psf_file
        },
        {
            'type': 'TwoDHist',
            'file_name': os.path.join('output', 'simple_psf_twodhiststats.pdf'),
            'number_bins_u': 3,
            'number_bins_v': 3,
        },
        {
            'type': 'TwoDHist',
            'file_name': os.path.join('output', 'simple_psf_twodhiststats_std.pdf'),
            'reducing_function': 'np.std',
            'number_bins_u': 3,
            'number_bins_v': 3,
        },
    ]

    os.remove(psf_file)
    os.remove(rho_psf_file)
    os.remove(shape_psf_file)
    piff.piffify(config, logger)

    # Test using the piffify executable
    os.remove(psf_file)
    os.remove(rho_psf_file)
    os.remove(shape_psf_file)
    config['verbose'] = 0
    with open('simple.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    p = subprocess.Popen( [piffify_exe, 'simple.yaml'] )
    p.communicate()

if __name__ == '__main__':
    test_Gaussian()
    test_Mean()
    test_single_image()
