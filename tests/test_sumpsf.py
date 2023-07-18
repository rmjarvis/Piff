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

import galsim
import numpy as np
import piff
import os
import fitsio

from piff_test_helper import timer

@timer
def test_trivial_sum1():
    """Test the trivial case of using a Sum of 1 component.
    """
    # This is essentially the same as test_single_image in test_simple.py

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_trivial_sum1.log')

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars. (Randomish, but make sure no blending.)
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]

    # Draw a Gaussian PSF at each location on the image.
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    targets = []
    for x,y in zip(x_list, y_list):
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        psf.drawImage(image=image[bounds], method='no_pixel', center=(x,y))
        targets.append(image[bounds])
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','trivial_sum1_im.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','trivial_sum1_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','trivial_sum1_psf.fits')
    stamp_size = 48
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'stamp_size' : stamp_size
        },
        'psf' : {
            'type' : 'Sum',
            'components': [
                {
                    'type' : 'Simple',
                    'model' : { 'type' : 'Gaussian',
                                'fastfit': True,
                                'include_pixel': False },
                    'interp' : { 'type' : 'Mean' },
                }
            ],
            'max_iter' : 10,
            'chisq_thresh' : 0.2,
        },
        'output' : { 'file_name' : psf_file },
    }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)

    assert type(psf) is piff.SumPSF
    assert len(psf.components) == 1
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.Gaussian
    assert type(psf.components[0].interp) is piff.Mean

    assert psf.chisq_thresh == 0.2
    assert psf.max_iter == 10

    for i, star in enumerate(psf.stars):
        target = targets[i]
        test_star = psf.drawStar(star)
        true_params = [ sigma, g1, g2 ]
        np.testing.assert_almost_equal(test_star.fit.params[0], true_params, decimal=4)
        # The drawn image should be reasonably close to the target.
        target = target[star.image.bounds]
        print('target max diff = ',np.max(np.abs(test_star.image.array - target.array)))
        np.testing.assert_allclose(test_star.image.array, target.array, atol=1.e-5)

        # test that draw works
        test_image = psf.draw(x=star['x'], y=star['y'], stamp_size=config['input']['stamp_size'],
                              flux=star.fit.flux, offset=star.fit.center/image.scale)
        # This image should be very close to the same values as test_star
        # Just make sure to compare over the same bounds.
        b = test_star.image.bounds & test_image.bounds
        print('draw/drawStar max diff = ',np.max(np.abs(test_image[b].array - test_star.image[b].array)))
        np.testing.assert_allclose(test_image[b].array, test_star.image[b].array, atol=1.e-9)

    # Outlier is a valid option
    config['psf']['outliers'] = {
        'type' : 'Chisq',
        'nsigma' : 5,
        'max_remove' : 3,
    }
    piff.piffify(config)
    psf2 = piff.read(psf_file)
    # This didn't remove anything, so the result is the same.
    # (Use the last star from the above loop for comparison.)
    test_star2 = psf2.drawStar(star)
    np.testing.assert_almost_equal(test_star2.fit.params[0], true_params, decimal=4)
    assert test_star2.image == test_star.image

    test_image2 = psf2.draw(x=star['x'], y=star['y'], stamp_size=config['input']['stamp_size'],
                           flux=star.fit.flux, offset=star.fit.center/image.scale)
    assert test_image2 == test_image

    # Error if components is missing
    config1 = config.copy()
    del config1['psf']['components']
    with np.testing.assert_raises(ValueError):
        piff.process(config)

@timer
def test_easy_sum2():
    """Test a fairly easy case with 2 components, one constant, the other linear.
    """

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_eaay_sum2.log')

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars. (Randomish, but make sure no blending.)
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]

    # The model is a sum of two Gaussians with very different size and interpolation patterns.
    # A wide Gaussian is the same everywhere
    # A narrow Gaussian has a linear slope in its parameters across the field of view.
    sigma1 = 2.3
    g1 = 0.23
    g2 = -0.17
    psf1 = galsim.Gaussian(sigma=sigma1).shear(g1=g1, g2=g2)
    targets = []
    for x,y in zip(x_list, y_list):
        # sigma ranges from 1.2 to 1.5
        sigma2 = 1.2 + 0.3 * x / 2048
        # dy ranges from -0.7 arcsec to +0.8 arcsec
        dy = -0.7 + 1.5 * y / 2048
        psf2 = galsim.Gaussian(sigma=sigma2).shift(0,dy) * 0.2
        bounds = galsim.BoundsI(int(x-33), int(x+33), int(y-33), int(y+33))
        psf = psf1 + psf2
        psf.drawImage(image=image[bounds], center=(x,y))
        targets.append(image[bounds])
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','easy_sum2_im.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','easy_sum2_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','easy_sum2_psf.fits')
    stamp_size = 48
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'stamp_size' : stamp_size,
            'noise': 1.e-12,
            'trust_pos': True,
        },
        'select' : {
            'max_snr': 1.e6,
        },
        'psf' : {
            'type' : 'Sum',
            'components': [
                {
                    'model' : { 'type' : 'Gaussian',
                                'include_pixel': True,
                                'fastfit': True,
                              },
                    'interp' : { 'type' : 'Mean' },
                },
                {
                    'model' : { 'type' : 'Gaussian',
                                'include_pixel': True,
                                'init': 'zero',
                                'centered': False,
                                'fit_flux': True,
                                'fastfit': True,
                              },
                    'interp' : { 'type' : 'Polynomial',
                                 'order': 1 },
                }
            ],
        },
        'output' : { 'file_name' : psf_file },
    }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)

    assert type(psf) is piff.SumPSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.Gaussian
    assert type(psf.components[0].interp) is piff.Mean
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Gaussian
    assert type(psf.components[1].interp) is piff.Polynomial

    # Defaults
    assert psf.chisq_thresh == 0.1
    assert psf.max_iter == 30

    test_stars = psf.interpolateStarList(psf.stars)
    test_stars = psf.drawStarList(test_stars)
    for i, test_star in enumerate(test_stars):
        print('star',i)
        target = targets[i]
        true_params1 = [ sigma1, g1, g2 ]
        print('comp0: ',test_star.fit.params[0],' =? ',true_params1)
        np.testing.assert_almost_equal(test_star.fit.params[0], true_params1, decimal=2)
        true_params2 = [ 0.2, 0, -0.7 + 1.5*y_list[i]/2048, 1.2 + 0.3*x_list[i]/2048, 0, 0 ]
        print('comp1: ',test_star.fit.params[1],' =? ',true_params2)
        np.testing.assert_almost_equal(test_star.fit.params[1], true_params2, decimal=2)

        # The drawn image should be reasonably close to the target.
        target = target[test_star.image.bounds]
        print('target max diff = ',np.max(np.abs(test_star.image.array - target.array)))
        np.testing.assert_allclose(test_star.image.array, target.array, atol=1.e-5)

        # Draw should produce something almost identical (modulo bounds).
        test_image = psf.draw(x=test_star['x'], y=test_star['y'],
                              stamp_size=config['input']['stamp_size'],
                              flux=test_star.fit.flux, offset=test_star.fit.center/image.scale)
        b = test_star.image.bounds & test_image.bounds
        print('image max diff = ',np.max(np.abs(test_image[b].array - test_star.image[b].array)))
        np.testing.assert_allclose(test_image[b].array, test_star.image[b].array, atol=1.e-8)

    # Repeat with both components being PixelGrid models.
    # Both pretty chunky, so this test doesn't take forever.
    config['psf']['components'][0]['model'] = {
        'type': 'PixelGrid',
        'scale': 1.04,  # 4x4 native pixels per grid pixel.
        'size': 10,     # Covers 40x40 original pixels.
    }
    if __name__ == '__main__':
        # The small Gaussian moves around the central ~8 pixels.
        # Make the grid big enough to capture the whole moving Gaussian.
        grid_size = 14
        tol = 3.e-4
    else:
        # For faster running on CI, use a bit smaller central piece with higher tolerance.
        grid_size = 10
        tol = 5.e-4
    config['psf']['components'][1]['model'] = {
        'type': 'PixelGrid',
        'scale': 0.26,  # native pixel scale.
        'size': grid_size,
        'init': 'zero',
        'centered': False,
        'fit_flux': True,
    }
    psf = piff.process(config, logger)
    assert type(psf) is piff.SumPSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.PixelGrid
    assert type(psf.components[0].interp) is piff.Mean
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.PixelGrid
    assert type(psf.components[1].interp) is piff.Polynomial

    for i, star in enumerate(psf.stars):
        print('star',i)
        target = targets[i]
        test_star = psf.drawStar(star)
        # Note: There is no useful test to make of the component parameters.
        #print('comp0: ',test_star.fit.params[0])
        #print('comp1: ',test_star.fit.params[1])

        # The drawn image should be reasonably close to the target.
        # Although not nearly as close as above.
        # Also, shrink the bounds a bit, since the PixelGrid edge treatment isn't great.
        b = star.image.bounds.withBorder(-8)
        target = target[b]
        print('target max diff = ',np.max(np.abs(test_star.image[b].array - target[b].array)))
        print('target max value = ',np.max(np.abs(target[b].array)))
        np.testing.assert_allclose(test_star.image[b].array, target[b].array, atol=tol)

        # Draw should produce something almost identical (modulo bounds).
        test_image = psf.draw(x=star['x'], y=star['y'], stamp_size=config['input']['stamp_size'],
                              flux=star.fit.flux, offset=np.array(star.fit.center)/image.scale)
        b = test_star.image.bounds & test_image.bounds
        print('image max diff = ',np.max(np.abs(test_image[b].array - test_star.image[b].array)))
        np.testing.assert_allclose(test_image[b].array, test_star.image[b].array, atol=1.e-7)



@timer
def test_mixed_pixel_sum2():
    """Test case one component including pixel, but not other one.
    """
    # This is identical to test_easy_sum2, except that component 2 is drawn and fit with
    # no pixel convolution.

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_mixel_pixel.log')
        logger = piff.config.setup_logger(verbose=2)

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars.
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]

    # The model is a sum of two Gaussians with very different size and interpolation patterns.
    # A wide Gaussian is the same everywhere
    # A narrow Gaussian has a linear slope in its parameters across the field of view.
    sigma1 = 2.3
    g1 = 0.23
    g2 = -0.17
    psf1 = galsim.Gaussian(sigma=sigma1).shear(g1=g1, g2=g2)
    targets = []
    for x,y in zip(x_list, y_list):
        # sigma ranges from 0.5 to 0.8
        sigma2 = 0.5 + 0.3 * x / 2048
        # dy ranges from -0.3 arcsec to +0.2 arcsec
        dy = -0.3 + 0.5 * y / 2048
        psf2 = galsim.Gaussian(sigma=sigma2).shift(0,dy) * 0.13
        bounds = galsim.BoundsI(int(x-33), int(x+33), int(y-33), int(y+33))
        psf1.drawImage(image=image[bounds], center=(x,y))
        psf2.drawImage(image=image[bounds], center=(x,y), add_to_image=True, method='no_pixel')
        targets.append(image[bounds])
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','mixed_pixel_im.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','mixed_pixel_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','mixed_pixel_psf.fits')
    stamp_size = 48
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'stamp_size' : stamp_size,
            'noise': 1.e-12,
            'trust_pos': True,
        },
        'select' : {
            'max_snr': 1.e6,
        },
        'psf' : {
            'type' : 'Sum',
            'components': [
                {
                    'model' : { 'type' : 'Gaussian',
                                'include_pixel': True,
                                'fastfit': True,
                              },
                    'interp' : { 'type' : 'Mean' },
                },
                {
                    'model' : { 'type' : 'Gaussian',
                                'include_pixel': False,
                                'init': '(0.1, 0.2)',
                                'centered': False,
                                'fit_flux': True,
                                'fastfit': True,
                              },
                    'interp' : { 'type' : 'Polynomial',
                                 'order': 1 },
                }
            ],
        },
        'output' : { 'file_name' : psf_file },
    }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)

    assert type(psf) is piff.SumPSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.Gaussian
    assert type(psf.components[0].interp) is piff.Mean
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Gaussian
    assert type(psf.components[1].interp) is piff.Polynomial

    for i, star in enumerate(psf.stars):
        print('star',i)
        target = targets[i]
        test_star = psf.drawStar(star)
        true_params1 = [ sigma1, g1, g2 ]
        print('comp0: ',test_star.fit.params[0],' =? ',true_params1)
        np.testing.assert_almost_equal(test_star.fit.params[0], true_params1, decimal=2)
        true_params2 = [ 0.13, 0, -0.3 + 0.5*y_list[i]/2048, 0.5 + 0.3*x_list[i]/2048, 0, 0 ]
        print('comp1: ',test_star.fit.params[1],' =? ',true_params2)
        np.testing.assert_almost_equal(test_star.fit.params[1], true_params2, decimal=2)

        # The drawn image should be reasonably close to the target.
        target = target[star.image.bounds]
        print('target max diff = ',np.max(np.abs(test_star.image.array - target.array)))
        np.testing.assert_allclose(test_star.image.array, target.array, atol=1.e-5)

        # Draw should produce something almost identical (modulo bounds).
        test_image = psf.draw(x=star['x'], y=star['y'], stamp_size=config['input']['stamp_size'],
                              flux=star.fit.flux, offset=star.fit.center/image.scale)
        b = test_star.image.bounds & test_image.bounds
        print('image max diff = ',np.max(np.abs(test_image[b].array - test_star.image[b].array)))
        np.testing.assert_allclose(test_image[b].array, test_star.image[b].array, atol=1.e-7)


    # Repeat with the first component being a PixelGrid.
    config['psf']['components'][0]['model'] = {
        'type': 'PixelGrid',
        'scale': 1.04,  # 4x4 pixel grid
        'size': 10,
    }
    psf = piff.process(config, logger)
    assert type(psf) is piff.SumPSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.PixelGrid
    assert type(psf.components[0].interp) is piff.Mean
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Gaussian
    assert type(psf.components[1].interp) is piff.Polynomial

    for i, star in enumerate(psf.stars):
        print('star',i)
        target = targets[i]
        test_star = psf.drawStar(star)
        #print('comp0: ',test_star.fit.params[0])
        true_params2 = [ 0.13, 0, -0.3 + 0.5*y_list[i]/2048, 0.5 + 0.3*x_list[i]/2048, 0, 0 ]
        print('comp1: ',test_star.fit.params[1],' =? ',true_params2)
        # Note: don't test these for exact match anymore.  The only thing we really care about
        # here is that the drawn image is reasonably close.

        # The drawn image should be reasonably close to the target.
        # Although not nearly as close as above.
        # Also, shrink the bounds a bit, since the PixelGrid edge treatment isn't great.
        b = star.image.bounds.withBorder(-8)
        target = target[b]
        print('target max diff = ',np.max(np.abs(test_star.image[b].array - target[b].array)))
        print('target max value = ',np.max(np.abs(target[b].array)))
        np.testing.assert_allclose(test_star.image[b].array, target[b].array, atol=3.e-4)

        # Draw should produce something almost identical (modulo bounds).
        test_image = psf.draw(x=star['x'], y=star['y'], stamp_size=config['input']['stamp_size'],
                              flux=star.fit.flux, offset=np.array(star.fit.center)/image.scale)
        b = test_star.image.bounds & test_image.bounds
        print('image max diff = ',np.max(np.abs(test_image[b].array - test_star.image[b].array)))
        np.testing.assert_allclose(test_image[b].array, test_star.image[b].array, atol=1.e-7)


if __name__ == '__main__':
    test_trivial_sum1()
    test_easy_sum2()
    test_mixed_pixel_sum2()
