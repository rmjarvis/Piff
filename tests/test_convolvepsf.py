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
from functools import lru_cache

from piff_test_helper import timer

@lru_cache(maxsize=1)
def make_screens():
    # Some parameters copied from psf_wf_movie.py in GalSim repo.
    Ellerbroek_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    Ellerbroek_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    Ellerbroek_interp = galsim.LookupTable(Ellerbroek_alts, Ellerbroek_weights,
                                           interpolant='linear')

    # Use given number of uniformly spaced altitudes
    alts = np.max(Ellerbroek_alts)*np.arange(6)/5.
    weights = Ellerbroek_interp(alts)  # interpolate the weights
    weights /= sum(weights)  # and renormalize

    spd = []  # Wind speed in m/s
    dirn = [] # Wind direction in radians
    r0_500 = [] # Fried parameter in m at a wavelength of 500 nm.
    u = galsim.UniformDeviate(1234)
    for i in range(6):
        spd.append(u() * 20)
        dirn.append(u() * 360*galsim.degrees)
        r0_500.append(0.1 * weights[i]**(-3./5))

    screens = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alts, rng=u,
                                screen_size=102.4, screen_scale=0.1)

    # Add in an optical screen
    screens.append(galsim.OpticalScreen(diam=8, defocus=0.7, astig1=-0.8, astig2=0.7,
                                        trefoil1=-0.6, trefoil2=0.5,
                                        coma1=0.5, coma2=0.7, spher=0.8, obscuration=0.4))

    aper = galsim.Aperture(diam=8, obscuration=0.4)

    return screens, aper


@timer
def test_trivial_convolve1():
    """Test the trivial case of using a Convolve of 1 component.
    """
    # This is essentially the same as test_single_image in test_simple.py

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_trivial_convolve1.log')

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
    image_file = os.path.join('output','trivial_convolve1_im.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','trivial_convolve1_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','trivial_convolve1_psf.fits')
    stamp_size = 48
    config = {
        'input': {
            'image_file_name': image_file,
            'cat_file_name': cat_file,
            'stamp_size': stamp_size
        },
        'psf': {
            'type': 'Convolve',
            'components': [
                {
                    'type': 'Simple',
                    'model': { 'type': 'Gaussian',
                               'fastfit': True,
                               'include_pixel': False
                             },
                    'interp': { 'type': 'Mean' },
                }
            ],
            'max_iter': 10,
            'chisq_thresh': 0.2,
        },
        'output': { 'file_name': psf_file },
    }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)

    assert type(psf) is piff.ConvolvePSF
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
        'type': 'Chisq',
        'nsigma': 5,
        'max_remove': 3,
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
def test_convolve_optatm():
    # Let reality be atmospheric + optical PSF with realistic variation.
    # And model this with a convolution of a Kolmogorov/GP SimplePSF and a fixed Optical PSF.

    if __name__ == '__main__':
        size = 2048
        nstars = 200
        noise = 20
    else:
        size = 1024
        nstars = 10
        noise = 2

    pixel_scale = 0.2
    im = galsim.ImageF(size, size, scale=pixel_scale)

    screens, aper = make_screens()

    rng = galsim.BaseDeviate(1234)
    x = rng.np.uniform(25, size-25, size=nstars)
    y = rng.np.uniform(25, size-25, size=nstars)

    for k in range(nstars):
        flux = 100000
        theta = ((x[k] - size/2) * pixel_scale * galsim.arcsec,
                 (y[k] - size/2) * pixel_scale * galsim.arcsec)

        psf = screens.makePSF(lam=500, aper=aper, exptime=100, flux=flux, theta=theta)
        psf.drawImage(image=im, center=(x[k],y[k]), method='phot', rng=rng, add_to_image=True)
        bounds = galsim.BoundsI(int(x[k]-33), int(x[k]+33), int(y[k]-33), int(y[k]+33))

    # Add a little noise
    noise = 10
    im.addNoise(galsim.GaussianNoise(rng=rng, sigma=noise))
    image_file = os.path.join('output', 'convolveatmpsf_im.fits')
    im.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x), dtype=dtype)
    data['x'] = x
    data['y'] = y
    cat_file = os.path.join('output','convolveatmpsf_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','convolveatmpsf.fits')
    stamp_size = 48
    config = {
        'input': {
            'image_file_name': image_file,
            'cat_file_name': cat_file,
            'stamp_size': 32,
            'noise': noise**2,
        },
        'select': {
            'max_snr': 1.e6,
            'max_edge_frac': 0.1,
            'hsm_size_reject': True,
        },
        'psf': {
            'type': 'Convolve',
            'components': [
                {
                    'model': { 'type': 'Kolmogorov',
                                'fastfit': True,
                             },
                    'interp': { 'type': 'GP',
                              },
                },
                {
                    'model': { 'type': 'Optical',
                               'atmo_type': 'None',
                               'lam': 500,
                               'diam': 8,
                               # These are the correct aberrations, not fitted.
                               'base_aberrations': [0,0,0,0,0.7,-0.8,0.7,0.5,0.7,-0.6,0.5,0.8],
                               'obscuration': 0.4,
                             },
                    'interp': { 'type': 'Mean', },
                }
            ],
            'outliers': {
                'type': 'Chisq',
                'nsigma': 5,
                'max_remove': 3,
            }
        },
        'output': {
            'file_name': psf_file,
       },
    }

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=1)
    else:
        logger = piff.config.setup_logger(log_file='output/test_convolve_optatm.log')
        logger = piff.config.setup_logger(verbose=1)

    psf = piff.process(config, logger)

    if __name__ == '__main__':
        config['output']['stats'] = [{
            'type': 'StarImages',
            'file_name': os.path.join('output','test_convolve_optatm_stars.png'),
            'nplot': 10,
            'adjust_stars': True,
        }]
        output = piff.Output.process(config['output'], logger=logger)
        output.write(psf, logger=logger)

    assert type(psf) is piff.ConvolvePSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.Kolmogorov
    assert type(psf.components[0].interp) is piff.GPInterp
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Optical
    assert type(psf.components[1].interp) is piff.Mean

    mean_max_rel_err = 0
    mean_mse = 0
    count = 0
    for i, star in enumerate(psf.stars):
        if star.is_flagged:
            continue
        test_star = psf.drawStar(star)

        b = star.image.bounds.withBorder(-8)
        max_diff = np.max(np.abs(test_star.image[b].array - star.image[b].array))
        max_val = np.max(np.abs(star.image[b].array))
        #print('max_diff / max_val = ',max_diff, max_val, max_diff / max_val)
        mse = np.sum((test_star.image[b].array - star.image[b].array)**2) / flux**2
        #print('mse = ',mse)
        mean_max_rel_err += max_diff/max_val
        mean_mse += mse
        count += 1

    mean_max_rel_err /= count
    mean_mse /= count
    print('mean maximum relative error = ',mean_max_rel_err)
    print('mean mean-squared error = ',mean_mse)
    assert mean_max_rel_err < 0.04
    assert mean_mse < 1.5e-5

    # Without the Optical component, it's quite a bit worse.
    config['psf'] = config['psf']['components'][0]
    psf = piff.process(config, logger)

    if __name__ == '__main__':
        config['output']['stats'][0]['file_name'] = os.path.join('output', 'test_convolve_optatm_stars2.png')
        output = piff.Output.process(config['output'], logger=logger)
        output.write(psf, logger=logger)

    mean_max_rel_err = 0
    mean_mse = 0
    count = 0
    for i, star in enumerate(psf.stars):
        if star.is_flagged:
            continue
        test_star = psf.drawStar(star)

        b = star.image.bounds.withBorder(-8)
        max_diff = np.max(np.abs(test_star.image[b].array - star.image[b].array))
        max_val = np.max(np.abs(star.image[b].array))
        #print('max_diff / max_val = ',max_diff / max_val)
        mse = np.sum((test_star.image[b].array - star.image[b].array)**2) / flux**2
        #print('mse = ',mse)
        mean_max_rel_err += max_diff/max_val
        mean_mse += mse
        count += 1

    mean_max_rel_err /= count
    mean_mse /= count
    print('mean maximum relative error = ',mean_max_rel_err)
    print('mean mean-squared error = ',mean_mse)
    assert mean_max_rel_err > 0.06
    assert mean_mse > 6.e-5


@timer
def test_convolve_pixelgrid():
    # Same as test_optatm, but use a PixelGrid for one of the components

    if __name__ == '__main__':
        size = 2048
        nstars = 200
        noise = 20
    else:
        size = 1024
        nstars = 10
        noise = 2

    pixel_scale = 0.2
    im = galsim.ImageF(size, size, scale=pixel_scale)

    screens, aper = make_screens()

    rng = galsim.BaseDeviate(1234)
    x = rng.np.uniform(25, size-25, size=nstars)
    y = rng.np.uniform(25, size-25, size=nstars)

    for k in range(nstars):
        flux = 100000
        theta = ((x[k] - size/2) * pixel_scale * galsim.arcsec,
                 (y[k] - size/2) * pixel_scale * galsim.arcsec)

        psf = screens.makePSF(lam=500, aper=aper, exptime=100, flux=flux, theta=theta)
        psf.drawImage(image=im, center=(x[k],y[k]), method='phot', rng=rng, add_to_image=True)
        bounds = galsim.BoundsI(int(x[k]-33), int(x[k]+33), int(y[k]-33), int(y[k]+33))

    # Add a little noise
    im.addNoise(galsim.GaussianNoise(rng=rng, sigma=noise))
    image_file = os.path.join('output', 'convolveatmpsf_im.fits')
    im.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x), dtype=dtype)
    data['x'] = x
    data['y'] = y
    cat_file = os.path.join('output','convolveatmpsf_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','convolveatmpsf.fits')
    config = {
        'input': {
            'image_file_name': image_file,
            'cat_file_name': cat_file,
            'stamp_size': 32,
            'noise': noise**2,
        },
        'select': {
            'max_snr': 1.e6,
            'max_edge_frac': 0.1,
            'hsm_size_reject': True,
        },
        'psf': {
            'type': 'Convolve',
            'max_iter': 5,
            'components': [
                {
                    'model': { 'type': 'PixelGrid',
                               'scale': pixel_scale,
                               'size': 17,
                             },
                    'interp': { 'type': 'Polynomial',
                                'order': 1,
                              },
                },
                {
                    'model': { 'type': 'Optical',
                               'atmo_type': 'None',
                               'lam': 500,
                               'diam': 8,
                               # These are the correct aberrations, not fitted.
                               'base_aberrations': [0,0,0,0,0.7,-0.8,0.7,0.5,0.7,-0.6,0.5,0.8],
                               'obscuration': 0.4,
                             },
                    'interp': { 'type': 'Mean', },
                }
            ],
            'outliers': {
                'type': 'Chisq',
                'nsigma': 5,
                'max_remove': 3,
            }
        },
        'output': {
            'file_name': psf_file,
       },
    }

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=1)
    else:
        logger = piff.config.setup_logger(log_file='output/test_convolve_pixelgrid.log')
        logger = piff.config.setup_logger(verbose=1)

    psf = piff.process(config, logger)

    if __name__ == '__main__':
        config['output']['stats'] = [{
            'type': 'StarImages',
            'file_name': os.path.join('output','test_convolve_pixelgrid_stars.png'),
            'nplot': 10,
            'adjust_stars': True,
        }]
        output = piff.Output.process(config['output'], logger=logger)
        output.write(psf, logger=logger)

    assert type(psf) is piff.ConvolvePSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.PixelGrid
    assert type(psf.components[0].interp) is piff.Polynomial
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Optical
    assert type(psf.components[1].interp) is piff.Mean

    mean_max_rel_err = 0
    mean_mse = 0
    count = 0
    for i, star in enumerate(psf.stars):
        if star.is_flagged:
            continue
        test_star = psf.drawStar(star)

        b = star.image.bounds.withBorder(-8)
        max_diff = np.max(np.abs(test_star.image[b].array - star.image[b].array))
        max_val = np.max(np.abs(star.image[b].array))
        #print('max_diff / max_val = ',max_diff, max_val, max_diff / max_val)
        mse = np.sum((test_star.image[b].array - star.image[b].array)**2) / flux**2
        #print('mse = ',mse)
        mean_max_rel_err += max_diff/max_val
        mean_mse += mse
        count += 1

    mean_max_rel_err /= count
    mean_mse /= count
    print('mean maximum relative error = ',mean_max_rel_err)
    print('mean mean-squared error = ',mean_mse)
    assert mean_max_rel_err < 0.04
    assert mean_mse < 2.e-5


@timer
def test_nested_convolve():
    # Test that ConvolvePSF can be nested inside another composite PSF (in particular one that
    # would use a convert_func).
    # This test is essentially the same as test_convolve_optatm, but with an additional convolution
    # with a DeltaFunction.
    # It should work whether this is done the straightforward way with a 3-component Convolve
    # or with two nested Convolves.

    #if __name__ == '__main__':
    if False:
        size = 2048
        nstars = 200
        noise = 20
    else:
        size = 1024
        nstars = 10
        noise = 2

    pixel_scale = 0.2
    im = galsim.ImageF(size, size, scale=pixel_scale)

    screens, aper = make_screens()

    rng = galsim.BaseDeviate(1234)
    x = rng.np.uniform(25, size-25, size=nstars)
    y = rng.np.uniform(25, size-25, size=nstars)

    for k in range(nstars):
        flux = 100000
        theta = ((x[k] - size/2) * pixel_scale * galsim.arcsec,
                 (y[k] - size/2) * pixel_scale * galsim.arcsec)

        psf = screens.makePSF(lam=500, aper=aper, exptime=100, flux=flux, theta=theta)
        psf.drawImage(image=im, center=(x[k],y[k]), method='phot', rng=rng, add_to_image=True)
        bounds = galsim.BoundsI(int(x[k]-33), int(x[k]+33), int(y[k]-33), int(y[k]+33))

    # Add a little noise
    noise = 10
    im.addNoise(galsim.GaussianNoise(rng=rng, sigma=noise))
    image_file = os.path.join('output', 'convolve_nested_im.fits')
    im.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x), dtype=dtype)
    data['x'] = x
    data['y'] = y
    cat_file = os.path.join('output','convolve_nested_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # First do it as a three-component convolution.
    psf_file = os.path.join('output','convolve_nested.fits')
    stamp_size = 48
    config = {
        'input': {
            'image_file_name': image_file,
            'cat_file_name': cat_file,
            'stamp_size': 32,
            'noise': noise**2,
        },
        'select': {
            'max_snr': 1.e6,
            'max_edge_frac': 0.1,
            'hsm_size_reject': True,
        },
        'psf': {
            'type': 'Convolve',
            'components': [
                {
                    'model': { 'type': 'Kolmogorov',
                               'fastfit': True,
                             },
                    'interp': { 'type': 'GP',
                              },
                },
                {
                    'model': { 'type': 'Optical',
                               'atmo_type': 'None',
                               'lam': 500,
                               'diam': 8,
                               # These are the correct aberrations, not fitted.
                               'base_aberrations': [0,0,0,0,0.7,-0.8,0.7,0.5,0.7,-0.6,0.5,0.8],
                               'obscuration': 0.4,
                             },
                    'interp': { 'type': 'Mean', },
                },
                {
                    'model': { 'type': 'Gaussian',
                               'init': 'delta',
                             },
                    'interp': { 'type': 'Mean', },
                },
            ],
            'outliers': {
                'type': 'Chisq',
                'nsigma': 5,
                'max_remove': 3,
            }
        },
        'output': {
            'file_name': psf_file,
       },
    }

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=1)
    else:
        logger = piff.config.setup_logger(log_file='output/test_convolve_optatm.log')
        logger = piff.config.setup_logger(verbose=1)

    psf = piff.process(config, logger)

    assert type(psf) is piff.ConvolvePSF
    assert len(psf.components) == 3
    assert type(psf.components[0]) is piff.SimplePSF
    assert type(psf.components[0].model) is piff.Kolmogorov
    assert type(psf.components[0].interp) is piff.GPInterp
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Optical
    assert type(psf.components[1].interp) is piff.Mean
    assert type(psf.components[2]) is piff.SimplePSF
    assert type(psf.components[2].model) is piff.Gaussian
    assert type(psf.components[2].interp) is piff.Mean

    mean_max_rel_err = 0
    mean_mse = 0
    count = 0
    for i, star in enumerate(psf.stars):
        if star.is_flagged:
            continue
        test_star = psf.drawStar(star)

        b = star.image.bounds.withBorder(-8)
        max_diff = np.max(np.abs(test_star.image[b].array - star.image[b].array))
        max_val = np.max(np.abs(star.image[b].array))
        mse = np.sum((test_star.image[b].array - star.image[b].array)**2) / flux**2
        mean_max_rel_err += max_diff/max_val
        mean_mse += mse
        count += 1

    mean_max_rel_err /= count
    mean_mse /= count
    print('mean maximum relative error = ',mean_max_rel_err)
    print('mean mean-squared error = ',mean_mse)
    assert mean_max_rel_err < 0.04
    assert mean_mse < 1.5e-5

    # Repeat using nested Convolves
    config['psf'] = {
        'type': 'Convolve',
        'components': [
            {
                'type': 'Convolve',
                'components': [
                    {
                        'model': { 'type': 'Kolmogorov',
                                   'fastfit': True,
                                 },
                        'interp': { 'type': 'GP',
                                  },
                    },
                    {
                        'model': { 'type': 'Optical',
                                   'atmo_type': 'None',
                                   'lam': 500,
                                   'diam': 8,
                                   # These are the correct aberrations, not fitted.
                                   'base_aberrations': [0,0,0,0,0.7,-0.8,0.7,0.5,0.7,-0.6,0.5,0.8],
                                   'obscuration': 0.4,
                                 },
                        'interp': { 'type': 'Mean', },
                    },
                ],
            },
            {
                'model': { 'type': 'Gaussian' },
                # Note: default init is delta, so don't need to specify it explicitily.
                'interp': { 'type': 'Mean', },
            },
        ],
    }

    psf = piff.process(config, logger)

    assert type(psf) is piff.ConvolvePSF
    assert len(psf.components) == 2
    assert type(psf.components[0]) is piff.ConvolvePSF
    assert type(psf.components[0].components[0]) is piff.SimplePSF
    assert type(psf.components[0].components[0].model) is piff.Kolmogorov
    assert type(psf.components[0].components[0].interp) is piff.GPInterp
    assert type(psf.components[0].components[1]) is piff.SimplePSF
    assert type(psf.components[0].components[1].model) is piff.Optical
    assert type(psf.components[0].components[1].interp) is piff.Mean
    assert type(psf.components[1]) is piff.SimplePSF
    assert type(psf.components[1].model) is piff.Gaussian
    assert type(psf.components[1].interp) is piff.Mean

    mean_max_rel_err = 0
    mean_mse = 0
    count = 0
    for i, star in enumerate(psf.stars):
        if star.is_flagged:
            continue
        test_star = psf.drawStar(star)

        b = star.image.bounds.withBorder(-8)
        max_diff = np.max(np.abs(test_star.image[b].array - star.image[b].array))
        max_val = np.max(np.abs(star.image[b].array))
        mse = np.sum((test_star.image[b].array - star.image[b].array)**2) / flux**2
        mean_max_rel_err += max_diff/max_val
        mean_mse += mse
        count += 1

    mean_max_rel_err /= count
    mean_mse /= count
    print('mean maximum relative error = ',mean_max_rel_err)
    print('mean mean-squared error = ',mean_mse)
    assert mean_max_rel_err < 0.04
    assert mean_mse < 1.5e-5



if __name__ == '__main__':
    test_trivial_convolve1()
    test_convolve_optatm()
    test_convolve_pixelgrid()
    test_nested_convolve()
