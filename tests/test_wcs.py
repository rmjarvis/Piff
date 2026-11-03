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
import fitsio
import os
import warnings
import coord
import pytest

from piff_test_helper import get_script_name, timer, CaptureLog

# Helper function for drawing an image of a number of Moffat stars
def drawImage(xsize, ysize, wcs, x, y, e1, e2, s):
    # Use GalSim to draw stars onto the images.
    nstars = len(x)
    gs_config = {
        'psf' : {
            'type' : 'Moffat',
            'beta' : 2.5,
            'ellip' : {
                'type' : 'G1G2',
                'g1' : e1.tolist(),
                'g2' : e2.tolist()
            },
            'half_light_radius' : s.tolist()
        },
        'image' : {
            'type' : 'Scattered',
            'xsize' : xsize,
            'ysize' : ysize,
            'wcs' : wcs,
            'nobjects' : nstars,
            'image_pos' : {
                'type' : 'XY',
                'x' : x.tolist(),
                'y' : y.tolist()
            }
        }
    }
    return galsim.config.BuildImage(gs_config)

@timer
def test_focal():
    """This test uses 2 input files and two catalogs, but does the interpolation over the
    whole field of view.
    """
    # Give them different wcs's.
    # The centers should be separated by ~0.25 arcsec/pixel * 2048 pixels / cos(dec) = 565 arcsec
    # The actual separation of 10 arcmin gives a bit of a gap between the chips.
    wcs1 = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wcs2 = galsim.TanWCS(
            galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees)
            )
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)

    if __name__ == '__main__':
        nstars = 20  # per ccd
    else:
        nstars = 3  # per ccd
    rng = np.random.RandomState(1234)
    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    u, v = field_center.project_rad(*wcs1._radec(x.copy(),y.copy()), projection='gnomonic')
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data1 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    np.testing.assert_array_equal(data1['x'] , x)
    np.testing.assert_array_equal(data1['y'] , y)
    np.testing.assert_array_equal(data1['e1'] , e1)
    np.testing.assert_array_equal(data1['e2'] , e2)
    np.testing.assert_array_equal(data1['s'] , s)
    im1 = drawImage(2048, 2048, wcs1, x, y, e1, e2, s)
    im1.write('output/test_focal_im1.fits')
    fitsio.write('output/test_focal_cat1.fits', data1, clobber=True)

    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    u, v = field_center.project_rad(*wcs2._radec(x.copy(),y.copy()), projection='gnomonic')
    # Same functions of u,v, but using the positions on chip 2
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data2 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im2 = drawImage(2048, 2048, wcs2, x, y, e1, e2, s)
    im2.write('output/test_focal_im2.fits')
    fitsio.write('output/test_focal_cat2.fits', data2, clobber=True)

    # Try to fit with the right model (Moffat) and interpolant (2nd order polyomial)
    # Should work very well, since no noise.
    config = {
        'input' : {
            'image_file_name' : 'output/test_focal_im?.fits',
            'cat_file_name' : 'output/test_focal_cat?.fits',
            'x_col' : 'x',
            'y_col' : 'y',
            'ra' : 0.,
            'dec' : -25.,
        },
        'psf' : {
            'type' : 'Simple',
            'model' : {
                'type' : 'Moffat',
                'beta' : 2.5
            },
            'interp' : {
                'type' : 'Polynomial',
                'order' : 2
            }
        }
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    psf = piff.process(config)

    for data, wcs in [(data1,wcs1), (data2,wcs2)]:
        for k in range(nstars):
            x = data['x'][k]
            y = data['y'][k]
            e1 = data['e1'][k]
            e2 = data['e2'][k]
            s = data['s'][k]
            #print('k,x,y = ',k,x,y)
            #print('  true s,e1,e2 = ',s,e1,e2)
            image_pos = galsim.PositionD(x,y)
            star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center)
            star = psf.drawStar(star)
            #print('  fitted s,e1,e2 = ',star.fit.params)
            np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)


@timer
def test_wrongwcs():
    """Same as test_focal, but the images are written out with the wrong wcs.
    """
    wcs1 = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wcs2 = galsim.TanWCS(
            galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wrong_wcs = galsim.TanWCS(
            galsim.AffineTransform(0.25, 0, 0, 0.25, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(0 * galsim.arcmin, -25 * galsim.degrees)
            )
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)

    if __name__ == '__main__':
        nstars = 20  # per ccd
    else:
        nstars = 3  # per ccd
    rng = np.random.RandomState(1234)
    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    u, v = field_center.project_rad(*wcs1._radec(x.copy(),y.copy()), projection='gnomonic')
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v
    data1 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im1 = drawImage(2048, 2048, wcs1, x, y, e1, e2, s)

    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    u, v = field_center.project_rad(*wcs2._radec(x.copy(),y.copy()), projection='gnomonic')
    # Same functions of u,v, but using the positions on chip 2
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v
    data2 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im2 = drawImage(2048, 2048, wcs2, x, y, e1, e2, s)

    # Put in the wrong wcs before writing them to files.
    im1.wcs = im2.wcs = wrong_wcs
    im1.write('output/test_wrongwcs_im1.fits')
    im2.write('output/test_wrongwcs_im2.fits')
    fitsio.write('output/test_wrongwcs_cat1.fits', data1, clobber=True)
    fitsio.write('output/test_wrongwcs_cat2.fits', data2, clobber=True)

    config = {
        'modules' : [ 'custom_wcs' ],
        'input' : {
            'dir' : 'output',
            # Normally more convenient to use a glob string, but an explicit list is also allowed.
            'image_file_name' : ['test_wrongwcs_im1.fits', 'test_wrongwcs_im2.fits'],
            'cat_file_name' : ['test_wrongwcs_cat1.fits', 'test_wrongwcs_cat2.fits'],
            'x_col' : 'x',
            'y_col' : 'y',
            'ra' : 0.,
            'dec' : -25.,
            # But here tell Piff the correct WCS to use.  This uses a custom WCS builder,
            # mostly so we can test the 'modules' option.  In practice, you might use a
            # galsim_extra Pixmappy WCS class.  Or maybe an LSST DM WCS.
            'wcs' : { 'type' : 'Custom' }
        },
        'psf' : {
            'type' : 'Simple',
            'model' : {
                'type' : 'Moffat',
                'beta' : 2.5
            },
            'interp' : {
                'type' : 'Polynomial',
                'order' : 2
            }
        },
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    psf = piff.process(config)

    for data, wcs in [(data1,wcs1), (data2,wcs2)]:
        for k in range(nstars):
            x = data['x'][k]
            y = data['y'][k]
            e1 = data['e1'][k]
            e2 = data['e2'][k]
            s = data['s'][k]
            #print('k,x,y = ',k,x,y)
            #print('  true s,e1,e2 = ',s,e1,e2)
            image_pos = galsim.PositionD(x,y)
            star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center)
            star = psf.drawStar(star)
            #print('  fitted s,e1,e2 = ',star.fit.params)
            np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)

@timer
def test_single():
    """Same as test_focal, but using the SingleCCD PSF type, which does a separate fit on each CCD.
    """
    wcs1 = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wcs2 = galsim.TanWCS(
            galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees)
            )
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)

    if __name__ == '__main__':
        nstars = 20  # per ccd
        logger = piff.config.setup_logger(verbose=2)
    else:
        nstars = 6  # per ccd
        logger = piff.config.setup_logger(log_file='output/test_single.log')
    rng = np.random.RandomState(1234)
    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    ra1, dec1 = wcs1.toWorld(x,y,units='rad')
    u, v = field_center.project_rad(ra1,dec1, projection='gnomonic')
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data1 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im1 = drawImage(2048, 2048, wcs1, x, y, e1, e2, s)
    im1.write('output/test_single_im1.fits')
    fitsio.write('output/test_single_cat1.fits', data1, clobber=True)

    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    ra2, dec2 = wcs2.toWorld(x,y,units='rad')
    u, v = field_center.project_rad(ra1,dec1, projection='gnomonic')
    # Same functions of u,v, but using the positions on chip 2
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data2 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im2 = drawImage(2048, 2048, wcs2, x, y, e1, e2, s)
    im2.write('output/test_single_im2.fits')
    fitsio.write('output/test_single_cat2.fits', data2, clobber=True)

    ra12 = np.concatenate([ra1,ra2])
    dec12 = np.concatenate([dec1,dec2])
    data12 = np.array(list(zip(ra12,dec12)), dtype=[('ra',float), ('dec',float)])
    fitsio.write('output/test_single_cat12.fits', data12, clobber=True)

    # Try to fit with the right model (Moffat) and interpolant (2nd order polyomial)
    # Should work very well, since no noise.
    config = {
        'input' : {
            # A third way to input these same file names.  Use GalSim config values and
            # explicitly specify the number of images to read
            'nimages' : 2,
            'image_file_name' : {
                'type' : 'FormattedStr',
                'format' : '%s/test_single_im%d.fits',
                'items' : [ 'output', '$image_num+1' ]
            },
            'cat_file_name' : {
                'type' : 'FormattedStr',
                'format' : '%s/test_single_cat%d.fits',
                'items' : [ 'output', '$image_num+1' ]
            },
            # Use chipnum = 1,2 rather than the default 0,1.
            'chipnum' : '$image_num+1',
            'x_col' : 'x',
            'y_col' : 'y',
            'ra' : 0.,
            'dec' : -25.,
        },
        'psf' : {
            'type' : 'SingleChip',
            'model' : {
                'type' : 'Moffat',
                'beta' : 2.5
            },
            'interp' : {
                'type' : 'Polynomial',
                'order' : 2
            }
        },
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    with CaptureLog(level=2) as cl:
        psf = piff.process(config, cl.logger)
    #print('without nproc, log = ',cl.output)
    assert "Building solution for chip 1" in cl.output
    assert "Building solution for chip 2" in cl.output

    for chipnum, data, wcs in [(1,data1,wcs1), (2,data2,wcs2)]:
        for k in range(nstars):
            x = data['x'][k]
            y = data['y'][k]
            e1 = data['e1'][k]
            e2 = data['e2'][k]
            s = data['s'][k]
            #print('k,x,y = ',k,x,y)
            #print('  true s,e1,e2 = ',s,e1,e2)
            image_pos = galsim.PositionD(x,y)
            star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center,
                                        chipnum=chipnum)
            star = psf.drawStar(star)
            #print('  fitted s,e1,e2 = ',star.fit.params)
            np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)

            # This is the more user-friendly way to do this.
            # Equivalent to ~machine precision.
            im = psf.draw(x, y, chipnum=chipnum)
            print('im = ',im)
            print('star im = ',star.data.image)
            print('max diff = ',np.max(np.abs(im.array - star.data.image.array)))
            np.testing.assert_almost_equal(im.array, star.data.image.array)

    # Make sure old pattern of giving wcs, pointing in fit() still works.
    objects, wcs_dict, pointing, bandpass = piff.Input.process(config['input'], logger=logger)
    assert bandpass is None
    with CaptureLog(level=2) as cl:
        psf_old = piff.PSF.process(config['psf'], logger=cl.logger)
        psf_old.fit(objects, wcs_dict, pointing, logger=cl.logger)
    assert "wcs should now be set with set_context" in cl.output

    for chipnum, data in [(1, data1), (2, data2)]:
        x = data['x'][0]
        y = data['y'][0]
        im = psf.draw(x, y, chipnum=chipnum)
        im_old = psf_old.draw(x, y, chipnum=chipnum)
        np.testing.assert_almost_equal(im_old.array, im.array)

    # SingleChip doesn't use code paths that hit these functions, but they are officially
    # required methods for PSF classes, so just check them directly.
    assert psf.fit_center is True
    raw1 = psf._getRawProfile(star)
    raw2 = psf.psf_by_chip[star['chipnum']]._getRawProfile(star)
    assert raw1 == raw2

    # Chipnum is required as a property to use SingleCCDPSF
    star1 = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center)
    with np.testing.assert_raises(ValueError):
        psf.drawStar(star1)
    star2 = piff.Star(star1.data, star.fit)  # If has a fit, it hits a different error
    with np.testing.assert_raises(ValueError):
        psf.drawStar(star2)

    # Check the more typical functions: psf.draw and psf.get_profile
    with np.testing.assert_raises(ValueError):
        psf.draw(x, y)
    with np.testing.assert_raises(ValueError):
        psf.get_profile(x, y)

    # Also gives an error if the chipnum is wrong.
    with np.testing.assert_raises(ValueError):
        psf.draw(x, y, chipnum=22)
    with np.testing.assert_raises(ValueError):
        psf.get_profile(x, y, chipnum=22)


@timer
def test_parallel():
    # Run the same test as test_single, but using nproc
    wcs1 = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wcs2 = galsim.TanWCS(
            galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees)
            )
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)

    if __name__ == '__main__':
        nstars = 20  # per ccd
    else:
        nstars = 6  # per ccd
    rng = np.random.RandomState(1234)
    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    ra1, dec1 = wcs1.toWorld(x,y,units='rad')
    u, v = field_center.project_rad(ra1,dec1, projection='gnomonic')
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data1 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im1 = drawImage(2048, 2048, wcs1, x, y, e1, e2, s)
    im1.write('output/test_parallel_im1.fits')

    x = rng.random_sample(nstars) * 2000 + 24
    y = rng.random_sample(nstars) * 2000 + 24
    ra2, dec2 = wcs2.toWorld(x,y,units='rad')
    u, v = field_center.project_rad(ra1,dec1, projection='gnomonic')
    # Same functions of u,v, but using the positions on chip 2
    e1 = 0.02 + 2.e-5 * u - 3.e-9 * u**2 + 2.e-9 * v**2
    e2 = -0.04 - 3.e-5 * v + 1.e-9 * u*v + 3.e-9 * v**2
    s = 0.3 + 8.e-9 * (u**2 + v**2) - 1.e-9 * u*v

    data2 = np.array(list(zip(x,y,e1,e2,s)),
                     dtype=[ ('x',float), ('y',float), ('e1',float), ('e2',float), ('s',float) ])
    im2 = drawImage(2048, 2048, wcs2, x, y, e1, e2, s)
    im2.write('output/test_parallel_im2.fits')

    ra12 = np.concatenate([ra1,ra2])
    dec12 = np.concatenate([dec1,dec2])
    data12 = np.array(list(zip(ra12,dec12)), dtype=[('ra',float), ('dec',float)])
    fitsio.write('output/test_parallel.fits', data12, clobber=True)

    # im3 is blank.  Will give errors trying to measure PSF from it.
    im3 = galsim.Image(2048,2048, wcs=wcs2)
    im3.write('output/test_parallel_im3.fits')

    psf_file = os.path.join('output','test_single.fits')
    config = {
        'input' : {
            # A third way to input these same file names.  Use GalSim config values and
            # explicitly specify the number of images to read
            'nimages' : 2,
            'image_file_name' : {
                'type' : 'FormattedStr',
                'format' : '%s/test_parallel_im%d.fits',
                'items' : [ 'output', '$image_num+1' ],
            },
            'cat_file_name' : 'output/test_parallel.fits',
            'chipnum' : '$image_num+1',
            'ra_col' : 'ra',
            'dec_col' : 'dec',
            'ra_units' : 'rad',
            'dec_units' : 'rad',
            'nproc' : -1,
        },
        'psf' : {
            'type' : 'SingleChip',
            'model' : {
                'type' : 'Moffat',
                'beta' : 2.5,
            },
            'interp' : {
                'type' : 'Polynomial',
                'order' : 2,
            },
            'nproc' : 2,
        },
        'output' : {
            'file_name' : psf_file,
        },
    }
    with CaptureLog(level=2) as cl:
        piff.piffify(config, logger=cl.logger)
    psf = piff.read(psf_file)

    for chipnum, data, wcs in [(1,data1,wcs1), (2,data2,wcs2)]:
        for k in range(nstars):
            x = data['x'][k]
            y = data['y'][k]
            e1 = data['e1'][k]
            e2 = data['e2'][k]
            s = data['s'][k]
            image_pos = galsim.PositionD(x,y)
            star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center,
                                        chipnum=chipnum)
            star = psf.drawStar(star)
            np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)

    # Finally, check that the logger properly captures the subprocess logs
    with CaptureLog(level=2) as cl:
        psf = piff.process(config, cl.logger)
    #print('with nproc=2, log = ',cl.output)
    assert "Processing catalog 1" in cl.output
    assert "Processing catalog 2" in cl.output
    assert "Building solution for chip 1" in cl.output
    assert "Building solution for chip 2" in cl.output

    # With verbose=0, there should be no output at all.  (There used to be a single newline.)
    with CaptureLog(level=0) as cl:
        piff.piffify(config, logger=cl.logger)
    assert len(cl.output) == 0

    # Check that errors in the solution get properly reported.
    config['input']['nimages'] = 3
    with CaptureLog(level=2) as cl:
        psf = piff.process(config, cl.logger)
    assert "No stars left to fit.  Cannot find PSF model." in cl.output
    assert "Solutions failed for chipnums: [3]" in cl.output

    # Check that errors in the multiprocessing input get properly reported.
    config['input']['ra_col'] = 'invalid'
    with CaptureLog(level=2) as cl:
        with np.testing.assert_raises(ValueError):
            psf = piff.process(config, cl.logger)
    assert "ra_col = invalid is not a column" in cl.output

    # With nproc=1, the error is raised directly.
    config['input']['nproc'] = 1
    config['verbose'] = 0
    with np.testing.assert_raises(ValueError):
        psf = piff.process(config)

    # But just the input error.  Not the one in fitting.
    config['psf']['nproc'] = 1
    config['input']['ra_col'] = 'ra'
    config['verbose'] = 1
    with CaptureLog(level=1) as cl:
        psf = piff.process(config, logger=cl.logger)
    assert "No stars left to fit.  Cannot find PSF model." in cl.output
    assert "Ignoring this failure and continuing on." in cl.output


@timer
def test_pickle():
    """Test the reading a file written with python 2 pickling is readable with python 2 or 3.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_pickle.log')

    # First, this is the output file written by the above test_single function on python 2.
    # Shoudl be trivially readable by python 2, but make sure it is also readable by python 3.
    with warnings.catch_warnings():
        if np.__version__ > '2':
            warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)
        psf = piff.read('input/test_single_py27.piff', logger=logger)

    wcs1 = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    wcs2 = galsim.TanWCS(
            galsim.AffineTransform(0.25, -0.02, 0.01, 0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(5 * galsim.arcmin, -25 * galsim.degrees)
            )

    data1 = fitsio.read('input/test_single_cat1.fits')
    data2 = fitsio.read('input/test_single_cat2.fits')
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)

    for chipnum, data, wcs in [(1,data1,wcs1), (2,data2,wcs2)]:
        for k in range(len(data)):
            x = data['x'][k]
            y = data['y'][k]
            e1 = data['e1'][k]
            e2 = data['e2'][k]
            s = data['s'][k]
            #print('k,x,y = ',k,x,y)
            #print('  true s,e1,e2 = ',s,e1,e2)
            image_pos = galsim.PositionD(x,y)
            star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center,
                                        chipnum=chipnum)
            star = psf.drawStar(star)
            #print('  fitted s,e1,e2 = ',star.fit.params)
            np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)


@timer
def test_olddes():
    # This is a DES Y3 PSF file that Matt Becker reported hadn't been readable with python 3.
    # The problem was it had been written with python 2's pickle, which isn't directly
    # compatible with python 3.  The code has been fixed to make it readable.  This unit
    # test is just to ensure that it remains so.
    # However, it only works if pixmappy is installed, so if not, just bail out.
    try:
        import pixmappy
    except ImportError:
        print('pixmappy not installed.  Skipping test_olddes()')
        return
    import copy

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_olddes.log')

    fname = os.path.join('input', 'D00240560_r_c01_r2362p01_piff.fits')
    with warnings.catch_warnings():
        # Numpy 2.4+ has started warning about a deprecation here.
        if np.__version__ > '2':
            warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)
        psf = piff.PSF.read(fname, logger=logger)

    print('psf.wcs = ',psf.wcs[0])
    print('(0,0) -> ',psf.wcs[0].toWorld(galsim.PositionD(0,0)))
    assert np.isclose(psf.wcs[0].toWorld(galsim.PositionD(0,0)).ra / galsim.degrees, 37.490941423)
    assert np.isclose(psf.wcs[0].toWorld(galsim.PositionD(0,0)).dec / galsim.degrees, -5.03391729)
    print('local at 0,0 = ',psf.wcs[0].local(galsim.PositionD(0,0)))
    print('area at 0,0 = ',psf.wcs[0].pixelArea(galsim.PositionD(0,0)),' = %f**2'%(
            psf.wcs[0].pixelArea(galsim.PositionD(0,0))**0.5))
    assert np.isclose(psf.wcs[0].pixelArea(galsim.PositionD(0,0)), 0.2628**2, rtol=1.e-3)
    image = psf.draw(x=103.3, y=592.0, logger=logger)
    print('image shape = ',image.array.shape)
    print('image near center = ',image.array[23:26,23:26])
    print('image sum = ',image.array.sum())
    assert np.isclose(image.array.sum(), 1.0, rtol=1.e-2)
    # The center values should be at least close to the following:
    regression_array = np.array([[0.02920381, 0.03528429, 0.03267081],
                                 [0.03597827, 0.04419591, 0.04229439],
                                 [0.03001573, 0.03743261, 0.03300782]])
    # Note: the centering mechanics have changed since this regression was set up to make the
    # nominal PSF center closer to the image center.  So the second slice changed from
    # 23:26 -> 22:25.
    np.testing.assert_allclose(image.array[23:26,22:25], regression_array, rtol=1.e-5)

    # Also check that it is picklable.
    psf2 = copy.deepcopy(psf)
    image2 = psf2.draw(x=103.3, y=592.0)
    np.testing.assert_equal(image2.array, image.array)

@timer
def test_newdes():
    # This is a DES Y6 PSF file made by Robert Gruendl using python 2, so
    # check that this also works correctly.
    try:
        import pixmappy
    except ImportError:
        print('pixmappy not installed.  Skipping test_newdes()')
        return
    # Also make sure pixmappy is recent enough to work.
    if 'exposure_file' not in pixmappy.GalSimWCS._opt_params:
        print('pixmappy not recent enough version.  Skipping test_newdes()')
        return
    import copy

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_newdes.log')

    fname = os.path.join('input', 'D00232418_i_c19_r5006p01_piff-model.fits')
    with warnings.catch_warnings():
        # This file was written with GalSim 2.1, and now raises a deprecation warning for 2.2.
        warnings.simplefilter("ignore", galsim.GalSimDeprecationWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        if np.__version__ > '2':
            warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)
        psf = piff.PSF.read(fname, logger=logger)

    print('psf.wcs = ',psf.wcs[0])
    print('(0,0) -> ',psf.wcs[0].toWorld(galsim.PositionD(0,0)))
    print(psf.wcs[0].toWorld(galsim.PositionD(0,0)).ra/galsim.degrees,
            psf.wcs[0].toWorld(galsim.PositionD(0,0)).dec/galsim.degrees)
    assert np.isclose(psf.wcs[0].toWorld(galsim.PositionD(0,0)).ra / galsim.degrees, 15.4729872672)
    assert np.isclose(psf.wcs[0].toWorld(galsim.PositionD(0,0)).dec / galsim.degrees, 1.95221895945)
    print('local at 0,0 = ',psf.wcs[0].local(galsim.PositionD(0,0)))
    print('area at 0,0 = ',psf.wcs[0].pixelArea(galsim.PositionD(0,0)),' = %f**2'%(
            psf.wcs[0].pixelArea(galsim.PositionD(0,0))**0.5))
    assert np.isclose(psf.wcs[0].pixelArea(galsim.PositionD(0,0)), 0.263021**2, rtol=1.e-3)
    image = psf.draw(x=103.3, y=592.0, logger=logger)
    print('image shape = ',image.array.shape)
    print('image near center = ',image.array[23:26,23:26])
    print('image sum = ',image.array.sum())
    assert np.isclose(image.array.sum(), 1.0, rtol=1.e-2)
    # The center values should be at least close to the following:
    regression_array = np.array([[0.03305565, 0.04500969, 0.0395154],
                                 [0.03765249, 0.05419811, 0.04867231],
                                 [0.02734579, 0.0418797, 0.03928504]])
    # Note: the centering mechanics have changed since this regression was set up to make the
    # nominal PSF center closer to the image center.  So the second slice changed from
    # 23:26 -> 22:25.
    np.testing.assert_allclose(image.array[23:26,22:25], regression_array, rtol=1.e-4)

    # Also check that it is picklable.
    psf2 = copy.deepcopy(psf)
    image2 = psf2.draw(x=103.3, y=592.0)
    np.testing.assert_equal(image2.array, image.array)

@timer
def test_des_wcs():
    """Test the get_nominal_wcs function.
    """
    # Read a random DES image
    image_file = 'input/DECam_00241238_01.fits.fz'
    print('read DES image ',image_file)
    im = galsim.fits.read(image_file, read_header=True)
    print(list(im.header.keys()))
    print('ra = ',im.header['TELRA'])
    print('dec = ',im.header['TELDEC'])
    ra = coord.Angle.from_hms(im.header['TELRA'])
    dec = coord.Angle.from_dms(im.header['TELDEC'])
    pointing = coord.CelestialCoord(ra,dec)

    print('raw wcs = ',im.wcs)
    print('world coord at center = ',im.wcs.toWorld(im.center))
    print('pointing = ',pointing)
    print('dist = ',pointing.distanceTo(im.wcs.toWorld(im.center)).deg)

    # This chip is near the edge, but check that we're at least close to the nominal pointing.
    assert pointing.distanceTo(im.wcs.toWorld(im.center)) < 1.2 * coord.degrees

    # Get the local affine approximation relative to the pointing center
    wcs1 = im.wcs.affine(world_pos=pointing)
    print('wcs1 = ',wcs1)
    print('u,v at image center = ',wcs1.toWorld(im.center))

    # A different approach.  Get the local wcs at the image center, and adjust
    # the origin to the pointing center.
    wcs2 = im.wcs.local(im.center).withOrigin(im.wcs.toImage(pointing))
    print('wcs2 = ',wcs2)
    print('u,v at image center = ',wcs2.toWorld(im.center))

    # Finally, compare with the mock up approximate version we have in piff
    wcs3 = piff.des.DECamInfo().get_nominal_wcs(chipnum=1)
    print('wcs3 = ',wcs3)
    print('u,v at image center = ',wcs3.toWorld(im.center))

    # Check that these are all vaguley similar.
    # wcs2 is probably the most accurate of these, since it Taylor expands the nonlinear
    # stuff at the image center.  wcs1 expands around a point way off the chip, so there
    # are expected to be some errors in this extrapolation.
    # And of course wcs3 is just an approximation, so it's only expected to be good to a
    # few arcsec or so.
    np.testing.assert_allclose(wcs3.jacobian().getMatrix(), wcs2.jacobian().getMatrix(),
                               rtol=0.02, atol=0.003)
    np.testing.assert_allclose(wcs3.toWorld(im.center).x, wcs2.toWorld(im.center).x,
                               rtol=0.02)
    np.testing.assert_allclose(wcs3.toWorld(im.center).y, wcs2.toWorld(im.center).y,
                               rtol=0.02)

    # As mentioned, wcs1 is not as close, but that's ok.
    np.testing.assert_allclose(wcs3.jacobian().getMatrix(), wcs1.jacobian().getMatrix(),
                               rtol=0.04, atol=0.002)
    np.testing.assert_allclose(wcs3.toWorld(im.center).x, wcs1.toWorld(im.center).x,
                               rtol=0.04)
    np.testing.assert_allclose(wcs3.toWorld(im.center).y, wcs1.toWorld(im.center).y,
                               rtol=0.04)

@timer
def test_gwcs():
    """Test the RomanGWCS class."""
    # Make a gwcs instance with a rotated TAN projection so we can make an equivalent galsim wcs.
    gwcs = pytest.importorskip('gwcs')
    import pickle
    from astropy import units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs.coordinate_frames import Frame2D, CelestialFrame
    center = galsim.CelestialCoord(53 * galsim.degrees, -27 * galsim.degrees)
    matrix = np.array([[-0.11, 0.005], [0.003, 0.108]])
    transform = ((models.Shift(-1024) & models.Shift(-1024))
                 | models.AffineTransformation2D(matrix=matrix / 3600)
                 | models.Pix2Sky_TAN()
                 | models.RotateNative2Celestial(53, -27, 180))
    native = gwcs.WCS(
        transform,
        input_frame=Frame2D(name='detector', unit=(u.pix, u.pix)),
        output_frame=CelestialFrame(reference_frame=ICRS(), name='world',
                                    unit=(u.deg, u.deg)),
    )
    native.bounding_box = ((0, 2047), (0, 2047))
    wcs = piff.roman.RomanGWCS(native)
    reference = galsim.TanWCS(
        galsim.AffineTransform(0.11, -0.005, 0.003, 0.108,
                               origin=galsim.PositionD(1025, 1025)), center)

    # Check that to and from world coordinates matches the reference wcs.
    positions = [galsim.PositionD(x, y) for x, y in
                 [(1, 1), (100.3, 800.7), (1025, 1025), (2048, 2048), (2050, -2)]]
    x = np.array([pos.x for pos in positions])
    y = np.array([pos.y for pos in positions])
    expected_ra, expected_dec = reference.xyToradec(x, y, units=galsim.degrees)
    ra, dec = wcs.xyToradec(x, y, units=galsim.degrees)
    np.testing.assert_allclose(ra, expected_ra, rtol=0, atol=1.e-10)
    np.testing.assert_allclose(dec, expected_dec, rtol=0, atol=1.e-10)
    np.testing.assert_allclose(wcs.radecToxy(ra, dec, units=galsim.degrees), [x, y],
                               rtol=0, atol=1.e-7)
    for pos in positions:
        world = wcs.toWorld(pos)
        assert world.distanceTo(reference.toWorld(pos)) / galsim.arcsec < 1.e-6
        recovered = wcs.toImage(world)
        np.testing.assert_allclose([recovered.x, recovered.y], [pos.x, pos.y],
                                   rtol=0, atol=1.e-7)

    # Local Jacobians and pixel areas agree with the independent tangent-plane projection.
    # The RomanGWCS calculates this with finite differences. GalSim's GSFits does an analytic
    # derivative. They match very well here.
    for pos in positions:
        np.testing.assert_allclose(wcs.jacobian(pos).getMatrix(),
                                   reference.jacobian(pos).getMatrix(),
                                   rtol=1.e-7, atol=1.e-8)
        np.testing.assert_allclose(wcs.pixelArea(pos), reference.pixelArea(pos), rtol=1.e-7)

    # Make two asdf files each with two gwcs objects stored at different paths.
    # The alternate at alternate.astrometry and the normal one in the second file include
    # a shift of 0.1 degrees in RA compared to the reference wcs.
    asdf = pytest.importorskip('asdf')
    asdf_names = [os.path.join('output', 'gwcs_input_%d.asdf' % i) for i in range(2)]
    second_native = gwcs.WCS(
        transform | (models.Shift(0.1) & models.Shift(0)),  # Shift RA by 0.1 degrees
        input_frame=native.input_frame, output_frame=native.output_frame)
    for file_name, native_wcs in zip(asdf_names, [native, second_native]):
        tree = {
            'roman': {'meta': {'wcs': native_wcs}},
            'alternate': {'astrometry': second_native},
        }
        with asdf.AsdfFile(tree) as af:
            af.write_to(file_name)

    # RomanGWCS loads the metadata from ASDF, with literal or directory/reference filenames.
    config = {
        'input': {
            'wcs': {
                'type': 'RomanGWCS',
                'file_name': asdf_names[0],
            }
        }
    }
    loaded = galsim.config.BuildWCS(config['input'], 'wcs', config)
    np.testing.assert_allclose(loaded.xyToradec(x, y, units='degrees'), [ra, dec],
                               rtol=0, atol=1.e-10)

    # Check using dir and reading file name with special GalSim config processing.
    config = {
        'input': {
            'wcs': {
                'type': 'RomanGWCS',
                'dir': 'output',
                'file_name': '@gwcs_name',
            }
        },
        'gwcs_name': os.path.basename(asdf_names[0]),
    }
    loaded = galsim.config.BuildWCS(config['input'], 'wcs', config)
    np.testing.assert_allclose(loaded.xyToradec(x, y, units='degrees'), [ra, dec],
                               rtol=0, atol=1.e-10)

    # The path option specifies a different place in the asdf tree.
    config = {
        'input': {
            'wcs': {
                'type': 'RomanGWCS',
                'file_name': asdf_names[0],
                'path': 'alternate.astrometry'
            }
        }
    }
    alternate = galsim.config.BuildWCS(config['input'], 'wcs', config)
    np.testing.assert_allclose(alternate.xyToradec(x, y, units='degrees'), [ra + 0.1, dec],
                               rtol=0, atol=1.e-10)

    # The file_name parsing can use the image_num to get a different wcs for each input image.
    image_name = os.path.join('output', 'gwcs_override.fits')
    cat_name = os.path.join('output', 'gwcs_override_cat.fits')
    image = galsim.Gaussian(sigma=0.17).drawImage(nx=64, ny=64, wcs=reference)
    image.wcs = galsim.PixelScale(0.5)
    image.write(image_name)
    fitsio.write(cat_name, np.array([(32.0, 32.0)], dtype=[('x', float), ('y', float)]),
                 clobber=True)
    config = {
        'input': {
            'image_file_name': [image_name, image_name],
            'cat_file_name': cat_name,
            'stamp_size': 25,
            'nproc': 2,
            'wcs': {
                'type': 'RomanGWCS',
                'dir': 'output',
                'file_name': '$"gwcs_input_%s.asdf"%image_num',
            }
        }
    }
    reader = piff.InputFiles(config['input'], base=config)
    stars = reader.makeStars(logger=piff.config.setup_logger(verbose=0))
    assert len(stars) == 2
    for i, star in enumerate(stars):
        expected_ra = ra + 0.1 * i
        np.testing.assert_allclose(star.image.wcs.xyToradec(x, y, units='degrees'),
                                   [expected_ra, dec], rtol=0, atol=1.e-10)

    # Origin shifts preserve the mapping in the corresponding image coordinates.
    # The WCS object also survives a serialization round trip.
    shift = galsim.PositionD(13, -8)
    shifted = wcs.shiftOrigin(shift)
    assert shifted.origin == wcs.origin + shift
    for copied in [wcs.copy(), pickle.loads(pickle.dumps(wcs))]:
        np.testing.assert_allclose(copied.xyToradec(x, y, units='degrees'), [ra, dec],
                                   rtol=0, atol=1.e-10)
    np.testing.assert_allclose(shifted.xyToradec(x + shift.x, y + shift.y, units='degrees'),
                               [ra, dec], rtol=0, atol=1.e-10)

    # Test using Piff to fit stars drawn with the GWCS.
    profile = galsim.Gaussian(sigma=0.17).shear(g1=0.08, g2=-0.04).withFlux(1000)
    stars = []
    for pos in positions[:4]:
        star = piff.Star.makeTarget(x=pos.x, y=pos.y, wcs=wcs, pointing=center,
                                    stamp_size=25, chipnum=0)
        profile.drawImage(image=star.image, center=pos, method='no_pixel')
        stars.append(star)
    psf = piff.SimplePSF(piff.Gaussian(include_pixel=False), piff.Mean())
    psf.set_context({0: loaded}, center)
    psf.fit(stars)
    drawn = psf.draw(800.3, 1200.7, stamp_size=25)
    expected = profile.withFlux(1).drawImage(bounds=drawn.bounds, wcs=reference,
                                            center=galsim.PositionD(800.3, 1200.7),
                                            method='no_pixel')
    np.testing.assert_allclose(drawn.array, expected.array, rtol=1.e-4, atol=1.e-7)

    # The PSF roundtrips correctly through a piff output file.
    file_name = os.path.join('output', 'gwcs.piff')
    psf.write(file_name)
    restored = piff.read(file_name)
    np.testing.assert_allclose(restored.wcs[0].xyToradec(x, y, units='degrees'), [ra, dec],
                               rtol=0, atol=1.e-10)
    np.testing.assert_array_equal(restored.draw(800.3, 1200.7, stamp_size=25).array, drawn.array)
    np.testing.assert_allclose(restored.wcs[0].radecToxy(ra, dec, units='degrees'), [x, y],
                               rtol=0, atol=1.e-7)

    # Full GWCS models cannot be represented by a FITS image header.
    with np.testing.assert_raises(NotImplementedError):
        drawn.write(os.path.join('output', 'gwcs_image.fits'))


if __name__ == '__main__':
    #import cProfile, pstats
    #pr = cProfile.Profile()
    #pr.enable()
    test_focal()
    test_wrongwcs()
    test_single()
    test_parallel()
    test_pickle()
    test_olddes()
    test_newdes()
    test_des_wcs()
    test_gwcs()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(20)
