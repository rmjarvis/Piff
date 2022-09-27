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

    # Chipnum is required as a property to use SingleCCDPSF
    star1 = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center)
    with np.testing.assert_raises(ValueError):
        psf.drawStar(star1)
    star2 = piff.Star(star1.data, star.fit)  # If has a fit, it hits a different error
    with np.testing.assert_raises(ValueError):
        psf.drawStar(star2)


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

    # Check that errors in the solution get properly reported.
    config['input']['nimages'] = 3
    with CaptureLog(level=2) as cl:
        psf = piff.process(config, cl.logger)
    assert "Removed 6 stars in initialize" in cl.output
    assert "No stars.  Cannot find PSF model." in cl.output
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
    assert "No stars.  Cannot find PSF model." in cl.output
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
    np.testing.assert_allclose(image.array[23:26,22:25], regression_array, rtol=1.e-5)

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

if __name__ == '__main__':
    #import cProfile, pstats
    #pr = cProfile.Profile()
    #pr.enable()
    test_focal()
    test_wrongwcs()
    test_single()
    test_pickle()
    test_olddes()
    test_newdes()
    test_des_wcs()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(20)
