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

from piff_test_helper import get_script_name, timer

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

    nstars = 10  # per ccd
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
        },
        'output' : {
            'file_name': 'output/test_focal.piff'
        }
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    piff.piffify(config)

    psf = piff.read('output/test_focal.piff')

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

    nstars = 10  # per ccd
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
        'output' : {
            'dir' : 'output',
            'file_name': 'test_wrongwcs.piff'
        }
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    piff.piffify(config)

    psf = piff.read('output/test_wrongwcs.piff')

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

    nstars = 10  # per ccd
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
    im1.write('output/test_single_im1.fits')
    fitsio.write('output/test_single_cat1.fits', data1, clobber=True)

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
    im2.write('output/test_single_im2.fits')
    fitsio.write('output/test_single_cat2.fits', data2, clobber=True)

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
        'output' : {
            'file_name': 'output/test_single.piff'
        },
    }
    if __name__ != '__main__':
        config['verbose'] = 0
    piff.piffify(config)

    psf = piff.read('output/test_single.piff')

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



if __name__ == '__main__':
    test_focal()
    test_wrongwcs()
    test_single()
