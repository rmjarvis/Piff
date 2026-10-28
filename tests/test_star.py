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
import fitsio
import os
import piff
import copy
import pickle
import pytest

from piff_test_helper import timer, make_gaussian_data


@timer
def test_init():
    """Test the basic initialization of a StarData object.
    """

    # Use an odd-sized image, so image.true_center and image.center are the same thing.
    # Otherwise, u,v below will be half-integer values.
    size = 63

    # Center the image at a non-trivial location to simulate this being a cutout from a larger
    # image.
    icen = 598
    jcen = 109

    # Use pixel scale = 1, so image_pos and focal pos are the same thing.
    image = galsim.Image(size,size, scale=1)
    field_pos = image.center

    # Update the bounds so the image is centered at icen, jcen.
    # Note: this also updates the wcs, so u,v at the center is still field_pos
    image.setCenter(icen, jcen)

    # Just draw something so it has non-trivial pixel values.
    galsim.Gaussian(sigma=5).drawImage(image)

    weight = galsim.ImageF(image.bounds, init_value=1)  # all weights = 1
    # To make below tests of weight pixel values useful, add the image to weight, so pixel
    # values are not all identical.
    weight += 0.2 * image

    image_pos = image.center

    properties = {
        'ra' : 34.1234,
        'dec' : -15.567,
        'color_ri' : 0.5,
        'color_iz' : -0.2,
        'chipnum' : 3
    }

    stardata = piff.StarData(image, image_pos, weight=weight, properties=properties)

    # Test attributes
    np.testing.assert_array_equal(stardata.image.array, image.array)
    np.testing.assert_array_equal(stardata.weight.array, weight.array)
    np.testing.assert_equal(stardata.image_pos, image_pos)

    # Test properties access viw properties attribute or directly with []
    for key, value in properties.items():
        np.testing.assert_equal(stardata.properties[key], value)
        np.testing.assert_equal(stardata[key], value)

    # Test the automatically generated properties
    print('image_pos = ',image_pos)
    print('image.wcs = ',image.wcs)
    print('props = ',stardata.properties)
    for key, value in [ ('x',image_pos.x), ('y',image_pos.y),
                        ('u',field_pos.x), ('v',field_pos.y) ]:
        np.testing.assert_equal(stardata.properties[key], value)
        np.testing.assert_equal(stardata[key], value)

    # Can't have x,y,u,v in properties dict here.
    np.testing.assert_raises(TypeError, piff.StarData, image, image_pos, weight=weight,
                             properties=stardata.properties)

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    np.testing.assert_array_equal(im.array, image.array)
    np.testing.assert_array_equal(wt.array, weight.array)
    np.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    # Note: This array() and then .T is like zip for Python lists.
    for data, wt, u, v in np.array(stardata.getDataVector()).T:
        # In this case, these should be integers, but round in case of numerical inaccuracy.
        iu = int(round(u))
        jv = int(round(v))
        # GalSim images access pixels as (x,y)
        np.testing.assert_equal(data, image(iu+icen,jv+jcen))
        np.testing.assert_equal(wt, weight(iu+icen,jv+jcen))
        # Numpy arrays access elements as [y,x]
        np.testing.assert_equal(data, image.array[jv+size//2, iu+size//2])
        np.testing.assert_equal(wt, weight.array[jv+size//2, iu+size//2])


@timer
def test_euclidean():
    """Test a slightly more complicated WCS and an object not centered at the center of the image.
    """

    # Make a non-trivial WCS
    wcs = galsim.AffineTransform(0.26, -0.02, 0.03, 0.28,
                                 world_origin=galsim.PositionD(912.4, -833.1))
    print('wcs = ',wcs)

    # Start with a larger image from which we will cut out the postage stamp
    full_image = galsim.Image(2048,2048, wcs=wcs)
    full_weight = galsim.ImageS(2048,2048, wcs=wcs, init_value=1)
    print('origin of full image is at u,v = ',full_image.wcs.toWorld(full_image.origin))
    print('center of full image is at u,v = ',full_image.wcs.toWorld(full_image.center))

    # Make a postage stamp cutout
    size = 64   # This time, use an even size.
    image_pos = galsim.PositionD(1083.9, 617.3)
    field_pos = wcs.toWorld(image_pos)
    icen = int(image_pos.x+0.5)
    jcen = int(image_pos.y+0.5)

    bounds = galsim.BoundsI(icen-size//2-1, icen+size//2, jcen-size//2-1, jcen+size//2)
    print('bounds center = ',bounds.center)
    print('icen, jcen = ',icen,jcen)
    assert bounds.center.x == icen
    assert bounds.center.y == jcen

    image = full_image[bounds]
    weight = full_weight[bounds]
    print('image_pos = ',image_pos)
    print('field pos (u,v) = ',field_pos)
    print('origin of ps image is at u,v = ',image.wcs.toWorld(image.origin))
    print('center of ps image is at u,v = ',image.wcs.toWorld(image.center))

    # Just draw something so it has non-trivial pixel values.
    galsim.Gaussian(sigma=5).drawImage(image)
    weight += image

    stardata = piff.StarData(image, image_pos, weight=weight)

    # Test properties
    print('props = ',stardata.properties)
    np.testing.assert_equal(stardata['x'], image_pos.x)
    np.testing.assert_equal(stardata['y'], image_pos.y)
    np.testing.assert_equal(stardata['u'], field_pos.x)
    np.testing.assert_equal(stardata['v'], field_pos.y)
    # Shouldn't matter whether we use the original wcs or the one in the postage stamp.
    np.testing.assert_equal(stardata['u'], image.wcs.toWorld(image_pos).x)
    np.testing.assert_equal(stardata['v'], image.wcs.toWorld(image_pos).y)
    print('image.wcs = ',image.wcs)
    print('image_pos = ',image_pos)
    print('field_pos = ',field_pos)
    print('image.wcs.toWorld(image_pos) = ',image.wcs.toWorld(image_pos))

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    np.testing.assert_array_equal(im.array, image.array)
    np.testing.assert_array_equal(wt.array, weight.array)
    np.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    for data, wt, u, v in np.array(stardata.getDataVector()).T:
        # u,v values should correspond to image coordinates via wcs
        uv = galsim.PositionD(u,v) + field_pos
        xy = wcs.toImage(uv)
        # These should now be integers, but round in case of numerical inaccuracy.
        ix = int(round(xy.x))
        jy = int(round(xy.y))
        np.testing.assert_equal(data, image(ix,jy))
        np.testing.assert_equal(wt, weight(ix,jy))


@timer
def test_celestial():
    """Test using a (realistic) CelestialWCS for the main image.
    """

    # Make a CelestialWCS.  The simplest kind to make from scratch is a TanWCS.
    affine = galsim.AffineTransform(0.26, -0.02, 0.03, 0.28,
                                    world_origin=galsim.PositionD(912.4, -833.1))
    ra = 13.2343 * galsim.hours
    dec = -39.8484 * galsim.degrees
    pointing = galsim.CelestialCoord(ra,dec)
    wcs = galsim.TanWCS(affine, world_origin=pointing)
    print('wcs = ',wcs)

    # Start with a larger image from which we will cut out the postage stamp
    full_image = galsim.Image(2048,2048, wcs=wcs)
    full_weight = galsim.ImageS(2048,2048, wcs=wcs, init_value=1)

    # Make a postage stamp cutout
    # This next bit is the same as we did for the EuclideanWCS
    size = 64
    image_pos = galsim.PositionD(1083.9, 617.3)
    sky_pos = wcs.toWorld(image_pos)
    if galsim.__version__ >= '2.0':
        u,v = pointing.project(sky_pos)
        field_pos = galsim.PositionD(u/galsim.arcsec, v/galsim.arcsec)
    else:
        field_pos = pointing.project(sky_pos)
    icen = int(image_pos.x)
    jcen = int(image_pos.y)

    bounds = galsim.BoundsI(icen-size//2+1, icen+size//2, jcen-size//2+1, jcen+size//2)
    image = full_image[bounds]
    weight = full_weight[bounds]

    galsim.Gaussian(sigma=5).drawImage(image)
    weight += image

    # With a CelestialWCS, we need to supply a pointing
    stardata = piff.StarData(image, image_pos, weight=weight, pointing=pointing)

    # Test properties
    print('props = ',stardata.properties)
    np.testing.assert_equal(stardata['x'], image_pos.x)
    np.testing.assert_equal(stardata['y'], image_pos.y)
    np.testing.assert_equal(stardata['u'], field_pos.x)
    np.testing.assert_equal(stardata['v'], field_pos.y)
    np.testing.assert_equal(stardata['ra'], sky_pos.ra/galsim.hours)
    np.testing.assert_equal(stardata['dec'], sky_pos.dec/galsim.degrees)

    # Can use calculateFieldPos directly. (Although seems unlikely people would do so.)
    field_pos2 = stardata.calculateFieldPos(image_pos, image.wcs, pointing)
    assert field_pos2 == field_pos

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    np.testing.assert_array_equal(im.array, image.array)
    np.testing.assert_array_equal(wt.array, weight.array)
    np.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    for data, wt, u, v in np.array(stardata.getDataVector()).T:
        # u,v values should correspond to image coordinates via wcs
        uv = galsim.PositionD(u,v) + field_pos
        if galsim.__version__ >= '2.0':
            radec = pointing.deproject(uv.x * galsim.arcsec, uv.y * galsim.arcsec)
        else:
            radec = pointing.deproject(uv)
        xy = wcs.toImage(radec)
        # These should now be integers, but round in case of numerical inaccuracy.
        ix = int(round(xy.x))
        jy = int(round(xy.y))
        np.testing.assert_equal(data, image(ix,jy))
        np.testing.assert_equal(wt, weight(ix,jy))

    # Missing pointing
    np.testing.assert_raises(TypeError, piff.StarData, image, image_pos, weight=weight)


@timer
def test_add_poisson():
    """Test the addPoisson() method
    """
    size = 23
    icen = 598
    jcen = 109
    image = galsim.Image(size,size, scale=0.25)
    image.setCenter(icen, jcen)
    image_pos = image.center

    # Make a signal image
    galsim.Gaussian(sigma=1.1, flux=100).drawImage(image)
    stardata1 = piff.StarData(image, image_pos, weight=4)

    # Make another stardata with a blank image
    image2 = galsim.ImageF(image.bounds, init_value=0)
    weight = galsim.ImageF(image.bounds, init_value=0.025)
    stardata2 = piff.StarData(image, image_pos, weight=weight)

    # Add poisson noise with gain=1
    test = stardata2.addPoisson(stardata1, gain=1)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array)

    # Note repeated application always adds to the original weight map, not the current one.
    test = test.addPoisson(stardata1, gain=3)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/3)

    test = test.addPoisson(stardata1, gain=2)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/2)

    # gain=None means don't change anything
    test = stardata2.addPoisson(stardata1)
    np.testing.assert_allclose(test.weight.array, stardata2.weight.array)

    # signal=None means use self for signal
    test = stardata1.addPoisson(gain=2)
    np.testing.assert_allclose(1./test.weight.array, 1./stardata1.weight.array + image.array/2)

    # If gain is in properties, use that
    stardata3 = piff.StarData(image, image_pos, weight=weight, properties=dict(gain=4))
    test = stardata3.addPoisson(stardata1)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/4)

    # But explicit gain takes precedence
    test = stardata3.addPoisson(stardata1, gain=0.3)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/0.3)

    # After one application with gain, the star remembers that value.
    test = stardata2.addPoisson(stardata2, gain=2)
    test = test.addPoisson(stardata1)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/2)

    # Error if signal is different shape
    small_image = galsim.Image(15,15, scale=0.25)
    stardata4 = piff.StarData(small_image, image_pos)
    with np.testing.assert_raises(ValueError):
        stardata2.addPoisson(stardata4, gain=1)

    # Equivalent to access function from Star class
    star = piff.Star(stardata2, None)
    test = star.addPoisson(stardata1, gain=3)
    np.testing.assert_allclose(1./test.weight.array, 1./weight.array + image.array/3)


@timer
def test_star():
    # Star class is a pretty thin wrapper of StarData and StarFit classes.

    # Same setup as for test_euclidean
    wcs = galsim.AffineTransform(0.26, -0.02, 0.03, 0.28,
                                 world_origin=galsim.PositionD(912.4, -833.1))
    size = 64
    image_pos = galsim.PositionD(1083.9, 617.3)
    field_pos = wcs.toWorld(image_pos)
    icen = int(image_pos.x+0.5)
    jcen = int(image_pos.y+0.5)
    bounds = galsim.BoundsI(icen-size//2-1, icen+size//2, jcen-size//2-1, jcen+size//2)
    image = galsim.Image(bounds, wcs=wcs)
    weight = galsim.ImageS(bounds, wcs=wcs, init_value=1)
    galsim.Gaussian(sigma=5).drawImage(image)
    weight += image

    properties = {
        'ra' : 34.1234,
        'dec' : -15.567,
        'color_ri' : 0.5,
        'color_iz' : -0.2,
        'chipnum' : 3
    }

    stardata = piff.StarData(image, image_pos, weight=weight, properties=properties)

    # Check access via star class
    star = piff.Star(stardata, None)  # fit is optional, but must be explicitly None
    for key, value in stardata.properties.items():
        np.testing.assert_equal(star[key], value)
    np.testing.assert_equal(star.image.array, image.array)
    np.testing.assert_equal(star.weight.array, weight.array)
    assert star.image_pos == image_pos
    print('field_pos = ',field_pos)
    print('star.field_pos = ',star.field_pos)
    assert star.field_pos == field_pos
    assert star.x == image_pos.x
    assert star.y == image_pos.y
    assert star.u == field_pos.x
    assert star.v == field_pos.y
    assert star.chipnum == 3
    assert star.flux == 1.
    assert star.center == (0,0)
    assert star.is_reserve == False
    assert star['color_ri'] == 0.5
    assert star['color_iz'] == -0.2
    with np.testing.assert_raises(KeyError):
        star['color_gi']

    star = star.withFlux(7)
    assert star.flux == 7.
    assert star.center == (0,0)

    star = star.withFlux(17, center=(102,22))
    assert star.flux == 17.
    assert star.center == (102,22)

    star = star.withFlux(center=(12,20))
    assert star.flux == 17.
    assert star.center == (12,20)

    star = star.withFlux(flux=2)
    assert star.flux == 2.
    assert star.center == (12,20)

    # Test withProperties
    star = star.withProperties(color_iz=-0.3, color_gi=1.1)
    assert star['color_ri'] == 0.5
    assert star['color_iz'] == -0.3
    assert star['color_gi'] == 1.1

    # Test using makeTarget
    star = piff.Star.makeTarget(properties=stardata.properties, image=image, weight=weight)
    np.testing.assert_equal(star.data['x'], image_pos.x)
    np.testing.assert_equal(star.data['y'], image_pos.y)
    np.testing.assert_equal(star.data['u'], field_pos.x)
    np.testing.assert_equal(star.data['v'], field_pos.y)
    # Shouldn't matter whether we use the original wcs or the one in the postage stamp.
    print('image.wcs = ',image.wcs)
    print('image_pos = ',image_pos)
    print('field_pos = ',field_pos)
    print('image.wcs.toWorld(image_pos) = ',image.wcs.toWorld(image_pos))
    print('in star.data: ',star.data['u'])
    print('in star.data: ',star.data['v'])
    np.testing.assert_equal(star.data['u'], image.wcs.toWorld(image_pos).x)
    np.testing.assert_equal(star.data['v'], image.wcs.toWorld(image_pos).y)
    im, wt, pos = star.data.getImage()
    np.testing.assert_array_equal(im.array, image.array)
    np.testing.assert_array_equal(wt.array, weight.array)
    np.testing.assert_equal(pos, image_pos)

    # Some invalid parameter checks
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, x=1, properties=stardata.properties,
                             image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, y=1, properties=stardata.properties,
                             image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, u=1, properties=stardata.properties,
                             image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, v=1, properties=stardata.properties,
                             image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, x=1, image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, y=1, image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, u=1, image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, image=image, weight=weight)
    np.testing.assert_raises(TypeError, piff.Star.makeTarget, x=1, y=2, scale=4, wcs=image.wcs,
                             image=image, weight=weight)


@timer
def test_io():
    np_rng = np.random.RandomState(1234)
    nstars = 100
    x = np_rng.random_sample(nstars) * 2048.
    y = np_rng.random_sample(nstars) * 2048.
    flux = np_rng.random_sample(nstars) * 1000.
    cenx = 2.*np_rng.random_sample(nstars) - 1.
    ceny = 2.*np_rng.random_sample(nstars) - 1.
    color_ri = np_rng.random_sample(nstars) * 1.4 - 0.8
    color_iz = np_rng.random_sample(nstars) * 1.9 - 0.6
    stars = [ piff.Star.makeTarget(x=x[i], y=y[i], scale=0.26, color_ri=color_ri[i],
                                   color_iz=color_iz[i]).withFlux(flux[i]) for i in range(nstars) ]
    for star in stars:
        star.data.image.array[:] = np_rng.random_sample(star.data.image.array.shape)
        star.data.weight = star.data.image.copy()
        star.data.weight.array[:] = np_rng.random_sample(star.data.image.array.shape)

    for do_params in [False, True]:
        # test both case we have params and case we do not
        if do_params:
            params = np_rng.random_sample((nstars, 3))
            params_var = np_rng.random_sample((nstars, 3))
            new_fluxes = np_rng.random_sample(nstars) * 10000
            for star, p, pv, f in zip(stars, params, params_var, new_fluxes):
                old_flux = star.fit.flux
                old_center = star.fit.center
                # first do test without specifying new flux
                star.fit = star.fit.newParams(p, params_var=pv)
                assert star.fit.flux != f
                assert star.fit.flux == old_flux
                assert star.fit.center == old_center
                # now specify new flux
                star.fit = star.fit.newParams(p, params_var=pv, flux=f)
                assert star.fit.flux == f
                assert star.fit.flux != old_flux
                assert star.fit.center == old_center
                # Invalid for new params to be different length than old.
                np.testing.assert_raises(ValueError, star.fit.newParams, p[:2])
                np.testing.assert_raises(ValueError, star.fit.newParams, np.arange(4))

        file_name = os.path.join('output','star_io.fits')
        print('Writing stars to ',file_name)
        with piff.writers.FitsWriter.open(file_name) as w:
            piff.Star.write(stars, w, 'stars')

        print('Reading from ',file_name)
        with piff.readers.FitsReader.open(file_name) as r:
            stars2 = piff.Star.read(r, 'stars')

        for s1, s2 in zip(stars,stars2):
            assert s1.data['x'] == s2.data['x']
            assert s1.data['y'] == s2.data['y']
            assert s1.data['u'] == s2.data['u']
            assert s1.data['v'] == s2.data['v']
            assert s1.data['color_ri'] == s2.data['color_ri']
            assert s1.data['color_iz'] == s2.data['color_iz']
            assert s1.data.properties == s2.data.properties
            assert s1.fit.flux == s2.fit.flux
            assert all(s1.fit.center == s2.fit.center)
            np.testing.assert_array_equal(s1.fit.params, s2.fit.params)
            np.testing.assert_array_equal(s1.fit.params_var, s2.fit.params_var)
            np.testing.assert_array_equal(s1.fit.get_params(None), s1.fit.params)
            np.testing.assert_array_equal(s1.fit.get_params_var(None), s1.fit.params_var)
            assert s1.data.image.bounds == s2.data.image.bounds
            assert s1.data.weight.bounds == s2.data.weight.bounds
            # The wcs doesn't have to match, but they should be locally equivalent.
            assert s1.data.image.wcs.jacobian() == s2.data.image.wcs.jacobian()
            assert s1.data.weight.wcs.jacobian() == s2.data.weight.wcs.jacobian()
            # The image and weight arrays are not serialized.
            #np.testing.assert_almost_equal(s1.data.image.array,s2.data.image.array)
            #np.testing.assert_almost_equal(s1.data.weight.array,s2.data.weight.array)

        # test read_coords_params
        if do_params:
            with fitsio.FITS(file_name, 'r') as fin:
                s2coords, s2params = piff.Star.read_coords_params(fin, 'stars')
            s2coords_manual = []
            s2params_manual = []
            for s in stars2:
                s2coords_manual.append(np.array([s.data[key] for key in ['u', 'v']]))
                s2params_manual.append(s.fit.params)
            s2coords_manual = np.array(s2coords_manual)
            s2params_manual = np.array(s2params_manual)
            np.testing.assert_array_equal(s2coords, s2coords_manual)
            np.testing.assert_array_equal(s2params, s2params_manual)
        else:
            # reading read_coords_params should fail if there is no params
            with np.testing.assert_raises(IOError):
                with fitsio.FITS(file_name, 'r') as fin:
                    s2coords, s2params = piff.Star.read_coords_params(fin, 'stars')

        with np.testing.assert_raises(IOError):
            with fitsio.FITS(file_name, 'r') as fin:
                s2coords, s2params = piff.Star.read_coords_params(fin, 'invalid')

@timer
def test_multifit_io():
    np_rng = np.random.RandomState(1234)
    nstars = 100
    nstars = 2
    x = np_rng.random_sample(nstars) * 2048.
    y = np_rng.random_sample(nstars) * 2048.
    flux = np_rng.random_sample(nstars) * 1000.
    color_ri = np_rng.random_sample(nstars) * 1.4 - 0.8
    color_iz = np_rng.random_sample(nstars) * 1.9 - 0.6

    stars = [ piff.Star.makeTarget(x=x[i], y=y[i], scale=0.26, color_ri=color_ri[i],
                                   color_iz=color_iz[i], flux=flux[i]) for i in range(nstars) ]
    for star in stars:
        star.data.image.array[:] = np_rng.random_sample(star.data.image.array.shape)
        star.data.weight = star.data.image.copy()
        star.data.weight.array[:] = np_rng.random_sample(star.data.image.array.shape)

    params1 = np_rng.random_sample((nstars, 3))
    params_var1 = np_rng.random_sample((nstars, 3))
    params2 = np_rng.random_sample((nstars, 5))
    params_var2 = np_rng.random_sample((nstars, 5))
    for k, star in enumerate(stars):
        params = [params1[k], params2[k]]
        params_var = [params_var1[k], params_var2[k]]
        star.fit = star.fit.withNew(params=params, params_var=params_var)

    file_name = os.path.join('output','star_multifit_io.fits')
    print('Writing stars to ',file_name)
    with piff.writers.FitsWriter.open(file_name) as w:
        piff.Star.write(stars, w, 'stars')

    print('Reading from ',file_name)
    with piff.readers.FitsReader.open(file_name) as r:
        stars2 = piff.Star.read(r, 'stars')

    for s1, s2 in zip(stars,stars2):
        assert s1.data['x'] == s2.data['x']
        assert s1.data['y'] == s2.data['y']
        assert s1.data['u'] == s2.data['u']
        assert s1.data['v'] == s2.data['v']
        assert s1.data['color_ri'] == s2.data['color_ri']
        assert s1.data['color_iz'] == s2.data['color_iz']
        assert s1.data.properties == s2.data.properties
        assert s1.fit.flux == s2.fit.flux
        assert all(s1.fit.center == s2.fit.center)
        assert len(s1.fit.params) == len(s2.fit.params)
        assert len(s1.fit.params_var) == len(s2.fit.params_var)
        for k in range(len(s1.fit.params)):
            np.testing.assert_array_equal(s1.fit.params[k], s2.fit.params[k])
            np.testing.assert_array_equal(s1.fit.params_var[k], s2.fit.params_var[k])
            np.testing.assert_array_equal(s1.fit.get_params(k), s1.fit.params[k])
            np.testing.assert_array_equal(s1.fit.get_params_var(k), s1.fit.params_var[k])

    # Check setting individual param nums with newParams
    params1 = np_rng.random_sample((nstars, 3))
    params_var1 = np_rng.random_sample((nstars, 3))
    params2 = np_rng.random_sample((nstars, 5))
    params_var2 = np_rng.random_sample((nstars, 5))
    for k, star in enumerate(stars):
        star.fit = star.fit.newParams(params1[k], params_var=params_var1[k], num=0)
        star.fit = star.fit.newParams(params2[k], params_var=params_var2[k], num=1)
        np.testing.assert_array_equal(star.fit.flatten_params(star.fit.params),
                                      np.concatenate([params1[k], params2[k]]))
        np.testing.assert_array_equal(star.fit.flatten_params(star.fit.params_var),
                                      np.concatenate([params_var1[k], params_var2[k]]))

        # Also check without params_var (leaves params_var as it was)
        star.fit = star.fit.newParams(params1[k//2], num=0)
        star.fit = star.fit.newParams(params2[k//2], num=1)
        np.testing.assert_array_equal(star.fit.flatten_params(star.fit.params),
                                      np.concatenate([params1[k//2], params2[k//2]]))
        np.testing.assert_array_equal(star.fit.flatten_params(star.fit.params_var),
                                      np.concatenate([params_var1[k], params_var2[k]]))

    # If a composite fit has a missing params_var entry for one component, newParams should
    # fill just that slot with zeros when updating the component params.
    fit = piff.StarFit(
        params=[np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
        params_var=[np.array([0.1, 0.2]), None],
    )
    fit = fit.newParams(np.array([6.0, 7.0, 8.0]), num=1)
    np.testing.assert_array_equal(fit.params[1], [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(fit.params_var[0], [0.1, 0.2])
    np.testing.assert_array_equal(fit.params_var[1], [0.0, 0.0, 0.0])


@timer
def test_shape_method():
    """Test changing the backend shape measurement method.
    """
    # Switching to the optional backends requires ngmix.
    pytest.importorskip('ngmix')

    # The shape measurement is hsm.
    assert piff.get_shape_method() == 'hsm'
    assert piff.get_shape_weight_fwhm() is None

    # star.measure_size_shape() is equivalent to star.hsm.
    # It caches the value on repeated calls to either one.
    star = make_gaussian_data(0.4, 0.01, -0.02, 1000, du=0.2, g1=0.1, g2=-0.05,
                              draw_method='no_pixel')
    hsm = star.measure_size_shape()
    assert hsm is star.hsm
    assert hsm is star.measure_size_shape()
    assert star.hsm[-1] == 0

    # After changing the shape measurement method, the generic function is equivalent
    # to the specific one.
    try:
        piff.set_shape_method('gauss')
        gauss = star.measure_size_shape()
        assert gauss is star.gauss
        assert gauss is not hsm
        assert gauss is star.measure_size_shape()
        assert star.gauss[-1] == 0

        # am use its own cache, independent of hsm and gauss.
        piff.set_shape_method('am')
        am = star.measure_size_shape()
        assert am is star.am
        assert am is not hsm
        assert am is not gauss
        assert am is star.measure_size_shape()
        assert am[-1] == 0

        # wmom caches each fwhm independently.
        piff.set_shape_method('wmom', weight_fwhm=0.3)
        wmom = star.measure_size_shape()
        assert wmom is star.wmom(0.3)
        assert wmom is not hsm
        assert wmom is not gauss
        assert star.measure_size_shape() is not star.wmom(0.4)
        assert star.measure_size_shape() is star.measure_size_shape()
        assert star.wmom(0.3)[-1] == 0
        # Both values are in the cache.
        assert 0.3 in star._wmom
        assert 0.4 in star._wmom

    finally:
        # Restore the default for other tests.
        piff.set_shape_method('hsm')

    # The various shape measurements are available regardless of the current backend setting.
    assert star.hsm is hsm
    assert star.gauss is gauss
    assert star.am is am
    assert star.wmom(0.3) is wmom

    # The cache survives changing to other methods and back.
    assert star.measure_size_shape() is hsm

    # An invalid method doesn't change anything.
    with np.testing.assert_raises(ValueError):
        piff.set_shape_method('unknown')
    assert piff.get_shape_method() == 'hsm'

@timer
def test_gauss():
    """Test the gauss shape measurement method.
    """
    pytest.importorskip('ngmix')
    piff.set_shape_method('gauss')
    try:
        # Check a few wcs geometries and profile sizes.
        for wcs in [galsim.PixelScale(0.11),
                    galsim.JacobianWCS(0.11, 0.013, -0.009, 0.105),
                    galsim.JacobianWCS(-0.11, 0.013, -0.009, 0.105)]:
            for sigma in [0.07, 0.3]:
                star = make_gaussian_data(sigma, 0.015, -0.012, 1000, wcs=wcs,
                                          g1=0.12, g2=-0.08, draw_method='no_pixel',
                                          nom_u0=0.2, nom_v0=0.4)
                result = star.measure_size_shape()
                assert result[-1] == 0
                # Measurements are in sky coords with arcsec units.
                # Should be basically exact here with no noise and no pixel response.
                np.testing.assert_allclose(result[:6],
                                           [1000, 0.015, -0.012, sigma, 0.12, -0.08],
                                           rtol=1.e-4, atol=1.e-5)

                # The cache survives a round trip through serialization. (e.g. pickle)
                # However, it won't be the same object anymore.
                restored = pickle.loads(pickle.dumps(star))
                assert restored._gauss is not None
                np.testing.assert_array_equal(restored._gauss, result)
                assert restored.measure_size_shape() is restored._gauss

        # The ngmix fitter isn't biased by zero-weight or non-finite pixels.
        # It just ignores these pixels.
        star = make_gaussian_data(0.3, 0.015, -0.012, 1000, du=0.11,
                                  g1=0.12, g2=-0.08, draw_method='no_pixel')
        star.image.array[10, 12] = np.nan
        star.data.weight.array[11, 12] = 0
        star.image.array[11, 12] = 1.e9
        star.data.weight.array[12, 12] = np.inf
        result = star.measure_size_shape()
        assert result[-1] == 0
        np.testing.assert_allclose(result[:6], [1000, 0.015, -0.012, 0.3, 0.12, -0.08],
                                   rtol=1.e-4, atol=1.e-5)

        # Fully masked and blank images report failure instead of successful measurements.
        failed = make_gaussian_data(0.3, 0, 0, 1000, du=0.11, draw_method='no_pixel')
        failed.data.weight.array[:] = 0
        assert failed.measure_size_shape()[-1] != 0
        blank = make_gaussian_data(0.3, 0, 0, 0, du=0.11, draw_method='no_pixel')
        assert blank.measure_size_shape()[-1] != 0

        # A broad, well-resolved profile measures a large centroid but flags it due to the
        # large centroid shift.
        distant = make_gaussian_data(0.5, 1.1, 0, 1000, du=0.11, nside=48,
                                     draw_method='no_pixel')
        result = distant.measure_size_shape()
        assert result[-1] & 4
        np.testing.assert_allclose(result[:4], [1000, 1.1, 0, 0.5], rtol=1.e-4, atol=1.e-5)

        # Negative flux has a different failure code.
        negative = make_gaussian_data(0.3, 0, 0, -1000, du=0.11, draw_method='no_pixel')
        assert negative.measure_size_shape()[-1] & 2

    finally:
        # Restore the default for other tests.
        piff.set_shape_method('hsm')


@timer
def test_am():
    """Check ngmix adaptive moments in world coordinates, caching, and failure flags."""
    pytest.importorskip('ngmix')
    piff.set_shape_method('am')
    try:
        # Resolved Gaussians recover flux, centroid, determinant radius, and reduced shear
        # in world coordinates, including for a sheared or reflected WCS.
        for wcs in [galsim.PixelScale(0.11),
                    galsim.JacobianWCS(0.11, 0.013, -0.009, 0.105),
                    galsim.JacobianWCS(-0.11, 0.013, -0.009, 0.105)]:
            for sigma in [0.2, 0.3]:
                star = make_gaussian_data(sigma, 0.015, -0.012, 1000, wcs=wcs,
                                          g1=0.12, g2=-0.08, draw_method='no_pixel',
                                          nom_u0=0.2, nom_v0=0.4)
                result = star.measure_size_shape()
                assert result[-1] == 0

                # These don't come out quite as accurate as gauss was able to do.
                np.testing.assert_allclose(result[0], 1000, rtol=1.e-4)
                np.testing.assert_allclose(result[1:3], [0.015, -0.012], atol=1.e-3)
                np.testing.assert_allclose(result[3:6], [sigma, 0.12, -0.08],
                                           rtol=1.e-4, atol=1.e-5)

                # The cache survives a round trip through serialization. (e.g. pickle)
                # However, it won't be the same object anymore.
                np.testing.assert_array_equal(star.measure_am(), result)
                restored = pickle.loads(pickle.dumps(star))
                np.testing.assert_array_equal(restored.am, result)
                assert restored.measure_size_shape() is restored.am

        # For a non-Gaussian profile, ngmix and HSM agree on the adaptive flux, size, and shear.
        profile = galsim.Moffat(beta=3, fwhm=0.7, flux=1000).shear(g1=0.12, g2=-0.08)
        image = profile.drawImage(nx=48, ny=48, scale=0.11, method='no_pixel')
        star = piff.Star(piff.StarData(image, image.true_center), None)
        assert star.am[-1] == star.hsm[-1] == 0
        np.testing.assert_allclose(star.am[:6], star.hsm[:6], rtol=1.e-3, atol=1.e-5)

        # Invalid data and masked outliers in the stamp corners do not affect the measurement.
        star = make_gaussian_data(0.3, 0.015, -0.012, 1000, du=0.11,
                                  g1=0.12, g2=-0.08, draw_method='no_pixel')
        expected = star.measure_am()
        star.image.array[0, 0] = np.nan
        star.weight.array[1, 0] = 0
        star.image.array[1, 0] = 1.e9
        star.weight.array[0, 1] = np.inf
        star.image.array[0, 1] = 1.e9
        result = star.measure_size_shape()
        assert result[-1] == 0
        np.testing.assert_allclose(result, expected, rtol=1.e-10, atol=1.e-10)

        # Fully masked, blank, negative-flux, and collapsed undersampled profiles report failure.
        masked = make_gaussian_data(0.3, 0, 0, 1000, du=0.11, draw_method='no_pixel')
        masked.weight.array[:] = 0
        assert masked.measure_size_shape()[-1] == 1
        for sigma, flux in [(0.3, 0), (0.3, -1000), (0.07, 1000)]:
            failed = make_gaussian_data(sigma, 0.015, -0.012, flux, du=0.11,
                                        g1=0.12, g2=-0.08, draw_method='no_pixel')
            assert failed.measure_size_shape()[-1] == 1

        # Successful measurements flag centroids displaced by more than one arcsecond.
        distant = make_gaussian_data(1.5, 1.1, 0, 1000, du=0.5, nside=48,
                                     draw_method='no_pixel')
        assert distant.measure_size_shape()[-1] == 4
    finally:
        # Restore the default for other tests.
        piff.set_shape_method('hsm')


@timer
def test_wmom():
    """Test the wmom shape measurement method.
    """
    pytest.importorskip('ngmix')

    # Invalid widths are rejected immediately without changing the global state.
    for invalid in [None, 0, -1, np.nan, np.inf]:
        with np.testing.assert_raises(ValueError) as error:
            piff.set_shape_method('wmom', weight_fwhm=invalid)
        assert 'shape_weight_fwhm' in str(error.exception)
        assert piff.get_shape_method() == 'hsm'

    # A reflected, sheared WCS and masked NaN exercise coordinates and pixel masking together.
    wcs = galsim.JacobianWCS(-0.06, 0.007, -0.005, 0.058)
    try:
        # Check a few wcs geometries and profile sizes.
        for wcs in [galsim.JacobianWCS(0.11, 0, 0, 0.11),
                    galsim.JacobianWCS(0.11, 0.013, -0.009, 0.105),
                    galsim.JacobianWCS(-0.11, 0.013, -0.009, 0.105)]:
            for sigma in [0.07, 0.3]:
                star = make_gaussian_data(sigma, 0.015, -0.012, 1000, wcs=wcs,
                                          g1=0.12, g2=-0.08, draw_method='no_pixel',
                                          nom_u0=0.2, nom_v0=0.4)
                # Put in a single nan pixel with weight=0 to make sure it handles it ok.
                # It's a direct moment, so missing pixels do bias the answer, but we'll check
                # the sum directly to see if it's right.
                star.data.weight.array[10, 12] = 0
                star.image.array[10, 12] = np.nan
                for fwhm in [0.3, 0.6]:
                    piff.set_shape_method('wmom', weight_fwhm=fwhm)

                    # Compute the appropriate sums directly. (Yeah, this is really more of a
                    # unit test for ngmix, but that's ok.)
                    result = star.measure_size_shape()
                    y, x = np.indices(star.image.array.shape)
                    x = x + star.image.xmin - star.x
                    y = y + star.image.ymin - star.y
                    u = wcs.dudx*x + wcs.dudy*y
                    v = wcs.dvdx*x + wcs.dvdy*y
                    wsigma = fwhm / np.sqrt(8*np.log(2))
                    window = np.exp(-(u*u+v*v)/(2*wsigma**2))
                    data = np.where(star.data.weight.array > 0, star.image.array, 0)
                    weighted = window*data
                    flux = weighted.sum()
                    mean = np.array([(weighted*u).sum(), (weighted*v).sum()])/flux
                    moment_matrix = np.array([
                        [(weighted*u*u).sum(), (weighted*u*v).sum()],
                        [(weighted*u*v).sum(), (weighted*v*v).sum()],
                    ])/flux
                    trace = np.trace(moment_matrix)
                    shape = galsim.Shear(e1=(moment_matrix[0, 0]-moment_matrix[1, 1])/trace,
                                         e2=2*moment_matrix[0, 1]/trace)
                    expected = [flux, *mean, np.linalg.det(moment_matrix)**0.25, shape.g1, shape.g2]
                    assert result[-1] == 0
                    np.testing.assert_allclose(result[:6], expected, rtol=1.e-4, atol=1.e-6)

                # The cache survives a round trip through serialization. (e.g. pickle)
                # However, it won't be the same object anymore.
                restored = pickle.loads(pickle.dumps(star))
                assert restored._wmom is not None
                assert 0.6 in restored._wmom
                np.testing.assert_array_equal(restored._wmom[0.6], result)
                assert restored.measure_size_shape() is restored._wmom[0.6]

        # Fully masked and blank images report failure instead of successful measurements.
        failed = make_gaussian_data(0.2, 0, 0, 1000, wcs=wcs, draw_method='no_pixel')
        failed.data.weight.array[:] = 0
        assert failed.measure_size_shape()[-1] != 0
        blank = make_gaussian_data(0.2, 0, 0, 0, wcs=wcs, draw_method='no_pixel')
        assert blank.measure_size_shape()[-1] != 0

        # Signed residuals can produce an indefinite moment matrix with positive flux and trace.
        # This image's moment matrix (ignoring the WCS) is proportional to:
        #     M ~ (20   0)
        #         ( 0  -2)
        # This results in a positive flux and positive trace, but an invalid ellipticity.
        #     |e| = 11/9
        # Ngmix lets this result pass through, but Piff flags it for having an invalid shape.
        # Note: Use a rotated WCS so neither ellipticity component is zero; otherwise ngmix
        # flags it before Piff can chck the ellipticity.
        wcs = galsim.JacobianWCS(0.06, 0.03, -0.03, 0.06)
        image = galsim.ImageD(31, 31, wcs=wcs)
        image.setCenter(0, 0)
        image[0, 0] = 100
        image[1, 0] = image[-1, 0] = 10
        image[0, 1] = image[0, -1] = -1
        weight = galsim.ImageD(image.bounds, init_value=1, wcs=wcs)
        unphysical = piff.Star(piff.StarData(image, galsim.PositionD(0, 0),
                                             weight=weight), None)
        assert unphysical.measure_size_shape()[-1] != 0

        # A sufficiently wide fwhm for the weight measures a large centroid but flags it due to
        # the large centroid shift.
        piff.set_shape_method('wmom', weight_fwhm=3)
        distant = make_gaussian_data(0.2, 1.1, 0, 1000, du=0.3, nside=48,
                                     draw_method='no_pixel')
        result = distant.measure_size_shape()
        assert result[1] > 1.  # measured du more than 1 arcsec.
        assert result[-1] & 4

        # Negative flux is checked by ngmix in this case, so the flag is just 1.
        negative = make_gaussian_data(0.3, 0, 0, -1000, du=0.11, draw_method='no_pixel')
        assert negative.measure_size_shape()[-1] & 1

    finally:
        # Restore the default for other tests.
        piff.set_shape_method('hsm')

# Helper function for test_shape_parallel
def _measure_shape_in_worker(star, logger):
    from threadpoolctl import threadpool_info
    measurement = star.measure_size_shape()
    threads = [pool['num_threads'] for pool in threadpool_info()]
    return (piff.get_shape_method(), piff.get_shape_weight_fwhm(), measurement,
            galsim.utilities.get_omp_threads(), threads)

@timer
def test_shape_parallel():
    """Check worker shape settings and thread limits without changing parent thread limits.
    """
    pytest.importorskip('ngmix')
    from threadpoolctl import threadpool_info
    try:
        for method in ['gauss', 'am', 'wmom']:
            star = make_gaussian_data(0.3, 0.015, -0.012, 1000, du=0.11,
                                      g1=0.12, g2=-0.08, draw_method='no_pixel')
            fwhm = 0.3 if method == 'wmom' else None
            piff.set_shape_method(method, weight_fwhm=fwhm)

            # Make sure to keep star that is passed to the worker unmeasured until the worker
            # gets it so the cache can't hide an error in the method being passed correctly.
            star_copy = copy.deepcopy(star)

            # Workers use one thread without changing the parent's thread limits.
            logger = piff.config.setup_logger(0)
            logger = piff.config.LoggerWrapper(logger)
            parent_omp_threads = galsim.utilities.get_omp_threads()
            parent_threads = {pool['filepath']: pool['num_threads']
                              for pool in threadpool_info()}
            results = piff.util.run_multi(
                _measure_shape_in_worker, nproc=2, raise_except=True,
                args=[(star,), (copy.deepcopy(star),)], logger=logger)
            assert galsim.utilities.get_omp_threads() == parent_omp_threads
            assert {pool['filepath']: pool['num_threads']
                    for pool in threadpool_info()} == parent_threads

            # Each backend agrees with the parent measurement and respects worker thread limits.
            expected = star_copy.measure_size_shape()
            for worker_method, worker_fwhm, measurement, omp_threads, threads in results:
                assert worker_method == method
                assert worker_fwhm == fwhm
                assert measurement[-1] == 0
                np.testing.assert_allclose(measurement, expected, rtol=1.e-4, atol=1.e-5)
                assert omp_threads == 1
                assert all(count == 1 for count in threads)
    finally:
        # Restore the default for other tests.
        piff.set_shape_method('hsm')


if __name__ == '__main__':
    test_init()
    test_euclidean()
    test_celestial()
    test_add_poisson()
    test_star()
    test_io()
    test_multifit_io()
    test_shape_method()
    test_gauss()
    test_am()
    test_wmom()
    test_shape_parallel()
