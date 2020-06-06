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

from piff_test_helper import timer


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
        with fitsio.FITS(file_name,'rw',clobber=True) as fout:
            piff.Star.write(stars, fout, extname='stars')

        print('Reading from ',file_name)
        with fitsio.FITS(file_name,'r') as fin:
            stars2 = piff.Star.read(fin, extname='stars')

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


if __name__ == '__main__':
    test_init()
    test_euclidean()
    test_celestial()
    test_add_poisson()
    test_star()
    test_io()
