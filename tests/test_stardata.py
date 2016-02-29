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
import numpy
import math
import piff

def test_init():
    """Test the basic initialization of a StarData object.
    """

    # Use an odd-sized image, so image.trueCenter() and image.center() are the same thing.
    # Otherwise, u,v below will be half-integer values.
    size = 63

    # Center the image at a non-trivial location to simulate this being a cutout from a larger
    # image.
    icen = 598
    jcen = 109

    # Use pixel scale = 1, so image_pos and focal pos are the same thing.
    image = galsim.Image(size,size, scale=1)
    field_pos = image.center()

    # Update the bounds so the image is centered at icen, jcen.
    # Note: this also updates the wcs, so u,v at the center is still field_pos
    image.setCenter(icen, jcen)  

    # Just draw something so it has non-trivial pixel values.
    galsim.Gaussian(sigma=5).drawImage(image)

    weight = galsim.ImageI(image.bounds, init_value=1)  # all weights = 1
    # To make below tests of weight pixel values useful, add the image to weight, so pixel
    # values are not all identical.

    image_pos = image.center()

    properties = {
        'ra' : 34.1234,
        'dec' : -15.567,
        'color_ri' : 0.5,
        'color_iz' : -0.2,
        'ccdnum' : 3
    }

    stardata = piff.StarData(image, image_pos, weight=weight, properties=properties)

    # Test attributes
    numpy.testing.assert_array_equal(stardata.image.array, image.array)
    numpy.testing.assert_array_equal(stardata.weight.array, weight.array)
    numpy.testing.assert_equal(stardata.image_pos, image_pos)

    # Test properties access viw properties attribute or directly with []
    for key, value in properties.items():
        numpy.testing.assert_equal(stardata.properties[key], value)
        numpy.testing.assert_equal(stardata[key], value)

    # Test the automatically generated properties
    print('image_pos = ',image_pos)
    print('image.wcs = ',image.wcs)
    print('props = ',stardata.properties)
    for key, value in [ ('x',image_pos.x), ('y',image_pos.y),
                        ('u',field_pos.x), ('v',field_pos.y) ]:
        numpy.testing.assert_equal(stardata.properties[key], value)
        numpy.testing.assert_equal(stardata[key], value)

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    numpy.testing.assert_array_equal(im.array, image.array)
    numpy.testing.assert_array_equal(wt.array, weight.array)
    numpy.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    # Note: This array() and then .T is like zip for Python lists.
    for data, wt, u, v in numpy.array(stardata.getDataVector()).T:
        # In this case, these should be integers, but round in case of numerical inaccuracy.
        iu = int(round(u))
        jv = int(round(v))
        # GalSim images access pixels as (x,y)
        numpy.testing.assert_equal(data, image(iu+icen,jv+jcen))
        numpy.testing.assert_equal(wt, weight(iu+icen,jv+jcen))
        # Numpy arrays access elements as [y,x]
        numpy.testing.assert_equal(data, image.array[jv+size//2, iu+size//2])
        numpy.testing.assert_equal(wt, weight.array[jv+size//2, iu+size//2])

    print("Passed basic initialization of StarData")


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
    print('origin of full image is at u,v = ',full_image.wcs.toWorld(full_image.origin()))
    print('center of full image is at u,v = ',full_image.wcs.toWorld(full_image.center()))

    # Make a postage stamp cutout
    size = 64   # This time, use an even size.
    image_pos = galsim.PositionD(1083.9, 617.3)
    field_pos = wcs.toWorld(image_pos)
    icen = int(image_pos.x)
    jcen = int(image_pos.y)

    bounds = galsim.BoundsI(icen-size//2+1, icen+size//2, jcen-size//2+1, jcen+size//2)
    image = full_image[bounds]
    weight = full_weight[bounds]

    print('image_pos = ',image_pos)
    print('field pos (u,v) = ',field_pos)
    print('origin of ps image is at u,v = ',image.wcs.toWorld(image.origin()))
    print('center of ps image is at u,v = ',image.wcs.toWorld(image.center()))
    
    # Just draw something so it has non-trivial pixel values.
    galsim.Gaussian(sigma=5).drawImage(image)
    weight += image

    stardata = piff.StarData(image, image_pos, weight=weight)

    # Test properties
    print('props = ',stardata.properties)
    numpy.testing.assert_equal(stardata['x'], image_pos.x)
    numpy.testing.assert_equal(stardata['y'], image_pos.y)
    numpy.testing.assert_equal(stardata['u'], field_pos.x)
    numpy.testing.assert_equal(stardata['v'], field_pos.y)
    # Shouldn't matter whether we use the original wcs or the one in the postage stamp.
    numpy.testing.assert_equal(stardata['u'], image.wcs.toWorld(image_pos).x)
    numpy.testing.assert_equal(stardata['v'], image.wcs.toWorld(image_pos).y)

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    numpy.testing.assert_array_equal(im.array, image.array)
    numpy.testing.assert_array_equal(wt.array, weight.array)
    numpy.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    for data, wt, u, v in numpy.array(stardata.getDataVector()).T:
        # u,v values should correspond to image coordinates via wcs
        uv = galsim.PositionD(u,v) + field_pos
        xy = wcs.toImage(uv)
        # These should now be integers, but round in case of numerical inaccuracy.
        ix = int(round(xy.x))
        jy = int(round(xy.y))
        numpy.testing.assert_equal(data, image(ix,jy))
        numpy.testing.assert_equal(wt, weight(ix,jy))

    print("Passed tests of StarData with EuclideanWCS")


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
    numpy.testing.assert_equal(stardata['x'], image_pos.x)
    numpy.testing.assert_equal(stardata['y'], image_pos.y)
    numpy.testing.assert_equal(stardata['u'], field_pos.x)
    numpy.testing.assert_equal(stardata['v'], field_pos.y)
    numpy.testing.assert_equal(stardata['ra'], sky_pos.ra)
    numpy.testing.assert_equal(stardata['dec'], sky_pos.dec)

    # Test access via getImage method:
    im, wt, pos = stardata.getImage()
    numpy.testing.assert_array_equal(im.array, image.array)
    numpy.testing.assert_array_equal(wt.array, weight.array)
    numpy.testing.assert_equal(pos, image_pos)

    # Test access via getDataVector method:
    for data, wt, u, v in numpy.array(stardata.getDataVector()).T:
        # u,v values should correspond to image coordinates via wcs
        uv = galsim.PositionD(u,v) + field_pos
        radec = pointing.deproject(uv)
        xy = wcs.toImage(radec)
        # These should now be integers, but round in case of numerical inaccuracy.
        ix = int(round(xy.x))
        jy = int(round(xy.y))
        numpy.testing.assert_equal(data, image(ix,jy))
        numpy.testing.assert_equal(wt, weight(ix,jy))

    print("Passed tests of StarData with CelestialWCS")


if __name__ == '__main__':
    test_init()
    test_euclidean()
    test_celestial()
