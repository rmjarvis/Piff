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
    image = galsim.Image(size,size, scale=1)  # Use odd
    image.setCenter(icen, jcen)
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


if __name__ == '__main__':
    test_init()

