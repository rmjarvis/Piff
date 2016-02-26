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
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

from __future__ import print_function
import galsim
import numpy
import piff

def test_Gaussian():
    """This is about the simplest possible model I could think of.  It just uses the
    HSM adaptive moments routine to measure the moments, and then it models the
    PSF as a Gaussian.
    """

    # Here is the true PSF
    sigma = 1.3
    e1 = 0.23
    e2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(e1=e1, e2=e2)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)
    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')

    # Fit the model from the image
    model = piff.Gaussian()
    model.fitImage(image)

    print('True sigma = ',sigma,', model sigma = ',model.sigma)
    print('True e1 = ',e1,', model e1 = ',model.shape.e1)
    print('True e2 = ',e2,', model e2 = ',model.shape.e2)

    # This test is pretty accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Gaussian.
    numpy.testing.assert_almost_equal(sigma, model.sigma, decimal=7)
    numpy.testing.assert_almost_equal(e1, model.shape.e1, decimal=7)
    numpy.testing.assert_almost_equal(e2, model.shape.e2, decimal=7)

    # Now test running it via the config parser
    config = {
        'model' : {
            'type' : 'Gaussian'
        }
    }
    logger = piff.config.setup_logger()
    model = piff.process_model(config, logger)
    model.fitImage(image)

    # Same tests.
    numpy.testing.assert_almost_equal(sigma, model.sigma, decimal=7)
    numpy.testing.assert_almost_equal(e1, model.shape.e1, decimal=7)
    numpy.testing.assert_almost_equal(e2, model.shape.e2, decimal=7)


if __name__ == '__main__':
    test_Gaussian()

