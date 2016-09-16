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
import piff
import numpy as np


def test_Kolmogorov():
    """Initial simple test of Kolmogorov mirroring test_Gaussian.
    """

    # Here is the true PSF
    fwhm = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Kolmogorov(fwhm=fwhm).shear(g1=g1, g2=g2)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)
    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')

    # Make a StarData instance for this image
    stardata = piff.StarData(image, image.trueCenter())
    star = piff.Star(stardata, None)

    # First try fastfit.
    model = piff.Kolmogorov(fastfit=True)
    fit = model.fit(star).fit

    print('True fwhm = ',fwhm,', model fwhm = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])
    print('True cenx = ', 0.0, ', model cenx = ', fit.center[0])
    print('True ceny = ', 0.0, ', model ceny = ', fit.center[1])

    # This test is fairly accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Kolmogorov.  Only wrinkle is that
    # HSM is adapted to Gaussians, not Kolmogorovs, so the requirements aren't as strict as
    # for Gaussian.
    np.testing.assert_allclose(fit.params[0], fwhm, rtol=1e-2)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=3)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=3)

    # Now try fastfit=False and up the accuracy requirements.
    model = piff.Kolmogorov(fastfit=False)
    fit = model.fit(star).fit

    print('True fwhm = ',fwhm,', model fwhm = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])

    # This test is fairly accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Kolmogorov.  Only wrinkle is that
    # HSM is adapted to Gaussians, not Kolmogorovs, so the requirements aren't as strict as
    # for Gaussian.
    np.testing.assert_almost_equal(fit.params[0], fwhm, decimal=5)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=5)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=5)

    # Now test running it via the config parser
    config = {
        'model' : {
            'type' : 'Kolmogorov'
        }
    }
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    model = piff.Model.process(config['model'], logger)
    fit = model.fit(star).fit

    # Same tests.
    np.testing.assert_allclose(fit.params[0], fwhm, rtol=1e-2)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=3)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=3)


if __name__ == '__main__':
    test_Kolmogorov()
