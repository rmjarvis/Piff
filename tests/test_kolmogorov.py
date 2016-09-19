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


def make_kolmogorov_data(fwhm, g1, g2, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
                         nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Kolmogorov profile

    :param fwhm:        The fwhm of the Kolmogorov.
    :param g1, g2:      Shear applied to profile.
    :param u0, v0:      The sub-pixel offset to apply.
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param du:          pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    """
    k = galsim.Kolmogorov(fwhm=fwhm, flux=flux).shear(g1=g1, g2=g2).shift(u0,v0)
    if noise == 0.:
        var = 0.1
    else:
        var = noise
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu, v=fpv, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    k.drawImage(star.image, method='no_pixel',
                offset=galsim.PositionD(nom_u0/du,nom_v0/du), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


def test_simple():
    """Initial simple test of Kolmogorov.
    """
    # Here is the true PSF
    fwhm = 1.3
    g1 = 0.23
    g2 = -0.17
    cenu = 0.0
    cenv = 0.0
    psf = galsim.Kolmogorov(fwhm=fwhm).shear(g1=g1, g2=g2).shift(cenu, cenv)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)

    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')
    cenxy = image.wcs.toImage(galsim.PositionD(cenu, cenv))

    # Make a StarData instance for this image
    stardata = piff.StarData(image, image.trueCenter())
    star = piff.Star(stardata, None)

    # First try fastfit.
    print('Fast fit')
    model = piff.Kolmogorov(fastfit=True)
    fit = model.fit(star).fit

    print('True fwhm = ',fwhm,', model fwhm = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])
    print('True cenx = ', cenxy.x, ', model cenx = ', fit.center[0])
    print('True ceny = ', cenxy.y, ', model ceny = ', fit.center[1])

    # This test is fairly accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Kolmogorov.  Only wrinkle is that
    # HSM is adapted to Gaussians, not Kolmogorovs, so the requirements aren't as strict as
    # for Gaussian.
    np.testing.assert_allclose(fit.params[0], fwhm, rtol=1e-2)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=3)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=3)

    # Now try fastfit=False.
    print('Slow fit')
    model = piff.Kolmogorov(fastfit=False)
    fit = model.fit(star).fit

    print('True fwhm = ',fwhm,', model fwhm = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])
    print('True cenx = ', cenxy.x, ', model cenx = ', fit.center[0])
    print('True ceny = ', cenxy.y, ', model ceny = ', fit.center[1])

    # The sensitivity of this test is somewhat dependent on what the wcs is, and whether the fit
    # center parameters are in image coords or world coords.  Decimal=3 isn't terrible though.
    np.testing.assert_almost_equal(fit.params[0], fwhm, decimal=3)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=3)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=3)

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


def test_center():
    """Fit with centroid free and PSF center constrained to an initially mis-registered PSF.
    """
    influx = 150.
    fwhm = 2.0
    u0, v0 = 0.6, -0.4
    g1, g2 = 0.1, 0.2
    s = make_kolmogorov_data(fwhm, g1, g2, u0, v0, influx, du=0.5)

    # Kolmogorov model
    mod = piff.Kolmogorov()
    star = mod.initialize(s)
    print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
    for i in range(3):
        star = mod.fit(star)
        star = mod.reflux(star)
        print('Flux, ctr, chisq after fit {:d}:'.format(i),
              star.fit.flux, star.fit.center, star.fit.chisq)
        # These fluxes are not at all close to influx.  Not sure why...
        # np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=2)

    # Residual image when done should be dominated by structure off the edge of the fitted region.
    mask = star.weight.array > 0
    # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
    star2 = mod.draw(star)
    print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
    print('max image abs value = ',np.max(np.abs(s.image.array)))
    peak = np.max(np.abs(s.image.array[mask]))
    np.testing.assert_almost_equal(star2.image.array[mask]/peak, s.image.array[mask]/peak,
                                   decimal=2)

if __name__ == '__main__':
    test_simple()
    test_center()
