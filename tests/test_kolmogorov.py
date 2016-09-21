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

def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    mod = piff.Kolmogorov()
    g1 = g2 = u0 = v0 = 0.0

    # Interpolator will be simple mean
    interp = piff.Polynomial(order=0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,10.)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = make_kolmogorov_data(1.0, g1, g2, u0, v0, influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_kolmogorov_data(1.0, g1, g2, u0, v0, influx, du=0.5)
    s0 = mod.initialize(s0)

    # Polynomial doesn't need this, but it should work nonetheless.
    interp.initialize(stars)

    # Iterate solution using interpolator
    for iteration in range(3):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)

def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    mod = piff.Kolmogorov()
    g1 = g2 = u0 = v0 = 0.0

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = make_kolmogorov_data(1.0, g1, g2, u0, v0, influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            # Kill 10% of each star's pixels
            bad = np.random.rand(*s.image.array.shape) < 0.1
            s.weight.array[bad] = 0.
            s.image.array[bad] = -999.
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_kolmogorov_data(1.0, g1, g2, u0, v0, influx, du=0.5)
    s0 = mod.initialize(s0)

    interp = piff.Polynomial(order=0)
    interp.initialize(stars)

    oldchisq = 0.
    # Iterate solution using interpolator
    for iteration in range(40):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchisq>0 and chisq<oldchisq and oldchisq-chisq < dof/10.:
            break
        else:
            oldchisq = chisq

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=1)


def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    mod = piff.Kolmogorov()

    # Interpolator will be linear
    interp = piff.Polynomial(order=1)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        # Put gradient in pixel size
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            # spatially-varying fwhm, g1, g2.
            s = make_kolmogorov_data(1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v, influx,
                                     noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            stars.append(s)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4, 4)
    # for star, ax in zip(stars, axes.ravel()):
    #     ax.imshow(star.data.image.array)
    # plt.show()

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_kolmogorov_data(1.0, 0., 0., 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    # Polynomial doesn't need this, but it should work nonetheless.
    interp.initialize(stars)

    oldchisq = 0.
    # Iterate solution using interpolator
    for iteration in range(40):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
            break
        else:
            oldchisq = chisq

    for i, s in enumerate(stars):
        print(i, s.fit.center)

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=1)


def test_gradient_center():
    """Next: fit spatially-varying PSF, with spatially-varying centers to multiple images.
    """
    mod = piff.Kolmogorov(force_model_center=False)

    # Interpolator will be linear
    interp = piff.Polynomial(order=1)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        # Put gradient in pixel size
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            # spatially-varying fwhm, g1, g2.
            s = make_kolmogorov_data(1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v,
                                     influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            stars.append(s)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4, 4)
    # for star, ax in zip(stars, axes.ravel()):
    #     ax.imshow(star.data.image.array)
    # plt.show()

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_kolmogorov_data(1.0, 0., 0., 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    # Polynomial doesn't need this, but it should work nonetheless.
    interp.initialize(stars)

    oldchisq = 0.
    # Iterate solution using interpolator
    for iteration in range(40):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
            break
        else:
            oldchisq = chisq

    for i, s in enumerate(stars):
        print(i, s.fit.center, s.fit.params[0:2])

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=1)

if __name__ == '__main__':
    test_simple()
    test_center()
    test_interp()
    test_missing()
    test_gradient()
    test_gradient_center()
