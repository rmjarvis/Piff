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
import os
import fitsio

from piff_test_helper import timer

fiducial_kolmogorov = galsim.Kolmogorov(half_light_radius=1.0)
fiducial_gaussian = galsim.Gaussian(half_light_radius=1.0)
fiducial_moffat = galsim.Moffat(half_light_radius=1.0, beta=3.0)

def make_data(gsobject, scale, g1, g2, u0, v0, flux, noise=0., pix_scale=1., fpu=0., fpv=0.,
              nside=32, nom_u0=0., nom_v0=0., rng=None, include_pixel=True):
    """Make a Star instance filled with a Kolmogorov profile

    :param gsobject     The fiducial gsobject profile to use.
    :param scale:       The scale to apply to the gsobject.
    :param g1, g2:      The shear to apply to the gsobject.
    :param u0, v0:      The sub-pixel offset to apply.
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param pix_scale:   pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    :param include_pixel:  Include integration over pixel.  [default: True]
    """
    k = gsobject.withFlux(flux).dilate(scale).shear(g1=g1, g2=g2).shift(u0, v0)
    if noise == 0.:
        var = 0.1
    else:
        var = noise
    star = piff.Star.makeTarget(x=nside/2+nom_u0/pix_scale, y=nside/2+nom_v0/pix_scale,
                                u=fpu, v=fpv, scale=pix_scale, stamp_size=nside)
    star.image.setOrigin(0,0)
    method = 'auto' if include_pixel else 'no_pixel'
    k.drawImage(star.image, method=method,
                offset=galsim.PositionD(nom_u0/pix_scale, nom_v0/pix_scale), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


@timer
def test_simple():
    """Initial simple test of Gaussian, Kolmogorov, and Moffat PSFs.
    """
    # Here is the true PSF
    scale = 1.3
    g1 = 0.23
    g2 = -0.17
    du = 0.1
    dv = 0.4
    for fiducial in [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]:
        print()
        print("fiducial = ", fiducial)
        print()
        psf = fiducial.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64, 64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        fiducial_star = piff.Star(stardata, None)

        # First try fastfit.
        print('Fast fit')
        model = piff.GSObjectModel(fiducial, fastfit=True, include_pixel=False)
        fit = model.fit(model.initialize(fiducial_star)).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-4)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-7)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-7)

        # Now try fastfit=False.
        print('Slow fit')
        model = piff.GSObjectModel(fiducial, fastfit=False, include_pixel=False)
        fit = model.fit(model.initialize(fiducial_star)).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-6)

        # Now test running it via the config parser
        config = {
            'model' : {
                'type' : 'GSObjectModel',
                'gsobj': repr(fiducial),
                'include_pixel': False
            }
        }
        if __name__ == '__main__':
            logger = piff.config.setup_logger(verbose=3)
        else:
            logger = piff.config.setup_logger(verbose=1)
        model = piff.Model.process(config['model'], logger)
        fit = model.fit(model.initialize(fiducial_star)).fit

        # Same tests.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-6)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__

        # Finally, we should also test with pixel convolution included.  This really only makes
        # sense for fastfit=False, since HSM FindAdaptiveMom doesn't account for the pixel shape
        # in its measurements.

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        psf.drawImage(image, method='auto')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        fiducial_star = piff.Star(stardata, None)


        print('Slow fit, pixel convolution included.')
        model = piff.GSObjectModel(fiducial, fastfit=False, include_pixel=True)
        star = model.initialize(fiducial_star)
        star = model.fit(star, fastfit=True)  # Get better results with one round of fastfit.
        fit = model.fit(star).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # Accuracy goals are a bit looser here since it's harder to fit with the pixel involved.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-6)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)


@timer
def test_center():
    """Fit with centroid free and PSF center constrained to an initially mis-registered PSF.
    """
    influx = 150.
    scale = 2.0
    u0, v0 = 0.6, -0.4
    g1, g2 = 0.1, 0.2
    for fiducial in [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]:
        print()
        print("fiducial = ", fiducial)
        print()
        s = make_data(fiducial, scale, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)

        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        star = mod.initialize(s)
        print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
        for i in range(3):
            star = mod.fit(star)
            star = mod.reflux(star)
            print('Flux, ctr, chisq after fit {:d}:'.format(i),
                  star.fit.flux, star.fit.center, star.fit.chisq)
            np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=14)

        # Residual image when done should be dominated by structure off the edge of the fitted
        # region.
        mask = star.weight.array > 0
        # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
        star2 = mod.draw(star)
        print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
        print('max image abs value = ',np.max(np.abs(s.image.array)))
        peak = np.max(np.abs(s.image.array[mask]))
        np.testing.assert_almost_equal(star2.image.array[mask]/peak, s.image.array[mask]/peak,
                                       decimal=14)

        # test copy_image
        star_copy = mod.draw(star, copy_image=True)
        star_nocopy = mod.draw(star, copy_image=False)
        star.image.array[0,0] = 132435
        assert star_nocopy.image.array[0,0] == star.image.array[0,0]
        assert star_copy.image.array[0,0] != star.image.array[0,0]
        assert star_copy.image.array[1,1] == star.image.array[1,1]

@timer
def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    influx = 150.
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
        niter = 3
        npos = 10
    else:
        fiducial_list = [fiducial_moffat]
        niter = 1  # Not actually any need for interating in this case.
        npos = 4
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        g1 = g2 = u0 = v0 = 0.0

        # Interpolator will be simple mean
        interp = piff.Polynomial(order=0)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,npos)
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            for v in positions:
                s = make_data(fiducial, 1.0, g1, g2, u0, v0, influx,
                              noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng, include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)
        s0 = mod.initialize(s0)

        # Polynomial doesn't need this, but it should work nonetheless.
        interp.initialize(stars)

        # Iterate solution using interpolator
        for iteration in range(niter):
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
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=3)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=3)


@timer
def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)
        g1 = g2 = u0 = v0 = 0.0

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        np_rng = np.random.RandomState(1234)
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                s = make_data(fiducial, 1.0, g1, g2, u0, v0, influx,
                              noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng, include_pixel=False)
                s = mod.initialize(s)
                # Kill 10% of each star's pixels
                bad = np_rng.rand(*s.image.array.shape) < 0.1
                s.weight.array[bad] = 0.
                s.image.array[bad] = -999.
                s = mod.reflux(s, fit_center=False) # Start with a sensible flux
                stars.append(s)

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, g1, g2, u0, v0, influx, pix_scale=0.5, include_pixel=False)
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
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=3)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=3)


@timer
def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)

        # Interpolator will be linear
        interp = piff.Polynomial(order=1)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            # Put gradient in pixel size
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                # spatially-varying fwhm, g1, g2.
                s = make_data(fiducial, 1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v, influx,
                                         noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng,
                                         include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(4, 4)
        # for star, ax in zip(stars, axes.ravel()):
        #     ax.imshow(star.data.image.array)
        # plt.show()

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, 0., 0., 0., 0., influx, pix_scale=0.5, include_pixel=False)
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
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


@timer
def test_gradient_center():
    """Next: fit spatially-varying PSF, with spatially-varying centers to multiple images.
    """
    if __name__ == '__main__':
        fiducial_list = [fiducial_gaussian, fiducial_kolmogorov, fiducial_moffat]
    else:
        fiducial_list = [fiducial_moffat]
    for fiducial in fiducial_list:
        print()
        print("fiducial = ", fiducial)
        print()
        mod = piff.GSObjectModel(fiducial, include_pixel=False)

        # Interpolator will be linear
        interp = piff.Polynomial(order=1)

        # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
        positions = np.linspace(0.,1.,4)
        influx = 150.
        stars = []
        rng = galsim.BaseDeviate(1234)
        for u in positions:
            # Put gradient in pixel size
            for v in positions:
                # Draw stars in focal plane positions around a unit ring
                # spatially-varying fwhm, g1, g2.
                s = make_data(fiducial, 1.0+u*0.1+0.1*v, 0.1*u, 0.1*v, 0.5*u, 0.5*v,
                              influx, noise=0.1, pix_scale=0.5, fpu=u, fpv=v, rng=rng,
                              include_pixel=False)
                s = mod.initialize(s)
                stars.append(s)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(4, 4)
        # for star, ax in zip(stars, axes.ravel()):
        #     ax.imshow(star.data.image.array)
        # plt.show()

        # Also store away a noiseless copy of the PSF, origin of focal plane
        s0 = make_data(fiducial, 1.0, 0., 0., 0., 0., influx, pix_scale=0.5, include_pixel=False)
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
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        peak = np.max(np.abs(s0.image.array))
        np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


@timer
def test_direct():
    """ Simple test for directly instantiated Gaussian, Kolmogorov, and Moffat without going through
    GSObjectModel explicitly.
    """
    # Here is the true PSF
    scale = 1.3
    g1 = 0.23
    g2 = -0.17
    du = 0.1
    dv = 0.4

    gsobjs = [galsim.Gaussian(sigma=1.0),
              galsim.Kolmogorov(half_light_radius=1.0),
              galsim.Moffat(half_light_radius=1.0, beta=3.0),
              galsim.Moffat(half_light_radius=1.0, beta=2.5, trunc=3.0)]

    models = [piff.Gaussian(fastfit=True, include_pixel=False),
              piff.Kolmogorov(fastfit=True, include_pixel=False),
              piff.Moffat(fastfit=True, beta=3.0, include_pixel=False),
              piff.Moffat(fastfit=True, beta=2.5, trunc=3.0, include_pixel=False)]

    for gsobj, model in zip(gsobjs, models):
        print()
        print("gsobj = ", gsobj)
        print()
        psf = gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        star = piff.Star(stardata, None)
        star = model.initialize(star)

        # First try fastfit.
        print('Fast fit')
        fit = model.fit(star).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        # These tests are more strict above.  The truncated Moffat included here but not there
        # doesn't work quite as well.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-4)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_direct_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__

    # repeat with fastfit=False

    models = [piff.Gaussian(fastfit=False, include_pixel=False),
              piff.Kolmogorov(fastfit=False, include_pixel=False),
              piff.Moffat(fastfit=False, beta=3.0, include_pixel=False),
              piff.Moffat(fastfit=False, beta=2.5, trunc=3.0, include_pixel=False)]

    for gsobj, model in zip(gsobjs, models):
        print()
        print("gsobj = ", gsobj)
        print()
        psf = gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)

        # This is only going to come out right if we (unphysically) don't convolve by the pixel.
        psf.drawImage(image, method='no_pixel')

        # Make a StarData instance for this image
        stardata = piff.StarData(image, image.true_center)
        star = piff.Star(stardata, None)
        star = model.initialize(star)

        print('Slow fit')
        fit = model.fit(star).fit

        print('True scale = ', scale, ', model scale = ', fit.params[0])
        print('True g1 = ', g1, ', model g1 = ', fit.params[1])
        print('True g2 = ', g2, ', model g2 = ', fit.params[2])
        print('True du = ', du, ', model du = ', fit.center[0])
        print('True dv = ', dv, ', model dv = ', fit.center[1])

        # This test is fairly accurate, since we didn't add any noise and didn't convolve by
        # the pixel, so the image is very accurately a sheared GSObject.
        np.testing.assert_allclose(fit.params[0], scale, rtol=1e-5)
        np.testing.assert_allclose(fit.params[1], g1, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.params[2], g2, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[0], du, rtol=0, atol=1e-5)
        np.testing.assert_allclose(fit.center[1], dv, rtol=0, atol=1e-5)

        # Also need to test ability to serialize
        outfile = os.path.join('output', 'gsobject_direct_test.fits')
        with fitsio.FITS(outfile, 'rw', clobber=True) as f:
            model.write(f, 'psf_model')
        with fitsio.FITS(outfile, 'r') as f:
            roundtrip_model = piff.GSObjectModel.read(f, 'psf_model')
        assert model.__dict__ == roundtrip_model.__dict__


if __name__ == '__main__':
    test_simple()
    test_center()
    test_interp()
    test_missing()
    test_gradient()
    test_gradient_center()
    test_direct()
