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
import numpy as np
import piff
import galsim

def make_gaussian_data(sigma, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
                       nom_u0=0., nom_v0=0., rng=None):
    """Make a StarData instance filled with a Gaussian profile

    :param sigma:       The sigma of the Gaussian
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
    g = galsim.Gaussian(sigma=sigma, flux=flux).shift(u0,v0)
    if noise == 0.:
        var = 0.1
    else:
        var = noise
    s = piff.StarData.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                 u=fpu, v=fpv, scale=du, stamp_size=nside)
    s.image.setOrigin(0,0)
    g.drawImage(s.image, method='no_pixel', use_true_center=False,
                offset=galsim.PositionD(nom_u0/du,nom_v0/du))
    s.weight = s.image.copy()
    s.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        s.image.addNoise(gn)
    return s


def test_simplest():
    """Fit a PSF to noiseless Gaussian data at same sampling
    """
    influx = 150.
    du = 0.5
    s = make_gaussian_data(2.0, 0., 0., influx, du=du)

    # Pixelized model with Lanczos 3 interp
    interp = piff.Lanczos(3)
    mod = piff.PixelModel(du, 32, interp, start_sigma=1.5)
    star = mod.makeStar(s, flux=np.sum(s.data))

    # Check that fitting the star can recover the right flux.
    # Note: this shouldn't match perfectly, since SimpleData draws this as a surface
    # brightness image, not integrated over pixels.  With GalSim drawImage, we can do better
    # by drawing the real flux image.  But even with this, we get 3 dp of accuracy.
    star = mod.fit(star)
    star = mod.reflux(star)
    print('Flux after fit 1:',star.fit.flux)
    np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=3)
    flux1 = star.fit.flux

    # It doesn't get any better after another iteration.
    star = mod.fit(star)
    star = mod.reflux(star)
    print('Flux after fit 2:',star.fit.flux)
    np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=3)
    np.testing.assert_almost_equal(star.fit.flux/flux1, 1.0, decimal=7)

    # Drawing the star should produce a nearly identical image to the original.
    star2 = mod.draw(star)
    print('max image abs diff = ',np.max(np.abs(star2.data.data-s.data)))
    print('max image abs value = ',np.max(np.abs(s.data)))
    np.testing.assert_almost_equal(star2.data.data, s.data, decimal=7)


def test_oversample():
    """Fit to oversampled data, decentered PSF.
    """
    influx = 150.
    du = 0.25
    nside = 64
    s = make_gaussian_data(2.0, 0.5, -0.25, influx, du=du, nside=nside)

    # Pixelized model with Lanczos 3 interp, coarser pix scale
    interp = piff.Lanczos(3)
    mod = piff.PixelModel(2*du, nside/2, interp, start_sigma=1.5)
    star = mod.makeStar(s, flux=np.sum(s.data))

    for i in range(2):
        star = mod.fit(star)
        star = mod.reflux(star)
        print('Flux after fit {:d}:'.format(i),star.fit.flux)
        np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=3)

    # Residual image should be checkeboard from limitations of interpolator.
    # So only agree to 3 dp.
    star2 = mod.draw(star)
    #print('star2 image = ',star2.data.data)
    #print('star.image = ',s.data)
    #print('diff = ',star2.data.data-s.data)
    print('max image abs diff = ',np.max(np.abs(star2.data.data-s.data)))
    print('max image abs value = ',np.max(np.abs(s.data)))
    peak = np.max(np.abs(s.data))
    np.testing.assert_almost_equal(star2.data.data/peak, s.data/peak, decimal=3)


def test_center():
    """Fit with centroid free and PSF center constrained to an initially mis-registered PSF.
    """
    influx = 150.
    s = make_gaussian_data(2.0, 0.6, -0.4, influx, du=0.5)

    # Pixelized model with Lanczos 3 interp, coarser pix scale, smaller
    # than the data
    interp = piff.Lanczos(3)
    # Want an odd-sized model when center=True
    mod = piff.PixelModel(0.5, 29, interp, force_model_center=True, start_sigma=1.5)
    star = mod.makeStar(s)
    star = mod.reflux(star, fit_center=False) # Start with a sensible flux
    star = mod.reflux(star) # and center too
    print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
    for i in range(3):
        star = mod.fit(star)
        star = mod.reflux(star)
        print('Flux, ctr, chisq after fit {:d}:'.format(i),
              star.fit.flux, star.fit.center, star.fit.chisq)
        # These fluxes are not at all close to influx.  Not sure why...
        #np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=2)

    # Residual image when done should be dominated by structure off the edge of the fitted region.
    # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
    star2 = mod.draw(star)
    print('max image abs diff = ',np.max(np.abs(star2.data.data-s.data)))
    print('max image abs value = ',np.max(np.abs(s.data)))
    peak = np.max(np.abs(s.data))
    np.testing.assert_almost_equal(star2.data.data/peak, s.data/peak, decimal=2)


def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelModel(0.5, 25, pixinterp, start_sigma=1.5, degenerate=False)

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
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

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
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
    print('max image abs value = ',np.max(np.abs(s0.data.data)))
    peak = np.max(np.abs(s0.data.data))
    np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=2)


def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelModel(0.5, 25, pixinterp, start_sigma=1.5)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.makeStar(s)
            # Kill 10% of each star's pixels
            bad = np.random.rand(*s.data.data.shape) < 0.1
            s.data.weight.array[bad] = 0.
            s.data.data[bad] = -999.
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

    if __name__ == "__main__":
        interps = [piff.Polynomial(order=0), piff.BasisPolynomial(order=0)]
    else:
        # The Polynomial interpolator works, but it's slow.  For the nosetests runs, skip it.
        interps = [piff.BasisPolynomial(order=0)]

    for interp in interps:
        # Interpolator will be simple mean
        interp = piff.Polynomial(order=0)

        oldchi = 0.
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
            if oldchi>0 and chisq<oldchi and oldchi-chisq < dof/10.:
                break
            else:
                oldchi = chisq

        # Now use the interpolator to produce a noiseless rendering
        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
        # Less than 2 dp of accuracy here!
        np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

        s1 = mod.draw(s1)
        print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
        print('max image abs value = ',np.max(np.abs(s0.data.data)))
        peak = np.max(np.abs(s0.data.data))
        np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=1)


def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelModel(0.5, 25, pixinterp, start_sigma=1.5, degenerate=False)

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
            s = make_gaussian_data(1.0+u*0.1, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

    oldchi = 0.
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
        if oldchi>0 and np.abs(oldchi-chisq) < dof/10.:
            break
        else:
            oldchi = chisq

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
    print('max image abs value = ',np.max(np.abs(s0.data.data)))
    peak = np.max(np.abs(s0.data.data))
    np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=1)


def test_undersamp():
    """Next: fit PSF to undersampled, dithered data with fixed centroids
    ***Doesn't work well! Need to work on the SV pruning***
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    du = 0.5
    mod = piff.PixelModel(0.25, 25, pixinterp, start_sigma=1.01)
    ##,force_model_center=True)

    # Interpolator will be constant
    interp = piff.Polynomial(order=0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Dither centers by 1 pixel
            phase = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase=(0.,0.)
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=du, fpu=u, fpv=v,
                                   nom_u0=phase[0], nom_v0=phase[1], rng=rng)
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            print("phase:",phase,'flux',s.fit.flux)
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

    oldchi = 0.
    # Iterate solution using interpolator
    for iteration in range(1): ###
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
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
            print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof, 'flux=',s.fit.flux)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < dof/10.:
            break
        else:
            oldchi = chisq

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
    print('max image abs value = ',np.max(np.abs(s0.data.data)))
    peak = np.max(np.abs(s0.data.data))
    np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=1)


def test_undersamp_shift():
    """Next: fit PSF to undersampled, dithered data with variable centroids,
    this time using chisq() and summing alpha,beta instead of fit() per star
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    influx = 150.
    du = 0.5
    mod = piff.PixelModel(0.3, 25, pixinterp, start_sigma=1.3, force_model_center=True)

    # Make a sample star just so we can pass the initial PSF into interpolator
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

    # Interpolator will be constant
    interp = piff.BasisPolynomial(0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Nominal star centers move by +-1/2 pix, real centers another 1/2 pix
            phase1 = (0.5 - np.random.rand(2))*du
            phase2 = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase1 = phase2 =(0.,0.)
            s = make_gaussian_data(1.0, phase2[0], phase2[1], influx, noise=0.1, du=du,
                                   fpu=u, fpv=v, nom_u0=phase1[0], nom_v0=phase1[1], rng=rng)
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            ###print("phase:",phase2,'flux',s.fit.flux)###
            stars.append(s)

    # BasisInterp needs to be initialized before solving.
    interp.initialize(stars)

    oldchi = 0.
    # Iterate solution using mean of chisq
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.chisq(s) for s in stars]
        # Solve for interpolated PSF function
        interp.solve(stars)
        # Refit and recenter all stars
        stars = [mod.reflux(interp.interpolate(s)) for s in stars]
        chisq = np.sum([s.fit.chisq for s in stars])
        dof   = np.sum([s.fit.dof for s in stars])
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < dof/10.:
            break
        else:
            oldchi = chisq

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
    print('max image abs value = ',np.max(np.abs(s0.data.data)))
    peak = np.max(np.abs(s0.data.data))
    np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=1)


def do_undersamp_drift(fit_centers=False):
    """Draw stars whose size and position vary across FOV.
    Fit to oversampled model with linear dependence across FOV.

    Argument fit_centers decides whether we are letting the PSF model
    center drift, or whether we re-fit the center positions of the stars.
    """
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    influx = 150.
    du = 0.5
    mod = piff.PixelModel(0.3, 25, pixinterp, start_sigma=1.3, force_model_center=fit_centers)

    # Make a sample star just so we can pass the initial PSF into interpolator
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.makeStar(s0)

    # Interpolator will be linear
    interp = piff.BasisPolynomial(1)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    stars = []
    np.random.seed(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Nominal star centers move by +-1/2 pix
            phase1 = (0.5 - np.random.rand(2))*du
            phase2 = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase1 = phase2 =(0.,0.)
            # PSF center will drift with v; size drifts with u
            s = make_gaussian_data(1.0+0.1*u, 0., 0.5*du*v, influx, noise=0.1, du=du,
                                   fpu=u, fpv=v, nom_u0=phase1[0], nom_v0=phase1[1], rng=rng)
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            ###print("phase:",phase2,'flux',s.fit.flux)###
            stars.append(s)

    # BasisInterp needs to be initialized before solving.
    interp.initialize(stars)

    oldchi = 0.
    # Iterate solution using mean of chisq
    for iteration in range(20):
        # Refit PSFs star by star:
        stars = [mod.chisq(s) for s in stars]
        # Solve for interpolated PSF function
        interp.solve(stars)
        # Refit and recenter all stars
        stars = [mod.reflux(interp.interpolate(s)) for s in stars]
        chisq = np.sum([s.fit.chisq for s in stars])
        dof   = np.sum([s.fit.dof for s in stars])
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < dof/10.:
            break
        else:
            oldchi = chisq

    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.data.data-s0.data.data)))
    print('max image abs value = ',np.max(np.abs(s0.data.data)))
    peak = np.max(np.abs(s0.data.data))
    np.testing.assert_almost_equal(s1.data.data/peak, s0.data.data/peak, decimal=1)

def test_undersamp_drift():
    do_undersamp_drift(True)
    do_undersamp_drift(False)

if __name__ == '__main__':
    #import cProfile, pstats
    #pr = cProfile.Profile()
    #pr.enable()
    test_simplest()
    test_oversample()
    test_center()
    test_interp()
    test_missing()
    test_gradient()
    test_undersamp()
    test_undersamp_shift()
    test_undersamp_drift()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('tottime').reverse_order()
    #ps.print_stats()
