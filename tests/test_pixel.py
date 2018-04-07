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
import yaml
import subprocess

from piff_test_helper import get_script_name, timer

def make_gaussian_data(sigma, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
                       nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Gaussian profile

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
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu, v=fpv, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    g.drawImage(star.image, method='no_pixel', use_true_center=False,
                offset=galsim.PositionD(nom_u0/du,nom_v0/du))
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


@timer
def test_simplest():
    """Fit a PSF to noiseless Gaussian data at same sampling
    """
    influx = 150.
    du = 0.5
    s = make_gaussian_data(2.0, 0., 0., influx, du=du)

    # Pixelized model with Lanczos 3 interp
    interp = piff.Lanczos(3)
    mod = piff.PixelGrid(du, 32, interp, start_sigma=1.5, force_model_center=False)
    star = mod.initialize(s).withFlux(flux=np.sum(s.image.array))

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
    print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
    print('max image abs value = ',np.max(np.abs(s.image.array)))
    np.testing.assert_almost_equal(star2.image.array, s.image.array, decimal=7)


@timer
def test_oversample():
    """Fit to oversampled data, decentered PSF.
    """
    influx = 150.
    du = 0.25
    nside = 64
    s = make_gaussian_data(2.0, 0.5, -0.25, influx, du=du, nside=nside)

    # Pixelized model with Lanczos 3 interp, coarser pix scale
    interp = 'Lanczos(3)'  # eval the string
    mod = piff.PixelGrid(2*du, nside//2, interp, start_sigma=1.5, force_model_center=False)
    star = mod.initialize(s).withFlux(flux=np.sum(s.image.array))

    for i in range(2):
        star = mod.fit(star)
        star = mod.reflux(star)
        print('Flux after fit {:d}:'.format(i),star.fit.flux)
        np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=3)

    # Residual image should be checkeboard from limitations of interpolator.
    # So only agree to 3 dp.
    star2 = mod.draw(star)
    #print('star2 image = ',star2.image.array)
    #print('star.image = ',s.image.array)
    #print('diff = ',star2.image.array-s.image.array)
    print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
    print('max image abs value = ',np.max(np.abs(s.image.array)))
    peak = np.max(np.abs(s.image.array))
    np.testing.assert_almost_equal(star2.image.array/peak, s.image.array/peak, decimal=3)


@timer
def test_center():
    """Fit with centroid free and PSF center constrained to an initially mis-registered PSF.
    """
    influx = 150.
    s = make_gaussian_data(2.0, 0.6, -0.4, influx, du=0.5)

    # Pixelized model with Lanczos 3 interp, coarser pix scale, smaller
    # than the data
    interp = None  # Default is Lanczos(3)
    # Want an odd-sized model when center=True
    mod = piff.PixelGrid(0.5, 29, interp, force_model_center=True, start_sigma=1.5)
    star = mod.initialize(s)
    print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
    for i in range(3):
        star = mod.fit(star)
        star = mod.reflux(star)
        print('Flux, ctr, chisq after fit {:d}:'.format(i),
              star.fit.flux, star.fit.center, star.fit.chisq)
        # These fluxes are not at all close to influx.  Not sure why...
        #np.testing.assert_almost_equal(star.fit.flux/influx, 1.0, decimal=2)

    # Residual image when done should be dominated by structure off the edge of the fitted region.
    mask = star.weight.array > 0
    # This comes out fairly close, but only 2 dp of accuracy, compared to 3 above.
    star2 = mod.draw(star)
    print('max image abs diff = ',np.max(np.abs(star2.image.array-s.image.array)))
    print('max image abs value = ',np.max(np.abs(s.image.array)))
    peak = np.max(np.abs(s.image.array[mask]))
    np.testing.assert_almost_equal(star2.image.array[mask]/peak, s.image.array[mask]/peak,
                                   decimal=2)


@timer
def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    if __name__ == '__main__':
        npos = 10
        size = 25
    else:
        npos = 5
        size = 15

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelGrid(0.5, size, pixinterp, start_sigma=1.5, degenerate=False)

    # Interpolator will be simple mean
    interp = piff.Polynomial(order=0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,npos)
    influx = 150.
    stars = []
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
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
    print('Flux, ctr after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    np.testing.assert_almost_equal(s1.fit.flux/influx, 1.0, decimal=2)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


@timer
def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    if __name__ == '__main__':
        bad_frac = 0.1
        size = 25
        interps = [piff.Polynomial(order=0), piff.BasisPolynomial(order=0)]
    else:
        bad_frac = 0.05
        size = 15
        # The Polynomial interpolator works, but it's slow.  For the nosetests runs, skip it.
        interps = [piff.BasisPolynomial(order=0)]

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelGrid(0.5, size, pixinterp, start_sigma=1.5, force_model_center=False)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np_rng = np.random.RandomState(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v, rng=rng)
            s = mod.initialize(s)
            # Kill 10% of each star's pixels
            bad = np_rng.rand(*s.image.array.shape) < bad_frac
            s.weight.array[bad] = 0.
            s.image.array[bad] = -999.
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    for interp in interps:
        # Interpolator will be simple mean
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


@timer
def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    if __name__ == '__main__':
        size = 25
    else:
        size = 15

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    mod = piff.PixelGrid(0.5, size, pixinterp, start_sigma=1.5,
                         degenerate=False, force_model_center=False)

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
            s = make_gaussian_data(1.0+u*0.1, 0., 0., influx, noise=0.1, du=0.5, fpu=u, fpv=v,
                                   rng=rng)
            s = mod.initialize(s)
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
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


@timer
def test_undersamp():
    """Next: fit PSF to undersampled, dithered data with fixed centroids
    ***Doesn't work well! Need to work on the SV pruning***
    """
    if __name__ == '__main__':
        size = 25
    else:
        size = 15

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    du = 0.5
    mod = piff.PixelGrid(0.25, size, pixinterp, start_sigma=1.01)
    ##,force_model_center=True)

    # Interpolator will be constant
    interp = piff.Polynomial(order=0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    np_rng = np.random.RandomState(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Dither centers by 1 pixel
            phase = (0.5 - np_rng.rand(2))*du
            if u==0. and v==0.:
                phase=(0.,0.)
            s = make_gaussian_data(1.0, 0., 0., influx, noise=0.1, du=du, fpu=u, fpv=v,
                                   nom_u0=phase[0], nom_v0=phase[1], rng=rng)
            s = mod.initialize(s)
            print("phase:",phase,'flux',s.fit.flux)
            stars.append(s)

    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    # Polynomial doesn't need this, but it should work nonetheless.
    interp.initialize(stars)

    oldchisq = 0.
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
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
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


@timer
def test_undersamp_shift():
    """Next: fit PSF to undersampled, dithered data with variable centroids,
    this time using chisq() and summing alpha,beta instead of fit() per star
    """
    if __name__ == '__main__':
        size = 25
    else:
        size = 15

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    influx = 150.
    du = 0.5
    mod = piff.PixelGrid(0.3, size, pixinterp, start_sigma=1.3, force_model_center=True)

    # Make a sample star just so we can pass the initial PSF into interpolator
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    # Interpolator will be constant
    interp = piff.BasisPolynomial(0)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    stars = []
    np_rng = np.random.RandomState(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Nominal star centers move by +-1/2 pix, real centers another 1/2 pix
            phase1 = (0.5 - np_rng.rand(2))*du
            phase2 = (0.5 - np_rng.rand(2))*du
            if u==0. and v==0.:
                phase1 = phase2 =(0.,0.)
            s = make_gaussian_data(1.0, phase2[0], phase2[1], influx, noise=0.1, du=du,
                                   fpu=u, fpv=v, nom_u0=phase1[0], nom_v0=phase1[1], rng=rng)
            s = mod.initialize(s)
            ###print("phase:",phase2,'flux',s.fit.flux)###
            stars.append(s)

    # BasisInterp needs to be initialized before solving.
    interp.initialize(stars)

    oldchisq = 0.
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
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
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


def do_undersamp_drift(fit_centers=False):
    """Draw stars whose size and position vary across FOV.
    Fit to oversampled model with linear dependence across FOV.

    Argument fit_centers decides whether we are letting the PSF model
    center drift, or whether we re-fit the center positions of the stars.
    """
    if __name__ == '__main__':
        size = 25
    else:
        size = 15

    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = piff.Lanczos(3)
    influx = 150.
    du = 0.5
    mod = piff.PixelGrid(0.3, size, pixinterp, start_sigma=1.3, force_model_center=fit_centers)

    # Make a sample star just so we can pass the initial PSF into interpolator
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = make_gaussian_data(1.0, 0., 0., influx, du=0.5)
    s0 = mod.initialize(s0)

    # Interpolator will be linear
    # Normally max order would be 1, but set it to 2 here, just to check that option.
    interp = piff.BasisPolynomial(order=1, max_order=2)

    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    stars = []
    np_rng = np.random.RandomState(1234)
    rng = galsim.BaseDeviate(1234)
    for u in positions:
        for v in positions:
            # Nominal star centers move by +-1/2 pix
            phase1 = (0.5 - np_rng.rand(2))*du
            phase2 = (0.5 - np_rng.rand(2))*du
            if u==0. and v==0.:
                phase1 = phase2 =(0.,0.)
            # PSF center will drift with v; size drifts with u
            s = make_gaussian_data(1.0+0.1*u, 0., 0.5*du*v, influx, noise=0.1, du=du,
                                   fpu=u, fpv=v, nom_u0=phase1[0], nom_v0=phase1[1], rng=rng)
            s = mod.initialize(s)
            ###print("phase:",phase2,'flux',s.fit.flux)###
            stars.append(s)

    # BasisInterp needs to be initialized before solving.
    interp.initialize(stars)

    oldchisq = 0.
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
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.:
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


@timer
def test_undersamp_drift():
    do_undersamp_drift(True)
    if __name__ == '__main__':
        do_undersamp_drift(False)


@timer
def test_single_image():
    """Test the whole process with a single image.

    Note: This test is based heavily on test_single_image in test_simple.py.
    """
    import os
    import fitsio
    np_rng = np.random.RandomState(1234)

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.2)

    # The (x,y) values will be on a grid 5 x 5 stars with a random sub-pixel offset.
    xvals = np.linspace(50., 1950., 5)
    yvals = np.linspace(50., 1950., 5)
    x_list, y_list = np.meshgrid(xvals, yvals)
    x_list = x_list.flatten()
    y_list = y_list.flatten()
    x_list = x_list + (np_rng.rand(len(x_list)) - 0.5)
    y_list = y_list + (np_rng.rand(len(x_list)) - 0.5)
    print('x_list = ',x_list)
    print('y_list = ',y_list)
    # Range of fluxes from 100 to 15000
    flux_list = 100. * np.exp(5. * np_rng.rand(len(x_list)))
    print('fluxes range from ',np.min(flux_list),np.max(flux_list))

    # Draw a Moffat PSF at each location on the image.
    # Have the truth values vary quadratically across the image.
    beta_fn = lambda x,y: 3.5 - 0.1*(x/1000) + 0.08*(y/1000)**2
    fwhm_fn = lambda x,y: 0.9 + 0.05*(x/1000) - 0.03*(y/1000) + 0.02*(x/1000)*(y/1000)
    e1_fn = lambda x,y: 0.02 - 0.01*(x/1000)
    e2_fn = lambda x,y: -0.03 + 0.02*(x/1000)**2 - 0.01*(y/1000)*2

    for x,y,flux in zip(x_list, y_list, flux_list):
        beta = beta_fn(x,y)
        fwhm = fwhm_fn(x,y)
        e1 = e1_fn(x,y)
        e2 = e2_fn(x,y)
        print(x,y,beta,fwhm,e1,e2)
        moffat = galsim.Moffat(fwhm=fwhm, beta=beta, flux=flux).shear(e1=e1, e2=e2)
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
        moffat.drawImage(image=image[bounds], offset=offset, method='no_pixel')
    print('drew image')

    # Add sky level and noise
    sky_level = 1000
    noise_sigma = 0.1  # Not much noise to keep this an easy test.
    image += sky_level
    image.addNoise(galsim.GaussianNoise(sigma=noise_sigma))

    # Write out the image to a file
    image_file = os.path.join('output','pixel_moffat_image.fits')
    image.write(image_file)
    print('wrote image')

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','pixel_moffat_cat.fits')
    fitsio.write(cat_file, data, clobber=True)
    print('wrote catalog')

    # Use InputFiles to read these back in
    config = { 'image_file_name': image_file,
               'cat_file_name': cat_file,
               'stamp_size': 32,
               'noise' : noise_sigma**2,
               'sky' : sky_level,
             }
    input = piff.InputFiles(config)
    assert input.image_file_name == [image_file]
    assert input.cat_file_name == [cat_file]

    # Check image
    assert len(input.images) == 1
    np.testing.assert_equal(input.images[0].array, image.array)

    # Check catalog
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == len(x_list)
    np.testing.assert_equal([pos.x for pos in input.image_pos[0]], x_list)
    np.testing.assert_equal([pos.y for pos in input.image_pos[0]], y_list)

    # Make stars
    orig_stars = input.makeStars()
    assert len(orig_stars) == len(x_list)
    assert orig_stars[0].image.array.shape == (32,32)

    # Make a test star, not at the location of any of the model stars to use for each of the
    # below tests.
    x0 = 1024  # Some random position, not where a star was originally.
    y0 = 133
    beta = beta_fn(x0,y0)
    fwhm = fwhm_fn(x0,y0)
    e1 = e1_fn(x0,y0)
    e2 = e2_fn(x0,y0)
    moffat = galsim.Moffat(fwhm=fwhm, beta=beta).shear(e1=e1, e2=e2)
    target_star = piff.Star.makeTarget(x=x0, y=y0, scale=image.scale)
    test_im = galsim.ImageD(bounds=target_star.image.bounds, scale=image.scale)
    moffat.drawImage(image=test_im, method='no_pixel', use_true_center=False)
    print('made test star')

    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
        order = 2
    else:
        logger = None
        order = 1

    # These tests are slow, and it's really just doing the same thing three times, so
    # only do the first one when running via nosetests.
    psf_file = os.path.join('output','pixel_psf.fits')
    if __name__ == '__main__':
        # Process the star data
        model = piff.PixelGrid(0.2, 16, start_sigma=0.9/2.355)
        interp = piff.BasisPolynomial(order=order)
        pointing = None     # wcs is not Celestial here, so pointing needs to be None.
        psf = piff.SimplePSF(model, interp)
        psf.fit(orig_stars, {0:input.images[0].wcs}, pointing, logger=logger)

        # Check that the interpolation is what it should be
        print('target.flux = ',target_star.fit.flux)
        test_star = psf.drawStar(target_star)
        print('flux = ', test_im.array.sum(), test_star.image.array.sum())
        print('max diff = ',np.max(np.abs(test_star.image.array-test_im.array)))
        np.testing.assert_almost_equal(test_star.image.array/2, test_im.array/2, decimal=3)

        # Check the convenience function that an end user would typically use
        image = psf.draw(x=x0, y=y0)
        np.testing.assert_almost_equal(image.array/2, test_im.array/2, decimal=3)

        # Round trip through a file
        psf.write(psf_file, logger)
        psf = piff.read(psf_file, logger)
        assert type(psf.model) is piff.PixelGrid
        assert type(psf.interp) is piff.BasisPolynomial
        test_star = psf.drawStar(target_star)
        np.testing.assert_almost_equal(test_star.image.array/2, test_im.array/2, decimal=3)

        # Check the convenience function that an end user would typically use
        image = psf.draw(x=x0, y=y0)
        np.testing.assert_almost_equal(image.array/2., test_im.array/2., decimal=3)

    # Do the whole thing with the config parser
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'x_col' : 'x',
            'y_col' : 'y',
            'noise' : noise_sigma**2,
            'sky' : sky_level,
            'stamp_size' : 48  # Bigger than we drew, but should still work.
        },
        'output' : {
            'file_name' : psf_file
        },
        'psf' : {
            'model' : {
                'type' : 'PixelGrid',
                'scale' : 0.2,
                'size' : 16,  # Much smaller than the input stamps, but this is plenty here.
                'start_sigma' : 0.9/2.355
            },
            'interp' : {
                'type' : 'BasisPolynomial',
                'order' : order
            },
        },
    }
    if __name__ == '__main__':
        config['verbose'] = 2
    else:
        config['verbose'] = 0

    print("Running piffify function")
    piff.piffify(config)
    psf = piff.read(psf_file)
    test_star = psf.drawStar(target_star)
    print("Max abs diff = ",np.max(np.abs(test_star.image.array - test_im.array)))
    np.testing.assert_almost_equal(test_star.image.array/2., test_im.array/2., decimal=3)

    # Test using the piffify executable
    with open('pixel_moffat.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    if __name__ == '__main__':
        print("Running piffify executable")
        if os.path.exists(psf_file):
            os.remove(psf_file)
        piffify_exe = get_script_name('piffify')
        p = subprocess.Popen( [piffify_exe, 'pixel_moffat.yaml'] )
        p.communicate()
        psf = piff.read(psf_file)
        test_star = psf.drawStar(target_star)
        np.testing.assert_almost_equal(test_star.image.array/2., test_im.array/2., decimal=3)

    # test copy_image property of drawStar and draw
    for draw in [psf.drawStar, psf.model.draw]:
        target_star_copy = psf.interp.interpolate(piff.Star(target_star.data.copy(), target_star.fit.copy()))  # interp is so that when we do psf.model.draw we have fit.params to work with

        test_star_copy = draw(target_star_copy, copy_image=True)
        test_star_nocopy = draw(target_star_copy, copy_image=False)
        # if we modify target_star_copy, then test_star_nocopy should be modified, but not test_star_copy
        target_star_copy.image.array[0,0] = 23456
        assert test_star_nocopy.image.array[0,0] == target_star_copy.image.array[0,0]
        assert test_star_copy.image.array[0,0] != target_star_copy.image.array[0,0]
        # however the other pixels SHOULD still be all the same value
        assert test_star_nocopy.image.array[1,1] == target_star_copy.image.array[1,1]
        assert test_star_copy.image.array[1,1] == target_star_copy.image.array[1,1]

    # check that drawing onto an image does not return a copy
    image = psf.draw(x=x0, y=y0)
    image_reference = psf.draw(x=x0, y=y0, image=image)
    image_reference.array[0,0] = 123456
    assert image.array[0,0] == image_reference.array[0,0]

@timer
def test_des_image():
    """Test the whole process with a DES CCD.
    """
    import os
    import fitsio

    image_file = 'input/DECam_00241238_01.fits.fz'
    cat_file = 'input/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits'
    orig_image = galsim.fits.read(image_file)
    psf_file = os.path.join('output','pixel_des_psf.fits')

    if __name__ == '__main__':
        # These match what Gary used in fit_des.py
        nstars = None
        scale = 0.15
        size = 31
        order = 2
        nsigma = 4
    else:
        # These are faster and good enough for the unit tests.
        nstars = 25
        scale = 0.26
        size = 15
        order = 1
        nsigma = 1.  # This needs to be low to make sure we do test outlier rejection here.
    stamp_size = 25

    # The configuration dict with the right input fields for the file we're using.
    start_sigma = 1.0/2.355  # TODO: Need to make this automatic somehow.
    config = {
        'input' : {
            'nstars': nstars,
            'image_file_name' : image_file,
            'image_hdu' : 1,
            'weight_hdu' : 3,
            'badpix_hdu' : 2,
            'cat_file_name' : cat_file,
            'cat_hdu' : 2,
            'x_col' : 'XWIN_IMAGE',
            'y_col' : 'YWIN_IMAGE',
            'sky_col' : 'BACKGROUND',
            'stamp_size' : stamp_size,
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            # Test explicitly specifying the wcs (although it is the same here as what is in the
            # image anyway).
            'wcs' : {
                'type': 'Fits',
                'file_name': image_file
            }
        },
        'output' : {
            'file_name' : psf_file,
        },
        'psf' : {
            'model' : {
                'type' : 'PixelGrid',
                'scale' : scale,
                'size' : size,
                'interp' : 'Lanczos(5)',
                'start_sigma' : start_sigma,
            },
            'interp' : {
                'type' : 'BasisPolynomial',
                'order' : order,
            },
            'outliers' : {
                'type' : 'Chisq',
                'nsigma' : nsigma,
                'max_remove' : 3
            }
        },
    }
    if __name__ == '__main__':
        config['verbose'] = 2
    else:
        config['verbose'] = 0

    # These tests are slow, and it's really just doing the same thing three times, so
    # only do the first one when running via nosetests.
    if __name__ == '__main__':
        # Start by doing things manually:
        logger = piff.config.setup_logger(2)

        # Largely copied from Gary's fit_des.py, but using the Piff input_handler to
        # read the input files.
        stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)
        if nstars is not None:
            stars = stars[:nstars]

        # Make model, force PSF centering
        model = piff.PixelGrid(scale=scale, size=size, interp=piff.Lanczos(3),
                               force_model_center=True, start_sigma=start_sigma,
                               logger=logger)

        # Interpolator will be zero-order polynomial.
        # Find u, v ranges
        interp = piff.BasisPolynomial(order=order, logger=logger)

        # Make a psf
        psf = piff.SimplePSF(model, interp)
        psf.fit(stars, wcs, pointing, logger=logger)

        # The difference between the images of the fitted stars and the originals should be
        # consistent with noise.  Keep track of how many don't meet that goal.
        n_bad = 0  # chisq/dof > 2
        n_marginal = 0  # chisq/dof > 1.1
        n_good = 0 # chisq/dof <= 1.1
        # Note: The 2 and 1.1 values here are very arbitrary!

        for s in psf.stars:
            fitted = psf.drawStar(s)
            orig_stamp = orig_image[fitted.image.bounds] - s['sky']
            fit_stamp = fitted.image

            x0 = int(s['x']+0.5)
            y0 = int(s['y']+0.5)
            b = galsim.BoundsI(x0-3,x0+3,y0-3,y0+3)
            #print('orig center = ',orig_stamp[b].array)
            #print('flux = ',orig_stamp.array.sum())
            #print('fit center = ',fit_stamp[b].array)
            #print('flux = ',fit_stamp.array.sum())
            flux = fitted.fit.flux
            #print('max diff/flux = ',np.max(np.abs(orig_stamp.array-fit_stamp.array))/flux)
            #np.testing.assert_almost_equal(fit_stamp.array/flux, orig_stamp.array/flux, decimal=2)
            weight = s.weight  # These should be 1/var_pix
            resid = fit_stamp - orig_stamp
            chisq = np.sum(resid.array**2 * weight.array)
            print('chisq = ',chisq)
            print('cf. star.chisq, dof = ',s.fit.chisq, s.fit.dof)
            assert abs(chisq - s.fit.chisq) < 1.e-3 * chisq
            if chisq > 2. * s.fit.dof:
                n_bad += 1
            elif chisq > 1.1 * s.fit.dof:
                n_marginal += 1
            else:
                n_good += 1

            # Check the convenience function that an end user would typically use
            offset = s.center_to_offset(s.fit.center)
            image = psf.draw(x=s['x'], y=s['y'], stamp_size=stamp_size,
                             flux=s.fit.flux, offset=offset)
            np.testing.assert_almost_equal(image.array, fit_stamp.array, decimal=4)

        print('n_good, marginal, bad = ',n_good,n_marginal,n_bad)
        # The real counts are 10 and 2.  So this says make sure any updates to the code don't make
        # things much worse.
        assert n_marginal <= 12
        assert n_bad <= 3

    # Use piffify function
    print('start piffify')
    piff.piffify(config)
    print('read stars')
    stars, wcs, pointing = piff.Input.process(config['input'])
    print('read psf')
    psf = piff.read(psf_file)
    stars = [psf.model.initialize(s) for s in stars]
    flux = stars[0].fit.flux
    offset = stars[0].center_to_offset(stars[0].fit.center)
    fit_stamp = psf.draw(x=stars[0]['x'], y=stars[0]['y'], stamp_size=stamp_size,
                         flux=flux, offset=offset)
    orig_stamp = orig_image[stars[0].image.bounds] - stars[0]['sky']
    # The first star happens to be a good one, so go ahead and test the arrays directly.
    np.testing.assert_almost_equal(fit_stamp.array/flux, orig_stamp.array/flux, decimal=2)

    # Test using the piffify executable
    with open('pixel_des.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    if __name__ == '__main__':
        if os.path.exists(psf_file):
            os.remove(psf_file)
        piffify_exe = get_script_name('piffify')
        print('start piffify executable')
        p = subprocess.Popen( [piffify_exe, 'pixel_des.yaml'] )
        p.communicate()
        print('read stars')
        stars, wcs, pointing = piff.Input.process(config['input'])
        print('read psf')
        psf = piff.read(psf_file)
        stars = [psf.model.initialize(s) for s in stars]
        flux = stars[0].fit.flux
        offset = stars[0].center_to_offset(stars[0].fit.center)
        fit_stamp = psf.draw(x=stars[0]['x'], y=stars[0]['y'], stamp_size=stamp_size,
                             flux=flux, offset=offset)
        orig_stamp = orig_image[stars[0].image.bounds] - stars[0]['sky']
        np.testing.assert_almost_equal(fit_stamp.array/flux, orig_stamp.array/flux, decimal=2)

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
    test_single_image()
    test_des_image()
    #pr.disable()
    #ps = pstats.Stats(pr).sort_stats('tottime')
    #ps.print_stats(20)
