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
import piff

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


def check_gp_poly(npca=0):
    """ Test that Gaussian Process interpolation works reasonably well for densely packed Kolmogorov
    star field with params that vary as polynomials.
    """
    bd = galsim.BaseDeviate(5647382910)
    ud = galsim.UniformDeviate(bd)
    nstars = 100
    upositions = [ud() for i in xrange(nstars)]
    vpositions = [ud() for i in xrange(nstars)]
    fluxes = [ud()*100.0 + 50 for i in xrange(nstars)]  # Uniform [50, 150]

    g1_fn = lambda u,v: 0.1*u - 0.1*v
    g2_fn = lambda u,v: -0.1*v + 0.1*u*v
    fwhm_fn = lambda u,v: 1.0-0.05*u+0.05*v
    u0_fn = lambda u,v: 0.5*u
    v0_fn = lambda u,v: 0.3*u+0.3*v

    mod = piff.Kolmogorov(force_model_center=False)  # Center is part of the PSF params.

    stars = []
    for u, v, flux in zip(upositions, vpositions, fluxes):
        s = make_kolmogorov_data(fwhm_fn(u,v), g1_fn(u,v), g2_fn(u,v), u0_fn(u,v), v0_fn(u,v), flux,
                                 noise=0.1, du=0.5, fpu=u, fpv=v, rng=bd)
        s = mod.initialize(s)
        stars.append(s)

    # Get noiseless copy of the PSF at the center of the FOV
    u,v = 0.5, 0.5
    s0 = make_kolmogorov_data(fwhm_fn(u,v), g1_fn(u,v), g2_fn(u,v), u0_fn(u,v), v0_fn(u,v), 1.0,
                              du=0.5, fpu=u, fpv=v)
    s0 = mod.initialize(s0)

    # theta is the inverse correlation length.  Sadly, the default settings for this in sklearn
    # lead to an exception for this dataset.  Fortunately, we can pass in an initial value (theta0)
    # and some bounds (thetaL and thetaU), and the GP framework will solve for the best value.
    interp = piff.GPInterp(thetaL=1e-6, theta0=1e0, thetaU=1e6, nugget=1e-4, npca=npca)
    interp.initialize(stars)

    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
    print()
    print("Initial state")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    # Go through and look for bad outliers that may confuse the GP interpolator
    old_stars = stars
    stars = []
    for i, s in enumerate(old_stars):
        # arbitrary cutoff, but for simulated data should be reasonable.
        if (s.fit.chisq / s.fit.dof) > 4:
            # print("star {0} has chisq/dof = {1}".format(i, s.fit.chisq/s.fit.dof))
            # try fitting again with lmfit instead of hsm...
            s1 = piff.Star(s.data, piff.StarFit(None))
            s1 = mod.fit(s1)
            s1 = mod.reflux(s1)
            # print("after refit, chisq/dof = {0}".format(s1.fit.chisq/s1.fit.dof))
            # if still bad, break loop, excluding star from further processing
            if (s1.fit.chisq / s1.fit.dof) > 4:
                # print("rejecting star {0}".format(i))
                continue
            stars.append(s1)
        else:
            # chisq/dof is okay, so keep this star.
            stars.append(s)

    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
        if (s.fit.chisq / s.fit.dof) > 4:
            print("chisq/dof = {0}".format(s.fit.chisq / s.fit.dof))
    print()
    print("After refitting / outlier rejection")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    oldchisq = 0.
    print()
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars)
        print('theta_ = ', interp.gp.theta_)
        if npca > 0:
            print('explained_variance_ratio = ', np.cumsum(interp._pca.explained_variance_ratio_))

        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        stars = interp.interpolateList(stars)
        for i, s in enumerate(stars):
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print("iteration: {}  chisq: {}  dof: {}".format(iteration, chisq, dof))
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.0:
            break
        else:
            oldchisq = chisq

    visualize = False
    if visualize:
        # Make a grid of output locations to visualize GP interpolation performance (for g1).
        uposout = np.linspace(0.0, 1.0, 21)
        vposout = np.linspace(0.0, 1.0, 21)
        uposout, vposout = np.meshgrid(uposout, vposout)
        interpstars = []
        for u, v in zip(uposout.ravel(), vposout.ravel()):
            interpstars.append(
                    make_kolmogorov_data(1.0, 0.0, 0.0, 0.0, 0.0, 1.0,du=0.5, fpu=u, fpv=v))
        interpstars = interp.interpolateList(interpstars)
        truth_g1 = [g1_fn(u, v) for (u, v) in zip(uposout.ravel(), vposout.ravel())]
        interp_g1 = [s.fit.params[3] for s in interpstars]

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 3))
        ax1 = fig.add_subplot(141)
        ax1.set_title("sampling")
        ax1.set_xlim((-0.2,1.2))
        ax1.set_ylim((-0.2,1.2))
        meas = ax1.scatter(upositions, vpositions,
                           c=[g1_fn(u,v) for u,v in zip(upositions, vpositions)],
                           vmin=-0.1, vmax=0.1)
        plt.colorbar(meas)

        ax2 = fig.add_subplot(142)
        ax2.set_title("truth")
        ax2.set_xlim((-0.2,1.2))
        ax2.set_ylim((-0.2,1.2))
        truth = ax2.scatter(uposout, vposout, c=truth_g1, vmin=-0.1, vmax=0.1)
        plt.colorbar(truth)

        ax3 = fig.add_subplot(143)
        ax3.set_title("interp")
        ax3.set_xlim((-0.2,1.2))
        ax3.set_ylim((-0.2,1.2))
        interp_scat = ax3.scatter(uposout, vposout, c=interp_g1, vmin=-0.1, vmax=0.1)
        plt.colorbar(interp_scat)

        ax4 = fig.add_subplot(144)
        ax4.set_title("resid")
        ax4.set_xlim((-0.2,1.2))
        ax4.set_ylim((-0.2,1.2))
        resid = ax4.scatter(uposout, vposout, c=[i-t for i,t in zip(interp_g1, truth_g1)],
                            vmin=-0.001, vmax=0.001)
        plt.colorbar(resid)
        plt.show()


    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print()
    print(("s0: "+"{:< 8.4f} "*len(s0.fit.params)).format(*s0.fit.params))
    print(("s1: "+"{:< 8.4f} "*len(s1.fit.params)).format(*s1.fit.params))
    print()
    print('Flux, ctr, chisq after interpolation: \n', s1.fit.flux, s1.fit.center, s1.fit.chisq)
    np.testing.assert_allclose(s1.fit.flux, 1.0, rtol=1e-3)

    s1 = mod.draw(s1)
    print()
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    print('min rtol = ', np.max(np.abs(s1.image.array - s0.image.array)/np.abs(s0.image.array)))
    np.testing.assert_allclose(s1.image.array, s0.image.array, rtol=1e-2)


def test_gp_poly():
    check_gp_poly(npca=0)
    check_gp_poly(npca=2)  # For this test, 2 PCs already describe ~99.7% of the variance.


def check_gp_gp(npca=0):
    """ Test that Gaussian Process interpolation works reasonably well for densely packed Kolmogorov
    star field with params that are also drawn from a Gaussian Process.
    """
    bd = galsim.BaseDeviate(1029384756)
    grid_spacing = 1
    ngrid = 21

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        ps = galsim.PowerSpectrum(lambda k:3e-4*k**(-5), lambda k:3e-4*k**(-5))
    ps.buildGrid(grid_spacing=grid_spacing, ngrid=ngrid, rng=bd)

    gridmin = (-ngrid/2. + 0.5) * grid_spacing
    gridmax = (ngrid/2. - 0.5) * grid_spacing
    uposout, vposout = np.meshgrid(np.arange(gridmin, gridmax+grid_spacing, grid_spacing),
                                   np.arange(gridmin, gridmax+grid_spacing, grid_spacing))
    g1_out, g2_out = ps.getShear((uposout.ravel(), vposout.ravel()))

    bd = galsim.BaseDeviate(5647382910)
    ud = galsim.UniformDeviate(bd)
    nstars = 200
    upositions = [ud()*20-10 for i in xrange(nstars)]
    vpositions = [ud()*20-10 for i in xrange(nstars)]
    fluxes = [ud()*100.0 + 50 for i in xrange(nstars)]  # Uniform [50, 150]
    g1s, g2s = ps.getShear((upositions, vpositions))

    mod = piff.Kolmogorov(force_model_center=False)  # Center is part of the PSF params.

    stars = []
    for u, v, g1, g2, flux in zip(upositions, vpositions, g1s, g2s, fluxes):
        s = make_kolmogorov_data(1.0, g1, g2, 0., 0., flux,
                                 noise=0.1, du=0.5, fpu=u, fpv=v, rng=bd)
        s = mod.initialize(s)
        stars.append(s)

    # Get noiseless copy of the PSF at the center of the FOV
    u,v = 0.0, 0.0
    g1_0, g2_0 = ps.getShear((0,0))
    s0 = make_kolmogorov_data(1.0, g1_0, g2_0, 0.0, 0.0, 1.0, du=0.5, fpu=u, fpv=v)
    s0 = mod.initialize(s0)

    interp = piff.GPInterp(thetaL=1e-6, theta0=1e0, thetaU=1e6, nugget=1e-4, npca=npca)
    interp.initialize(stars)

    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
    print()
    print("Initial state")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    # Go through and look for bad outliers that may confuse the GP interpolator
    old_stars = stars
    stars = []
    for i, s in enumerate(old_stars):
        # arbitrary cutoff, but for simulated data should be reasonable.
        if (s.fit.chisq / s.fit.dof) > 4:
            # print("star {0} has chisq/dof = {1}".format(i, s.fit.chisq/s.fit.dof))
            # try fitting again with lmfit instead of hsm...
            s1 = piff.Star(s.data, piff.StarFit(None))
            s1 = mod.fit(s1)
            s1 = mod.reflux(s1)
            # print("after refit, chisq/dof = {0}".format(s1.fit.chisq/s1.fit.dof))
            # if still bad, break loop, excluding star from further processing
            if (s1.fit.chisq / s1.fit.dof) > 4:
                # print("rejecting star {0}".format(i))
                continue
            stars.append(s1)
        else:
            # chisq/dof is okay, so keep this star.
            stars.append(s)

    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
        if (s.fit.chisq / s.fit.dof) > 4:
            print("chisq/dof = {0}".format(s.fit.chisq / s.fit.dof))
    print()
    print("After refitting / outlier rejection")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    oldchisq = 0.
    print()
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars)
        print('theta_ = ', interp.gp.theta_)
        if npca > 0:
            print('explained_variance_ratio = ', np.cumsum(interp._pca.explained_variance_ratio_))

        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        stars = interp.interpolateList(stars)
        for i, s in enumerate(stars):
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print("iteration: {}  chisq: {}  dof: {}".format(iteration, chisq, dof))
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.0:
            break
        else:
            oldchisq = chisq

    visualize = False
    if visualize:
        # Make a grid of output locations to visualize GP interpolation performance (for g1).
        interpstars = []
        for u, v in zip(uposout.ravel(), vposout.ravel()):
            interpstars.append(
                    make_kolmogorov_data(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, du=0.5, fpu=u, fpv=v))
        interpstars = interp.interpolateList(interpstars)
        interp_g1 = [s.fit.params[3] for s in interpstars]

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 3))
        ax1 = fig.add_subplot(141)
        ax1.set_title("sampling")
        ax1.set_xlim((-14,14))
        ax1.set_ylim((-14,14))
        meas = ax1.scatter(upositions, vpositions, c=g1s, vmin=-0.05, vmax=0.05)
        plt.colorbar(meas)

        ax2 = fig.add_subplot(142)
        ax2.set_title("truth")
        ax2.set_xlim((-14,14))
        ax2.set_ylim((-14,14))
        truth = ax2.scatter(uposout, vposout, c=g1_out, vmin=-0.05, vmax=0.05)
        plt.colorbar(truth)

        ax3 = fig.add_subplot(143)
        ax3.set_title("interp")
        ax3.set_xlim((-14,14))
        ax3.set_ylim((-14,14))
        interp_scat = ax3.scatter(uposout, vposout, c=interp_g1, vmin=-0.05, vmax=0.05)
        plt.colorbar(interp_scat)

        ax4 = fig.add_subplot(144)
        ax4.set_title("resid")
        ax4.set_xlim((-14,14))
        ax4.set_ylim((-14,14))
        resid = ax4.scatter(uposout.ravel(), vposout.ravel(),
                            c=[i-t for i,t in zip(interp_g1, g1_out)],
                            vmin=-0.01, vmax=0.01)
        plt.colorbar(resid)
        plt.show()

    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print()
    print(("s0: "+"{:< 8.4f} "*len(s0.fit.params)).format(*s0.fit.params))
    print(("s1: "+"{:< 8.4f} "*len(s1.fit.params)).format(*s1.fit.params))
    print()
    print('Flux, ctr, chisq after interpolation: \n', s1.fit.flux, s1.fit.center, s1.fit.chisq)
    np.testing.assert_allclose(s1.fit.flux, 1.0, rtol=1e-3)

    s1 = mod.draw(s1)
    print()
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    print('min rtol = ', np.max(np.abs(s1.image.array - s0.image.array)/np.abs(s0.image.array)))
    np.testing.assert_allclose(s1.image.array, s0.image.array, rtol=1e-1)


def test_gp_gp():
    check_gp_gp(npca=0)
    check_gp_gp(npca=2)  # capture 99.3% of the variance in 2 PCs


if __name__ == '__main__':
    test_gp_poly()
    test_gp_gp()
