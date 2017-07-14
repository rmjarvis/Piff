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
import warnings
import galsim
import numpy as np
import piff
import os
import subprocess
import yaml
import fitsio

from piff_test_helper import get_script_name, timer

fiducial_kolmogorov = galsim.Kolmogorov(half_light_radius=1.0)
mod = piff.GSObjectModel(fiducial_kolmogorov, force_model_center=False, include_pixel=False,
                         fastfit=True)

star_type = np.dtype([('u', float),
                      ('v', float),
                      ('hlr', float),
                      ('g1', float),
                      ('g2', float),
                      ('u0', float),
                      ('v0', float),
                      ('flux', float)])


def make_star(hlr, g1, g2, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
              nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Kolmogorov profile

    :param hlr:         The half_light_radius of the Kolmogorov.
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
    k = galsim.Kolmogorov(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2).shift(u0,v0)
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


def make_constant_psf_params(ntrain, nvalidate, nvisualize):
    """ Make training/testing data for a constant PSF.
    """

    bd = galsim.BaseDeviate(5647382910)
    ud = galsim.UniformDeviate(bd)

    training_data = np.recarray((ntrain,), dtype=star_type)
    validate_data = np.recarray((nvalidate,), dtype=star_type)

    hlr, g1, g2, u0, v0 = 0.4, 0.03, 0.06, 0.1, 0.2
    for i in range(ntrain):
        u = ud()
        v = ud()
        flux = ud()*50+100
        training_data[i] = np.array([u, v, hlr, g1, g2, u0, v0, flux])

    for i in range(nvalidate):
        u = ud()*0.5 + 0.25
        v = ud()*0.5 + 0.25
        flux = 1.0
        validate_data[i] = np.array([u, v, hlr, g1, g2, u0, v0, flux])

    vis_data = np.recarray((nvisualize, nvisualize), dtype=star_type)
    u = v = np.linspace(0, 1, nvisualize)
    u, v = np.meshgrid(u, v)
    vis_data['u'] = u
    vis_data['v'] = v
    vis_data['hlr'] = hlr
    vis_data['g1'] = g1
    vis_data['g2'] = g2
    vis_data['u0'] = u0
    vis_data['v0'] = v0
    vis_data['flux'] = flux

    return training_data, validate_data, vis_data


def make_polynomial_psf_params(ntrain, nvalidate, nvisualize):
    """ Make training/testing data for PSF with params varying as polynomials.
    """
    bd = galsim.BaseDeviate(5772156649+314159)
    ud = galsim.UniformDeviate(bd)

    training_data = np.recarray((ntrain,), dtype=star_type)
    validate_data = np.recarray((nvalidate,), dtype=star_type)

    # Make randomish Chebyshev polynomial coefficients
    # 5 Different arrays (hlr, g1, g2, u0, v0), and up to 3rd order in each of x and y.
    coefs = np.empty((4, 4, 5), dtype=float)
    for (i, j, k), _ in np.ndenumerate(coefs):
        coefs[i, j, k] = 2*ud() - 1.0

    for i in range(ntrain):
        u = ud()
        v = ud()
        flux = ud()*50+100
        vals = np.polynomial.chebyshev.chebval2d(u, v, coefs)/6  # range is [-0.5, 0.5]
        hlr = vals[0] * 0.1 + 0.35
        g1 = vals[1] * 0.1
        g2 = vals[2] * 0.1
        u0 = vals[3]
        v0 = vals[4]
        training_data[i] = np.array([u, v, hlr, g1, g2, u0, v0, flux])

    for i in range(nvalidate):
        u = ud()*0.5 + 0.25
        v = ud()*0.5 + 0.25
        flux = 1.0
        vals = np.polynomial.chebyshev.chebval2d(u, v, coefs)/6  # range is [-0.5, 0.5]
        hlr = vals[0] * 0.1 + 0.35
        g1 = vals[1] * 0.1
        g2 = vals[2] * 0.1
        u0 = vals[3]
        v0 = vals[4]
        validate_data[i] = np.array([u, v, hlr, g1, g2, u0, v0, flux])

    vis_data = np.recarray((nvisualize*nvisualize), dtype=star_type)
    u = v = np.linspace(0, 1, nvisualize)
    u, v = np.meshgrid(u, v)
    for i, (u1, v1) in enumerate(zip(u.ravel(), v.ravel())):
        vals = np.polynomial.chebyshev.chebval2d(u1, v1, coefs)/6  # range is [-0.5, 0.5]
        hlr = vals[0] * 0.1 + 0.35
        g1 = vals[1] * 0.1
        g2 = vals[2] * 0.1
        u0 = vals[3]
        v0 = vals[4]
        vis_data[i] = np.array([u1, v1, hlr, g1, g2, u0, v0, 1.0])

    return training_data, validate_data, vis_data.reshape((nvisualize, nvisualize))


def make_grf_psf_params(ntrain, nvalidate, nvisualize):
    """ Make training/testing data for PSF with params drawn from isotropic Gaussian random field.
    """
    bd = galsim.BaseDeviate(5772156649+2718281828)
    ud = galsim.UniformDeviate(bd)

    ntotal = ntrain + nvalidate + nvisualize**2
    params = np.recarray((ntotal,), dtype=star_type)

    # Training
    us = [ud() for i in range(ntrain)]
    vs = [ud() for i in range(ntrain)]
    fluxes = [ud()*50+100 for i in range(ntrain)]
    # Validate
    us += [ud()*0.5+0.25 for i in range(nvalidate)]
    vs += [ud()*0.5+0.25 for i in range(nvalidate)]
    fluxes += [1.0] * nvalidate
    # Visualize
    umesh, vmesh = np.meshgrid(np.linspace(0, 1, nvisualize), np.linspace(0, 1, nvisualize))
    us += list(umesh.ravel())
    vs += list(vmesh.ravel())
    fluxes += [1.0] * nvisualize**2

    # Next, generate input data by drawing from a single Gaussian Random Field.
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(np.array([us, vs]).T))
    cov = np.exp(-0.5*dists**2/0.3**2)  # Use 0.3 as arbitrary scale length.

    params['u'] = us
    params['v'] = vs
    # independently draw hlr, g1, g2, u0, v0
    np.random.seed(1234567890)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params['hlr'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05+0.6
        params['g1'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05
        params['g2'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05
        params['u0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3
        params['v0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3
    params['flux'] = fluxes

    training_data = params[:ntrain]
    validate_data = params[ntrain:ntrain+nvalidate]
    vis_data = params[ntrain+nvalidate:].reshape((nvisualize, nvisualize))

    return training_data, validate_data, vis_data


def make_anisotropic_grf_psf_params(ntrain, nvalidate, nvisualize):
    """ Make training/testing data for PSF with params drawn from anisotropic Gaussian random field.
    """
    bd = galsim.BaseDeviate(314159+2718281828)
    ud = galsim.UniformDeviate(bd)

    ntotal = ntrain + nvalidate + nvisualize**2
    params = np.recarray((ntotal,), dtype=star_type)

    # Training
    us = [ud() for i in range(ntrain)]
    vs = [ud() for i in range(ntrain)]
    fluxes = [ud()*50+100 for i in range(ntrain)]
    # Validate
    us += [ud()*0.5+0.25 for i in range(nvalidate)]
    vs += [ud()*0.5+0.25 for i in range(nvalidate)]
    fluxes += [1.0] * nvalidate
    # Visualize
    umesh, vmesh = np.meshgrid(np.linspace(0, 1, nvisualize), np.linspace(0, 1, nvisualize))
    us += list(umesh.ravel())
    vs += list(vmesh.ravel())
    fluxes += [1.0] * nvisualize**2

    # Next, generate input data by drawing from a single Gaussian Random Field.
    from scipy.spatial.distance import pdist, squareform
    # Use an anisotropic covariance.
    var1 = 0.1**2
    var2 = 0.2**2
    corr = 0.7
    cov = np.array([[var1, np.sqrt(var1*var2)*corr],
                    [np.sqrt(var1*var2)*corr, var2]])

    dists = squareform(pdist(np.array([us, vs]).T, metric='mahalanobis', VI=np.linalg.inv(cov)))
    bigcov = np.exp(-0.5*dists**2)

    params['u'] = us
    params['v'] = vs
    # independently draw hlr, g1, g2, u0, v0
    np.random.seed(1234567890)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params['hlr'] = np.random.multivariate_normal([0]*ntotal, bigcov)*0.05+0.5
        params['g1'] = np.random.multivariate_normal([0]*ntotal, bigcov)*0.05
        params['g2'] = np.random.multivariate_normal([0]*ntotal, bigcov)*0.05
        params['u0'] = np.random.multivariate_normal([0]*ntotal, bigcov)*0.1
        params['v0'] = np.random.multivariate_normal([0]*ntotal, bigcov)*0.1
    params['flux'] = fluxes

    training_data = params[:ntrain]
    validate_data = params[ntrain:ntrain+nvalidate]
    vis_data = params[ntrain+nvalidate:].reshape((nvisualize, nvisualize))

    return training_data, validate_data, vis_data


def params_to_stars(params, noise=0.0, rng=None):
    stars = []
    for param in params.ravel():
        u, v, hlr, g1, g2, u0, v0, flux = param
        s = make_star(hlr, g1, g2, u0, v0, flux, noise=noise, du=0.2, fpu=u, fpv=v, rng=rng)
        s = mod.initialize(s)
        stars.append(s)
    return stars


def iterate(stars, interp):
    """Iteratively improve the global PSF model.
    """
    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
    print()
    print()
    print("Initial state")
    print("-------------")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    oldchisq = 0.
    print()
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        stars = interp.interpolateList(stars)
        for i, s in enumerate(stars):
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print("iteration: {}  chisq: {}  dof: {}".format(iteration, chisq, dof))
        if oldchisq>0 and abs(oldchisq-chisq) < dof/10.0:
            break
        else:
            oldchisq = chisq
    print(interp.gp.kernel_)


def display_old(training_data, vis_data, interp):
    """Display training samples, true PSF, model PSF, and residual over field-of-view.
    """
    interpstars = params_to_stars(vis_data, noise=0.0)

    # Make a grid of output locations to visualize GP interpolation performance (for g1).
    ctruth = np.array(vis_data['g1']).ravel()
    interpstars = interp.interpolateList(interpstars)
    cinterp = np.array([s.fit.params[3] for s in interpstars])

    vmin = np.min(ctruth)
    vmax = np.max(ctruth)
    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 3))
    ax1 = fig.add_subplot(141)
    ax1.set_title("sampling")
    ax1.set_xlim((-0.2,1.2))
    ax1.set_ylim((-0.2,1.2))
    meas = ax1.scatter(training_data['u'], training_data['v'],
                       c=training_data['g1'], vmin=vmin, vmax=vmax)

    plt.colorbar(meas)

    ax2 = fig.add_subplot(142)
    ax2.set_title("truth")
    ax2.set_xlim((-0.2,1.2))
    ax2.set_ylim((-0.2,1.2))
    truth = ax2.scatter(vis_data['u'], vis_data['v'], c=ctruth, vmin=vmin, vmax=vmax)
    plt.colorbar(truth)

    ax3 = fig.add_subplot(143)
    ax3.set_title("interp")
    ax3.set_xlim((-0.2,1.2))
    ax3.set_ylim((-0.2,1.2))
    interp_scat = ax3.scatter(vis_data['u'], vis_data['v'], c=cinterp, vmin=vmin, vmax=vmax)
    plt.colorbar(interp_scat)

    ax4 = fig.add_subplot(144)
    ax4.set_title("resid")
    ax4.set_xlim((-0.2,1.2))
    ax4.set_ylim((-0.2,1.2))
    resid = ax4.scatter(vis_data['u'], vis_data['v'], c=(cinterp-ctruth),
                        vmin=vmin/10, vmax=vmax/10)
    plt.colorbar(resid)
    plt.show()


def display(training_data, vis_data, interp):
    """Display training samples, true PSF, model PSF, and residual over field-of-view.
    """
    import matplotlib.pyplot as plt
    interpstars = params_to_stars(vis_data, noise=0.0)
    interpstars = interp.interpolateList(interpstars)


    fig, axarr = plt.subplots(5, 4, figsize=(7, 10))

    rows = ['u0', 'v0', 'hlr', 'g1', 'g2']
    for irow, var in enumerate(rows):
        # Make a grid of output locations to visualize GP interpolation performance (for g1).
        ctruth = np.array(vis_data[var]).ravel()
        cinterp = np.array([s.fit.params[irow] for s in interpstars])

        vmin = np.min(ctruth)
        vmax = np.max(ctruth)
        if vmin == vmax:
            vmin -= 0.01
            vmax += 0.01

        ax1 = axarr[irow, 0]
        ax1.set_title("sampling")
        ax1.set_xlim((-0.2,1.2))
        ax1.set_ylim((-0.2,1.2))
        ax1.scatter(training_data['u'], training_data['v'],
                    c=training_data[var], vmin=vmin, vmax=vmax)

        ax2 = axarr[irow, 1]
        ax2.set_title("truth")
        ax2.set_xlim((-0.2,1.2))
        ax2.set_ylim((-0.2,1.2))
        ax2.scatter(vis_data['u'], vis_data['v'], c=ctruth, vmin=vmin, vmax=vmax)

        ax3 = axarr[irow, 2]
        ax3.set_title("interp")
        ax3.set_xlim((-0.2,1.2))
        ax3.set_ylim((-0.2,1.2))
        ax3.scatter(vis_data['u'], vis_data['v'], c=cinterp, vmin=vmin, vmax=vmax)

        ax4 = axarr[irow, 3]
        ax4.set_title("resid")
        ax4.set_xlim((-0.2,1.2))
        ax4.set_ylim((-0.2,1.2))
        ax4.scatter(vis_data['u'], vis_data['v'], c=(cinterp-ctruth),
                            vmin=vmin/10, vmax=vmax/10)

    for ax in axarr.ravel():
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    plt.show()


def validate(validate_stars, interp):
    """ Check that global PSF model sufficiently interpolates some stars.
    """
    # Noiseless copy of the PSF at the center of the FOV
    for s0 in validate_stars:
        s0 = mod.initialize(s0)
        s0 = mod.fit(s0)
        s0 = mod.reflux(s0)

        s1 = interp.interpolate(s0)
        s1 = mod.reflux(s1)
        print()
        print(("s0: "+"{:< 8.4f} "*len(s0.fit.params)).format(*s0.fit.params))
        print(("s1: "+"{:< 8.4f} "*len(s1.fit.params)).format(*s1.fit.params))
        print("s0 flux:", s0.fit.flux)
        print("s1 flux:", s1.fit.flux)
        print()
        print('Flux, ctr, chisq after interpolation: \n', s1.fit.flux, s1.fit.center, s1.fit.chisq)
        np.testing.assert_allclose(s1.fit.flux, s0.fit.flux, rtol=1e-2)

        s1 = mod.draw(s1)
        print()
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        print('min rtol = ', np.max(np.abs(s1.image.array - s0.image.array)/s0.image.array.max()))
        np.testing.assert_allclose(s1.image.array, s0.image.array,
                                   rtol=0, atol=s0.image.array.max()*0.01)

        if False:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(s0.image.array)
            axes[1].imshow(s1.image.array)
            axes[2].imshow(s1.image.array - s0.image.array)
            plt.show()


def check_gp(training_data, validation_data, visualization_data,
             kernel, npca=0, optimize=False, file_name=None, rng=None,
             visualize=False, check_config=False):
    """ Solve for global PSF model, test it, and optionally display it.
    """
    stars = params_to_stars(training_data, noise=0.03, rng=rng)
    validate_stars = params_to_stars(validation_data, noise=0.0, rng=rng)
    interp = piff.GPInterp(kernel=kernel, optimize=optimize, npca=npca)
    interp.initialize(stars)
    iterate(stars, interp)
    if visualize:
        display(training_data, visualization_data, interp)
    validate(validate_stars, interp)

    if check_config:
        config = {
            'interp' : {
                'type' : 'GPInterp',
                'kernel' : kernel,
                'npca' : npca,
                'optimize' : optimize
            }
        }
        logger = piff.config.setup_logger()
        interp3 = piff.Interp.process(config['interp'], logger)
        iterate(stars, interp3)
        validate(validate_stars, interp3)

    # Check that we can write interp to disk and read back in.
    if file_name is not None:
        testfile = os.path.join('output', file_name)
        with fitsio.FITS(testfile, 'rw', clobber=True) as f:
            interp.write(f, 'interp')
        with fitsio.FITS(testfile, 'r') as f:
            interp2 = piff.GPInterp.read(f, 'interp')
        print("Revalidating after i/o.")
        X = np.vstack([training_data['u'], training_data['v']]).T
        np.testing.assert_allclose(interp.gp.kernel(X), interp2.gp.kernel(X))
        np.testing.assert_allclose(interp.gp.kernel.theta, interp2.gp.kernel.theta)
        np.testing.assert_allclose(interp.gp.kernel_.theta, interp2.gp.kernel_.theta)
        np.testing.assert_allclose(interp.gp.alpha_, interp2.gp.alpha_, rtol=1e-6, atol=1.e-7)
        np.testing.assert_allclose(interp.gp.X_train_, interp2.gp.X_train_)
        np.testing.assert_allclose(interp.gp.y_train_mean, interp2.gp.y_train_mean)
        validate(validate_stars, interp2)


@timer
def test_constant_psf():
    rng = galsim.BaseDeviate(572958179)
    ntrain, nvalidate, nvisualize = 100, 1, 21
    training_data, validation_data, visualization_data = \
        make_constant_psf_params(ntrain, nvalidate, nvisualize)

    kernel = "1*RBF(1.0, (1e-1, 1e3))"
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5...
    kernel += " + WhiteKernel(1e-5, (1e-7, 1e-1))"

    if __name__ == '__main__':
        npcas = [0, 2]
        optimizes = [True, False]
    else:
        npcas = [0]
        optimizes = [False]

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data, kernel,
                     npca=npca, optimize=optimize, rng=rng, check_config=True)


@timer
def test_polynomial_psf():
    rng = galsim.BaseDeviate(1203985)
    ntrain, nvalidate, nvisualize = 200, 1, 21
    training_data, validation_data, visualization_data = \
        make_polynomial_psf_params(ntrain, nvalidate, nvisualize)
    kernel = "1*RBF(0.3, (1e-1, 1e3))"
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5, so add that amount of
    # white noise
    kernel += " + WhiteKernel(1e-5, (1e-7, 1e-1))"

    if __name__ == '__main__':
        npcas = [0, 2]
        optimizes = [True, False]
    else:
        npcas = [0]
        optimizes = [True]

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data, kernel,
                     npca=npca, optimize=optimize, rng=rng)


@timer
def test_grf_psf():
    rng = galsim.BaseDeviate(987654334587656)
    ntrain, nvalidate, nvisualize = 100, 1, 21
    training_data, validation_data, visualization_data = \
        make_grf_psf_params(ntrain, nvalidate, nvisualize)

    kernel = "1*RBF(0.3, (1e-1, 1e1))"
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5, so add that amount of
    # white noise
    kernel += " + WhiteKernel(1e-5, (1e-7, 1e-1))"

    if __name__ == '__main__':
        npcas = [0, 5]
        optimizes = [True, False]
        check_config = True
    else:
        npcas = [0]
        optimizes = [False]
        check_config = False

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data, kernel,
                     npca=npca, optimize=optimize, file_name="test_gp_grf.fits", rng=rng,
                     check_config=check_config)

    # Check ExplicitKernel here too
    #
    # We could in principal use any function of dx, dy here, for instance a galsim.LookupTable2D.
    # For simplicity, though, just assert a Gaussian == SquaredExponential with scale-length
    # of 0.3.
    kernel = "ExplicitKernel('np.exp(-0.5*(du**2+dv**2)/0.3**2)')"
    kernel += " + WhiteKernel(1e-5)"
    # No optimize loop, since ExplicitKernel is not optimizable.
    for npca in npcas:
        check_gp(training_data, validation_data, visualization_data, kernel,
                 npca=npca, file_name="test_explicit_grf.fits", rng=rng,
                 check_config=check_config)

    # Try out an AnisotropicRBF on the isotropic data too.
    kernel = "1*AnisotropicRBF(scale_length=[0.3, 0.3])"
    kernel += " + WhiteKernel(1e-5)"
    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data, kernel,
                     npca=npca, optimize=optimize,
                     file_name="test_aniso_isotropic_grf.fits", rng=rng,
                     check_config=check_config)


@timer
def test_anisotropic_rbf_kernel():
    rng = galsim.BaseDeviate(5867943)
    ntrain, nvalidate, nvisualize = 250, 1, 21
    training_data, validation_data, visualization_data = \
        make_anisotropic_grf_psf_params(ntrain, nvalidate, nvisualize)
    var1 = 0.1**2
    var2 = 0.2**2
    corr = 0.7
    cov = np.array([[var1, np.sqrt(var1*var2)*corr],
                    [np.sqrt(var1*var2)*corr, var2]])
    invLam = np.linalg.inv(cov)

    kernel = "0.1*AnisotropicRBF(invLam={0!r})".format(invLam)
    kernel += "+ WhiteKernel(1e-5, (1e-7, 1e-2))"

    print(kernel)

    if __name__ == '__main__':
        npcas = [0, 5]
        optimizes = [True, False]
        check_config = True
    else:
        npcas = [0]
        optimizes = [False]
        check_config = False

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data, kernel,
                     npca=npca, optimize=optimize, file_name="test_anisotropic_rbf.fits",
                     rng=rng, check_config=check_config)


@timer
def test_yaml():
    # Take DES test image, and test doing a psf run with GP interpolator
    # Use config parser:
    psf_file = os.path.join('output','gp_psf.fits')
    config = {
        'input' : {
            # These can be regular strings
            'image_file_name' : 'input/DECam_00241238_01.fits.fz',
            # Or any GalSim str value type.  e.g. FormattedStr
            'cat_file_name' : {
                'type': 'FormattedStr',
                'format': '%s/DECam_%08d_%02d_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits',
                'items': [
                    'input',    # dir
                    241238,     # expnum
                    1           # chipnum
                ]
            },

            # What hdu is everything in?
            'image_hdu' : 1,
            'badpix_hdu' : 2,
            'weight_hdu' : 3,
            'cat_hdu' : 2,

            # What columns in the catalog have things we need?
            'x_col' : 'XWIN_IMAGE',
            'y_col' : 'YWIN_IMAGE',
            'ra' : 'TELRA',
            'dec' : 'TELDEC',
            'gain' : 'GAINA',
            'sky_col' : 'BACKGROUND',

            # How large should the postage stamp cutouts of the stars be?
            'stamp_size' : 31,
        },
        'psf' : {
            'model' : { 'type' : 'GSObjectModel',
                        'fastfit' : True,
                        'gsobj' : 'galsim.Gaussian(sigma=1.0)' },
            'interp' : { 'type' : 'GPInterp',
                         'keys' : ['u', 'v'],
                         'kernel' : 'RBF(200.0)',
                         'optimize' : False,}
        },
        'output' : { 'file_name' : psf_file },
    }

    # using piffify executable
    config['verbose'] = 0
    with open('gp.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'gp.yaml'] )
    p.communicate()
    piff.read(psf_file)

    # Doesn't actually check results, just checks that everything runs.


@timer
def test_anisotropic_limit():
    """Test that AnisotropicRBF with isotropic covariance equals RBF"""

    kernel1 = "RBF(0.45)"
    kernel2 = "AnisotropicRBF(scale_length=[0.45, 0.45])"

    gp1 = piff.GPInterp(kernel=kernel1)
    gp2 = piff.GPInterp(kernel=kernel2)

    X = np.random.rand(1000, 2)
    np.testing.assert_allclose(gp1.gp.kernel(X), gp2.gp.kernel(X))


@timer
def test_guess():
    rng = galsim.BaseDeviate(8675309)
    ntrain, nvalidate, nvisualize = 100, 1, 21
    training_data, validation_data, visualization_data = \
        make_grf_psf_params(ntrain, nvalidate, nvisualize)

    inferred_scale_length = []
    if __name__ == '__main__':
        guesses =  [0.03, 0.1, 0.3, 1.0, 3.0]
        rtol = 0.02
    else:
        guesses = [0.03, 0.3, 3.0]
        rtol = 0.03
    for guess in guesses:
        # noise of 0.3 turns out to be pretty significant here.
        stars = params_to_stars(training_data, noise=0.3, rng=rng)
        kernel = "1*RBF({0}, (1e-1, 1e1))".format(guess)
        kernel += " + WhiteKernel(1e-5, (1e-7, 1e-1))"
        interp = piff.GPInterp(kernel=kernel, normalize=False)
        stars = [mod.fit(s) for s in stars]
        stars = interp.initialize(stars)
        interp.solve(stars)

        # A bit complicated, but this extracts the scale-length
        inferred_scale_length.append(np.exp(interp.gp.kernel_.theta[1]))

    # Check that the inferred scale length is close to the input value of 0.3
    np.testing.assert_allclose(inferred_scale_length, 0.3, rtol=0.15)
    # More interesting however, is how independent is the optimization wrt the initial value.
    # So check that the standard deviation of the results is much smaller than the value.
    np.testing.assert_array_less(np.std(inferred_scale_length), 0.3*rtol)


@timer
def test_anisotropic_guess():
    rng = galsim.BaseDeviate(8675309)
    # ntrain, nvalidate, nvisualize = 100, 1, 1
    # training_data, validation_data, visualization_data = \
    #     make_grf_psf_params(ntrain, nvalidate, nvisualize)
    ntrain, nvalidate, nvisualize = 100, 1, 1
    training_data, validation_data, visualization_data = \
        make_anisotropic_grf_psf_params(ntrain, nvalidate, nvisualize)

    var1s = []
    var2s = []
    corrs = []

    if __name__ == '__main__':
        guesses =  [0.03, 0.1, 0.3, 1.0, 3.0]
        rtol = 0.05
    else:
        guesses = [0.03, 0.3, 3.0]
        rtol = 0.10
    for guess in guesses:
        # noise of 0.3 turns out to be pretty significant here.
        stars = params_to_stars(training_data, noise=0.03, rng=rng)
        kernel = "1*AnisotropicRBF(scale_length={0!r})".format([guess, guess])
        kernel += " + WhiteKernel(1e-5, (1e-7, 1e-1))"
        interp = piff.GPInterp(kernel=kernel)
        stars = [mod.fit(s) for s in stars]
        stars = interp.initialize(stars)
        interp.solve(stars)

        invLam = interp.gp.kernel_.get_params()['k1__k2__invLam']
        Lam = np.linalg.inv(invLam)
        var1s.append(Lam[0, 0])
        var2s.append(Lam[1, 1])
        corrs.append(Lam[0, 1] / np.sqrt(Lam[0, 0]*Lam[1, 1]))
        print(var1s[-1], var2s[-1], corrs[-1])
    # Check that the inferred correlation is close to the input correlation with params:
    # var1 = 0.1**2, var2 = 0.2**2, corr = 0.7
    np.testing.assert_allclose(var1s, 0.1**2, rtol=1.0)  # Only get right order-of-magnitude or so
    np.testing.assert_allclose(var2s, 0.2**2, rtol=1.0)  # Only get right order-of-magnitude or so
    np.testing.assert_allclose(corrs, 0.7, rtol=0.1)  # This one works much better
    # More interesting however, is how independent is the optimization wrt the initial value.
    # So check that the standard deviation of the results is small.
    np.testing.assert_array_less(np.std(var1s), 0.1**2*rtol)
    np.testing.assert_array_less(np.std(var2s), 0.2**2*rtol)
    np.testing.assert_array_less(np.std(corrs), 0.7*rtol)


if __name__ == '__main__':
    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    test_constant_psf()
    test_polynomial_psf()
    test_grf_psf()
    test_anisotropic_rbf_kernel()
    test_yaml()
    test_anisotropic_limit()
    test_guess()
    test_anisotropic_guess()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('tottime')
    # ps.print_stats(25)
