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
from sklearn.gaussian_process import kernels


fiducial_kolmogorov = galsim.Kolmogorov(half_light_radius=1.0)

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


def make_constant_psf_params(ntrain, nvalidate, nvis):
    # every data set defined on unit square field-of-view.
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

    vis_data = np.recarray((nvis, nvis), dtype=star_type)
    u = v = np.linspace(0, 1, nvis)
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


def make_polynomial_psf_params(ntrain, nvalidate, nvis):
    # every data set defined on unit square field-of-view.
    bd = galsim.BaseDeviate(5772156649)
    ud = galsim.UniformDeviate(bd)

    training_data = np.recarray((ntrain,), dtype=star_type)
    validate_data = np.recarray((nvalidate,), dtype=star_type)

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

    vis_data = np.recarray((nvis*nvis), dtype=star_type)
    u = v = np.linspace(0, 1, nvis)
    u, v = np.meshgrid(u, v)
    for i, (u1, v1) in enumerate(zip(u.ravel(), v.ravel())):
        vals = np.polynomial.chebyshev.chebval2d(u1, v1, coefs)/6  # range is [-0.5, 0.5]
        hlr = vals[0] * 0.1 + 0.35
        g1 = vals[1] * 0.1
        g2 = vals[2] * 0.1
        u0 = vals[3]
        v0 = vals[4]
        vis_data[i] = np.array([u1, v1, hlr, g1, g2, u0, v0, 1.0])

    return training_data, validate_data, vis_data.reshape((nvis, nvis))


def make_grf_psf_params(ntrain, nvalidate, nvis):
    # Gaussian Random Field data.
    bd = galsim.BaseDeviate(5772156649)
    ud = galsim.UniformDeviate(bd)

    ntotal = ntrain + nvalidate + nvis**2
    params = np.recarray((ntotal,), dtype=star_type)

    # Training
    us = [ud() for i in range(ntrain)]
    vs = [ud() for i in range(ntrain)]
    fluxes = [ud()*50+100 for i in range(ntrain)]
    # Validate
    us += [ud()*0.5+0.25 for i in range(nvalidate)]
    vs += [ud()*0.5+0.25 for i in range(nvalidate)]
    fluxes += [1.0]
    # Visualize
    umesh, vmesh = np.meshgrid(np.linspace(0, 1, nvis), np.linspace(0, 1, nvis))
    us += list(umesh.ravel())
    vs += list(vmesh.ravel())
    fluxes += [1.0] * nvis**2

    # Next, generate input data by drawing from a single Gaussian Random Field.
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(np.array([us, vs]).T))
    cov = np.exp(-0.5*dists**2/0.3**2)  # Use 0.3 as arbitrary scale length.

    params['u'] = us
    params['v'] = vs
    # independently draw hlr, g1, g2, u0, v0
    np.random.seed(1234567890)
    params['hlr'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05+0.4
    params['g1'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05
    params['g2'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05
    params['u0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3
    params['v0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3
    params['flux'] = fluxes

    training_data = params[:ntrain]
    validate_data = params[ntrain:ntrain+nvalidate]
    vis_data = params[ntrain+nvalidate:].reshape((nvis, nvis))

    return training_data, validate_data, vis_data


def params_to_stars(params, mod, noise=0.0, rng=None):
    stars = []
    for param in params.ravel():
        u, v, hlr, g1, g2, u0, v0, flux = param
        # I think the following line is bounded between
        s = make_star(hlr, g1, g2, u0, v0, flux, noise=noise, du=0.2, fpu=u, fpv=v, rng=rng)
        s = mod.initialize(s)
        stars.append(s)
    return stars


def iterate(stars, mod, interp):
    chisq = 0.0
    dof = 0
    for s in stars:
        chisq += s.fit.chisq
        dof += s.fit.dof
    print()
    print("Initial state")
    print("chisq: {0}    dof: {1}".format(chisq, dof))
    print("chisq/dof: {0}".format(chisq/dof))

    oldchisq = 0.
    print()
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars)
        # print('theta_ = ', interp.gp.theta_)
        # if npca > 0:
        #     print('explained_variance_ratio = ', np.cumsum(interp._pca.explained_variance_ratio_))

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
        if oldchisq>0 and abs(oldchisq-chisq) < dof/10.0:
            break
        else:
            oldchisq = chisq
    print(interp.gp.kernel_)


def display(training_data, vis_data, mod, interp):
    interpstars = params_to_stars(vis_data, mod, noise=0.0)

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


def validate(validate_stars, mod, interp):
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
        np.testing.assert_allclose(s1.fit.flux, s0.fit.flux, rtol=2e-3)

        s1 = mod.draw(s1)
        print()
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        print('min rtol = ', np.max(np.abs(s1.image.array - s0.image.array)/np.abs(s0.image.array)))
        np.testing.assert_allclose(s1.image.array, s0.image.array, rtol=2e-2)


def check_constant_psf(npca, optimizer, visualize=False):
    # Simplest possible interpolation: the PSF is constant.
    bd = galsim.BaseDeviate(5610472938)
    ntrain = 100
    training_data, validate_data, vis_data = make_constant_psf_params(ntrain, 1, 21)

    mod = piff.GSObjectModel(fiducial_kolmogorov, force_model_center=False)

    stars = params_to_stars(training_data, mod, noise=0.03, rng=bd)
    validate_stars = params_to_stars(validate_data, mod, noise=0.0)

    kernel = kernels.RBF(1.0, (1e-1, 1e3))  # Should be nearly flat covariance...
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5...
    kernel += kernels.WhiteKernel(1e-5, (1e-7, 1e-1))
    interp = piff.SKLearnGPInterp(kernel, optimizer=optimizer, npca=npca)

    interp.initialize(stars)

    iterate(stars, mod, interp)

    if visualize:
        display(training_data, vis_data, mod, interp)

    validate(validate_stars, mod, interp)


def test_constant_psf():
    check_constant_psf(0, None)
    check_constant_psf(0, 'fmin_l_bfgs_b')
    check_constant_psf(2, None)
    check_constant_psf(2, 'fmin_l_bfgs_b')


def check_polynomial_psf(npca, optimizer, visualize=False):
    bd = galsim.BaseDeviate(5610472938)
    ntrain = 100
    training_data, validate_data, vis_data = make_polynomial_psf_params(ntrain, 1, 21)

    mod = piff.GSObjectModel(fiducial_kolmogorov, force_model_center=False)

    stars = params_to_stars(training_data, mod, noise=0.03, rng=bd)
    validate_stars = params_to_stars(validate_data, mod, noise=0.0)

    kernel = kernels.RBF(1.0, (1e-1, 1e3))  # Should be nearly flat covariance...
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5...
    kernel += kernels.WhiteKernel(1e-5, (1e-7, 1e-1))
    interp = piff.SKLearnGPInterp(kernel, optimizer=optimizer, npca=npca)

    interp.initialize(stars)

    iterate(stars, mod, interp)

    if visualize:
        display(training_data, vis_data, mod, interp)

    validate(validate_stars, mod, interp)


def test_polynomial_psf():
    check_polynomial_psf(0, None)
    check_polynomial_psf(0, 'fmin_l_bfgs_b')
    check_polynomial_psf(5, None)
    check_polynomial_psf(5, 'fmin_l_bfgs_b')


def check_grf_psf(npca, optimizer, visualize=False):
    bd = galsim.BaseDeviate(12020569031)
    ntrain = 100
    training_data, validate_data, vis_data = make_grf_psf_params(ntrain, 1, 21)

    mod = piff.GSObjectModel(fiducial_kolmogorov, force_model_center=False)

    stars = params_to_stars(training_data, mod, noise=0.03, rng=bd)
    validate_stars = params_to_stars(validate_data, mod, noise=0.0)

    kernel = kernels.RBF(0.3, (1e-1, 1e0))
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5...
    kernel += kernels.WhiteKernel(1e-5, (1e-7, 1e-1))
    interp = piff.SKLearnGPInterp(kernel, optimizer=optimizer, npca=npca)

    interp.initialize(stars)

    iterate(stars, mod, interp)

    if visualize:
        display(training_data, vis_data, mod, interp)

    validate(validate_stars, mod, interp)


def test_grf_psf():
    check_grf_psf(0, None, visualize=False)
    check_grf_psf(0, 'fmin_l_bfgs_b', visualize=False)
    check_grf_psf(5, None, visualize=False)
    check_grf_psf(5, 'fmin_l_bfgs_b', visualize=False)


def check_empirical_kernel(npca, visualize):
    bd = galsim.BaseDeviate(12020569031)
    ntrain, nvalidate, nvis = 100, 1, 21

    training_data, validate_data, vis_data = make_grf_psf_params(ntrain, nvalidate, nvis)

    mod = piff.GSObjectModel(fiducial_kolmogorov, force_model_center=False)

    stars = params_to_stars(training_data, mod, noise=0.03, rng=bd)
    validate_stars = params_to_stars(validate_data, mod, noise=0.0)


    # We could in principal use any function of dx, dy here.  For simplicity, just assert a
    # Gaussian = SquaredExponential.
    def fn(dx, dy):
        return np.exp(-0.5*(dx**2+dy**2)/0.2**2)

    kernel = piff.EmpiricalKernel(fn)
    # We probably aren't measuring fwhm, g1, g2, etc. to better than 1e-5...
    kernel += kernels.WhiteKernel(1e-5)
    interp = piff.SKLearnGPInterp(kernel, optimizer=None, npca=npca)

    interp.initialize(stars)

    iterate(stars, mod, interp)

    if visualize:
        display(training_data, vis_data, mod, interp)

    validate(validate_stars, mod, interp)


def test_empirical_kernel():
    check_empirical_kernel(0, False)
    check_empirical_kernel(5, False)


if __name__ == '__main__':
    test_constant_psf()
    test_polynomial_psf()
    test_grf_psf()
    test_empirical_kernel()
