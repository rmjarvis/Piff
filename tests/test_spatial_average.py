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

def make_star0(u, v, hlr, g1, g2, u0, v0):
    """Make a Star instance filled with a Kolmogorov profile

    :param hlr:         The half_light_radius of the Kolmogorov.
    :param g1, g2:      Shear applied to profile.
    :param u0, v0:      The sub-pixel offset to apply.
    :param u, v:        Coordinate
    """

    k = galsim.Kolmogorov(half_light_radius=1., flux=1.)

    
    star = piff.Star.makeTarget(x=None, y=None, u=u, v=v,
                                properties={}, wcs=None, scale=None,
                                stamp_size=24, image=None,
                                pointing=None, flux=1.)
        
    k.drawImage(star.image, method='auto')
    fit = piff.StarFit(np.array([u0, v0, hlr, g1, g2]), flux=1.)
    star = piff.Star(star.data, fit)
    return star

def make_average(Coord=None):
    if Coord is None:
        x = np.linspace(0,1,20)
        x, y = np.meshgrid(x,x)
        x = x.reshape(len(x)**2)
        y = y.reshape(len(y)**2)
    else:
        x = Coord[:,0]
        y = Coord[:,1]
    
    average = 0.01 + 1e-1*x**2 - 1e-2*x*y + 1e-1*y**2
    params = np.recarray((len(x),), dtype=star_type)
    keys = ['u0', 'v0', 'hlr', 'g1', 'g2']
    for key in keys:
        params[key] = average
    params['u'] = x
    params['v'] = y
    params['flux'] = np.ones_like(average)
    import pylab as plt
    for key in keys:
        plt.scatter(params['u'], params['v'], c=params[key], lw=0, s=20)
        plt.title(key)
        plt.colorbar()
        plt.show()
    return params

def make_grf_psf_params_average(ntrain, nvalidate, nvisualize, scale_length=0.3):
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
    cov = np.exp(-0.5*dists**2/scale_length**2)

    params['u'] = us
    params['v'] = vs

    average = make_average(Coord=np.array([us,vs]).T)

    # independently draw hlr, g1, g2, u0, v0
    np.random.seed(1234567890)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params['hlr'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05+0.6 + average['hlr']
        params['g1'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05 + average['g1']
        params['g2'] = np.random.multivariate_normal([0]*ntotal, cov)*0.05 + average['g2']
        params['u0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3 + average['u0']
        params['v0'] = np.random.multivariate_normal([0]*ntotal, cov)*0.3 + average['v0']
    params['flux'] = fluxes

    training_data = params[:ntrain]
    validate_data = params[ntrain:ntrain+nvalidate]
    vis_data = params[ntrain+nvalidate:].reshape((nvisualize, nvisualize))

    average = make_average()
    
    return training_data, validate_data, vis_data, average


def params_to_stars(params, noise=0.0, rng=None):
    stars = []
    for param in params.ravel():
        u, v, hlr, g1, g2, u0, v0, flux = param
        s = make_star(hlr, g1, g2, u0, v0, flux, noise=noise, du=0.2, fpu=u, fpv=v, rng=rng)
        try:
            s = mod.initialize(s)
        except:
            print("Failed to initialize star at ",u,v)
        else:
            stars.append(s)
    return stars

def params_to_stars0(params, noise=0.0, rng=None):
    stars = []
    for param in params.ravel():
        u, v, hlr, g1, g2, u0, v0, flux = param
        s = make_star0(u, v, hlr, g1, g2, u0, v0)
        stars.append(s)
    return stars


def iterate_2pcf(stars, interp, stars0=None):
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

    interp.initialize(stars)
    oldchisq = 0.
    print()
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars, stars0)
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


def validate(validate_stars, interp, rtol=0.02):
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
        np.testing.assert_allclose(s1.fit.flux, s0.fit.flux, rtol=rtol)

        s1 = mod.draw(s1)
        print()
        print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
        print('max image abs value = ',np.max(np.abs(s0.image.array)))
        print('min rtol = ', np.max(np.abs(s1.image.array - s0.image.array)/s0.image.array.max()))
        np.testing.assert_allclose(s1.image.array, s0.image.array,
                                   rtol=0, atol=s0.image.array.max()*rtol*2)

        if False:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(s0.image.array)
            axes[1].imshow(s1.image.array)
            axes[2].imshow(s1.image.array - s0.image.array)
            plt.show()


def check_gp_2pcf(training_data, validation_data, visualization_data,
                  kernel, npca=0, average_data=None, optimize=False, file_name=None, rng=None,
                  visualize=True, check_config=False, rtol=0.02):
    """ Solve for global PSF model, test it, and optionally display it.
    """
    stars = params_to_stars(training_data, noise=0.03, rng=rng)
    if average_data is not None: 
        stars0 = params_to_stars0(average_data)
    else:
        stars0 = None
    validate_stars = params_to_stars(validation_data, noise=0.0, rng=rng)
    interp = piff.GPInterp2pcf(kernel=kernel, optimize=optimize, npca=npca, white_noise=1e-5)
    interp.initialize(stars)
    iterate_2pcf(stars, interp, stars0=stars0)
    if visualize:
        display(training_data, visualization_data, interp)
    validate(validate_stars, interp, rtol=rtol)

    if check_config:
        config = {
            'interp' : {
                'type' : 'GPInterp2pcf',
                'kernel' : kernel,
                'npca' : npca,
                'optimize' : optimize,
                'white_noise': 1e-5
            }
        }
        print(config)
        logger = piff.config.setup_logger()
        interp3 = piff.Interp.process(config['interp'], logger)
        iterate_2pcf(stars, interp3)
        validate(validate_stars, interp3, rtol=rtol)

    # Check that we can write interp to disk and read back in.
    if file_name is not None:
        testfile = os.path.join('output', file_name)
        with fitsio.FITS(testfile, 'rw', clobber=True) as f:
            interp.write(f, 'interp')
        with fitsio.FITS(testfile, 'r') as f:
            interp2 = piff.GPInterp2pcf.read(f, 'interp')
        print("Revalidating after i/o.")
        X = np.vstack([training_data['u'], training_data['v']]).T
        for i in range(interp.nparams):
            np.testing.assert_allclose(interp.kernels[i].__call__(X), interp2.kernels[i].__call__(X))
            np.testing.assert_allclose(interp._init_theta[i], interp2._init_theta[i])
            np.testing.assert_allclose(interp.kernels[i].theta, interp2.kernels[i].theta)
            np.testing.assert_allclose(interp._X, interp2._X)
            np.testing.assert_allclose(interp._mean[i], interp2._mean[i],atol=1e-12)
        validate(validate_stars, interp2, rtol=rtol)


@timer
def test_grf_psf():
    if __name__ == '__main__':
        ntrain = 200
        npcas = [0]
        optimizes = [True, False]
        check_config = True
    else:
        ntrain = 100
        npcas = [0]
        optimizes = [False]
        check_config = False
    nvalidate, nvisualize = 1, 21
    rng = galsim.BaseDeviate(987654334587656)

    training_data, validation_data, visualization_data, average_data = make_grf_psf_params_average(
            ntrain, nvalidate, nvisualize)

    kernel = "1*RBF(0.3, (1e-1, 1e1))"

    for npca in npcas:
        for optimize in optimizes:
            check_gp_2pcf(training_data, validation_data, visualization_data, kernel,
                          npca=npca, optimize=optimize, average_data=average_data, file_name="test_gp_grf.fits", rng=rng,
                          check_config=check_config)

if __name__ == '__main__':


    test_grf_psf()

    #training_data, validation_data, visualization_data, average_data = make_grf_psf_params_average(100, 1, 21)
    #stars0 = params_to_stars0(average_data)

