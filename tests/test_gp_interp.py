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
import treegp
import numpy as np
import piff
import os
import copy
import fitsio
from scipy.linalg import cholesky, cho_solve
from sklearn.model_selection import train_test_split

from piff_test_helper import get_script_name, timer

kolmogorov = galsim.Kolmogorov(half_light_radius=1., flux=1.)

def get_correlation_length_matrix(correlation_length, g1, g2):
    """
    Produce correlation matrix to introduce anisotropy in kernel. 
    Used same parametrization as shape measurement in weak-lensing 
    because this is mathematicaly equivalent (anistropic kernel 
    will have an elliptical shape).

    :param correlation_length: Correlation lenght of the kernel.
    :param g1, g2:             Shear applied to isotropic kernel.
    """
    if abs(g1)>1 or abs(g2)>1:
        raise ValueError('abs value of g1 and g2 must be lower than one')
    e = np.sqrt(g1**2 + g2**2)
    q = (1-e) / (1+e)
    m = galsim.Shear(g1=g2, g2=g2).getMatrix() * correlation_length
    L = m.dot(m) * q
    return L

def make_single_star(u, v, size, g1, g2, size_err, g1_err, g2_err):
    """Make a Star instance filled with a Kolmogorov profile

    :param u, v:           Star coordinate.
    :param size:           Star size.
    :param g1, g2:         Shear applied to profile.
    :param size_err:       Size error.
    :param g1_err, g2_err: Shear error. 
    """
    star = piff.Star.makeTarget(x=None, y=None, u=u, v=v,
                                properties={}, wcs=None, scale=0.26,
                                stamp_size=24, image=None,
                                pointing=None, flux=1.)

    kolmogorov.drawImage(star.image, method='auto')
    fit = piff.StarFit(np.array([size, g1, g2]),
                       params_var=np.array([size_err**2, g1_err**2, g2_err**2]),
                       flux=1.)
    final_star = piff.Star(star.data, fit)

    return final_star

def return_gp_predict(y, X1, X2, kernel, factor):
    """Compute interpolation with gaussian process for a given kernel.

    :param y:      The dependent responses.  (n_samples, n_targets)
    :param X1:     The independent covariates.  (n_samples, 2)
    :param X2:     The independent covariates at which to interpolate.  (n_samples, 2)
    :param kernel: sklearn.gaussian_process kernel.
    :param factor: Cholesky decomposition of sklearn.gaussian_process kernel.
    """
    HT = kernel.__call__(X2, Y=X1)
    alpha = cho_solve(factor, y, overwrite_b=False)
    y_predict = np.dot(HT,alpha.reshape((len(alpha),1))).T[0]
    return y_predict

def make_gaussian_random_fields(kernel, nstars, noise_level=1e-3,
                                xlim=-10, ylim=10, seed=30352010,
                                test_size=0.20, vmax=8, plot=False):
    """
    Make psf params as gaussian random fields.

    :param kernel:      sklearn kernel to used for generating
                        the data.
    :param nstars:      number of stars to generate.
    :param noise_level: quantity of noise to add to the data.
    :param xlim:        x limit of the field.
    :param ylim:        y limit of the field.
    :param seed:        seed of the generator.
    :param test_size:   size ratio of the test sample.
    :param plot:        set to true to have plot of the field.
    :param vmax=8       max value for the color map.
    """
    np.random.seed(seed)

    # generate star coordinate
    if nstars<1500:
        nstars_interp = nstars
    else:
        nstars_interp = 1500
    u_interp = np.random.uniform(-xlim, xlim, nstars_interp)
    v_interp = np.random.uniform(-ylim, ylim, nstars_interp)
    coord_interp = np.array([u_interp, v_interp]).T

    # generate covariance matrix
    kernel = treegp.eval_kernel(kernel)
    cov_interp = kernel.__call__(coord_interp)

    # generate gaussian random fields
    size_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)
    g1_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)
    g2_interp = np.random.multivariate_normal([0]*nstars_interp, cov_interp)

    if nstars<1500:
        size = size_interp
        g1 = g1_interp
        g2 = g2_interp
        u = u_interp
        v = v_interp
        coord = coord_interp
    else:
        # Interp on stars position using a gp interp with truth kernel.
        # Trick to have more stars faster as a gaussian random field.
        u = np.random.uniform(-xlim, xlim, nstars)
        v = np.random.uniform(-ylim, ylim, nstars)
        coord = np.array([u,v]).T

        K = kernel.__call__(coord_interp) + np.eye(nstars_interp)*1e-10
        factor = (cholesky(K, overwrite_a=True, lower=False), False)

        size = return_gp_predict(size_interp, coord_interp, coord, kernel, factor)
        g1 = return_gp_predict(g1_interp, coord_interp, coord, kernel, factor)
        g2 = return_gp_predict(g2_interp, coord_interp, coord, kernel, factor)

    # add noise on psfs parameters
    size += np.random.normal(scale=noise_level, size=nstars)
    g1 += np.random.normal(scale=noise_level, size=nstars)
    g2 += np.random.normal(scale=noise_level, size=nstars)

    size_err = np.ones(nstars)*noise_level
    g1_err = np.ones(nstars)*noise_level
    g2_err = np.ones(nstars)*noise_level

    # create stars
    stars = []
    for i in range(nstars):
        star = make_single_star(u[i], v[i],
                                size[i], g1[i], g2[i],
                                size_err[i], g1_err[i], g2_err[i])
        stars.append(star)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(u, v, c=size, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g1, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g2, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)

    # split training / validation
    stars_training, stars_validation = train_test_split(stars, test_size=test_size, random_state=42)

    return stars_training, stars_validation

def check_gp(stars_training, stars_validation, kernel, optimizer,
             min_sep=None, max_sep=None, nbins=20, l0=3000., rows=None,
             plotting=False, atol=4e-2, rtol=1e-3, test_star_fit=False):
    """ Solve for global PSF model, test it, and optionally display it.
    """
    interp = piff.GPInterp(kernel=kernel, optimizer=optimizer,
                           normalize=True, white_noise=0., l0=l0,
                           n_neighbors=4, average_fits=None, rows=rows,
                           nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                           logger=None)

    assert interp.property_names == ('u', 'v')

    interp.initialize(stars_training)
    interp.solve(stars=stars_training, logger=None)

    if not test_star_fit:
        stars_test = interp.interpolateList(stars_validation)
    else:
        stars_v = copy.deepcopy(stars_validation)
        for s in stars_v:
            s.fit = None
        stars_test = interp.interpolateList(stars_v)

    xtest = np.array([interp.getProperties(star) for star in stars_validation])
    y_validation = np.array([star.fit.params for star in stars_validation])
    y_err = np.sqrt(np.array([star.fit.params_var for star in stars_validation]))

    y_test = np.array([star.fit.params for star in stars_test])

    np.testing.assert_allclose(y_test, y_validation, atol=atol)

    if optimizer != 'none':
        truth_hyperparameters = np.exp(interp._init_theta)
        fitted_hyperparameters = np.exp(
                np.array([gp._optimizer._kernel.theta for gp in interp.gps]))
        np.testing.assert_allclose(np.mean(fitted_hyperparameters, axis=0),
                                   np.mean(truth_hyperparameters, axis=0),
                                   rtol=rtol)

    # Invalid kernel (can't use an instantiated kernel object for the kernel here)
    with np.testing.assert_raises(TypeError):
        piff.GPInterp(kernel=interp.gps[0].kernel, optimizer=optimizer)
    # Invalid optimizer
    with np.testing.assert_raises(ValueError):
        piff.GPInterp(kernel=kernel, optimizer='invalid')
    # Invalid number of kernels. (Can't tell until initialize)
    if isinstance(kernel, str):
        interp2 = piff.GPInterp(kernel=[kernel] * 4, optimizer=optimizer)
        with np.testing.assert_raises(ValueError):
            interp2.initialize(stars_training)

    # Check I/O.
    file_name = os.path.join('output', 'test_gp.fits')
    with fitsio.FITS(file_name,'rw',clobber=True) as fout:
        interp.write(fout, extname='gp')
    with fitsio.FITS(file_name,'r') as fin:
        interp2 = piff.Interp.read(fin, extname='gp')

    stars_test = interp2.interpolateList(stars_validation)
    y_test = np.array([star.fit.params for star in stars_test])
    np.testing.assert_allclose(y_test, y_validation, atol=atol)

    if plotting:
        import matplotlib.pyplot as plt
        title = ["size", "$g_1$", "$g_2$"]
        for j in range(3):
            plt.figure()
            plt.title('%s validation'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_validation[:,j], vmin=-4e-2, vmax=4e-2,
                        cmap=plt.cm.seismic)
            plt.colorbar()
            plt.figure()
            plt.title('%s test (gp interp)'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_test[:,j], vmin=-4e-2, vmax=4e-2,
                        cmap=plt.cm.seismic)
            plt.colorbar()

        if optimizer in ['isotropic', 'anisotropic']:
            if optimizer == 'isotropic':
                for gp in interp.gps:
                    plt.figure()
                    plt.scatter(gp._optimizer._2pcf_dist, gp._optimizer._2pcf)
                    plt.plot(gp._optimizer._2pcf_dist, gp._optimizer._2pcf_fit)
                    plt.plot(gp._optimizer._2pcf_dist,
                             np.ones_like(gp._optimizer._2pcf_dist)*4e-4,'b--')
                    plt.ylim(0,7e-4)
            else:
                for gp in interp.gps:
                    EXT = [np.min(gp._optimizer._2pcf_dist[:,0]),
                                  np.max(gp._optimizer._2pcf_dist[:,0]),
                           np.min(gp._optimizer._2pcf_dist[:,1]),
                                  np.max(gp._optimizer._2pcf_dist[:,1])]
                    CM = plt.cm.seismic
                    MAX = np.max(gp._optimizer._2pcf)
                    N = int(np.sqrt(len(gp._optimizer._2pcf)))

                    plt.figure(figsize=(10,5) ,frameon=False)
                    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)

                    plt.subplot(1,2,1)
                    plt.imshow(gp._optimizer._2pcf.reshape(N,N), extent=EXT,
                               interpolation='nearest', origin='lower',
                               vmin=-MAX, vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    plt.title('Measured 2-PCF',fontsize=16)

                    plt.subplot(1,2,2)
                    plt.imshow(gp._optimizer._2pcf_fit.reshape(N,N), extent=EXT,
                               interpolation='nearest',
                               origin='lower',vmin=-MAX,vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi\'$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)

        plt.show()

@timer
def test_gp_interp_isotropic():

    if __name__ == "__main__":
        atol = 4e-2
        rtol = 3e-1
        nstars = [1600, 1600, 4000, 4000]
    else:
        atol = 4e-1
        rtol = 5e-1
        nstars = [160, 160, 400, 400]

    noise_level = 1e-3
    LIM = [10, 10, 20, 20]

    kernels = [["4e-4 * RBF(4.)", "4e-4 * RBF(4.)", "4e-4 * RBF(4.)"],
               "4e-4 * RBF(4.)",
               "4e-4 * RBF(4.)",
               "4e-4 * VonKarman(20.)"]

    optimizer = ['none',
                 'likelihood',
                 'isotropic',
                 'isotropic']

    rows = [[0,1,2], None, None, None]

    test_star_fit = [True, False, False, False]

    for i in range(len(kernels)):

        if i!=0:
            K = kernels[i]
        else:
            K = kernels[i][0]

        stars_training, stars_validation = make_gaussian_random_fields(
                K, nstars[i], xlim=-LIM[i], ylim=LIM[i],
                seed=30352010, vmax=4e-2, noise_level=noise_level)

        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], rows=rows[i],
                 atol=atol, rtol=rtol, test_star_fit=test_star_fit[i],
                 plotting=False)

@timer
def test_gp_interp_anisotropic():

    if __name__ == "__main__":
        atol = 4e-2
        rtol = 3e-1
        nstars = [1600, 4000, 1600, 4000]
    else:
        atol = 4e-1
        rtol = 5e-1
        nstars = [160, 500, 160, 500]

    noise_level = 1e-4

    L1 = get_correlation_length_matrix(4., 0.3, 0.3)
    invL1 = np.linalg.inv(L1)
    L2 = get_correlation_length_matrix(20., 0.3, 0.3)
    invL2 = np.linalg.inv(L2)

    kernels = ["4e-4 * AnisotropicRBF(invLam={0!r})".format(invL1),
               "4e-4 * AnisotropicRBF(invLam={0!r})".format(invL1),
               "4e-4 * AnisotropicVonKarman(invLam={0!r})".format(invL2),
               "4e-4 * AnisotropicVonKarman(invLam={0!r})".format(invL2)]

    optimizer = ['none',
                 'anisotropic',
                 'none',
                 'anisotropic']

    for i in range(len(kernels)):

        stars_training, stars_validation = make_gaussian_random_fields(
                kernels[i], nstars[i], xlim=-20, ylim=20,
                seed=30352010, vmax=4e-2,
                noise_level=noise_level)
        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], min_sep=0., max_sep=5., nbins=11,
                 l0=20., atol=atol, rtol=rtol, plotting=False)

@timer
def test_yaml():

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_gp.log')

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
            'stamp_size' : 21,
            },
        'psf' : {
            'model' : { 'type' : 'GSObjectModel',
                        'fastfit' : True,
                        'gsobj' : 'galsim.Gaussian(sigma=1.0)' },
            'interp' : { 'type' : 'GPInterp',
                         'keys' : ['u', 'v'],
                         'optimizer' : 'none',
                         'kernel' : 'RBF(200.0)'}
            },
        'output' : { 'file_name' : psf_file },
        }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)
    assert type(psf.model) == piff.GSObjectModel
    assert type(psf.interp) == piff.GPInterp
    print('nstars = ',len(psf.stars))
    target = psf.stars[17]
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, target.fit.params, decimal=3)
    # This should also work if the target doesn't have a fit yet.
    print('interpolate ',piff.Star(target.data,None))
    test_star = psf.interp.interpolate(piff.Star(target.data,None))
    np.testing.assert_almost_equal(test_star.fit.params, target.fit.params, decimal=3)


if __name__ == "__main__":
    test_gp_interp_isotropic()
    test_gp_interp_anisotropic()
    test_yaml()
