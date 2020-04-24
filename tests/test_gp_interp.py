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
from sklearn.model_selection import train_test_split

from piff_test_helper import get_script_name, timer

kolmogorov = galsim.Kolmogorov(half_light_radius=1., flux=1.)

def return_var_map(weight, xi):
    """
    function to return variance of 2-pcf.

    :param weight: weight, output of treegp.
    :param xi:     2-pcf, ouput of treegp.
    """
    N = int(np.sqrt(len(xi)))
    var = np.diag(np.linalg.inv(weight))
    VAR = np.zeros(N*N)
    I = 0
    for i in range(N*N):
        if xi[i] !=0:
            VAR[i] = var[I]
            I+=1
        if I == len(var):
            break
    VAR = VAR.reshape(N,N) + np.flipud(np.fliplr(VAR.reshape(N,N)))
    if N%2 == 1:
        VAR[N/2, N/2] /= 2.
    return VAR


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
    u = np.random.uniform(-xlim, xlim, nstars)
    v = np.random.uniform(-ylim, ylim, nstars)
    coord = np.array([u,v]).T

    # generate covariance matrix
    kernel = treegp.eval_kernel(kernel)
    cov = kernel.__call__(coord)

    # generate gaussian random fields 
    size = np.random.multivariate_normal([0]*nstars, cov)
    g1 = np.random.multivariate_normal([0]*nstars, cov)
    g2 = np.random.multivariate_normal([0]*nstars, cov)

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
        import pylab as plt
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
             min_sep=None, max_sep=None, nbins=20, l0=3000.,
             plotting=False):
    """ Solve for global PSF model, test it, and optionally display it.
    """
    interp = piff.GPInterp(kernel=kernel, optimizer=optimizer,
                           normalize=True, white_noise=0., l0=l0,
                           n_neighbors=4, average_fits=None,
                           nbins=nbins, min_sep=min_sep, max_sep=max_sep,
                           logger=None)

    interp.initialize(stars_training)
    interp.solve(stars=stars_training, logger=None)
    stars_test = interp.interpolateList(stars_validation)

    xtest = np.array([interp.getProperties(star) for star in stars_validation])
    y_validation = np.array([star.fit.params for star in stars_validation])
    y_err = np.sqrt(np.array([star.fit.params_var for star in stars_validation]))
            
    y_test = np.array([star.fit.params for star in stars_test])

    np.testing.assert_allclose(y_test, y_validation, atol = 4e-2)

    if optimizer is not 'none':
        truth_hyperparameters = np.exp(interp._init_theta)
        fitted_hyperparameters = np.exp(np.array([gp._optimizer._kernel.theta for gp in interp.gps]))
        np.testing.assert_allclose(fitted_hyperparameters, truth_hyperparameters, rtol = 2.)
            
    if plotting:
        import pylab as plt
        title = ["size", "$g_1$", "$g_2$"]
        for j in range(3):
            plt.figure()
            plt.title('%s validation'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_validation[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.figure()
            plt.title('%s test'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_test[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
            plt.colorbar()
        
        if optimizer in ['two-pcf', 'anisotropic']:
            if optimizer is 'two-pcf':
                for gp in interp.gps:
                    plt.figure()
                    plt.scatter(gp._optimizer._2pcf_dist, gp._optimizer._2pcf)
                    plt.plot(gp._optimizer._2pcf_dist, gp._optimizer._2pcf_fit)
                    plt.plot(gp._optimizer._2pcf_dist, np.ones_like(gp._optimizer._2pcf_dist)*4e-4,'b--')
                    plt.ylim(0,7e-4)
            else:
                for gp in interp.gps:
                    EXT = [np.min(gp._optimizer._2pcf_dist[:,0]), np.max(gp._optimizer._2pcf_dist[:,0]),
                           np.min(gp._optimizer._2pcf_dist[:,1]), np.max(gp._optimizer._2pcf_dist[:,1])]
                    CM = plt.cm.seismic
                    MAX = np.max(gp._optimizer._2pcf)
                    N = int(np.sqrt(len(gp._optimizer._2pcf)))

                    plt.figure(figsize=(14,5) ,frameon=False)
                    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)

                    plt.subplot(1,3,1)
                    plt.imshow(gp._optimizer._2pcf.reshape(N,N), extent=EXT, interpolation='nearest', origin='lower',
                               vmin=-MAX, vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    plt.title('Measured 2-PCF',fontsize=16)
                
                    plt.subplot(1,3,2)
                    plt.imshow(gp._optimizer._2pcf_fit.reshape(N,N), extent=EXT, interpolation='nearest',
                               origin='lower',vmin=-MAX,vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi\'$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    
                    var = return_var_map(gp._optimizer._2pcf_weight, gp._optimizer._2pcf)
                    cm_residual = plt.matplotlib.cm.get_cmap('RdBu',10)
                    Res = gp._optimizer._2pcf[gp._optimizer._2pcf_mask] - gp._optimizer._2pcf_fit[gp._optimizer._2pcf_mask]
                    chi2 = Res.dot(gp._optimizer._2pcf_weight).dot(Res)
                    dof = np.sum(gp._optimizer._2pcf_mask) - 4.
                    pull = (gp._optimizer._2pcf.reshape(N,N) - gp._optimizer._2pcf_fit.reshape(N,N)) / np.sqrt(var)
                    plt.title('Fitted 2-PCF'%(chi2/dof),fontsize=16)

                    plt.subplot(1,3,3)
                
                    plt.imshow(pull, extent=EXT, interpolation='nearest', 
                               origin='lower', vmin=-5., vmax=+5., cmap=cm_residual)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\frac{\\xi-\\xi\'}{\sigma_{\\xi}}$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    plt.title('Pull',fontsize=16)
    
        plt.show()

@timer
def test_gp_interp_isotropic():

    nstars = 1000
    noise_level = 1e-3

    kernels = ["4e-4 * RBF(1.)", 
               "4e-4 * RBF(1.)", 
               "4e-4 * RBF(1.)", 
               "4e-4 * VonKarman(20.)",
               "4e-4 * VonKarman(20.)",
               "4e-4 * VonKarman(20.)"]

    optimizer = ['none',
                 'log-likelihood', 
                 'two-pcf',
                 'none',
                 'log-likelihood',
                 'two-pcf']

    for i in range(len(kernels)):
        stars_training, stars_validation = make_gaussian_random_fields(kernels[i], nstars, xlim=-10, ylim=10,
                                                                       seed=30352010, vmax=4e-2,
                                                                       noise_level=noise_level)
        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], plotting=False)

@timer
def test_gp_interp_anisotropic():
    
    nstars = 1400
    noise_level = 1e-3

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
        stars_training, stars_validation = make_gaussian_random_fields(kernels[i], nstars, xlim=-10, ylim=10,
                                                                       seed=30352010, vmax=4e-2,
                                                                       noise_level=noise_level)
        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], min_sep=0., max_sep=5., nbins=11, l0=20., plotting=False)

if __name__ == "__main__":

    test_gp_interp_isotropic()
    test_gp_interp_anisotropic()
