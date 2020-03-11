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
import yaml
import fitsio
from sklearn.model_selection import train_test_split

from piff_test_helper import get_script_name, timer

#TO REMOVE:
import pylab as plt


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

def make_gaussian_random_fields(kernel, nstars, noise_level=1e-3, 
                                xlim=-10, ylim=10, seed=30352010,
                                test_size=0.20, vmax=8, plot=False):
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
        plt.figure()
        plt.scatter(u, v, c=size, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g1, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)
        plt.figure()
        plt.scatter(u, v, c=g2, vmin=-vmax, vmax=vmax, cmap=plt.cm.seismic)

    # split training / validation
    stars_training, stars_validation = train_test_split(stars, test_size=test_size, random_state=42)

    return stars_training, stars_validation

@timer
def test_gp_interp_isotropic():

    nstars = 1400
    noise_level = 1e-3

    kernels = ["4e-4 * RBF(1.)", 
               "4e-4 * RBF(1.)", 
               "4e-4 * RBF(1.)", 
               "4e-4 * VonKarman(10.)",
               "4e-4 * VonKarman(10.)",
               "4e-4 * VonKarman(10.)"]

    optimize = [False, True, True, False, True, True]

    optimizer = ['log-likelihood', 
                 'log-likelihood', 
                 'two-pcf',
                 'log-likelihood',
                 'log-likelihood',
                 'two-pcf']

    for t in range(3):
        i = 3+t
        print(kernels[i], optimize[i], optimizer[i])
        stars_training, stars_validation = make_gaussian_random_fields(kernels[i], nstars, xlim=-10, ylim=10,
                                                                       seed=30352010, vmax=4e-2,
                                                                       noise_level=noise_level)

        interp = piff.GPInterp(kernel=kernels[i], optimize=optimize[i], optimizer=optimizer[i],
                               normalize=True, white_noise=0.,
                               anisotropic=False, p0=[3000., 0.,0.],
                               n_neighbors=4, average_fits=None,
                               nbins=20, min_sep=None, max_sep=None,
                               logger=None)

        interp.initialize(stars_training)
        interp.solve(stars=stars_training, logger=None)
        stars_test = interp.interpolateList(stars_validation)

        xtest = np.array([interp.getProperties(star) for star in stars_validation])
        y_validation = np.array([star.fit.params for star in stars_validation])
        y_err = np.sqrt(np.array([star.fit.params_var for star in stars_validation]))
            
        y_test = np.array([star.fit.params for star in stars_test])

        print(np.mean(y_test - y_validation, axis=0))
        print(np.std(y_test - y_validation, axis=0))
        print("")
        
        for gp in interp.gps:
            if optimize[i]:
                print(np.exp(gp._optimizer._kernel.theta))
            else:
                print("Truth:", np.exp(gp.kernel.theta))
        print("")

        #np.testing.assert_allclose(y_test, y_validation, atol = 2e-2)

        #if optimize[i]:
        #    truth_hyperparameters = np.exp(interp._init_theta)
        #    fitted_hyperparameters = np.exp(np.array([gp._optimizer._kernel.theta for gp in interp.gps]))
        #    np.testing.assert_allclose(fitted_hyperparameters, truth_hyperparameters, rtol = 0.3)
            
        title = ["size", "$g_1$", "$g_2$"]
        for j in range(3):
            plt.figure()
            plt.title('%s validation'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_validation[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
            plt.figure()
            plt.title('%s test'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_test[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
        
        if optimizer[i] == 'two-pcf':
            for gp in interp.gps:
                plt.figure()
                plt.scatter(gp._optimizer._2pcf_dist, gp._optimizer._2pcf)
                plt.plot(gp._optimizer._2pcf_dist, gp._optimizer._2pcf_fit)
                plt.plot(gp._optimizer._2pcf_dist, np.ones_like(gp._optimizer._2pcf_dist)*4e-4,'b--')
                plt.ylim(0,7e-4)
            
        plt.show()
    return interp

@timer
def test_gp_interp_anisotropic():
    print("to do")


if __name__ == "__main__":

    interp = test_gp_interp_isotropic()
    #test_gp_interp_anisotropic()
