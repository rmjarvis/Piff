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
    k = galsim.Kolmogorov(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2).shift(u0,v0)
    if noise == 0.:
        var = 1.e-6
    else:
        var = noise**2
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu, v=fpv, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    k.drawImage(star.image, method='no_pixel',
                offset=galsim.PositionD(nom_u0/du,nom_v0/du), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


def make_constant_psf_params(ntrain, nvalidate, nvisualize):
    """ Make training/testing data for a constant PSF.
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
    nstars_interp = 1500
    u_interp = np.random.uniform(-xlim, xlim, nstars_interp)
    v_interp = np.random.uniform(-ylim, ylim, nstars_interp)
    coord_interp = np.array([u_interp, v_interp]).T

    # generate covariance matrix
    kernel = treegp.eval_kernel(kernel)
    cov_interp = kernel.__call__(coord_interp)

    # generate gaussian random fields
    size_interp = np.random.multivariate_normal([0]*1500, cov_interp)
    g1_interp = np.random.multivariate_normal([0]*1500, cov_interp)
    g2_interp = np.random.multivariate_normal([0]*1500, cov_interp)

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
        np.testing.assert_allclose(np.mean(fitted_hyperparameters, axis=0),
                                   np.mean(truth_hyperparameters, axis=0),
                                   rtol = 3e-1)

    if plotting:
        import matplotlib.pyplot as plt
        title = ["size", "$g_1$", "$g_2$"]
        for j in range(3):
            plt.figure()
            plt.title('%s validation'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_validation[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.figure()
            plt.title('%s test (gp interp)'%(title[j]), fontsize=18)
            plt.scatter(xtest[:,0], xtest[:,1], c=y_test[:,j], vmin=-4e-2, vmax=4e-2, cmap=plt.cm.seismic)
            plt.colorbar()

        if optimizer in ['two-pcf', 'anisotropic']:
            if optimizer == 'two-pcf':
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

                    plt.figure(figsize=(10,5) ,frameon=False)
                    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)

                    plt.subplot(1,2,1)
                    plt.imshow(gp._optimizer._2pcf.reshape(N,N), extent=EXT, interpolation='nearest', origin='lower',
                               vmin=-MAX, vmax=MAX, cmap=CM)
                    cbar = plt.colorbar()
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                    cbar.set_label('$\\xi$',fontsize=20)
                    plt.xlabel('$\\theta_X$',fontsize=20)
                    plt.ylabel('$\\theta_Y$',fontsize=20)
                    plt.title('Measured 2-PCF',fontsize=16)

                    plt.subplot(1,2,2)
                    plt.imshow(gp._optimizer._2pcf_fit.reshape(N,N), extent=EXT, interpolation='nearest',
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

    noise_level = 1e-3

    nstars = [1600, 1600, 4000, 4000]
    LIM = [10, 10, 20, 20]

    kernels = ["4e-4 * RBF(4.)",
               "4e-4 * RBF(4.)",
               "4e-4 * RBF(4.)",
               "4e-4 * VonKarman(20.)"]

    optimizer = ['none',
                 'log-likelihood',
                 'two-pcf',
                 'two-pcf']

    for i in range(len(kernels)):

        stars_training, stars_validation = make_gaussian_random_fields(kernels[i], nstars[i], xlim=-LIM[i], ylim=LIM[i],
                                                                       seed=30352010, vmax=4e-2,
                                                                       noise_level=noise_level)
        check_gp(stars_training, stars_validation, kernels[i],
                 optimizer[i], plotting=False)

@timer
def test_gp_interp_anisotropic():

    noise_level = 1e-4

    nstars = [1600, 4000, 1600, 4000]

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

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data,
                     kernel + " + WhiteKernel(1e-5, (1e-6, 1e-1))",
                     npca=npca, optimize=optimize, file_name="test_gp_vonkarman.fits", rng=rng,
                     check_config=check_config, rtol=rtol)
            check_gp_2pcf(training_data, validation_data, visualization_data, kernel,
                          npca=npca, optimize=optimize, anisotropic=False, file_name="test_gp_vonkarman.fits", rng=rng,
                          check_config=check_config, rtol=rtol)

    check_gp_2pcf(training_data, validation_data, visualization_data, anisotropic_kernel,
                  npca=0, optimize=False, anisotropic=True, file_name="test_gp_vonkarman.fits", rng=rng,
                  check_config=check_config, rtol=rtol)

@timer
def test_gp_with_kernels():

    if __name__ == '__main__':
        ntrain = 1000
        npcas = [0]
        optimizes = [True, False]
        check_config = True
        rtol = 0.02
    else:
        ntrain = 500
        npcas = [0]
        optimizes = [False]
        check_config = False
        rtol = 0.05
    nvalidate, nvisualize = 1, 20
    rng = galsim.BaseDeviate(987654334587656)

    training_data, validation_data, visualization_data = make_vonkarman_and_rbf_psf_params(
            ntrain, nvalidate, nvisualize)
    
    kernel = ["0.01*RBF(0.7, (1e-1, 1e1))",
              "0.01*RBF(0.7, (1e-1, 1e1))",
              "0.01*VonKarman(0.7, (1e-1, 1e1))",
              "0.01*VonKarman(0.7, (1e-1, 1e1))",
              "0.01*VonKarman(0.7, (1e-1, 1e1))"]

    for npca in npcas:
        for optimize in optimizes:
            check_gp_2pcf(training_data, validation_data, visualization_data, kernel,
                          npca=npca, optimize=optimize, file_name="test_gp_vonkarman.fits", rng=rng,
                          check_config=check_config, rtol=rtol)


@timer
def test_anisotropic_rbf_kernel():
    if __name__ == '__main__':
        ntrain = 250
        npcas = [0]
        optimizes = [True, False]
        check_config = True
        nvalidate = 1
        nvisualize = 21
        rtol = 0.03
    else:
        ntrain = 100
        npcas = [0]
        optimizes = [False, True]
        check_config = False
        nvalidate = 1
        nvisualize = 1
        rtol = 0.05
    rng = galsim.BaseDeviate(5867943)

    training_data, validation_data, visualization_data = make_anisotropic_grf_psf_params(
            ntrain, nvalidate, nvisualize)
    var1 = 0.1**2
    var2 = 0.2**2
    corr = 0.7
    cov = np.array([[var1, np.sqrt(var1*var2)*corr],
                    [np.sqrt(var1*var2)*corr, var2]])
    invLam = np.linalg.inv(cov)

    kernel = "0.1*AnisotropicRBF(invLam={0!r})".format(invLam)
    #kernel_vk= "0.1*AnisotropicVonKarman(invLam={0!r})".format(invLam)

    print(kernel)

    for npca in npcas:
        for optimize in optimizes:
            check_gp(training_data, validation_data, visualization_data,
                     kernel+ "+ WhiteKernel(1e-5, (1e-7, 1e-2))",
                     npca=npca, optimize=optimize, file_name="test_anisotropic_rbf.fits",
                     rng=rng, check_config=check_config)
            check_gp_2pcf(training_data, validation_data, visualization_data, kernel,
                          npca=npca, anisotropic=True, optimize=optimize, rng=rng,
                          check_config=check_config, rtol=rtol)

@timer
def test_vonkarman_kernel():
    from scipy import special
                    
    corr_lenght = [1.,10.,100.,1000.]
    kernel_amp = [1e-4,1e-3,1e-2,1.]
    dist = np.linspace(0,10,100)
    coord = np.array([dist,dist]).T

    dist = np.linspace(0.01,10,100)
    coord_corr = np.array([dist,np.zeros_like(dist)]).T
    
    def _vonkarman_kernel(param,x):
        A = (x[:,0]-x[:,0][:,None])
        B = (x[:,1]-x[:,1][:,None])
        distance = np.sqrt(A*A + B*B)
        Filter = distance != 0.
        K = np.zeros_like(distance)
        K[Filter] = param[0]**2 * ((distance[Filter]/param[1])**(5./6.) *
                                   special.kv(-5./6.,2*np.pi*distance[Filter]/param[1]))
        dist = np.linspace(1e-4,1.,100)
        div = 5./6.
        lim0 = special.gamma(div) /(2 * (np.pi**div) )
        K[~Filter] = param[0]**2 * lim0
        K /= lim0
        return K
        
    def _vonkarman_corr_function(param, distance):
        div = 5./6.
        lim0 = (2 * (np.pi**div) ) / special.gamma(div) 
        return param[0]**2 * lim0 * ((distance/param[1])**(5./6.)) * special.kv(-5./6.,2*np.pi*distance/param[1])
    
    for corr in corr_lenght:
        for amp in kernel_amp:
        
            kernel = "%.10f * VonKarman(length_scale=%f)"%((amp**2,corr))

            interp = piff.GPInterp2pcf(kernel=kernel,
                                       normalize=False,
                                       white_noise=0.)
            ker = interp.kernel_template[0]

            ker_piff = ker.__call__(coord)
            corr_piff = ker.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]

            ker_test = _vonkarman_kernel([amp,corr],coord)
            corr_test = _vonkarman_corr_function([amp,corr], dist) 

            np.testing.assert_allclose(ker_piff, ker_test, atol=1e-12)
            np.testing.assert_allclose(corr_piff, corr_test, atol=1e-12)

@timer
def test_anisotropic_vonkarman_kernel():
    from scipy import special
    from scipy.spatial.distance import pdist, squareform

    corr_length = [1., 30. ,30. ,30., 30.]
    g1 = [0, 0.4, 0.4, -0.4, -0.4]
    g2 = [0, 0.4, -0.4, 0.4, -0.4]
    kernel_amp = [1e-4, 1e-3, 1e-2, 1., 1.]
    dist = np.linspace(0,10,100)
    coord = np.array([dist,dist]).T

    dist = np.linspace(-10,10,21)
    #coord_corr = np.array([dist,np.zeros_like(dist)]).T

    X, Y = np.meshgrid(dist,dist)
    x = X.reshape(len(dist)**2)
    y = Y.reshape(len(dist)**2)
    coord_corr = np.array([x, y]).T

    def _anisotropic_vonkarman_kernel(x, sigma, corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        invL = np.linalg.inv(L)
        dists = pdist(x, metric='mahalanobis', VI=invL)
        K = dists **(5./6.) *  special.kv(5./6., 2*np.pi * dists)
        lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
        K = squareform(K)
        np.fill_diagonal(K, lim0)
        K /= lim0
        K *= sigma**2
        return K
        
    def _anisotropic_vonkarman_corr_function( x, y, sigma, 
                                              corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        l = np.linalg.inv(L)
        dist_a = (l[0,0]*x*x) + (2*l[0,1]*x*y) + (l[1,1]*y*y)
        z = np.zeros_like(dist_a)
        Filter = dist_a != 0.
        z[Filter] = dist_a[Filter]**(5./12.) *  special.kv(5./6., 2*np.pi * np.sqrt(dist_a[Filter]))
        lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
        if np.sum(Filter) != len(z):
            z[~Filter] = lim0
        z /= lim0
        return z*sigma**2
    
    for i in range(5):
        L = get_correlation_length_matrix(corr_length[i], g1[i], g2[i])
        inv_L = np.linalg.inv(L)
        ker = kernel_amp[i]**2 * piff.AnisotropicVonKarman(invLam=inv_L)
        ker_piff = ker.__call__(coord)
        corr_piff = ker.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]
        ker_test = _anisotropic_vonkarman_kernel(coord, kernel_amp[i], corr_length[i], g1[i], g2[i])
        corr_test = _anisotropic_vonkarman_corr_function(x, y, kernel_amp[i],
                                                         corr_length[i], g1[i], g2[i])
        np.testing.assert_allclose(ker_piff, ker_test, atol=1e-12)
        np.testing.assert_allclose(corr_piff, corr_test, atol=1e-12)
        
        hyperparameter = ker.theta
        theta = hyperparameter[1:]
        L1 = np.zeros_like(inv_L)
        L1[np.diag_indices(2)] = np.exp(theta[:2])
        L1[np.tril_indices(2, -1)] = theta[2:]
        invLam = np.dot(L1, L1.T)
        np.testing.assert_allclose(inv_L, invLam, atol=1e-12)
>>>>>>> master

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
    assert type(psf.model) is piff.GSObjectModel
    assert type(psf.interp) is piff.GPInterp
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
