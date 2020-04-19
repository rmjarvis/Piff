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

"""
.. module:: gp_interp
"""

import numpy as np
import warnings
import copy

import treegp

from .interp import Interp
from .star import Star, StarFit


class GPInterp(Interp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface.

    :param keys:        A list of star attributes to interpolate from
    :param kernel:      A string that can be eval-ed to make a
                        sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                        sklearn.gaussian_process.kernels will work, as well as the repr of a
                        custom piff AnisotropicRBF or ExplicitKernel object.  [default: 'RBF()']
    :param optimize:    Boolean indicating whether or not to try and optimize the kernel by
                        maximizing the marginal likelihood.  [default: True]
    :param normalize:   Whether to normalize the interpolation parameters to have a mean of 0.
                        Normally, the parameters being interpolated are not mean 0, so you would
                        want this to be True, but if your parameters have an a priori mean of 0,
                        then subtracting off the realized mean would be invalid.  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), kernel='RBF()', 
                 optimize=True, optimizer='two-pcf',
                 anisotropic=False, normalize=True, p0=[3000., 0.,0.],
                 white_noise=0., n_neighbors=4, average_fits=None,
                 nbins=20, min_sep=None, max_sep=None, 
                 rows=None, logger=None):

        self.keys = keys
        self.optimize = optimize
        self.optimizer = optimizer
        self.anisotropic = anisotropic
        if not self.anisotropic:
            self.robust_fit = False
        else:
            self.robust_fit = True
        self.p0 = p0
        self.n_neighbors = n_neighbors
        self.average_fits = average_fits
        self.nbins = nbins
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.normalize = normalize
        self.white_noise = white_noise
        self.kernel = kernel
        self.rows = rows

        self.kwargs = {
            'keys': keys,
            'optimize': optimize,
            'kernel': kernel
        }

        if isinstance(kernel,str):
            self.kernel_template = [kernel]
        else:
            if type(kernel) is not list and type(kernel) is not np.ndarray:
                raise TypeError("kernel should be a string a list or a numpy.ndarray of string")
            else:
                self.kernel_template = [ker for ker in kernel]
        
        if self.optimizer not in ['two-pcf', 'log-likelihood']:
            raise ValueError("Only two-pcf and log-likelihood are supported for optimizer. Current value: %s"%(self.optimizer))


    def _fit(self, X, y, y_err=None, logger=None):
        """Update the GaussianProcessRegressor with data
        :param X:  The independent covariates.  (n_samples, n_features)
        :param y:  The dependent responses.  (n_samples, n_targets)
        """
        for i in range(self.nparams):
            self.gps[i].initialize(X, y[:,i], y_err=y_err[:,i])
            self.gps[i].solve()

    def _predict(self, Xstar):
        """ Predict responses given covariates.
        :param X:  The independent covariates at which to interpolate.  (n_samples, n_features).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        ystar = np.array([gp.predict(Xstar) for gp in self.gps]).T
        return ystar

    def getProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the independent variables for the
        interpolation.

        Take self.keys from star.data

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A np vector of these properties.
        """
        return np.array([star.data[key] for key in self.keys])

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param stars:   A list of Star instances to interpolate between
        :param logger:  A logger object for logging debug info. [default: None]
        """
        if self.rows is None:
            self.nparams = len(stars[0].fit.params)
            self.rows = range(self.nparams)
        else:
            self.nparams = len(self.rows)

        if len(self.kernel_template)==1:
            self.kernels = [self.kernel_template[0] for i in range(self.nparams)]
        else:
            if len(self.kernel_template)!= self.nparams:
                raise ValueError("numbers of kernel provided should be 1 (same for all parameters) or " \
                                 "equal to the number of params (%i), number kernel provided: %i" \
                                 %((self.nparams,len(self.kernel_template))))
            else:
                self.kernels = [copy.deepcopy(ker) for ker in self.kernel_template]
        self.gps = []

        for i in range(self.nparams):

            gp = treegp.GPInterpolation(kernel=self.kernels[i],
                                        optimize=self.optimize, optimizer=self.optimizer,
                                        anisotropic=self.anisotropic, normalize=self.normalize,
                                        robust_fit=self.robust_fit, p0=self.p0,
                                        white_noise=self.white_noise, n_neighbors=self.n_neighbors,
                                        average_fits=self.average_fits, indice_meanify = i,
                                        nbins=self.nbins, min_sep=self.min_sep, max_sep=self.max_sep)
            self.gps.append(gp)
            
        self._init_theta = np.array([gp.kernel_template.theta for gp in self.gps])

        return stars

    def solve(self, stars=None, logger=None):
        """Set up this GPInterp object.

        :param stars:    A list of Star instances to interpolate between
        :param logger:   A logger object for logging debug info. [default: None]
        """
        X = np.array([self.getProperties(star) for star in stars])
        y = np.array([star.fit.params for star in stars])
        y_err = np.sqrt(np.array([star.fit.params_var for star in stars]))

        y = np.array([y[:,i] for i in self.rows]).T
        y_err = np.array([y_err[:,i] for i in self.rows]).T

        self._X = X
        self._y = y

        if self.white_noise > 0:
            y_err = np.sqrt(y_err**2 + self.white_noise**2)
        self._y_err = y_err

        self._fit(X, y, y_err=y_err, logger=logger)
        self.kernels = [gp.kernel for gp in self.gps]

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        # because of sklearn formatting, call interpolateList and take 0th entry
        return self.interpolateList([star], logger=logger)[0]

    def interpolateList(self, stars, logger=None):
        """Perform the interpolation for a list of stars.

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """
        Xstar = np.array([self.getProperties(star) for star in stars])
        gp_y = self._predict(Xstar)
        fitted_stars = []
        for y0, star in zip(gp_y, stars):
            if star.fit is None:
                fit = StarFit(y0)
            else:
                y0_updated = star.fit.params
                for j in range(self.nparams):
                    y0_updated[self.rows[j]] = y0[j] 
                fit = star.fit.newParams(y0_updated)
            fitted_stars.append(Star(star.data, fit))
        return fitted_stars

    def _finish_write(self, fits, extname):
        # Note, we're only storing the training data and hyperparameters here, which means the
        # Cholesky decomposition will have to be re-computed when this object is read back from
        # disk.
        init_theta = np.array([self._init_theta[i] for i in range(self.nparams)])
        fit_theta = np.array([ker.theta for ker in self.kernels])

        dtypes = [('INIT_THETA', init_theta.dtype, init_theta.shape),
                  ('FIT_THETA', fit_theta.dtype, fit_theta.shape),
                  ('X', self._X.dtype, self._X.shape),
                  ('Y', self._y.dtype, self._y.shape),
                  ('Y_ERR', self._y_err.dtype, self._y_err.shape),
                  ('ROWS', self.rows.dtype,  self.rows.shape)]

        # TO DO: need to see how I propagate meanify

                  #('X0', self._X0.dtype, self._X0.shape),
                  #('Y0', self._y0.dtype, self._y0.shape)]

        data = np.empty(1, dtype=dtypes)
        data['INIT_THETA'] = init_theta
        data['FIT_THETA'] = fit_theta
        data['X'] = self._X
        data['Y'] = self._y
        data['Y_ERR'] = self._y_err
        data['ROWS'] = self.rows
        #data['X0'] = self._X0
        #data['Y0'] = self._y0

        fits.write_table(data, extname=extname+'_kernel')

    def _finish_read(self, fits, extname):
        data = fits[extname+'_kernel'].read()
        # Run fit to set up GP, but don't actually do any hyperparameter optimization. Just
        # set the GP up using the current hyperparameters.

        init_theta = np.atleast_1d(data['INIT_THETA'][0])
        fit_theta = np.atleast_1d(data['FIT_THETA'][0])

        self._X = np.atleast_1d(data['X'][0])
        self._y = np.atleast_1d(data['Y'][0])
        self._y_err = np.atleast_1d(data['Y_ERR'][0])
        self.rows = np.atleast_1d(data['ROWS'][0])

        self._init_theta = init_theta
        self.nparams = len(init_theta)

        # TO DO : see what to do with mean function #
        #self._X0 = np.atleast_1d(data['X0'][0])
        #self._y0 = np.atleast_1d(data['Y0'][0])
        #self._spatial_average = self._build_average_meanify(self._X)

        #if self.normalize:
        #    self._mean = np.mean(self._y - self._spatial_average, axis=0)
        #else:
        #    self._mean = np.zeros(self.nparams)
        if len(self.kernel_template)==1:
            self.kernels = [copy.deepcopy(self.kernel_template[0]) for i in range(self.nparams)]
        else:
            if len(self.kernel_template)!= self.nparams:
                raise ValueError("numbers of kernel provided should be 1 (same for all parameters) or " \
                "equal to the number of params (%i), number kernel provided: %i"%((self.nparams,len(self.kernel_template))))
            else:
                self.kernels = [copy.deepcopy(ker) for ker in self.kernel_template]

        self.gps = []
        for i in range(self.nparams):

            gp = treegp.GPInterpolation(kernel=self.kernels[i],
                                        optimize=self.optimize, optimizer='log-likelihood',
                                        anisotropic=False, normalize=self.normalize,
                                        robust_fit=False, p0=[3000., 0.,0.],
                                        white_noise=self.white_noise, n_neighbors=4, average_fits=None,
                                        nbins=20, min_sep=None, max_sep=None)
            gp.kernel.clone_with_theta(fit_theta[i])
            self.gps.append(gp)
