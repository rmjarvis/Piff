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
    An interpolator that models the underlying field as a Gaussian process.

    Gaussian process regression, also known as “Kriging,” assumes that the parameters
    are drawn from a multi-dimensional Gaussian random field across the (u, v) space.
    It requires an estimate of the spatial covariance function of p, commonly referred to
    as the kernel.

    The interpolation estimate at an arbitrary location (u, v) is the minimum variance
    unbiased estimate from the Gaussian distribution at that location conditioned on the
    values of the parameters measured at all the PSF stars.

    The implemention of this class use the treegp module. (https://github.com/PFLeget/treegp)
    It can use any of the kernels defined in scikit-learn (https://scikit-learn.org/)
    to define the covariance matrix as well as a custom VonKarman kernel defined in treegp.
    The default kernel is the so-called "squared exponential" or "radial basis function" kernel,
    known as RBF in scikit-learn.

    The default behavior involves measuring the radial two-point correlation function with
    TreeCorr (https://github.com/rmjarvis/TreeCorr) and then fitting the hyper-parameters of
    the kernel that best fits this measurement.  This can be done either isotropically
    or anisotropically.  There are also options to use the traditional maximum likelihood
    optimization or no optimization if preferred.  See the ``optimizer`` parameter below.

    :param keys:         A list of keys for properties that will be interpolated.  Must be 2
                         properties, which can be used to calculate a 2-point correlation
                         function. [default: ('u','v')]
    :param kernel:       A string that can be evaled to make a
                         ``sklearn.gaussian_process.kernels.Kernel`` object. Could be also a list
                         of Kernels objects (one per PSF param).  The reprs of sklear Kernels
                         will work, as well as the repr of a custom treegp VonKarman object.
                         [default: 'RBF(1)']
    :param optimizer:    Indicates which techniques to use for optimizing the kernel. Four options
                         are available:

                            * "isotropic" = use an isotropic radial 2-point correlation function
                              estimated by TreeCorr.
                            * "anisotropic" = use an anisotropic two-dimensional 2-point
                              correlation function estimated by TreeCorr.
                            * "likelihood" = use the classical Gaussian process maximum
                              likelihood method.
                            * "none" = don't do any kernal optimization.

                         [default: "isotropic"]
    :param rows:         A list of integer which indicates which rows of Star.fit.param
                         need to be interpolated using GPs. [default: None, which means all rows]
    :param normalize:    Whether to normalize the interpolation parameters to have a mean of 0.
                         Normally, the parameters being interpolated are not mean 0, so you would
                         want this to be True, but if your parameters have an a priori mean of 0,
                         then subtracting off the realized mean would be invalid.  [default: True]
    :param white_noise:  A float value that indicate the ammount of white noise that you want to
                         use during the gp interpolation. This is an additional uncorrelated noise
                         added in quadrature to the measured uncertainties of the PSF parameters.
                         This should be given as a "sigma" value, not a variance. [default: 0.]
    :param nbins:        Number of bins (in each direction using a 2D correlation function)
                         used in TreeCorr to compute the 2-point correlation function. Used only if
                         optimizer is "isotropic" or "anisotropic". [default: 20]
    :param min_sep:      Minimum separation between pairs when computing 2-point correlation
                         function in arcsec (or more generally the same units as the keys).
                         Compute automaticaly if it is not given. Used only if optimizer is
                         "isotropic" or "anisotropic".  [default: None]
    :param max_sep:      Maximum separation between pairs when computing 2-point correlation
                         function in arcsec (or more generally the same units as the keys).
                         Compute automaticaly if it is not given. Used only if optimizer is
                         "isotropic" or "anisotropic".  [default: None]
    :param l0:           Initial guess for correlation length when optimzer is "anisotropic" in
                         arcsec (or more generally the same units as the keys). [default: 3000.]
    :param average_fits: A fits file that have the spatial average functions of PSF parameters
                         build in it. Build using meanify and piff output across different
                         exposures. See meanify documentation for details. [default: None]
    :param n_neighbors:  Number of neighbors to use for interpolating the spatial average using
                         a KNeighbors interpolation. Used only if average_fits is not None.
                         [defaulf: 4]
    :param logger:       A logger object for logging debug info. [default: None]
    """

    # treegp currently uses slightly different names for these
    treegp_alias = {
        'none' : 'none',
        'likelihood' : 'log-likelihood',
        'isotropic' : 'two-pcf',
        'anisotropic' : 'anisotropic'
    }

    def __init__(self, keys=('u','v'), kernel='RBF(1)',
                 optimizer='isotropic', normalize=True, l0=3000.,
                 white_noise=0., n_neighbors=4, average_fits=None,
                 nbins=20, min_sep=None, max_sep=None,
                 rows=None, logger=None):

        self.keys = keys
        self.optimizer = optimizer
        self.l0 = l0
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
            'optimizer': optimizer,
            'kernel': kernel
        }

        if isinstance(kernel,str):
            self.kernel_template = [kernel]
        elif isinstance(kernel, (list, np.ndarray)):
            self.kernel_template = [ker for ker in kernel]
        else:
            raise TypeError("kernel should be a string a list or a numpy.ndarray of string")

        if self.optimizer not in ['anisotropic', 'isotropic', 'likelihood', 'none']:
            raise ValueError("Only anisotropic, isotropic, likelihood, and " \
                             "none are supported for optimizer. Current value: %s"%(self.optimizer))

    @property
    def property_names(self):
        """List of properties used by this interpolant.
        """
        return self.keys

    def _fit(self, X, y, y_err=None, logger=None):
        """Update the GaussianProcess with data

        :param X:  The independent covariates.  (n_samples, n_features)
        :param y:  The dependent responses.  (n_samples, n_targets)
        :param y_err: Error of y. (n_samples, n_targets)
        :param logger:  A logger object for logging debug info. [default: None]
        """
        # If the treecorr libomp is different from galsim's libomp linkage, then
        # calling treecorr from here can result in a seg fault.  I couldn't figure
        # out a reliable way to avoid thiss -- it happened on GHA's MacOs system.
        # But the parallel computation in TreeCorr here isn't actually particularly
        # useful -- it's not a tall pole in this calculation.  So to be safe, just
        # set the treecorr max_threads to 1 during this calculation.
        from treecorr import set_max_omp_threads
        set_max_omp_threads(1)
        for i in range(self.nparams):
            self.gps[i].initialize(X, y[:,i], y_err=y_err[:,i])
            self.gps[i].solve()
        set_max_omp_threads(None)

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
            self.rows = np.arange(0, self.nparams, 1).astype(int)
        else:
            self.nparams = len(self.rows)
            self.rows = np.array(self.rows)

        if len(self.kernel_template)==1:
            self.kernels = [self.kernel_template[0] for i in range(self.nparams)]
        elif len(self.kernel_template) == self.nparams:
            self.kernels = [ker for ker in self.kernel_template]
        else:
            raise ValueError("numbers of kernel provided should be 1 (same for all parameters) "
                             "or equal to the number of params (%i), number kernel provided: %i"
                                %((self.nparams,len(self.kernel_template))))
        self.gps = []

        for i in range(self.nparams):

            gp = treegp.GPInterpolation(kernel=self.kernels[i],
                                        optimizer=self.treegp_alias[self.optimizer],
                                        normalize=self.normalize,
                                        p0=[self.l0, 0, 0], white_noise=self.white_noise,
                                        n_neighbors=self.n_neighbors,
                                        average_fits=self.average_fits, indice_meanify = i,
                                        nbins=self.nbins,
                                        min_sep=self.min_sep, max_sep=self.max_sep)
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
                if star.fit.params is None:
                    y0_updated = np.zeros(self.nparams)
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
                  ('ROWS', self.rows.dtype,  self.rows.shape),
                  ('OPTIMIZER', str, len(self.optimizer))]

        data = np.empty(1, dtype=dtypes)
        data['INIT_THETA'] = init_theta
        data['FIT_THETA'] = fit_theta
        data['X'] = self._X
        data['Y'] = self._y
        data['Y_ERR'] = self._y_err
        data['ROWS'] = self.rows
        data['OPTIMIZER'] = self.optimizer

        fits.write_table(data, extname=extname+'_kernel')

    def _finish_read(self, fits, extname):
        data = fits[extname+'_kernel'].read()
        # Run fit to set up GP, but don't actually do any hyperparameter optimization. Just
        # set the GP up using the current hyperparameters.
        # Need to give back average fits files if needed.

        init_theta = np.atleast_1d(data['INIT_THETA'][0])
        fit_theta = np.atleast_1d(data['FIT_THETA'][0])

        self._X = np.atleast_1d(data['X'][0])
        self._y = np.atleast_1d(data['Y'][0])
        self._y_err = np.atleast_1d(data['Y_ERR'][0])
        self.rows = np.atleast_1d(data['ROWS'][0])

        self._init_theta = init_theta
        self.nparams = len(init_theta)
        self.optimizer = data['OPTIMIZER'][0]

        if len(self.kernel_template)==1:
            self.kernels = [self.kernel_template[0] for i in range(self.nparams)]
        else:
            self.kernels = [ker for ker in self.kernel_template]

        self.gps = []
        for i in range(self.nparams):

            gp = treegp.GPInterpolation(kernel=self.kernels[i],
                                        optimizer=self.treegp_alias[self.optimizer],
                                        normalize=self.normalize,
                                        p0=[3000., 0.,0.],
                                        white_noise=self.white_noise, n_neighbors=4,
                                        average_fits=None,
                                        nbins=20, min_sep=None, max_sep=None)
            gp.kernel_template.clone_with_theta(fit_theta[i])
            gp.initialize(self._X, self._y[:,i], y_err=self._y_err[:,i])
            self.gps.append(gp)

