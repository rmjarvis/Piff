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
.. module:: sklearn_gp_interp
"""

import numpy as np
import warnings

from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter

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
    :param npca:        Number of principal components to keep.  [default: 0, which means don't
                        decompose PSF parameters into principle components]
    :param normalize:   Whether to normalize the interpolation parameters to have a mean of 0.
                        Normally, the parameters being interpolated are not mean 0, so you would
                        want this to be True, but if your parameters have an a priori mean of 0,
                        then subtracting off the realized mean would be invalid.  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), kernel='RBF()', optimize=True, npca=0, normalize=True,
                 logger=None):
        from sklearn.gaussian_process import GaussianProcessRegressor

        self.keys = keys
        self.kernel = kernel
        self.npca = npca
        self.degenerate_points = False

        self.kwargs = {
            'keys': keys,
            'optimize': optimize,
            'npca': npca,
            'kernel': kernel
        }
        optimizer = 'fmin_l_bfgs_b' if optimize else None
        self.gp = GaussianProcessRegressor(self._eval_kernel(self.kernel), optimizer=optimizer,
                                           normalize_y=normalize)

    @staticmethod
    def _eval_kernel(kernel):
        # Some import trickery to get all subclasses of sklearn.gaussian_process.kernels.Kernel
        # into the local namespace without doing "from sklearn.gaussian_process.kernels import *"
        # and without importing them all manually.
        def recurse_subclasses(cls):
            out = []
            for c in cls.__subclasses__():
                out.append(c)
                out.extend(recurse_subclasses(c))
            return out
        clses = recurse_subclasses(Kernel)
        for cls in clses:
            module = __import__(cls.__module__, globals(), locals(), cls)
            execstr = "{0} = module.{0}".format(cls.__name__)
            exec(execstr, globals(), locals())

        from numpy import array

        try:
            k = eval(kernel)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to evaluate kernel string {0!r}.  "
                               "Original exception: {1}".format(kernel, e))
        return k

    def _fit(self, X, y, logger=None):
        """Update the GaussianProcessRegressor with data
        :param X:  The independent covariates.  (n_samples, n_features)
        :param y:  The dependent responses.  (n_samples, n_targets)
        """
        # Save these for potential read/write.
        self._X = X
        self._y = y
        if self.npca > 0:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.npca)
            self._pca.fit(y)
            y = self._pca.transform(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Sometimes the next line emits a warning along the lines of:
            # UserWarning: fmin_l_bfgs_b terminated abnormally with the  state:
            # {'grad': array([-0.29692092, -4.153523  ,  2.9923153 ]),
            # 'task': b'ABNORMAL_TERMINATION_IN_LNSRCH', 'funcalls': 70, 'nit': 2, 'warnflag': 2}
            # As far as I can tell, it's not actually harmful, so just ignore it.
            self.gp.fit(X, y)

    def _predict(self, Xstar):
        """ Predict responses given covariates.
        :param X:  The independent covariates at which to interpolate.  (n_samples, n_features).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        ystar = self.gp.predict(Xstar)
        if self.npca > 0:
            ystar = self._pca.inverse_transform(ystar)
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
        return stars

    def solve(self, stars=None, logger=None):
        """Set up this GPInterp object.

        :param stars:    A list of Star instances to interpolate between
        :param logger:   A logger object for logging debug info. [default: None]
        """
        X = np.array([self.getProperties(star) for star in stars])
        y = np.array([star.fit.params for star in stars])
        self._fit(X, y, logger=logger)

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
        y = self._predict(Xstar)
        fitted_stars = []
        for y0, star in zip(y, stars):
            if star.fit is None:
                fit = StarFit(y)
            else:
                fit = star.fit.newParams(y0)
            fitted_stars.append(Star(star.data, fit))
        return fitted_stars

    def _finish_write(self, fits, extname):
        # Note, we're only storing the training data and hyperparameters here, which means the
        # Cholesky decomposition will have to be re-computed when this object is read back from
        # disk.
        init_theta = self.gp.kernel.theta
        fit_theta = self.gp.kernel_.theta
        dtypes = [('INIT_THETA', init_theta.dtype, init_theta.shape),
                  ('FIT_THETA', fit_theta.dtype, fit_theta.shape),
                  ('X', self._X.dtype, self._X.shape),
                  ('Y', self._y.dtype, self._y.shape)]

        data = np.empty(1, dtype=dtypes)
        data['INIT_THETA'] = init_theta
        data['FIT_THETA'] = fit_theta
        data['X'] = self._X
        data['Y'] = self._y

        fits.write_table(data, extname=extname+'_kernel')

    def _finish_read(self, fits, extname):
        data = fits[extname+'_kernel'].read()
        # Run fit to set up GP, but don't actually do any hyperparameter optimization.  Just
        # set the GP up using the current hyperparameters.
        self.gp.kernel.theta = np.atleast_1d(data['FIT_THETA'][0])
        old_optimizer, self.gp.optimizer = self.gp.optimizer, None
        self._fit(data['X'][0], data['Y'][0])
        self.gp.optimizer = old_optimizer
        # Now that gp is setup, we can restore it's initial kernel.
        self.gp.kernel.theta = np.atleast_1d(data['INIT_THETA'][0])
