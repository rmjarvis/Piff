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

from .interp import Interp
from .star import Star, StarFit

import numpy as np

class GPInterp(Interp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface.
    """
    def __init__(self, theta0=1e-1, thetaL=None, thetaU=None, nugget=None, logger=None):
        """Create the GP interpolator.

        :param logger:      A logger object for logging debug info. [default: None]
        :param theta0:      Double array_like, optional
            An array with shape (n_features, ) or (1, ).
            The parameters in the autocorrelation model.
            If thetaL and thetaU are also specified, theta0 is considered as
            the starting point for the maximum likelihood estimation of the
            best set of parameters.
            Default assumes isotropic autocorrelation model with theta0 = 1e-1.
        :param thetaL:      Double array_like, optional
            An array with shape matching theta0's.
            Lower bound on the autocorrelation parameters for maximum
            likelihood estimation.
            Default is None, so that it skips maximum likelihood estimation and
            it uses theta0.
        :param thetaU:      Double array_like, optional
            An array with shape matching theta0's.
            Upper bound on the autocorrelation parameters for maximum
            likelihood estimation.
            Default is None, so that it skips maximum likelihood estimation and
            it uses theta0.
        """

        self.gp_kwargs = {
            'theta0' : theta0,
            'thetaL' : thetaL,
            'thetaU' : thetaU,
            'nugget' : nugget
        }

        from sklearn.gaussian_process import GaussianProcess
        self.gp = GaussianProcess(**self.gp_kwargs)

    def _fit(self, locations, targets, logger=None):
        """Update the Gaussian Process Regressor with data (and solve for hyperparameters?)

        :param locations:   The locations for interpolating. (n_samples, n_features).
                            (In sklearn parlance, this is 'X'.)
        :param targets:     The target values. (n_samples, n_targets).
                            (In sklearn parlance, this is 'y'.)
        """
        # from sklearn.preprocessing import StandardScaler
        # self._X_scaler = StandardScaler()
        # self._Y_scaler = StandardScaler()
        # self._scaled_locations = self._X_scaler.fit_transform(locations)
        # self._scaled_targets = self._Y_scaler.fit_transform(targets)
        # self.gp.fit(self._scaled_locations, self._scaled_targets)
        self.gp.fit(locations, targets)
        # print("locations.shape = {}".format(locations.shape))
        # print("targets.shape = {}".format(targets.shape))
        # print("theta_ = {}".format(self.gp.theta_))
        if logger:
            logger.debug("theta_ = {}".format(self.gp.theta_))
            logger.debug("GP updated!")

    def _predict(self, locations, logger=None):
        """Predict from gp.

        :param locations:   The locations for interpolating. (n_samples, n_features).
                            In sklearn parlance, this is 'X'

        :returns:   Regressed parameters y (n_samples, n_targets)
        """
        # print("locations.shape = {}".format(locations.shape))
        return self.gp.predict(locations)

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param stars:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.solve(stars, logger=logger)
        return self.interpolateList(stars)

    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        locations = np.array([(star.u, star.v) for star in star_list])
        targets = np.array([star.fit.params for star in star_list])
        self._fit(locations, targets)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.
        Calls interpolateList because sklearn prefers list input anyways.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        # because of sklearn formatting, call interpolateList and take 0th entry
        return self.interpolateList([star], logger=logger)[0]

    def interpolateList(self, star_list, logger=None):
        """Perform the interpolation for a list of stars.

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """

        locations = np.array([(star.u, star.v) for star in star_list])
        targets = self._predict(locations)
        star_list_fitted = []
        for yi, star in zip(targets, star_list):
            if star.fit is None:
                fit = StarFit(yi)
            else:
                fit = star.fit.newParams(yi)
            star_list_fitted.append(Star(star.data, fit))
        return star_list_fitted
