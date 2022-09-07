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
.. module:: knn_interp
"""

import numpy as np
import galsim

from .interp import Interp
from .star import Star, StarFit

class kNNInterp(Interp):
    """
    An interpolator that uses sklearn KNeighborsRegressor to interpolate a
    single surface

    :param keys:        A list of star attributes to interpolate from [default: ('u', 'v')]
    :param n_neighbors: Number of neighbors used for interpolation. [default: 15]
    :param weights:     Weight function used in prediction. Possible values are 'uniform',
                        'distance', and a callable function which accepts an array of distances
                        and returns an array of the same shape containing the weights.
                        [default: 'uniform']
    :param algorithm:   Algorithm used to compute nearest neighbors. Possible values are
                        'ball_tree', 'kd_tree', 'brute', and 'auto', which tries to determine the
                        best choice. [default: 'auto']
    :param p:           Power parameter of distance metrice. p=2 is default euclidean distance,
                        p=1 is manhattan. [default: 2]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), n_neighbors=15, weights='uniform', algorithm='auto',
                 p=2,logger=None):
        self.kwargs = {
            'keys': keys,
            }
        self.knr_kwargs = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'p': p,
            }
        self.kwargs.update(self.knr_kwargs)

        self.keys = keys

        from sklearn.neighbors import KNeighborsRegressor
        self.knn = KNeighborsRegressor(**self.knr_kwargs)

    @property
    def property_names(self):
        """List of properties used by this interpolant.
        """
        return self.keys

    def _fit(self, locations, targets, logger=None):
        """Update the Neighbors Regressor with data

        :param locations:   The locations for interpolating. (n_samples, n_features).
                            (In sklearn parlance, this is 'X'.)
        :param targets:     The target values. (n_samples, n_targets).
                            (In sklearn parlance, this is 'y'.)
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.knn.fit(locations, targets)
        self.locations = locations
        logger.debug('locations updated to shape: %s', self.locations.shape)
        self.targets = targets
        logger.debug('targets updated to shape: %s', self.targets.shape)

    def _predict(self, locations, logger=None):
        """Predict from knn.

        :param locations:   The locations for interpolating. (n_samples, n_features).
                            In sklearn parlance, this is 'X'

        :returns:   Regressed parameters y (n_samples, n_targets)
        """
        logger = galsim.config.LoggerWrapper(logger)
        regression = self.knn.predict(locations)
        logger.debug('Regression shape: %s', regression.shape)
        return regression

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
        :param logger:      A logger object for logging debug info. [default: None]
        """
        return stars

    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        locations = np.array([self.getProperties(star) for star in star_list])
        targets = np.array([star.fit.params for star in star_list])
        self._fit(locations, targets)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position. Calls interpolateList because sklearn prefers list input anyways

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
        logger = galsim.config.LoggerWrapper(logger)
        locations = np.array([self.getProperties(star) for star in star_list])
        targets = self._predict(locations)
        star_list_fitted = []
        for yi, star in zip(targets, star_list):
            if star.fit is None:
                fit = StarFit(yi)
            else:
                fit = star.fit.newParams(yi)
            star_list_fitted.append(Star(star.data, fit))
        return star_list_fitted

    def _finish_write(self, fits, extname):
        """Write the solution to a FITS binary table.

        Save the knn params and the locations and targets arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension with the interp information.
        """

        dtypes = [('LOCATIONS', self.locations.dtype, self.locations.shape),
                  ('TARGETS', self.targets.dtype, self.targets.shape),
                  ]
        data = np.empty(1, dtype=dtypes)
        # assign
        data['LOCATIONS'] = self.locations
        data['TARGETS'] = self.targets

        # write to fits
        fits.write_table(data, extname=extname + '_solution')

    def _finish_read(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension with the interp information.
        """
        data = fits[extname + '_solution'].read()

        # self.locations and self.targets assigned in _fit
        self._fit(data['LOCATIONS'][0], data['TARGETS'][0])
