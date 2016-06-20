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

from __future__ import print_function

from .interp import Interp
from .starfit import Star, StarFit

import numpy

from sklearn.neighbors import KNeighborsRegressor

class kNNInterp(Interp):
    """
    An interpolator that uses sklearn KNeighborsRegressor to interpolate a
    single surface
    """
    def __init__(self, logger=None, **knr_kwargs):
        """Create the kNN interpolator

        :param attr_interp: A list of star attributes to interpolate from
        :param attr_target: A list of star attributes to interpolate to
        :param logger:      A logger object for logging debug info. [default: None]
        :param knr_kwargs:  Arguments that will be passed to the regressor.
        """

        self.kwargs = {
            'n_neighbors': 15,
            'weights': 'uniform',
            'algorithm': 'auto',
            'p': 2,
            }
        self.kwargs.update(knr_kwargs)
        self.knn = {}

    def build(self, attr_interp, attr_target, logger=None):
        self.attr_interp = numpy.array(attr_interp)
        self.attr_target = numpy.array(attr_target)
        for target in self.attr_target:
            self.knn[target] = KNeighborsRegressor(**self.kwargs)

    def _fit(self, X, y, logger=None):
        """Update the Neighbors Regressor with data

        :param X:   The locations for interpolating. (n_samples, n_features)
        :param y:   The target values. (n_samples, n_targets)
        """
        for key, yi in zip(self.attr_target, y.T):
            self.knn[key].fit(X, yi)
        if logger:
            logger.debug('knn updated to keys: %s', self.knn.keys())
        self.X = X
        if logger:
            logger.debug('X updated to shape: %s', self.X.shape)
        self.y = y
        if logger:
            logger.debug('y updated to shape: %s', self.y.shape)

    def _predict(self, X, logger=None):
        """Predict from knn.

        :param X:   The locations for interpolating. (n_samples, n_features)

        :returns:   Regressed parameters y (n_samples, n_targets)
        """
        regression = numpy.array([self.knn[key].predict(X) for key in self.attr_target]).T
        if logger:
            logger.debug('Regression shape: %s', regression.shape)
        return regression

    def getProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the independent variables for the
        interpolation.

        Take self.attr_interp from star.data

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A numpy vector of these properties.
        """
        return numpy.array([star.data[key] for key in self.attr_interp])

    def getFitProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the dependent variables for the
        interpolation.

        Take self.attr_target from star.fit

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A numpy vector of these properties.
        """
        return numpy.array([star.fit[key] for key in self.attr_target])

    def initialize(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.solve(star_list, logger=logger)

    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        X = numpy.array([self.getProperties(star) for star in star_list])
        y = numpy.array([self.getFitProperties(star) for star in star_list])
        self._fit(X, y)

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

        X = numpy.array([self.getProperties(star) for star in star_list])
        # y = numpy.array([self.getFitProperties(star) for star in star_list])
        y = self._predict(X)
        star_list_fitted = []
        for yi, star in zip(y, star_list):
            if star.fit is None:
                fit = StarFit(yi)
            else:
                try:
                    fit = star.fit.newParams(yi)
                except TypeError:
                    if logger:
                        logger.info('Warning, stars interpolated to fewer params! %s', len(fit))
                    fit = StarFit(yi)
            star_list_fitted.append(Star(star.data, fit))
        return star_list_fitted

    def writeSolution(self, fits, extname):
        """Write the solution to a FITS binary table.

        Save the knn params and the X and y arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        dtypes = [('X', self.X.dtype, self.X.shape), ('Y', self.y.dtype, self.y.shape),
                  ('ATTR_TARGET', self.attr_target.dtype, self.attr_target.shape),
                  ('ATTR_INTERP', self.attr_interp.dtype, self.attr_interp.shape),]
        data = numpy.empty(1, dtype=dtypes)
        # assign
        data['X'] = self.X
        data['Y'] = self.y
        data['ATTR_TARGET'] = self.attr_target
        data['ATTR_INTERP'] = self.attr_interp

        # put the knn params in the header?
        header = self.kwargs

        # write to fits
        fits.write_table(data, extname=extname, header=header)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        The extension should contain the same values as are saved
        in the writeSolution method.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        # header = fits[extname].read_header()
        data = fits[extname].read()

        self.X = data['X'][0]
        self.y = data['Y'][0]
        self.attr_target = data['ATTR_TARGET'][0]
        self.attr_interp = data['ATTR_INTERP'][0]

        # self.kwargs = header

        self.build(self.attr_interp, self.attr_target)
        self._fit(self.X, self.y)
