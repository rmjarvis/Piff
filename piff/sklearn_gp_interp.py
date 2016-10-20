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

from .interp import Interp
from .star import Star, StarFit

import numpy as np

class GPInterp(Interp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface.
    """
    def __init__(self, kernel=None, npca=0, logger=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.kernel = kernel
        self.npca = npca
        self.gp = GaussianProcessRegressor(self.kernel)

    def _fit(self, X, y, logger=None):
        if self.npca > 0:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.npca, whiten=True)
            self._pca.fit(y)
            y = self._pca.transform(y)
        self.gp.fit(X, y)

    def _predict(self, Xstar):
        ystar = self.gp.predict(Xstar)
        if self._npca > 0:
            ystar = self._pca.inverse_transform(ystar)
        return ystar

    def initialize(self, stars, logger=None):
        self.solve(stars, logger=logger)
        return self.interpolateList(stars)

    def solve(self, star_list=None, logger=None):
        X = np.array([(star.u, star.v) for star in star_list])
        y = np.array([star.fit.params for star in star_list])
        self._fit(X, y, logger=logger)

    def interpolate(self, star, logger=None):
        return self.interpolateList([star], logger=logger)

    def interpolateList(self, stars, logger=None):
        X = np.array([(star.u, star.v) for star in stars])
        y = self._predict(X)
        fitted_stars = []
        for y0, star in zip(y, stars):
            if star.fit is None:
                fit = StarFit(y)
            else:
                fit = star.fit.newParams(y)
            fitted_stars.append(Star(star.data, fit))
        return fitted_stars
