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
    def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', npca=0, logger=None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.kernel = kernel
        self.npca = npca
        self.gp = GaussianProcessRegressor(self.kernel, optimizer=optimizer)

    def _fit(self, X, y, logger=None):
        if self.npca > 0:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.npca, whiten=True)
            self._pca.fit(y)
            y = self._pca.transform(y)
        self.gp.fit(X, y)

    def _predict(self, Xstar):
        ystar = self.gp.predict(Xstar)
        if self.npca > 0:
            ystar = self._pca.inverse_transform(ystar)
        return ystar

    def initialize(self, stars, logger=None):
        self.solve(stars, logger=logger)
        return self.interpolateList(stars)

    def solve(self, stars=None, logger=None):
        X = np.array([(star.u, star.v) for star in stars])
        y = np.array([star.fit.params for star in stars])
        self._fit(X, y, logger=logger)

    def interpolate(self, star, logger=None):
        return self.interpolateList([star], logger=logger)[0]

    def interpolateList(self, stars, logger=None):
        X = np.array([(star.u, star.v) for star in stars])
        y = self._predict(X)
        fitted_stars = []
        for y0, star in zip(y, stars):
            if star.fit is None:
                fit = StarFit(y)
            else:
                fit = star.fit.newParams(y0)
            fitted_stars.append(Star(star.data, fit))
        return fitted_stars


from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter

class EmpiricalKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, fn):
        self.fn = fn
        assert self.fn(0, 0) == 1

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise RuntimeError("Cannot evaluate gradient for EmpiricalKernel.")

        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        # Only writen for 2D covariance at the moment
        xshift = np.subtract.outer(X[:,0], Y[:,0])
        yshift = np.subtract.outer(X[:,1], Y[:,1])
        return self.fn(xshift, yshift)


class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, invLam):
        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)

        if eval_gradient:
            if self.hyperparameter_cho_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                L_grad = np.zeros((self.ntheta, self.ndim, self.ndim), dtype=float)
                L_grad[(np.arange(self.ndim),)+self._d] = self._L[self._d]
                L_grad[(np.arange(self.ndim, self.ntheta),)+self._t] = 1.0

                half_invLam_grad = np.dot(L_grad, self._L.T)
                invLam_grad = half_invLam_grad + np.transpose(half_invLam_grad, (0, 2, 1))

                dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_grad = np.einsum("ijk,lkm,ijm->ijl", dX, invLam_grad, dX)
                K_gradient = -0.5 * K[:, :, np.newaxis] * dist_grad
                return K, K_gradient
        else:
            return K

    @property
    def hyperparameter_cho_factor(self):
        return Hyperparameter("ChoFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    @property
    def bounds(self):
        return np.array([(-5, 5)]*int(self.ntheta))
