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
.. module:: kernel
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import _check_length_scale


class ExplicitKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A kernel that wraps an arbitrary python function.

    :param  fn:  String that can be combined with 'lambda du,dv:' to eval into a lambda expression.
                 For example, fn="np.exp(-0.5*np.sqrt(du**2+dv**2)/0.1**2)" would make a Gaussian
                 kernel with scale length of 0.1.
    """
    def __init__(self, fn):
        self.fn = fn

        self.kwargs = {
            'fn': fn
        }

        self._fn = self._eval_fn(fn)
        self.theta = []

    @staticmethod
    def _eval_fn(fn):
        # Some potentially useful imports for the eval string.
        import math
        import numpy
        import numpy as np
        return eval("lambda du,dv:" + fn)

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise RuntimeError("Cannot evaluate gradient for ExplicitKernel.")

        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        shift = [np.subtract.outer(X[:,i], Y[:,i]) for i in range(X.shape[1])]
        return self._fn(*shift)


class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a radial basis function (essentially a
    squared exponential or Gaussian) but with arbitrary anisotropic covariance.

    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length.
    For optimization, it's necessary to reparameterize the inverse covariance matrix in such a way
    as to ensure that it's always positive definite.

    To this end, we define `theta` (abbreviated `th` below) such that::

        invLam = L * L.T
        L = [[exp(th[0])  0              0           ...    0                 0           ]
             [th[n]       exp(th[1])]    0           ...    0                 0           ]
             [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
             [...         ...            ...         ...    ...               ...         ]
             [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
             [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]

    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.

    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicRBF.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

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
            if self.hyperparameter_cholesky_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # dK_pq/dth_k = -0.5 * K_pq *
                #               ((x_p_i-x_q_i) * dInvLam_ij/dth_k * (x_q_j - x_q_j))
                # dInvLam_ij/dth_k = dL_ij/dth_k * L_ij.T  +  L_ij * dL_ij.T/dth_k
                # dL_ij/dth_k is a matrix with all zeros except for one element.  That element is
                # L_ij if k indicates one of the theta parameters landing on the Cholesky diagonal,
                # and is 1.0 if k indicates one of the thetas in the lower triangular region.
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
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

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

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds


class VonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """VonKarman kernel.

    Parameters:
        length_scale : float or array with shape (n_features,), default: 1.0
            The length scale of the kernel. If a float, an isotropic kernel is
            used. If an array, an anisotropic kernel is used where each dimension
            of l defines the length-scale of the respective feature dimension.

        length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
            The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters:
            X : array, shape (n_samples_X, n_features)
                Left argument of the returned kernel k(X, Y)

            Y : array, shape (n_samples_Y, n_features), (optional, default=None)
                Right argument of the returned kernel k(X, Y). If None, k(X, X)
                if evaluated instead.

            eval_gradient : bool (optional, default=False)
                Determines whether the gradient with respect to the kernel
                hyperparameter is determined. Only supported when Y is None.

        Returns:
            K : array, shape (n_samples_X, n_samples_Y)
                Kernel k(X, Y)

            K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
                The gradient of the kernel k(X, X) with respect to the
                hyperparameter of the kernel. Only returned when eval_gradient
                is True.
        """
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                         special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            K = squareform(K)

            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")

            dists = cdist(X, Y, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                       special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                raise ValueError(
                    "Gradient can only be evaluated with isotropic VonKarman kernel for the moment.")
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

class AnisotropicVonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a Von-Karman correlation function
    with an arbitrary anisotropic covariance.

    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length. For
    optimization, it's necessary to reparameterize the inverse covariance matrix in such a way as
    to ensure that it's always positive definite.

    To this end, we define `theta` (abbreviated `th` below) such that::

        invLam = L * L.T
        L = [[exp(th[0])  0              0           ...    0                 0           ]
             [th[n]       exp(th[1])]    0           ...    0                 0           ]
             [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
             [...         ...            ...         ...    ...               ...         ]
             [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
             [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]

    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.

    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicVonKarman.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            K = squareform(K)
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can not be evaluated.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            raise ValueError(
                "Gradient can not be evaluated.")
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

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

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds
