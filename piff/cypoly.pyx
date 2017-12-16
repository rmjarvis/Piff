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
.. module:: cyanalytic
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, log, pi
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1] cypoly(
        np.ndarray[np.float64_t, ndim=2] X,
        np.ndarray[np.float64_t, ndim=1] coef,
        np.ndarray[np.int64_t, ndim=2] indices):
    """Deals with complicated sparse polynomials

    :param X:           array of values [nstar, nvar] we make polynomial out of
                        that is one-hot (ie the first term of each entry is 1)
    :param coef:        sets of coefficients [ncoef]
    :param indices:     sets which indices we are multiplying together [ncoef, norder]

    :returns y:         [nstar] values of the polynomial
    """

    cdef unsigned int nstar = X.shape[0]
    cdef unsigned int nvar = X.shape[1]
    cdef unsigned int ncoef = coef.shape[0]
    cdef unsigned int norder = indices.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(nstar, dtype=np.float64)

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef np.float64_t term

    for i in range(0, nstar):
        for j in range(0, ncoef):
            term = 1
            for k in range(0, norder):
                term *= X[i, indices[j, k]]
            term *= coef[j]
            y[i] += term

    return y

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] cypoly_full(
        np.ndarray[np.float64_t, ndim=2] X,
        np.ndarray[np.int64_t, ndim=2] indices):
    """Deals with complicated sparse polynomials

    :param X:           array of values [nstar, nvar] we make polynomial out of
                        that is one-hot (ie the first term of each entry is 1)
    :param indices:     sets which indices we are multiplying together [ncoef, norder]

    :returns Xpoly:     [nstar, ncoef] values of the polynomial
    """

    cdef unsigned int nstar = X.shape[0]
    cdef unsigned int nvar = X.shape[1]
    cdef unsigned int ncoef = indices.shape[0]
    cdef unsigned int norder = indices.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] Xpoly = np.zeros((nstar, ncoef), dtype=np.float64)

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef np.float64_t term

    for i in range(0, nstar):
        for j in range(0, ncoef):
            term = 1
            for k in range(0, norder):
                term *= X[i, indices[j, k]]
            Xpoly[i, j] = term

    return Xpoly
