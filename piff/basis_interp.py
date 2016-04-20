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
.. module:: interp
"""

from __future__ import print_function
from interp import Interpolator
from starfit import Star
import numpy

class BasisInterpolator(Interpolator):
    """An Interpolator class that works whenever the interpolating functions are
    linear sums of basis functions.  Does things the "slow way" to be stable to
    degenerate fits to individual stars, instead of fitting to parameter sets
    produced by single stars.

    First time coding this we will assume that each element of the PSF parameter
    vector p is a linear combination of the same set of basis functions across the
    focal plane,
    p_i = \sum_{j} q_{ij} K_j(u,v,other stellar params).

    The _basis_ argument to the constructor is an object that will return the vector
    K when given the StarData as a function argument. It should also have a method
    basis.constant(c) which returns the coefficient vector that generates a constant
    value c across the focal plane (which will be used for initialization of the q
    vector)

    Internally we'll store the interpolation coefficients in a 2d array of dimensions
    (nparams, nbases)

    """
    def __init__(self, basis, star):
        """Initialize a new linear interpolator.
        :param basis:   An object which returns values of the basis functions for
        a specified star.
        :param star:    A Star instance whose parameter vector will be assumed to
        specify the initial PSF for all stars in the subsequent solve().
        """
        self._basis = basis
        self.q = star.fit.params[:,numpy.newaxis] * basis.constant(1.)[numpy.newaxis,:]
        
    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given some data.
        The StarFit element of each Star in the list is assumed to hold valid
        alpha and beta members specifying depending of chisq on differential
        changes to its parameter vector.

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """

        # Empty A and B
        A = numpy.zeros( self.q.shape+self.q.shape, dtype=float)
        B = numpy.zeros_like(self.q)
        
        for s in star_list:
            # Get the basis function values at this star
            K = self._basis(s.data)
            # Sum contributions into A, B
            B += s.fit.beta[:,numpy.newaxis] * K
            tmp = s.fit.alpha[:,:,numpy.newaxis] * K
            A += K[numpy.newaxis,:,numpy.newaxis,numpy.newaxis] * tmp[:,numpy.newaxis,:,:]
        # Reshape to have single axis for all q's
        B = B.flatten()
        nq = B.shape[0]
        A = A.reshape(nq,nq)
        dq = numpy.linalg.solve(A,B)
        self.q += dq.reshape(self.q.shape)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        K = self._basis(star.data)
        p = numpy.dot(self.q,K)
        return Star(star.data, star.fit.newParams(p))


class UVPolyBasis(object):
    """A class to generate polynomials in u & v as bases for LinearInterpolator.
    For now, it's assuming 2d, and order is max sum of powers.
    Nothing fancy with domain yet either.
    """

    def __init__(self,order):
        """Set up 2d polynomial basis function calculator

        :param order: maximum sum of orders of u,v
        """
        self._order = order
        i = numpy.arange(order+1, dtype=int)[:,numpy.newaxis] * numpy.ones(order+1,dtype=int)
        j = numpy.arange(order+1, dtype=int) * numpy.ones(order+1,dtype=int)[:,numpy.newaxis]
        self._mask = (i+j) <= order
        return

    def __call__(self,sdata):
        """Return 1d array of polynomial basis values for this star

        :param sdata:  A StarData instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        upow = numpy.ones(self._order+1, dtype=float)
        upow[1:] = sdata['u']
        upow = numpy.cumprod(upow)
        vpow = numpy.ones(self._order+1, dtype=float)
        vpow[1:] = sdata['v']
        vpow = numpy.cumprod(vpow)
        uvpow = upow[:,numpy.newaxis] * vpow
        return uvpow[self._mask]

    def constant(self,c=1.):
        """Return 1d array of coefficients that represent a polynomial
        with constant value c
        """
        out = numpy.zeros( (self._order+1)*(self._order+2)/2, dtype=float)
        out[0] = c  # The constant term is always first.
        return out
