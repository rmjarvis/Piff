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
from .interp import Interpolator
from .starfit import Star
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
    vector), and a basis.getKeys() method that extracts the vector of quantities
    used as inputs to the basis functions.

    The property degenerate_points is set to True to indicate that this interpolator
    uses the alpha/beta quadratic form of chisq for each sample, rather than assuming
    that a best-fit parameter vector is available at every sample.
    
    Internally we'll store the interpolation coefficients in a 2d array of dimensions
    (nparams, nbases)
    """

    def __init__(self, basis, logger=None):
        """Initialize a new linear interpolator.
        :param basis:   An object which returns values of the basis functions for
        a specified star.
        """
        self._basis = basis
        self.degenerate_points = True  # This Interpolator uses chisq quadratic forms
        self.q = None
        
    def getKeys(self, sdata):
        """Extract the quantities to use as interpolation keys for a particular star's data.
        Obtains this from the Basis object.

        :param sdata:        A StarData instances from which to extract the properties used
                             for interpolation.

        :returns:            A numpy vector of these properties.
        """
        return self._basis.getKeys(sdata)
        
    def initialize(self, star_list, logger=None):
        """Initialize the interpolator and the parameter values in the Stars,
        prefatory to any solve iterations.  This class will initialize everything
        to have constant PSF parameter vector taken from the first Star in the list.

        :param star_list:   A list of Star instances to use to initialize.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           A new list of Stars which have their parameters initialized.
        """

        c = star_list[0].fit.params.copy()
        self.q = c[:,numpy.newaxis] * self._basis.constant(1.)[numpy.newaxis,:]

        return [Star(s.data, s.fit.newParams(c)) for s in star_list]
    
    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given some data.
        The StarFit element of each Star in the list is assumed to hold valid
        alpha and beta members specifying depending of chisq on differential
        changes to its parameter vector.

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """

        if self.q is None:
            raise RuntimeError("Attempt to solve() before initialize() of BasisInterpolator")

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
        if self.q is None:
            raise RuntimeError("Attempt to interpolate() before initialize() of BasisInterpolator")

        K = self._basis(star.data)
        p = numpy.dot(self.q,K)
        return Star(star.data, star.fit.newParams(p))


class PolyBasis(object):
    """A class to generate polynomials bases for LinearInterpolator.
    All combinations of powers of keys that have total order <=maxorder
    are used.  Maximum orders for each key can be specified.  Ranges
    for each key can be given which are rescaled into the [-1,1] interval that
    will help keep polynomial arguments at O(1).
    """

    def __init__(self, maxorder, keys=('u','v'), orders=None, ranges=None):
        """Set up a polynomial basis function calculator.

        :param maxorder: maximum sum of orders of all keys
        :param keys:     array of keys for StarData properties that will be used as the
                         polynomial arguments.  Defaults to using focal plane position (u,v)
        :param orders:   Maximum allowed order for each key.  Can be a single value (applied
                         to all keys) or an array matching number of keys.  Each value is
                         either a non-negative integer, or None, which will default to maxorder.
        :param ranges:   Range to be linearly remapped to [-1,1] interval before calculating
                         polynomials.  Can be a single tuple (which will be used for all dimensions)
                         or an array with a tuple for each key.  Any value of None will default
                         to range=(-1,1), i.e. no rescaling.
        """
        self._maxorder = maxorder
        self._keys = keys
        if orders is None:
            self._orders = (self._maxorder,) * len(keys)
        elif type(orders) is int:
            self._orders = (orders,) * len(keys)
        elif not len(orders)==len(keys):
            raise ValueError('Number of provided orders does not match number of keys')
        else:
            # Replace all None entries in the orders array with maxorder
            self._orders=()
            for o in orders:
                if o is None:
                    self._orders = self._orders + (self._maxorder,)
                else:
                    self._orders = self._orders + (o,)
        if self._maxorder<0 or numpy.any(numpy.array(self._orders) < 0):
            # Exception if we have any requests for negative orders
            raise ValueError('Negative polynomial order specified')

        # Now build a mask that picks the desired polynomial products
        # Start with 1d arrays giving orders in all dimensions
        ord = [numpy.arange(o+1,dtype=int) for o in self._orders]
        # Nifty trick to produce n-dim array holding total order
        sumorder = reduce(numpy.add, numpy.ix_(*ord))
        self._mask = sumorder <= self._maxorder

        # Set up the ranges: save the additive and multiplicative factors
        if ranges is None:
            # All dimensions take default
            rr = ((-1.,1.),) * len(keys)
        elif type(ranges[0]) is int and type(ranges[1]) is int:
            # Replicate a single range pair
            rr = (ranges,) * len(keys)
        elif not len(ranges)==len(keys):
             raise ValueError('Number of provided ranges does not match number of keys')

        # Copy all ranges, None means -1,1
        left=[]
        right=[]
        for r in rr:
            if r is None:
                left.append(-1.)
                right.append(1.)
            else:
                left.append(r[0])
                right.append(r[1])
        self._center = (numpy.array(right)+numpy.array(left))/2.
        self._scale =  (numpy.array(right)-numpy.array(left))/2.
                    
    def getKeys(self,sdata):
        return numpy.array([sdata[k] for k in self._keys], dtype=float)
    
    def __call__(self,sdata):
        """Return 1d array of polynomial basis values for this star

        :param sdata:  A StarData instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        # Get the interpolation key values
        vals = self.getKeys(sdata)
        # Rescale to nominal (-1,1) interval
        vals = self._scale * (vals-self._center)
        # Make 1d arrays of all needed powers of keys
        pows1d = []
        for i,o in enumerate(self._orders):
            p = numpy.ones(o+1,dtype=float)
            p[1:] = vals[i]
            pows1d.append(numpy.cumprod(p))
        # Use trick to produce outer product of all these powers
        pows2d = reduce(numpy.multiply, numpy.ix_(*pows1d))
        # Return linear array of terms making total power constraint
        return pows2d[self._mask]

    def constant(self,c=1.):
        """Return 1d array of coefficients that represent a polynomial
        with constant value c
        """
        out = numpy.zeros( numpy.count_nonzero(self._mask), dtype=float)
        out[0] = c  # The constant term is always first.
        return out
