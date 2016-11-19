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
from .interp import Interp
from .star import Star, StarFit
import numpy as np

class BasisInterp(Interp):
    """An Interp class that works whenever the interpolating functions are
    linear sums of basis functions.  Does things the "slow way" to be stable to
    degenerate fits to individual stars, instead of fitting to parameter sets
    produced by single stars.

    First time coding this we will assume that each element of the PSF parameter
    vector p is a linear combination of the same set of basis functions across the
    focal plane,
    p_i = \sum_{j} q_{ij} K_j(u,v,other stellar params).

    The property degenerate_points is set to True to indicate that this interpolator
    uses the alpha/beta quadratic form of chisq for each sample, rather than assuming
    that a best-fit parameter vector is available at every sample.

    Internally we'll store the interpolation coefficients in a 2d array of dimensions
    (nparams, nbases)

    Note: This is an abstract base class.  The concrete class you probably want to use
    is BasisPolynomial.
    """
    def __init__(self):
        self.degenerate_points = True  # This Interpolator uses chisq quadratic forms
        self.q = None

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        This class will initialize everything to have constant PSF parameter vector taken
        from the first Star in the list.

        :param stars:       A list of Star instances to use to initialize.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           A new list of Stars which have their parameters initialized.
        """
        c = stars[0].fit.params.copy()
        self.q = c[:,np.newaxis] * self.constant(1.)[np.newaxis,:]
        stars = self.interpolateList(stars)
        return stars

    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        raise NotImplemented("Cannot call `basis` for abstract base class BasisInterp. "
                             "You probably want to use BasisPolynomial.")

    def constant(self, value=1.):
        """Return 1d array of coefficients that represent a polynomial with constant value.

        :param value:  The value to use as the constant term.  [default: 1.]

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        raise NotImplemented("Cannot call `constant` for abstract base class BasisInterp. "
                             "You probably want to use BasisPolynomial.")

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data.
        The StarFit element of each Star in the list is assumed to hold valid
        alpha and beta members specifying depending of chisq on differential
        changes to its parameter vector.

        :param stars:       A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        if self.q is None:
            raise RuntimeError("Attempt to solve() before initialize() of BasisInterp")

        # Empty A and B
        A = np.zeros( self.q.shape+self.q.shape, dtype=float)
        B = np.zeros_like(self.q)

        for s in stars:
            # Get the basis function values at this star
            K = self.basis(s)
            # Sum contributions into A, B
            B += s.fit.beta[:,np.newaxis] * K
            tmp = s.fit.alpha[:,:,np.newaxis] * K
            A += K[np.newaxis,:,np.newaxis,np.newaxis] * tmp[:,np.newaxis,:,:]
        # Reshape to have single axis for all q's
        B = B.flatten()
        nq = B.shape[0]
        A = A.reshape(nq,nq)
        if logger:
            logger.debug('Beginning solution of matrix size %d',A.shape[0])
        dq = np.linalg.solve(A,B)
        if logger:
            logger.debug('...finished solution')
        self.q += dq.reshape(self.q.shape)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance holding the interpolated parameters
        """
        if self.q is None:
            raise RuntimeError("Attempt to interpolate() before initialize() of BasisInterp")

        K = self.basis(star)
        p = np.dot(self.q,K)
        if star.fit is None:
            fit = StarFit(p)
        else:
            fit = star.fit.newParams(p)
        return Star(star.data, fit)


class BasisPolynomial(BasisInterp):
    """A version of the Polynomial interpolator that works with BasisModels and can use the
    quadratic form of the chisq information it calculates.  It works better than the regular
    Polynomial interpolator when there is missing or degenerate information.

    The order is the highest power of a key to be used.  This can be the same for all keys
    or you may provide a list of separate order values to be used for each key.  (e.g. you
    may want to use 2nd order in the positions, but only 1st order in the color).

    All combinations of powers of keys that have total order <= maxorder are used.
    The maximum order is normally the maximum order of any given key's order, but you may
    specify a larger value.  (e.g. to use 1, x, y, xy, you would specify order=1, maxorder=2.)

    Ranges for each key can be given which are rescaled into the [-1,1] interval that
    will help keep polynomial arguments at O(1).

    :param order:       The order to use for each key.  Can be a single value (applied to all
                        keys) or an array matching number of keys.
    :param keys:        List of keys for properties that will be used as the polynomial arguments.
                        [default: ('u','v')]
    :param ranges:      Range for each key to be linearly remapped to [-1,1] interval before
                        calculating polynomials.  Can be a single tuple (which will be used for all
                        keys) or a list with a tuple for each key. [default: None]
    :param maxorder:    The maximum total order to use for cross terms between keys.
                        [default: None, which uses the maximum value of any individual key's order]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, order, keys=('u','v'), ranges=None, maxorder=None, logger=None):
        super(BasisPolynomial, self).__init__()

        self._keys = keys
        if hasattr(order,'len'):
            if not len(order)==len(keys):
                raise ValueError('Number of provided orders does not match number of keys')
            self._orders = order
        else:
            self._orders = (order,) * len(keys)

        if maxorder is None:
            self._maxorder = np.max(self._orders)
        else:
            self._maxorder = maxorder

        if self._maxorder<0 or np.any(np.array(self._orders) < 0):
            # Exception if we have any requests for negative orders
            raise ValueError('Negative polynomial order specified')

        # TODO: Need to update the Interp write command to handle lists.
        #       Or write a custom BasisPolynomial.write function.
        self.kwargs = {
            'order' : order,
        }

        # Now build a mask that picks the desired polynomial products
        # Start with 1d arrays giving orders in all dimensions
        ord_ranges = [np.arange(order+1,dtype=int) for order in self._orders]
        # Nifty trick to produce n-dim array holding total order
        sumorder = sum(np.ix_(*ord_ranges))
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
        else:
            rr = ranges

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

        self._center = (np.array(right)+np.array(left))/2.
        self._scale =  (np.array(right)-np.array(left))/2.

    def getProperties(self, star):
        return np.array([star.data[k] for k in self._keys], dtype=float)

    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        # Get the interpolation key values
        vals = self.getProperties(star)
        # Rescale to nominal (-1,1) interval
        vals = self._scale * (vals-self._center)
        # Make 1d arrays of all needed powers of keys
        pows1d = []
        for i,o in enumerate(self._orders):
            p = np.ones(o+1,dtype=float)
            p[1:] = vals[i]
            pows1d.append(np.cumprod(p))
        # Use trick to produce outer product of all these powers
        pows2d = np.ix_(*pows1d)[0]
        for p in np.ix_(*pows1d)[1:]:
            pows2d = pows2d * p

        # Return linear array of terms making total power constraint
        return pows2d[self._mask]

    def constant(self, value=1.):
        """Return 1d array of coefficients that represent a polynomial with constant value.

        :param value:  The value to use as the constant term.  [default: 1.]

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        out = np.zeros( np.count_nonzero(self._mask), dtype=float)
        out[0] = value  # The constant term is always first.
        return out

    def _finish_write(self, fits, extname):
        """Write the solution to a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        if self.q is None:
            raise RuntimeError("Solution not set yet.  Cannot write this BasisPolynomial.")

        dtypes = [ ('q', float, self.q.shape) ]
        data = np.zeros(1, dtype=dtypes)
        data['q'] = self.q
        fits.write_table(data, extname=extname + '_solution')

    def _finish_read(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interpolator information.
        """
        data = fits[extname + '_solution'].read()
        self.q = data['q'][0]

