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
.. module:: pixelmodel
"""

from __future__ import print_function
import numpy as np

class PixelModel(Model):
    """A PSF modeled as interpolation between a grid of points.

    The parameters of the model are the values at the grid points.  The grid is in uv
    space, with the pitch and size specified on construction. Optionally a boolean
    mask array can be passed specifying which tells which grid elements are non-zero.
    Interpolation will always assume values of zero outside of grid.

    PixelModel also needs an Interpolant on construction to specify how to determine
    values between grid points.

    """

    def __init__(self, du, n_side, interp, mask=None):
        """Constructor for PixelModel defines the PSF pitch, size, and interpolator.

        If a mask is given, n_side is ignored, and the PSF origin is taken to be
        at pixel [shape/2].  

        :param du: PSF model grid pitch (in uv units)
        :param n_side: number of PSF model points on each side of square array
        :param interp: an Interpolator to be used
        :param mask: optional square boolean 2d array, True where we want a non-zero value

        """
        self.du = du
        self.interp = interp
        if mask is None:
            if n_side <= 0:
                raise AttributeError("Non-positive PixelModel size {:d}".format(n_side))
            self.nx = n_side
            self.ny = n_side
            # Make a map of 2d position to index in 1d array
            self._indices = np.arange(n_side*n_side, dtype=int).reshape(n_side,n_side)
            self._params = np.zeros(n_side*n_side, dtype=float) # 1d parameter vector
        else:
            self.ny, self.nx = mask.shape
            self._params = np.zeros(np.count_nonzero(mask), dtype=float) #1d parameter vector
            # Map of 2d position to index in 1d array has -1 in unused grid positions
            self._indices = np.full( (self.ny,self.nx), -1, dtype=int)
            self._indices[mask] = np.arange(len(self._params), dtype=int)
        return
        
    def fitStar(self, star):
        """Fit the model parameters to the data for a single star.

        :param star:    A StarData instance

        :returns: self (for convenience of stringing together operations)
        """
        alpha, beta, gamma = self.getFit(star)
        dparam = np.solve(alpha, beta)
        # ??? Trap exception for singular matrix here?
        # ??? dparam = scipy.linalg.solve(alpha, beta, sym_pos=True) would be faster
        self.params += dparam
        return self

    def getFit(self, star):
        """Return dependence of chi^2 = -2 log L(D|p) on parameters for single star.
        Returns the quadratic form chi^2 = dp^T*alpha*dp - 2*beta*dp + gamma,

        linear models, an approximation more generally.

        :param star:   A StarData instance

        :returns: alpha, beta, gamma 
        """

        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.getDataVector()
        coeffs, psfy, psfx = interp.getCoefficients(u/du, v/dv)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        # First, shift psfy, psfx to reference a 0-indexed array
        psfy += self.ny/2
        psfx += self.nx/2
        # Mask references to invalid pixels
        # ???
        # Look up param index for each referenced pixel
        index1d = np.where(usable, self._indices[psfy, psfx], -1)
        pvals = np.where(index1d>=0, self.params[index1d],0.)
        resid = data - np.sum( coeffs * pvals, axis=1)

        rw = resid * weight
        gamma = np.sum(resid * rw)

        # Accumulate alpha and beta point by point.  I don't
        # know how to do purely with numpy calls instead of a loop.
        beta = np.zeros_like(self.params)
        alpha = np.zeros( (len(self.params),len(self.params)), dtype=float)
        for i in range(len(data)):
            usable = index1d[i,:]>=0
            ii = index1d[i,usable]
            cc = coeffs[i,usable]
            # beta_j += resid_i * weight_i * coeff_{ij}
            beta[ii] += rw[i] * cc
            # alpha_jk += weight_i * coeff_ij * coeff_ik
            dalpha = cc[np.newaxis,:]*cc[:,np.newaxis] * weight[i]
            iouter = np.broadcast_to(ii, (len(ii),len(ii)))
            alpha[iouter.flatten(), iouter.T.flatten()] += dalpha.flatten()
        return alpha, beta, gamma
        
    def drawImage(self, image, pos=None):
        """Draw the model on the given image.

        :param image:   A galsim.Image on which to draw the model.
        :param pos:     The position on the image at which to place the nominal center.
                        [default: None, which means to use the center of the image.]

        :returns: image
        """
        # ??? u, v = meshgrid of image coordinates
        coeffs, psfy, psfx = interp.getCoefficients(u/du, v/dv)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        # First, shift psfy, psfx to reference a 0-indexed array
        psfy += self.ny/2
        psfx += self.nx/2
        # Mask references to invalid pixels
        # ???
        # Look up param index for each referenced pixel
        index1d = np.where(usable, self._indices[psfy, psfx], -1)
        pvals = np.where(index1d>=0, self.params[index1d],0.)
        data = np.sum( coeffs * pvals, axis=1)
        # ??? fill image with data
        return image

    def getParameters(self):
        """Get the parameters of the model, to be used by the interpolator.

        :returns: a numpy array of the model parameters
        """
        return self.params.copy()

    def setParameters(self, params):
        """Set the parameters of the model, typically provided by an interpolator.

        :param params:  A numpy array of the model parameters

        :returns: self
        """
        if len(params) != len(self.params):
            raise AttributeError("Wrong number of parameters")
        self.params = params
        return self

def Interpolant(object):
    """Interface for interpolators
    """
    
    def range(self):
        """Size of interpolation kernel

        :returns: Maximum distance from target to source pixel.
        """
        raise NotImplemented("Derived classes must define the range function")

    def __call__(self, u, v):
        """Calculate interpolation coefficient for vector of target points

        Outputs will be 3 matrices, each of dimensions (nin, nkernel) where nin is
        number of input coordinates and nkernel is number of points in kernel footprint.
        The coeff matrix gives interpolation coefficients, then the y and x integer matrices
        give the grid point to which each coefficient is applied.
        
        :param u: 1d array of target u coordinates
        :param v: 1d array of target v coordinates

        :returns: coeff, y, x
        """
        raise NotImplemented("Derived classes must define the __call__ function")
    
def Lanczos(Interpolant):
    """Lanczos interpolator in 2 dimensions.
    """
    def __init__(self, order=3):
        """Initialize with the order of the filter
        """
        self.order = order
        return
    
    def range(self):
        return self.order

    def __call__(self, u, v):
        # Get integer and fractional parts of u, v
        # Set up coordinate arrays
        # Calculate Lanczos function
        return coeffs, x, y
