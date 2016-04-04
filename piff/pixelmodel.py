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

    The parameters of the model are the values at the grid points, although the constraint
    for unit flux means that not all grid points are free parameters.  The grid is in uv
    space, with the pitch and size specified on construction. Optionally a boolean
    mask array can be passed specifying which tells which grid elements are non-zero.
    Interpolation will always assume values of zero outside of grid.  Integral of PSF is
    forced to unity and, optionally, centroid is forced to origin.  As a consequence 1 (or 3)
    of the PSF pixel values will be missing from the parameter vector as they are determined
    by the flux (and centroid) constraints. And there is more covariance between pixel values.

    PixelModel also needs an Interpolant on construction to specify how to determine
    values between grid points.

    """

    def __init__(self, du, n_side, interp, mask=None, force_model_center=False):
        """Constructor for PixelModel defines the PSF pitch, size, and interpolator.

        If a mask is given, n_side is ignored, and the PSF origin is taken to be
        at pixel [shape/2].  

        :param du: PSF model grid pitch (in uv units)
        :param n_side: number of PSF model points on each side of square array
        :param interp: an Interpolator to be used
        :param mask: optional square boolean 2d array, True where we want a non-zero value
        :param force_model_center: If True, PSF model centroid is fixed at origin and
        PSF fitting will marginalize over stellar position.  If False, stellar position is
        fixed at input value and the fitted PSF may be off-center.
        """
        self.du = du
        self.interp = interp
        self._force_model_center = force_model_center
        if mask is None:
            if n_side <= 0:
                raise AttributeError("Non-positive PixelModel size {:d}".format(n_side))
            self._mask = np.ones( (n_side,n_side), dtype=bool)
        else:
            self._mask = mask

        self.ny, self.nx = self._mask.shape
        self._nparams = np.count_nonzero(mask)
        self._nparams -=1  # The flux constraint will remove 1 degree of freedom
        self._constraints = 1
        if self._force_model_center:
            self._nparams -= 2 # Centroid constraint will remove 2 more degrees of freedom
            self._constraints += 2

        # Now we need to make a 2d array whose entries are the indices of
        # each pixel in the 1d parameter array.  We will put the central
        # pixels (and first to top & right) at the front of the array
        # because we will be chopping these off when we enforce the
        # flux (and center) conditions on the PSF.
        # In this array, a negative entry is a pixel that is not being
        # fit (and always assumed to be zero, for interpolation purposes).
        self._indices = np.where( self._mask, self._constraints, -1)
        self._origin = (self.ny/2, self.nx/2)
        if not self._mask[self._origin]:
            raise AttributeError("Not happy with central PSF pixel being masked")
        self._indices[self._origin] = 0    # Central pixel for flux constraint
        if self._force_model_center:
            right = (self._origin[0], self._origin[1]+1)
            up = (self._origin[0]+1, self._origin[1])
            if not (self._mask[right] and self._mask[up]):
                raise AttributeError("Not happy with near-central PSF pixels being masked")
            self._indices[right] = 1
            self._indices[up] = 2
        free_param = self._indices >= self._constraints
        self._indices[free_param] = np.arange(self._constraints,
                                              self._constraints+self._nparams,
                                              dtype=int)

        # Next job is to create the flux/center constraint conditions.
        # ??? Could have some type of window function here, for now just
        # ??? using unweighted flux & centroid
        A = np.zeros( (self._constraints, self._constraints + self._nparams), dtype=float)
        B = np.zeros( (self._constraints,) dtype=float)
        A[0,:] = 1.
        B[0] = 1.  # That's the flux constraint - sum pixels to unity.  ??? Pixel area factor???
        
        if self._force_model_center:
            # Generate linear center constraints too
            delta_u = np.arange( -self._origin[1], self._indices.shape[1]-self._origin[1])
            A[1,:] = self._1dFrom2d(np.ones(self._indices.shape, dtype=float) * delta_u)
            B[1] = 0.
            delta_v = np.arange( -self._origin[0], self._indices.shape[0]-self._origin[0])
            A[2,:] = self._1dFrom2d(np.ones(self._indices.shape, dtype=float) * delta_vv[:,np.newaxis])
            B[2] = 0.
        
        ainv = np.linalg.inverse(A[:,:self._constraints])
        self._a = np.dot(ainv, A[:, self._constraints:])
        self._b = np.dot(ainv, B)
        # Now our constraints are that p0 = _b - _a * p1 where p0 are the (1 or 3) constrained
        # pixel values and p1 are the remaining free ones.
        return

    def _1dFrom2d(self, in2d, out1d):
        """Make a 1d array from a 2d array, using the model's
        mapping from the 2d psf grid to the 1d parameter array.

        :param in2d:    A 2d array matching the PSFs sample grid

        :returns:       A 1d array of the length of number of grid points in use

        :returns  None
        """
        out1d = np.zeros( (self._constraints + self._nparams,), dtype=in2d.dtype)
        out1d[self._indices[self._mask]] = in2d[self._mask]
        return out1d

    def _2dFrom1d(self, in1d, out2d):
        """Make a 2d array of the PSF from a 1d list of grid points, using the model's
        mapping from the 2d psf to the 1d parameter array.

        :param in1d:    A 1d array of values for PSF grid points in use

        :returns:       A 2d array representing the PSF, with zeros for grid points not in mask.
        """

        i = np.zeros( (in1d.shape[0]+1,), dtype=int)
        # The i array is the input array supplemented by a zero for pixels outside of mask
        i[:-1] = in1d
        # Now index the 1d array by the index array altered to point to the extra zero element
        # where the mask is False:
        return i[ np.where(self._mask, self._indices, len(i)-1)]

    def fillPSF(self, star, in2d):
        """ Initialize the PSF for a star from a given 2d uv-plane array.
        Sets elements outside the mask to zero, renormalizes to enforce flux
        condition, and checks centering condition if force_model_center=True

        :param star:    A Star instance to initialize
        :param in2d:    2d input array matching the PSF uv-plane grid

        :returns: None
        """
        params = self._1dFrom2d(in2d)

        # Renormalize to get unity flux
        params /= np.sum(params)
        # ??? Allow for other flux norms, pixel size, check centering ???

        star.params[:] = params[self._constraints:]  # Omit the constrained pixels
        return
        
    def makeStar(self, data, flux=0., center=(0.,0.)):
        """Create a Star instance that PixelModel can manipulate.

        :param data:    A StarData instance
        :param flux:    Initial estimate of stellar flux
        :param center:  Initial estimate of stellar center in world coord system

        :returns: Star instance
        """
        return Star(data, np.zeros(self._nparams, dtype=float), flux, center)

    def fit(self, star):
        """Fit the model parameters to the data for a single Star, updating its parameters,
        flux, and (optionally) center.

        :param star:    A Star instance

        :returns: None
        """
        self.chisq(star)  # Get chisq for linearized model and solve for parameters
        # That call will also update the flux (and center) of the Star
        dparam = np.solve(star.alpha, star.beta)
        # ??? Trap exception for singular matrix here?
        # ??? dparam = scipy.linalg.solve(alpha, beta, sym_pos=True) would be faster
        star.params += dparam
        return

    def chisq(self, star):
        """Calculate dependence of chi^2 = -2 log L(D|p) on PSF parameters for single star.
        as a quadratic form chi^2 = dp^T*alpha*dp - 2*beta*dp + gamma,
        where dp is the *shift* from current parameter values.  Marginalization over
        flux (and, optionally, center) are done by this routine. Results are saved in
        alpha,beta,gamma,flux, (center) attributes of Star.

        :param star:   A Star instance

        :returns: None
        """

        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()
        if self._force_model_center:
            coeffs, dcdu, dcdv, psfy, psfx = interp(u/du, v/du)
            dcdu /= du
            dcdv /= du
        else:
            coeffs, psfy, psfx = interp(u/du, v/du)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        # First, shift psfy, psfx to reference a 0-indexed array
        psfy += self._origin[0]
        psfx += self._origin[1]
        # Mark references to invalid pixels with nopsf array
        # First note which pixels are referenced outside of grid:
        nopsf = np.logical_or(psfy < 0, psfy >= self.ny)
        nopsf = np.logical_or(nopsf, psfx<0)
        nopsf = np.logical_or(nopsf, psfx>=self.nx)
        # Set them to reference pixel 0
        psfx = np.where(nopsf, 0, psfx)
        psfy = np.where(nopsf, 0, psfy)
        # Then read all indices, setting invalid ones to -1
        index1d = np.where(nopsf, -1, self._indices[psfy, psfx])
        # All invalid pixels now have index of -1; record and set to zero
        nopsf = index1d < 0
        index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)
        if self._force_model_center:
            dcdu = np.where(nopsf, 0., dcdu)
            dcdv = np.where(nopsf, 0., dcdv)

        # Look up param index for each referenced pixel
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
        
    def draw(self, star)
        """Fill the star's pixel data array with a rendering of the PSF specified by
        its current parameters, flux, and center.

        :param star:   A Star instance

        :returns: None
       """
        # ??? u, v = meshgrid of image coordinates
        coeffs, psfy, psfx = interp(u/du, v/dv)
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

    def derivatives(self, u, v):
        """Calculate interpolation coefficient for vector of target points, and
        their derivatives with respect to shift in u, v position of the star.

        Outputs will be 5 matrices, each of dimensions (nin, nkernel) where nin is
        number of input coordinates and nkernel is number of points in kernel footprint.
        The coeff matrix gives interpolation coefficients; then there are derivatives of the
        kernel with respect to u and v; then the y and x integer matrices
        give the grid point to which each coefficient is applied.
        
        :param u: 1d array of target u coordinates
        :param v: 1d array of target v coordinates

        :returns: coeff, dcdu, dcdv, y, x
        """
        raise NotImplemented("Derived classes must define the derivatives function")
        
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

    def derivatives(self, u, v):
        # Get integer and fractional parts of u, v
        # Set up coordinate arrays
        # Calculate Lanczos function and derivatives
        return coeffs, dcdu, dcdv, x, y
    
