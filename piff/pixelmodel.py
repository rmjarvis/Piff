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
        # For later convenience, add some columns of zeros to _a so it can multiply
        # into arrays containing flux (and center) shift
        tmp = np.zeros( (self._a.shape[0], self._a.shape[1]+self._constraints),
                        dtype=float)
        tmp[:,:self._a.shape[1]] = self._a
        self._a = tmp
        
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

    def _indexFromPsfxy(self, psfx, psfy)
        """ Turn arrays of coordinates of the PSF array into a single same-shape
        array of indices into a 1d parameter vector.  The index is <0 wherever
        the psf x,y values were outside the PSF mask.

        :param psfx:  array (any shape) of integer x displacements from origin
        of the PSF grid
        :param psfy:  array of integer y locations in PSF grid

        :returns: same shape array, filled with indices into 1d array
        """

        if not psfx.shape==psfy.shape:
            raise TypeError("psfx and psfy arrays are not same shape")
        
        # First, shift psfy, psfx to reference a 0-indexed array
        y = psfy + self._origin[0]
        x = psfx + self._origin[1]
        # Mark references to invalid pixels with nopsf array
        # First note which pixels are referenced outside of grid:
        nopsf = np.logical_or(y < 0, y >= self.ny)
        nopsf = np.logical_or(nopsf, x<0)
        nopsf = np.logical_or(nopsf, x>=self.nx)
        # Set them to reference pixel 0
        x = np.where(nopsf, 0, x)
        y = np.where(nopsf, 0, y)
        # Then read all indices, setting invalid ones to -1
        return np.where(nopsf, -1, self._indices[y, x])

    def _fullPsf1d(self, star):
        """ Using stored PSF parameters, create full 1d array of PSF grid
        point values by applying the flux (and center) constraints to generate
        the dependent values

        :param star:  A Star instance whose parameters to use
        
        :returns: 1d array of all PSF values at grid points in mask
        """
        constrained = self._b - np.dot(self._a[:self._nparams], star.params)
        return np.concatenate(constrained, star.params)
        

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
        # ??? Subtract star.center from u, v ???
        if self._force_model_center:
            coeffs, dcdu, dcdv, psfx, psfy = interp.derivs(u/du, v/du)
            dcdu /= du
            dcdv /= du
        else:
            coeffs, psfx, psfy = interp(u/du, v/du)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)
        if self._force_model_center:
            dcdu = np.where(nopsf, 0., dcdu)
            dcdv = np.where(nopsf, 0., dcdv)

        # Multiply kernel (and derivs) by current PSF element values
        # to get current estimates
        pvals = self._fullPsf1d(star)[index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        if self._force_model_center:
            dmdu = star.flux * np.sum(dcdu*pvals, axis=1)
            dmdv = star.flux * np.sum(dcdv*pvals, axis=1)
        resid = data - mod*star.flux

        # Now begin construction of alpha/beta/gamma that give
        # chisq vs linearized model.
        rw = resid * weight
        gamma = np.sum(resid * rw)

        # To begin with, we build alpha and beta over all PSF points
        # within mask, *and* the flux (and center) shifts.  Then
        # will eliminate the constrained PSF points, and then
        # marginalize over the flux (and center).

        # Augment the coeffs and index1d vectors with extra column(s)
        # for the shift in flux (and center), so it will be
        # the derivative of model w.r.t. augmented parameter set
        derivs = np.zeros( (coeffs.shape[0], coeffs.shape[1]+self._constraints),
                           dtype=float)
        indices = np.zeros( (index1d.shape[0], index1d.shape[1]+self._constraints),
                            dtype=int)
        derivs[:, :coeffs.shape[1]] = star.flux * coeffs  #derivs wrt PSF elements
        indices[:,:index1d.shape[1]] = index1d

        # Add derivs wrt flux
        derivs[:,coeffs.shape[1]] = mod
        indices[:,coeffs.shape[1]] = len(pvals)
        if self._force_model_center:
            # Derivs w.r.t. center shift:
            derivs[:,coeffs.shape[1]+1] = dmdu
            derivs[:,coeffs.shape[1]+2] = dmdv
            indices[:,coeffs.shape[1]+1] = len(pvals)+1
            indices[:,coeffs.shape[1]+2] = len(pvals)+2
        
        # Accumulate alpha and beta point by point.  I don't
        # know how to do it purely with numpy calls instead of a loop over data points
        nderivs = len(pvals) + self._constraints
        beta = np.zeros(nderivs, dtype=float)
        alpha = np.zeros( (nderivs,nderivs), dtype=float)
        for i in range(len(data)):
            ii = indices[i,:]
            cc = derivs[i,:]
            # beta_j += resid_i * weight_i * coeff_{ij}
            beta[ii] += rw[i] * cc
            # alpha_jk += weight_i * coeff_ij * coeff_ik
            dalpha = cc[np.newaxis,:]*cc[:,np.newaxis] * weight[i]
            iouter = np.broadcast_to(ii, (len(ii),len(ii)))
            alpha[iouter.flatten(), iouter.T.flatten()] += dalpha.flatten()

        # Next we eliminate the first _constraints PSF values from the parameters
        # using the linear constraints that dp0 = - _a * dp1 
        s0 = slice(None, self._constraints)  # parameters to eliminate
        s1 = slice(self._constraints, None)  # parameters to keep
        beta = beta[s1] - np.dot(beta[s0], self._a).T
        alpha = alpha[s1,s1] \
          - np.dot( self._a.T, alpha[s0,s1]) \
          - np.dot( alpha[s1,s0].T, self._a) \
          + np.dot( self._a.T, np.dot(alpha[s0,s0],self._a))

        # Now we marginalize over the flux (and center). These shifts are at
        # the back end of the parameter array.
        # But first we need to apply a prior to the shift of flux (and center)
        # to avoid numerical instabilities when these are degenerate because of
        # missing pixel data or otherwise unspecified PSF
        # ??? make these properties of the Model???
        fractional_flux_prior = 0.1 # prior of 10% on pre-existing flux
        center_shift_prior = 0.1*self.du #prior of 0.1 uv-plane pixels
        alpha[self._nparams, self.n_params] += (fractional_flux_prior*star.flux)**(-2.)
        alpha[self._nparams, self.n_params] += (center_shift_prior)**(-2.)

        s0 = slice(None, self._nparams)  # parameters to keep
        s1 = slice(self._nparams, None)  # parameters to marginalize
        a11inv = np.linalg.inverse(alpha[s1,s1])
        # Calculate shift in flux - ??? Note that this is the solution for shift
        # when PSF parameters do *not* move; so if we subsequently update
        # the PSF params, we miss shifts due to covariances between flux and PSF.
        df = np.dot(a11inv, beta[s1])
        star.flux += df[0]
        if self._force_model_center:
            star.center = (star.center[0] + df[1],
                           star.center[1] + df[2])  # ?? check u,v ordering

        # Now get the final alpha, beta, gamma for the remaining PSF params
        star.gamma = gamma - np.dot(beta[s1].T,np.dot(a11inv, beta[s1]))
        tmp = np.dot(a11inv, alpha[s1,s0])
        star.beta = beta[s0] - np.dot(beta[s1].T,tmp)
        star.alpha = alpha[s0,s0] - np.dot(alpha[s0,s1],tmp)

        return 
        
    def draw(self, star)
        """Fill the star's pixel data array with a rendering of the PSF specified by
        its current parameters, flux, and center.

        :param star:   A Star instance

        :returns: None
       """
        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()
        # ??? Subtract star.center from u, v ???
        coeffs, psfx, psfy = interp(u/du, v/du)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)

        pvals = self._fullPsf1d(star)[index1d]
        model = self.flux * np.sum(coeffs*pvals, axis=1)

        star.data.setData(model)
        return

    def reflux(self, star):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.

        :param star:   A Star instance

        :returns: chi-squared, dof of the fit to the data. ??
        """
        # This will be an iterative process if the centroid is free.
        max_iterations = 10    # Max iteration count
        chisq_tolerance = 0.01 # Quit when chisq changes less than this
        for iteration in range(max_iterations):
            # Start by getting all interpolation coefficients for all observed points
            data, weight, u, v = star.data.getDataVector()
            u -= star.center[0]
            v -= star.center[1]
            if self._force_model_center:
                coeffs, dcdu, dcdv, psfx, psfy = interp.derivs(u/du, v/du)
                dcdu /= du
                dcdv /= du
            else:
                coeffs, psfx, psfy = interp(u/du, v/du)
            # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
            index1d = self._indexFromPsfxy(psfx, psfy)
            # All invalid pixel references now have negative index; record and set to zero
            nopsf = index1d < 0
            index1d = np.where(nopsf, 0, index1d)
            # And null the coefficients for such pixels
            coeffs = np.where(nopsf, 0., coeffs)
            if self._force_model_center:
                dcdu = np.where(nopsf, 0., dcdu)
                dcdv = np.where(nopsf, 0., dcdv)

            # Multiply kernel (and derivs) by current PSF element values
            # to get current estimates
            pvals = self._fullPsf1d(star)[index1d]
            mod = np.sum(coeffs*pvals, axis=1)
            if self._force_model_center:
                dmdu = star.flux * np.sum(dcdu*pvals, axis=1)
                dmdv = star.flux * np.sum(dcdv*pvals, axis=1)
            resid = data - mod*star.flux

            derivs = np.vstack( (mod, dmdu, dmdv), axis=1)
            # Now begin construction of alpha/beta/gamma that give
            # chisq vs linearized model.
            rw = resid * weight
            gamma = np.sum(resid * rw)
            beta = np.sum( derivs*rw, axis=0)
            alpha = np.dot( derivs.T * weight, derivs)
            df = np.linalg.solve(alpha, beta)
            dchi = np.dot(beta, df)
            chisq = gamma - dchi
            # update the flux (and center) of the star
            star.flux += df[0]
            if self._force_model_center:
                star.center = (star.center[0]+df[1],
                               star.center[1]+df[2])
            if abs(dchi) < chisq_tolerance or not _force_model_center:
                # Quit iterating
                return chisq, np.count_nonzero(weight) - self._constraints

        raise RuntimeError("Maximum number of iterations exceeded in PixelModel.reflux()")

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
        # Here is range of pixels to use in each dimension relative to ceil(u,v)
        self._duv = np.arange(-self.order, self.order, dtype=int)
        # And here are flattened arrays of u, v displacement for whole footprint
        self._du = np.ones( (2*self.order,2*self.order), dtype=int) * self._duv
        self._du = self._du.flatten()
        self._dv = np.ones( (2*self.order,2*self.order), dtype=int) * \
          self._duv[:,np.newaxis]
        self._dv.flatten()
        return
    
    def range(self):
        return self.order

    @classmethod
    def _sinc(cls, u):
        """Calculate sinc for elements of array u

        :param u   Numpy array of floats

        :returns:  Array of sinc(u)
        """
        return np.where( abs(u)>0.001, sin(u)/u, 1-(np.PI*np.PI/6.)*u*u)

    @classmethod
    def _dsinc(cls, u):
        """Calculate d sinc(u) / du for elements of array u

        :param u   Numpy array of floats

        :returns:  Array of derivatives 
        """
        return np.where( abs(u)>0.001, (u*cos(u)-sin(u))/(u*u), u/3.)
        
    def __call__(self, u, v):
        # Get integer and fractional parts of u, v
        u_ceil = np.ceil(u).astype(int)
        v_ceil = np.ceil(v).astype(int)
        # Make arrays giving coordinates of grid points within footprint
        x = u_ceil + self._du
        y = v_ceil[:,np.newaxis] + self._dv
        # Make npts x (2*order) arrays holding 1d displacements
        # to be arguments of the 1d kernel functions
        argu = (u_ceil-u)[:,np.newaxis] + self._duv
        argv = (v_ceil-v)[:,np.newaxis] + self._duv
        # Calculate the Lanczos function each axis:
        ku = self._sinc(argu) * self._sinc(argu/self.order)
        kv = self._sinc(argv) * self._sinc(argv/self.order)
        # Then take outer products to produce kernel
        coeffs = (ku[:,np.newaxis,:] * kv[:,:,np.newaxis]).reshape(x.shape)
        return coeffs, x, y

    def derivatives(self, u, v):
        # Get integer and fractional parts of u, v
        # Set up coordinate arrays
        # Calculate Lanczos function and derivatives
        return coeffs, dcdu, dcdv, x, y
    
