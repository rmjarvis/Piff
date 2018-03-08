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
from past.builtins import basestring
import numpy as np
import galsim
import scipy.linalg
import warnings

from .model import Model
from .star import Star, StarFit

class PixelGrid(Model):
    """A PSF modeled as interpolation between a grid of points.

    The parameters of the model are the values at the grid points, although the constraint
    for unit flux means that not all grid points are free parameters.  The grid is in uv
    space, with the pitch and size specified on construction. Optionally a boolean
    mask array can be passed specifying which tells which grid elements are non-zero.
    Interpolation will always assume values of zero outside of grid.  Integral of PSF is
    forced to unity and, optionally, centroid is forced to origin.  As a consequence 1 (or 3)
    of the PSF pixel values will be missing from the parameter vector as they are determined
    by the flux (and centroid) constraints. And there is more covariance between pixel values.

    PixelGrid also needs an PixelInterpolant on construction to specify how to determine
    values between grid points.

    Stellar data is assumed either to be in flux units (with default sb=False), such that
    flux is defined as sum of pixel values; or in surface brightness units (sb=True), such
    that flux is (sum of pixels)*(pixel area).  Internally the sb convention is used.

    Convention of this code is that coordinates are (u,v).  All 2d forms of the PSF use
    this indexing order also.  StarData classes can use whatever they want, we only
    access them via 1d arrays.

    """
    def __init__(self, scale, size, interp=None, mask=None, start_sigma=1.,
                 force_model_center=True, degenerate=True, logger=None):
        """Constructor for PixelGrid defines the PSF pitch, size, and interpolator.

        :param scale:       Pixel scale of the PSF model (in arcsec)
        :param size:        Number of pixels on each side of square grid.
        :param interp:      An Interpolator to be used [default: Lanczos(3)]
        :param mask:        Optional square boolean 2d array, True where we want a non-zero value
                            [default: None]
        :param start_sigma: sigma of a 2d Gaussian installed as 1st guess for all stars
                            [default: 1.]
        :param force_model_center: If True, PSF model centroid is fixed at origin and
                            PSF fitting will marginalize over stellar position.  If False, stellar
                            position is fixed at input value and the fitted PSF may be off-center.
                            [default: True]
        :param degenerate:  Is it possible that individual stars give degenerate PSF sol'n?
                            If False, it runs faster, but fails on degeneracies. [default: True]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Building Pixel model with the following parameters:")
        logger.debug("scale = %s",scale)
        logger.debug("size = %s",size)
        logger.debug("interp = %s",interp)
        logger.debug("mask = %s",mask)
        logger.debug("start_sigma = %s",start_sigma)
        logger.debug("force_model_center = %s",force_model_center)
        logger.debug("degenerate = %s",degenerate)

        self.du = scale
        self.pixel_area = self.du*self.du
        if interp is None: interp = Lanczos(3)
        elif isinstance(interp, basestring): interp = eval(interp)
        self.interp = interp
        self._force_model_center = force_model_center
        self._degenerate = degenerate

        # These are the kwargs that can be serialized easily.
        # TODO: Add interp to this, so it can be specified in the yaml file and read/written.
        self.kwargs = {
            'scale' : scale,
            'size' : size,
            'start_sigma' : start_sigma,
            'force_model_center' : force_model_center,
            'degenerate' : degenerate
        }

        if mask is None:
            if size <= 0:
                raise ValueError("Non-positive PixelGrid size {:d}".format(size))
            self._mask = np.ones( (size,size), dtype=bool)
        else:
            if mask.shape != (size,size):
                raise ValueError("Shape of input mask does not match size {:d}".format(size))
            self._mask = mask

        self.ny, self.nx = self._mask.shape
        self._nparams = np.count_nonzero(self._mask)
        self._nparams -= 1  # The flux constraint will remove 1 degree of freedom
        self._constraints = 1
        if self._force_model_center:
            self._nparams -= 2 # Centroid constraint will remove 2 more degrees of freedom
            self._constraints += 2
        logger.debug("nparams = %d, constraints = %d",self._nparams, self._constraints)

        # Now we need to make a 2d array whose entries are the indices of
        # each pixel in the 1d parameter array.  We will put the central
        # pixels (and first to top & right) at the front of the array
        # because we will be chopping these off when we enforce the
        # flux (and center) conditions on the PSF.
        # In this array, a negative entry is a pixel that is not being
        # fit (and always assumed to be zero, for interpolation purposes).
        self._indices = np.where( self._mask, self._constraints, -1)
        self._origin = (self.ny//2, self.nx//2)
        if not self._mask[self._origin]:
            raise ValueError("Not happy with central PSF pixel being masked")
        self._indices[self._origin] = 0    # Central pixel for flux constraint
        if self._force_model_center:
            u1 = (self._origin[0]+1, self._origin[1])
            v1 = (self._origin[0],   self._origin[1]+1)
            if not (self._mask[u1] and self._mask[v1]):
                raise ValueError("Not happy with near-central PSF pixels being masked")
            self._indices[u1] = 1
            self._indices[v1] = 2
        free_param = self._indices >= self._constraints
        self._indices[free_param] = np.arange(self._constraints,
                                              self._constraints+self._nparams,
                                              dtype=int)

        # Next job is to create the flux/center constraint conditions.
        # ??? Could have some type of window function here, for now just
        # ??? using unweighted flux & centroid
        A = np.zeros( (self._constraints, self._constraints + self._nparams), dtype=float)
        B = np.zeros( (self._constraints,), dtype=float)
        A[0,:] = 1.
        B[0] = 1./self.pixel_area  # That's the flux constraint - sum(pixels) * pixel_area=1

        if self._force_model_center:
            # Generate linear center constraints too
            delta_u = np.arange( -self._origin[0], self._indices.shape[0]-self._origin[0])
            A[1,:] = self._1dFrom2d(np.ones(self._indices.shape, dtype=float) \
                                    * delta_u[:,np.newaxis])
            B[1] = 0.
            delta_v = np.arange( -self._origin[1], self._indices.shape[1]-self._origin[1])
            A[2,:] = self._1dFrom2d(np.ones(self._indices.shape, dtype=float)
                                    * delta_v[np.newaxis,:])
            B[2] = 0.

        ainv = np.linalg.inv(A[:,:self._constraints])
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

        # Now create a parameter array for a Gaussian that will be used to initialize new stars
        u = np.arange( -self._origin[0], self._indices.shape[0]-self._origin[0]) * self.du
        v = np.arange( -self._origin[1], self._indices.shape[1]-self._origin[1]) * self.du
        rsq = (u*u)[:,np.newaxis] + (v*v)[np.newaxis,:]
        gauss = np.exp(-rsq / (2.* start_sigma * start_sigma))
        if self._force_model_center:
            # If we are enforcing centering then we need to have symmetry about origin
            # This means if PSF has even number of dimensions, need to null the hanger-on
            if self._indices.shape[0]%2==0:
                gauss[0,:] = 0.
            if self._indices.shape[1]%2==0:
                gauss[1,:] = 0.
        params = self._1dFrom2d(gauss)
        # Renormalize to get unity flux
        params /= np.sum(params)*self.pixel_area
        self._initial_params = params[self._constraints:]

        return

    def _1dFrom2d(self, in2d):
        """Make a 1d array from a 2d array, using the model's
        mapping from the 2d psf grid to the 1d parameter array.

        :param in2d:    A 2d array matching the PSFs sample grid

        :returns:       A 1d array of the length of number of grid points in use

        :returns  None
        """
        out1d = np.zeros( (self._constraints + self._nparams,), dtype=in2d.dtype)
        out1d[self._indices[self._mask]] = in2d[self._mask]
        return out1d

    def _2dFrom1d(self, in1d):
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

    def _indexFromPsfxy(self, psfx, psfy):
        """ Turn arrays of coordinates of the PSF array into a single same-shape
        array of indices into a 1d parameter vector.  The index is <0 wherever
        the psf x,y values were outside the PSF mask.

        :param psfx:  array (any shape) of integer x displacements from origin
        of the PSF grid
        :param psfy:  array of integer y locations in PSF grid

        :returns: same shape array, filled with indices into 1d array
        """

        if not psfx.shape==psfy.shape:
            raise ValueError("psfx and psfy arrays are not same shape")

        # First, shift psfy, psfx to reference a 0-indexed array
        y = psfy + self._origin[0]
        x = psfx + self._origin[1]
        # Mark references to invalid pixels with nopsf array
        # First note which pixels are referenced outside of grid:
        nopsf = (y < 0) | (y >= self.ny) | (x < 0) | (x >= self.nx)
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
        constrained = self._b - np.dot(self._a[:,:self._nparams], star.fit.params)
        return np.concatenate((constrained, star.fit.params))

    def fillPSF(self, star, in2d):
        """ Initialize the PSF for a star from a given 2d uv-plane array.
        Sets elements outside the mask to zero, renormalizes to enforce flux
        condition, and checks centering condition if force_model_center=True

        :param star:    A Star instance to initialize
        :param in2d:    2d input array matching the PSF uv-plane grid

        :returns: None
        ??? Return a new one instead???
        """
        params = self._1dFrom2d(in2d)

        # Renormalize to get unity flux
        params /= np.sum(params)*self.pixel_area
        # ??? check centering ???

        star.fit.params[:] = params[self._constraints:]  # Omit the constrained pixels

    def initialize(self, star, mask=True, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param mask:    If True, set data.weight to zero at pixels that are outside
                        the range of the model.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a star instance with the appropriate initial fit values
        """
        var = np.zeros(len(self._initial_params))
        fit = StarFit(self._initial_params, star.fit.flux, star.fit.center, params_var=var)
        if mask:
            # Null weight at pixels where interpolation coefficients
            # come up short of specified fraction of the total kernel
            required_kernel_fraction = 0.7

            _, _, u, v = star.data.getDataVector()
            # Subtract star.fit.center from u, v:
            u -= fit.center[0]
            v -= fit.center[1]
            coeffs, psfx, psfy = self.interp(u/self.du, v/self.du)
            # Turn the (psfx,psfy) coordinates into an index into 1d parameter vector.
            index1d = self._indexFromPsfxy(psfx, psfy)
            # All invalid pixel references now have negative index;
            # Null the coefficients for such pixels
            coeffs = np.where(index1d < 0, 0., coeffs)
            use = np.sum(coeffs,axis=1) > required_kernel_fraction
            data = star.data.maskPixels(use)
        star = Star(data, fit)
        # Update the flux to something close to right.
        star = self.reflux(star, fit_center=False, logger=logger)
        return star

    def fit(self, star, logger=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated fit information
        """
        star1 = self.chisq(star)  # Get chisq Taylor expansion for linearized model
        ### Check for non-pos-def
        ###S = np.linalg.svd(star1.fit.alpha,compute_uv=False)
        ###print("  .in fit(), min SV:",np.min(S))###
        ###U,S,Vt = np.linalg.svd(star1.fit.alpha,compute_uv=True)
        ###print("  ..in fit(), min SV:",np.min(S))###

        # star1 has marginalized over flux (& center, if free), and updated these
        # for best linearized fit at the input parameter values.
        if self._degenerate:
            # Do SVD and retain
            # input values for degenerate parameter combinations
            # U,S,Vt = np.linalg.svd(star1.fit.alpha)
            S,U = np.linalg.eigh(star1.fit.alpha)
            # Invert, while zeroing small elements of S.
            # "Small" will be taken to be causing a small chisq change
            # when corresponding PSF component changes by the full flux of PSF
            small = 0.2 * self.pixel_area * self.pixel_area
            if np.any(S < -small):
                print("negative: ",np.min(S),"small:",small)###
                raise ValueError("Negative singular value in alpha matrix")
            # Leave values that are close to zero equal to zero in inverse.
            nonzero = np.abs(S) > small
            invs = np.zeros_like(S)
            invs[nonzero] = 1./S[nonzero]

            ###print('S/zero:',S.shape,np.count_nonzero(np.abs(S)<=small),'small=',small) ###
            ###print(' ',np.max(S[np.abs(S)<=small]),np.min(S[np.abs(S)>small])) ##
            # answer = V * S^{-1} * U^T * beta
            # dparam = np.dot(Vt.T, invs * np.dot(U.T,star1.fit.beta))
            dparam = np.dot(U, invs * np.dot(U.T,star1.fit.beta))
        else:
            # If it is known there are no degeneracies, we can skip SVD
            # Note: starting with scipy 1.0, the generic version of this got extremely slow.
            # Like 10x slower than scipy 0.19.1.  cf. https://github.com/scipy/scipy/issues/7847
            # So the assume_a='pos' bit is really important until they fix that.
            # Unfortunately, our matrices aren't necessarily always positive definite.  If not,
            # we switch to the svd method, which might be overkill, but is cleaner than switching
            # to LU for non-posdef, but then SV for fully singular.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    dparam = scipy.linalg.solve(star1.fit.alpha, star1.fit.beta, assume_a='pos',
                                                check_finite=False)
            except (RuntimeWarning, np.linalg.LinAlgError) as e:  # pragma: no cover
                if logger:
                    logger.warning('Caught %s',str(e))
                    logger.warning('Switching to non-posdef method')
                dparam = scipy.linalg.solve(star1.fit.alpha, star1.fit.beta)

        # Create new StarFit, update the chisq value.  Note no beta is returned as
        # the quadratic Taylor expansion was about the old parameters, not these.
        var = np.zeros(len(star1.fit.params + dparam))
        starfit2 = StarFit(star1.fit.params + dparam,
                           params_var = var,
                           flux = star1.fit.flux,
                           center = star1.fit.center,
                           alpha = star1.fit.alpha,  # Inverse covariance matrix
                           chisq = star1.fit.chisq \
                                   + np.dot(dparam, np.dot(star1.fit.alpha, dparam)) \
                                   - 2 * np.dot(star1.fit.beta, dparam))
        return Star(star1.data, starfit2)

    def chisq(self, star, logger=None):
        """Calculate dependence of chi^2 = -2 log L(D|p) on PSF parameters for single star.
        as a quadratic form chi^2 = dp^T*alpha*dp - 2*beta*dp + chisq,
        where dp is the *shift* from current parameter values.  Marginalization over
        flux (and center, if free) should be done by this routine. Returned Star
        instance has the resultant alpha, beta, chisq, flux, center) attributes,
        but params vector has not have been updated yet (could be degenerate).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated StarFit
        """

        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()
        if not star.data.values_are_sb:
            # If the images are flux instead of surface brightness, convert
            # them into SB
            star_pix_area = star.data.pixel_area
            data /= star_pix_area
            weight *= star_pix_area*star_pix_area

        # Subtract star.fit.center from u, v:
        u -= star.fit.center[0]
        v -= star.fit.center[1]

        if self._force_model_center:
            coeffs, dcdu, dcdv, psfx, psfy = self.interp.derivatives(u/self.du, v/self.du)
            dcdu /= self.du
            dcdv /= self.du
        else:
            coeffs, psfx, psfy = self.interp(u/self.du, v/self.du)

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
            dmdu = star.fit.flux * np.sum(dcdu*pvals, axis=1)
            dmdv = star.fit.flux * np.sum(dcdv*pvals, axis=1)
        resid = data - mod*star.fit.flux

        # Now begin construction of alpha/beta/chisq that give
        # chisq vs linearized model.
        rw = resid * weight
        chisq = np.sum(resid * rw)
        dof = np.count_nonzero(weight) - self._constraints

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
        derivs[:, :coeffs.shape[1]] = star.fit.flux * coeffs  #derivs wrt PSF elements
        indices[:,:index1d.shape[1]] = index1d

        # Add derivs wrt flux
        derivs[:,coeffs.shape[1]] = mod
        dflux_index = self._nparams + self._constraints
        indices[:,coeffs.shape[1]] = dflux_index
        if self._force_model_center:
            # Derivs w.r.t. center shift:
            derivs[:,coeffs.shape[1]+1] = dmdu
            derivs[:,coeffs.shape[1]+2] = dmdv
            indices[:,coeffs.shape[1]+1] = dflux_index+1
            indices[:,coeffs.shape[1]+2] = dflux_index+2

        # Accumulate alpha and beta point by point.  I don't
        # know how to do it purely with numpy calls instead of a loop over data points
        nderivs = self._nparams + 2*self._constraints
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
          - np.dot( alpha[s1,s0], self._a) \
          + np.dot( self._a.T, np.dot(alpha[s0,s0],self._a))

        # Now we marginalize over the flux (and center). These shifts are at
        # the back end of the parameter array.
        # But first we need to apply a prior to the shift of flux (and center)
        # to avoid numerical instabilities when these are degenerate because of
        # missing pixel data or otherwise unspecified PSF
        # ??? make these properties of the Model???
        fractional_flux_prior = 0.5 # prior of 50% on pre-existing flux ???
        center_shift_prior = 0.5*self.du #prior of 0.5 uv-plane pixels ???
        alpha[self._nparams, self._nparams] += (fractional_flux_prior*star.fit.flux)**(-2.)
        if self._force_model_center:
            alpha[self._nparams+1, self._nparams+1] += (center_shift_prior)**(-2.)
            alpha[self._nparams+2, self._nparams+2] += (center_shift_prior)**(-2.)

        s0 = slice(None, self._nparams)  # parameters to keep
        s1 = slice(self._nparams, None)  # parameters to marginalize
        a11inv = np.linalg.inv(alpha[s1,s1])
        # Calculate shift in flux - ??? Note that this is the solution for shift
        # when PSF parameters do *not* move; so if we subsequently update
        # the PSF params, we miss shifts due to covariances between flux and PSF.

        df = np.dot(a11inv, beta[s1])
        outflux = star.fit.flux + df[0]
        if self._force_model_center:
            outcenter = (star.fit.center[0] + df[1],
                         star.fit.center[1] + df[2])
        else:
            outcenter = star.fit.center

        # Now get the final alpha, beta, chisq for the remaining PSF params
        outchisq = chisq - np.dot(beta[s1].T,np.dot(a11inv, beta[s1]))
        if logger:
            logger.info('chisq = %f -> %f  dof = %d'%(chisq,outchisq,dof))
        tmp = np.dot(a11inv, alpha[s1,s0])
        outbeta = beta[s0] - np.dot(beta[s1].T,tmp)
        outalpha = alpha[s0,s0] - np.dot(alpha[s0,s1],tmp)

        var = np.zeros(len(star.fit.params))
        outfit = StarFit(star.fit.params,
                         params_var = var,
                         flux = outflux,
                         center = outcenter,
                         chisq = outchisq,
                         dof = dof,
                         alpha = outalpha,
                         beta = outbeta)

        return Star(star.data, outfit)

    def draw(self, star):
        """Create new Star instance that has StarData filled with a rendering
        of the PSF specified by the current StarFit parameters, flux, and center.
        Coordinate mapping of the current StarData is assumed.

        :param star:   A Star instance

        :returns:      New Star instance with rendered PSF in StarData
        """
        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector(include_zero_weight=True)
        # Subtract star.fit.center from u, v
        u -= star.fit.center[0]
        v -= star.fit.center[1]

        coeffs, psfx, psfy = self.interp(u/self.du, v/self.du)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)

        pvals = self._fullPsf1d(star)[index1d]
        model = star.fit.flux * np.sum(coeffs*pvals, axis=1)
        if not star.data.values_are_sb:
            # Change data from surface brightness into flux
            model *= star.data.pixel_area

        return Star(star.data.setData(model,include_zero_weight=True), star.fit)

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.  DOF in the result assume
        only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance, with updated flux, center, chisq, dof, worst
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Reflux for star:")
        logger.debug("    flux = %s",star.fit.flux)
        logger.debug("    center = %s",star.fit.center)
        logger.debug("    props = %s",star.data.properties)
        logger.debug("    image = %s",star.data.image)
        #logger.debug("    image = %s",star.data.image.array)
        #logger.debug("    weight = %s",star.data.weight.array)
        logger.debug("    image center = %s",star.data.image(star.data.image.center))
        logger.debug("    weight center = %s",star.data.weight(star.data.weight.center))

        # This will be an iterative process if the centroid is free.
        max_iterations = 100    # Max iteration count

        chisq_thresh = 1.e-3     # Quit when chisq changes less than this (fractionally)
        do_center = fit_center and self._force_model_center
        flux = star.fit.flux
        center = star.fit.center
        prev_chisq = 1.e500
        for iteration in range(max_iterations):
            logger.debug("Start iteration %d",iteration)
            # Start by getting all interpolation coefficients for all observed points
            data, weight, u, v = star.data.getDataVector()
            if not star.data.values_are_sb:
                # If the images are flux instead of surface brightness, convert
                # them into SB
                star_pix_area = star.data.pixel_area
                data /= star_pix_area
                weight *= star_pix_area*star_pix_area
            u -= center[0]
            v -= center[1]
            if do_center:
                coeffs, dcdu, dcdv, psfx, psfy = self.interp.derivatives(u/self.du, v/self.du)
                dcdu /= self.du
                dcdv /= self.du
            else:
                coeffs, psfx, psfy = self.interp(u/self.du, v/self.du)
            # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
            index1d = self._indexFromPsfxy(psfx, psfy)
            # All invalid pixel references now have negative index; record and set to zero
            nopsf = index1d < 0
            index1d = np.where(nopsf, 0, index1d)
            # And null the coefficients for such pixels
            coeffs = np.where(nopsf, 0., coeffs)
            if do_center:
                dcdu = np.where(nopsf, 0., dcdu)
                dcdv = np.where(nopsf, 0., dcdv)

            # Multiply kernel (and derivs) by current PSF element values
            # to get current estimates
            pvals = self._fullPsf1d(star)[index1d]
            mod = np.sum(coeffs*pvals, axis=1)
            if do_center:
                dmdu = flux * np.sum(dcdu*pvals, axis=1)
                dmdv = flux * np.sum(dcdv*pvals, axis=1)
                derivs = np.vstack( (mod, dmdu, dmdv)).T
            else:
                derivs = mod.reshape(mod.shape+(1,))
                # derivs should end up with shape (npts, nconstraints)
            resid = data - mod*flux
            logger.debug("total pixels = %s, nopsf = %s",len(pvals),np.sum(nopsf))

            # Now begin construction of alpha/beta/chisq that give
            # chisq vs linearized model.
            rw = resid * weight
            chisq = np.sum(resid * rw)
            logger.debug("initial chisq = %s",chisq)
            beta = np.dot(derivs.T,rw)
            alpha = np.dot(derivs.T*weight, derivs)
            try:
                df = np.linalg.solve(alpha, beta)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if do_center:
                    logger.debug("Caught exception %s",e)
                    logger.debug("Turning off centering and retrying")
                    do_center = False
                    continue
                else:
                    raise
            dchi = np.dot(beta, df)
            chisq = chisq - dchi
            logger.debug("chisq -= %s => %s",dchi,chisq)
            # Record worst single pixel chisq:
            resid -= np.dot(derivs,df)
            rw = resid * weight
            worst_chisq = np.max(resid * rw)
            logger.debug("worst_chisq = %s",worst_chisq)

            # update the flux (and center) of the star
            logger.debug("initial flux = %s",flux)
            flux += df[0]
            logger.debug("flux += %s => %s",df[0],flux)
            logger.debug("center = %s",center)
            if do_center:
                center = (center[0]+df[1],
                          center[1]+df[2])
                logger.debug("center += (%s,%s) => %s",df[1],df[2],center)
            dof = np.count_nonzero(weight) - self._constraints
            logger.debug("dchi, dof, do_center = %s, %s, %s", dchi, dof, do_center)
            if dchi < chisq_thresh * chisq or not do_center:
                # Done with iterations.  Return new Star with updated information
                var = np.zeros(len(star.fit.params))
                return Star(star.data, StarFit(star.fit.params,
                                               params_var = var,
                                               flux = flux,
                                               center = center,
                                               chisq = chisq,
                                               worst_chisq = worst_chisq,
                                               dof = dof,
                                               alpha = star.fit.alpha,
                                               beta = star.fit.beta))
            # If chisq went up, turn off centering.  There are a number of failure modes
            # to this algorithm that can lead to oscillatory behavior, so if we start doing
            # that, just turn off the centering for subsequent iterations.
            if chisq > prev_chisq:
                assert do_center  # The logic of the above test means this should be True here.
                do_center = False
                center = (center[0]-df[1], center[1]-df[2])  # undo the last centroid update.
                logger.debug("chisq increased in reflux.  Turning off centering.")
            prev_chisq = chisq

        raise RuntimeError("Maximum number of iterations exceeded in PixelGrid.reflux()")


class PixelInterpolant(object):
    """Interface for interpolators
    """
    def range(self):
        """Size of interpolation kernel

        :returns: Maximum distance from target to source pixel.
        """
        raise NotImplementedError("Derived classes must define the range function")

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
        raise NotImplementedError("Derived classes must define the __call__ function")

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
        raise NotImplementedError("Derived classes must define the derivatives function")


class Lanczos(PixelInterpolant):
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
        self._dv = self._dv.flatten()

    def range(self):
        return self.order

    def _kernel1d(self, u):
        """ Calculate the 1d interpolation kernel at each value in array u.

        :param u: 1d array of (u_dest-u_src) spanning the footprint of the kernel.

        :returns: interpolation kernel values at these grid points
        """
        # Normalize Lanczos to unit sum over kernel elements
        k = np.sinc(u) * np.sinc(u/self.order)
        return k / np.sum(k,axis=1)[:,np.newaxis]

    def __call__(self, u, v):
        return self._calculate(u,v,derivs=False)

    def derivatives(self, u, v):
        return self._calculate(u,v,derivs=True)

    def _calculate(self, u, v, derivs=False):
        """ Routine which does the kernel calculations.  Uses finite differences to
        calculate derivatives, if requested.

        :param u,v:    1d arrays of coordinates to which we are interpolating
        :param derivs: Set to true if outputs should include derivatives w.r.t. u,v

        :returns: coeffs, [dcoeff/du, dcoeff/dv,] x, y where each is a 2d array of
        dimensions (len(u), # of kernel elements), holding coefficients of each grid point
        for interpolation to each destination point, [derivatives of these], and x,y are
        the integer coordinates of the grid points.
        """

        # Get integer and fractional parts of u, v
        u_ceil = np.ceil(u).astype(int)
        v_ceil = np.ceil(v).astype(int)
        # Make arrays giving coordinates of grid points within footprint
        x = u_ceil[:,np.newaxis] + self._du[np.newaxis,:]
        y = v_ceil[:,np.newaxis] + self._dv[np.newaxis,:]
        # Make npts x (2*order) arrays holding 1d displacements
        # to be arguments of the 1d kernel functions
        argu = (u_ceil-u)[:,np.newaxis] + self._duv
        argv = (v_ceil-v)[:,np.newaxis] + self._duv
        # Calculate the Lanczos function each axis:
        ku = self._kernel1d(argu)
        kv = self._kernel1d(argv)
        # Then take outer products to produce kernel
        coeffs = (ku[:,np.newaxis,:] * kv[:,:,np.newaxis]).reshape(x.shape)

        if derivs:
            # Take derivatives with respect to u
            duv = 0.01   # Step for finite differences
            dku = (self._kernel1d(argu+duv)-self._kernel1d(argu-duv)) / (2*duv)
            dcdu = (dku[:,np.newaxis,:] * kv[:,:,np.newaxis]).reshape(x.shape)
            # and v
            dkv = (self._kernel1d(argv+duv)-self._kernel1d(argv-duv)) / (2*duv)
            dcdv = (ku[:,np.newaxis,:] * dkv[:,:,np.newaxis]).reshape(x.shape)
            return coeffs, dcdu, dcdv, x, y
        else:
            return coeffs, x, y


class Bilinear(PixelInterpolant):
    """Lanczos interpolator in 2 dimensions.
    """
    def __init__(self):
        """Initialize - "order" is the range, 1 pixel here
        """
        self.order = 1
        # Here is range of pixels to use in each dimension relative to ceil(u,v)
        self._duv = np.arange(-self.order, self.order, dtype=int)
        # And here are flattened arrays of u, v displacement for whole footprint
        self._du = np.ones( (2*self.order,2*self.order), dtype=int) * self._duv
        self._du = self._du.flatten()
        self._dv = np.ones( (2*self.order,2*self.order), dtype=int) * \
          self._duv[:,np.newaxis]
        self._dv = self._dv.flatten()

    def range(self):
        return self.order

    def _kernel1d(self, u):
        """ Calculate the 1d interpolation kernel at each value in array u.

        :param u: 1d array of (u_dest-u_src) spanning the footprint of the kernel.

        :returns: interpolation kernel values at these grid points
        """
        return 1. - np.abs(u)

    def __call__(self, u, v):
        return self._calculate(u,v,derivs=False)

    def derivatives(self, u, v):
        return self._calculate(u,v,derivs=True)

    def _calculate(self, u, v, derivs=False):
        """ Routine which does the kernel calculations.  Uses finite differences to
        calculate derivatives, if requested.

        :param u,v:    1d arrays of coordinates to which we are interpolating
        :param derivs: Set to true if outputs should include derivatives w.r.t. u,v

        :returns: coeffs, [dcoeff/du, dcoeff/dv,] x, y where each is a 2d array of
        dimensions (len(u), # of kernel elements), holding coefficients of each grid point
        for interpolation to each destination point, [derivatives of these], and x,y are
        the integer coordinates of the grid points.
        """

        # Get integer and fractional parts of u, v
        u_ceil = np.ceil(u).astype(int)
        v_ceil = np.ceil(v).astype(int)
        # Make arrays giving coordinates of grid points within footprint
        x = u_ceil[:,np.newaxis] + self._du[np.newaxis,:]
        y = v_ceil[:,np.newaxis] + self._dv[np.newaxis,:]
        # Make npts x (2*order) arrays holding 1d displacements
        # to be arguments of the 1d kernel functions
        argu = (u_ceil-u)[:,np.newaxis] + self._duv
        argv = (v_ceil-v)[:,np.newaxis] + self._duv
        # Calculate the 1d function each axis:
        ku = self._kernel1d(argu)
        kv = self._kernel1d(argv)
        # Then take outer products to produce kernel
        coeffs = (ku[:,np.newaxis,:] * kv[:,:,np.newaxis]).reshape(x.shape)

        if derivs:
            # Take derivatives with respect to u
            duv = 0.01   # Step for finite differences
            dku = (self._kernel1d(argu+duv)-self._kernel1d(argu-duv)) / (2*duv)
            dcdu = (dku[:,np.newaxis,:] * kv[:,:,np.newaxis]).reshape(x.shape)
            # and v
            dkv = (self._kernel1d(argv+duv)-self._kernel1d(argv-duv)) / (2*duv)
            dcdv = (ku[:,np.newaxis,:] * dkv[:,:,np.newaxis]).reshape(x.shape)
            return coeffs, dcdu, dcdv, x, y
        else:
            return coeffs, x, y
