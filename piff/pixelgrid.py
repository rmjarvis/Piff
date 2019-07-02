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

from galsim import Lanczos
from .model import Model
from .star import Star, StarData, StarFit

class PixelGrid(Model):

    _method = 'no_pixel'

    """A PSF modeled as interpolation between a grid of points.

    The parameters of the model are the values at the grid points, although the constraint
    for unit flux means that not all grid points are free parameters.  The grid is in uv
    space, with the pitch and size specified on construction.  Interpolation will always
    assume values of zero outside of grid.

    PixelGrid also needs an PixelInterpolant on construction to specify how to determine
    values between grid points.

    Stellar data is assumed either to be in flux units (with default sb=False), such that
    flux is defined as sum of pixel values; or in surface brightness units (sb=True), such
    that flux is (sum of pixels)*(pixel area).  Internally the sb convention is used.

    Convention of this code is that coordinates are (u,v).  All 2d forms of the PSF use
    this indexing order also.  StarData classes can use whatever they want, we only
    access them via 1d arrays.

    """
    def __init__(self, scale, size, interp=None, start_sigma=1.,
                 force_model_center=True, degenerate=True, logger=None):
        """Constructor for PixelGrid defines the PSF pitch, size, and interpolator.

        :param scale:       Pixel scale of the PSF model (in arcsec)
        :param size:        Number of pixels on each side of square grid.
        :param interp:      An Interpolant to be used [default: Lanczos(3); currently only
                            Lanczos(n) is implemented for any n]
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
        logger.debug("start_sigma = %s",start_sigma)
        logger.debug("force_model_center = %s",force_model_center)
        logger.debug("degenerate = %s",degenerate)

        self.scale = scale
        self.size = size
        self.pixel_area = self.scale*self.scale
        if interp is None: interp = Lanczos(3)
        elif isinstance(interp, basestring): interp = eval(interp)
        self.interp = interp
        self._force_model_center = force_model_center
        self._degenerate = degenerate

        # These are the kwargs that can be serialized easily.
        self.kwargs = {
            'scale' : scale,
            'size' : size,
            'start_sigma' : start_sigma,
            'force_model_center' : force_model_center,
            'degenerate' : degenerate,
            'interp' : repr(self.interp),
        }

        if size <= 0:
            raise ValueError("Non-positive PixelGrid size {:d}".format(size))

        self._nparams = size*size
        logger.debug("nparams = %d",self._nparams)

        # Now we need to make a 2d array whose entries are the indices of
        # each pixel in the 1d parameter array.
        self._indices = np.arange(self._nparams, dtype=int).reshape(size,size)

        # Now create a parameter array for a Gaussian that will be used to initialize new stars
        self._origin = (self.size//2, self.size//2)
        u = np.arange( -self._origin[0], self._indices.shape[0]-self._origin[0]) * self.scale
        v = np.arange( -self._origin[1], self._indices.shape[1]-self._origin[1]) * self.scale
        rsq = (u*u)[:,np.newaxis] + (v*v)[np.newaxis,:]
        gauss = np.exp(-rsq / (2.* start_sigma * start_sigma))
        params = self._1dFrom2d(gauss)
        # Renormalize to get unity flux
        params /= np.sum(params)*self.pixel_area
        self._initial_params = params
    def _1dFrom2d(self, in2d):
        """Make a 1d array from a 2d array, using the model's
        mapping from the 2d psf grid to the 1d parameter array.

        :param in2d:    A 2d array matching the PSFs sample grid

        :returns:       A 1d array of the length of number of grid points in use

        :returns  None
        """
        out1d = np.zeros( (self._nparams,), dtype=in2d.dtype)
        out1d[self._indices] = in2d
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
        return i[self._indices]

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
        nopsf = (y < 0) | (y >= self.size) | (x < 0) | (x >= self.size)
        # Set them to reference pixel 0
        x = np.where(nopsf, 0, x)
        y = np.where(nopsf, 0, y)
        # Then read all indices, setting invalid ones to -1
        return np.where(nopsf, -1, self._indices[y, x])

    def initialize(self, star, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a star instance with the appropriate initial fit values
        """
        var = np.zeros(len(self._initial_params))

        data, weight, u, v = star.data.getDataVector()
        # Start with the sum of pixels as initial estimate of flux.
        flux = np.sum(data)
        # Subtract center from u, v:
        Ix = np.sum(data * u) / flux
        Iy = np.sum(data * v) / flux
        center = (Ix,Iy)
        u -= center[0]
        v -= center[1]

        # We will limit the calculations to |u|, |v| < maxuv
        self.maxuv = self.size/2. * self.scale

        # Null weight at pixels where interpolation coefficients
        # come up short of specified fraction of the total kernel
        required_kernel_fraction = 0.7
        coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)
        # Turn the (psfx,psfy) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index;
        # Null the coefficients for such pixels
        coeffs = np.where(index1d < 0, 0., coeffs)
        use = np.sum(coeffs,axis=1) > required_kernel_fraction
        stardata = star.data.maskPixels(use)

        starfit = StarFit(self._initial_params, flux, center, params_var=var)
        return Star(star.data, starfit)

    def fit(self, star, logger=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated fit information
        """
        star1 = self.chisq(star)  # Get chisq Taylor expansion for linearized model

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
        var = np.zeros(len(star1.fit.params))
        starfit2 = StarFit(star1.fit.params + dparam,
                           params_var = var,
                           flux = star1.fit.flux,
                           center = star1.fit.center,
                           alpha = star1.fit.alpha,  # Inverse covariance matrix
                           chisq = star1.fit.chisq \
                                   + np.dot(dparam, np.dot(star1.fit.alpha, dparam)) \
                                   - 2 * np.dot(star1.fit.beta, dparam))

        star = Star(star1.data, starfit2)
        self.normalize(star)
        return star

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
        mask = (np.abs(u) < self.maxuv) & (np.abs(v) < self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]

        coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)

        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        alt_index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)

        # Multiply kernel (and derivs) by current PSF element values
        # to get current estimates
        pvals = star.fit.params[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        resid = data - mod*star.fit.flux

        # Now begin construction of alpha/beta/chisq that give
        # chisq vs linearized model.
        rw = resid * weight
        chisq = np.sum(resid * rw)
        dof = np.count_nonzero(weight)

        derivs = star.fit.flux * coeffs  #derivs wrt PSF elements

        # Accumulate alpha and beta point by point.  I don't
        # know how to do it purely with numpy calls instead of a loop over data points
        beta = np.zeros(self._nparams, dtype=float)
        alpha = np.zeros( (self._nparams,self._nparams), dtype=float)
        for i in range(len(data)):
            ii = index1d[i,:]
            cc = derivs[i,:]
            # Select only those with ii >= 0
            cc = cc[ii>=0]
            ii = ii[ii>=0]
            # beta_j += resid_i * weight_i * coeff_{ij}
            beta[ii] += rw[i] * cc
            # alpha_jk += weight_i * coeff_ij * coeff_ik
            dalpha = cc[np.newaxis,:]*cc[:,np.newaxis] * weight[i]
            iouter = np.broadcast_to(ii, (len(ii),len(ii)))
            alpha[iouter.flatten(), iouter.T.flatten()] += dalpha.flatten()

        var = np.zeros(len(star.fit.params))
        outfit = StarFit(star.fit.params,
                         params_var = var,
                         flux = star.fit.flux,
                         center = star.fit.center,
                         chisq = chisq,
                         dof = dof,
                         alpha = alpha,
                         beta = beta)

        return Star(star.data, outfit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      The fit parameters for a given star.

        :returns: a galsim.GSObject instance
        """
        im = galsim.Image(params.reshape(self.size,self.size), scale=self.scale)
        return galsim.InterpolatedImage(im, x_interpolant=self.interp,
                                        normalization='sb', use_true_center=False)

    def normalize(self, star):
        """Make sure star.fit.params are normalized properly.

        Note: This modifies the input star in place.
        """
        # Backwards compatibility check.
        # We used to only keep nparams - 1 or nparams - 3 values in fit.params.
        # If this is the case, fix it up to match up with our new convention.
        nparams1 = len(star.fit.params)
        nparams2 = self.size**2
        if nparams1 < nparams2:
            # Difference is either 1 or 3.  If not, something very weird happened.
            assert nparams2 - nparams1 in [1,3]

            # First copy over the parameters into the full array
            temp = np.zeros((self.size,self.size))
            mask = np.ones((self.size,self.size), dtype=bool)
            origin = (self.size//2, self.size//2)
            mask[origin] = False
            if nparams2 == nparams1 + 3:
                mask[origin[0]+1,origin[1]] = False
                mask[origin[0],origin[1]+1] = False
            temp[mask] = star.fit.params

            # Now populate the masked pixels
            delta_u = np.arange(-origin[0], self.size-origin[0])
            delta_v = np.arange(-origin[1], self.size-origin[1])
            u, v = np.meshgrid(delta_u, delta_v)
            if nparams2 == nparams1 + 3:
                # Do off-origin pixels first so that the centroid is 0,0.
                temp[origin[0]+1, origin[1]] = -np.sum(v*temp)
                temp[origin[0], origin[1]+1] = -np.sum(u*temp)

            # Now the center from the total flux == 1
            temp[origin] = 1./self.pixel_area - np.sum(temp)

            star.fit.params = temp.flatten()

        star.fit.params /= np.sum(star.fit.params)*self.pixel_area

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  DOF in the result assume only flux (& center) are free parameters.

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
        delta_u = np.arange(-self._origin[0], self.size-self._origin[0])
        delta_v = np.arange(-self._origin[1], self.size-self._origin[1])
        u, v = np.meshgrid(delta_u, delta_v)
        temp = star.fit.params.reshape(self.size,self.size)
        params_cenu = np.sum(u*temp)/np.sum(temp)
        params_cenv = np.sum(v*temp)/np.sum(temp)

        # Make sure input is properly normalized
        self.normalize(star)

        flux = star.fit.flux
        center = star.fit.center

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
        mask = (np.abs(u) < self.maxuv) & (np.abs(v) < self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]
        if self._force_model_center:
            coeffs, psfx, psfy, dcdu, dcdv = self.interp_calculate(u/self.scale, v/self.scale, True)
            dcdu /= self.scale
            dcdv /= self.scale
        else:
            coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)
        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        alt_index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        coeffs = np.where(nopsf, 0., coeffs)
        if self._force_model_center:
            dcdu = np.where(nopsf, 0., dcdu)
            dcdv = np.where(nopsf, 0., dcdv)

        # Multiply kernel (and derivs) by current PSF element values
        # to get current estimates
        pvals = star.fit.params[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        if self._force_model_center:
            dmdu = flux * np.sum(dcdu*pvals, axis=1)
            dmdv = flux * np.sum(dcdv*pvals, axis=1)
            derivs = np.vstack( (mod, dmdu, dmdv)).T
        else:
            derivs = mod.reshape(mod.shape+(1,))
        resid = data - mod*flux
        logger.debug("total pixels = %s, nopsf = %s",len(pvals),np.sum(nopsf))

        # Now begin construction of alpha/beta/chisq that give
        # chisq vs linearized model.
        rw = resid * weight
        chisq = np.sum(resid * rw)
        logger.debug("initial chisq = %s",chisq)
        beta = np.dot(derivs.T,rw)
        alpha = np.dot(derivs.T*weight, derivs)
        df = np.linalg.solve(alpha, beta)

        dchi = np.dot(beta, df)
        logger.debug("chisq -= %s => %s",dchi,chisq-dchi)
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
        if self._force_model_center:
            center = (center[0]+df[1],
                      center[1]+df[2])
            logger.debug("center += (%s,%s) => %s",df[1],df[2],center)

            center = (center[0]+params_cenu*self.scale,
                      center[1]+params_cenv*self.scale)
            logger.debug("params cen = %s,%s.  center => %s",params_cenu,params_cenv,center)

        dof = np.count_nonzero(weight)
        logger.debug("dchi, dof, do_center = %s, %s, %s", dchi, dof, self._force_model_center)

        var = np.zeros(len(star.fit.params))
        # Update to the expected new chisq value.
        chisq = chisq - dchi
        return Star(star.data, StarFit(star.fit.params,
                                       params_var = var,
                                       flux = flux,
                                       center = center,
                                       chisq = chisq,
                                       worst_chisq = worst_chisq,
                                       dof = dof,
                                       alpha = star.fit.alpha,
                                       beta = star.fit.beta))

    def interp_calculate(self, u, v, derivs=False):
        """Calculate interpolation coefficient for vector of target points

        Outputs will be 3 matrices, each of dimensions (nin, nkernel) where nin is
        number of input coordinates and nkernel is number of points in kernel footprint.
        The coeff matrix gives interpolation coefficients, then the y and x integer matrices
        give the grid point to which each coefficient is applied.

        :param u:       1d array of target u coordinates
        :param v:       1d array of target v coordinates
        :param derivs:  whether to also return derivatives (default: False)

        :returns: coeff, x, y[, dcdu, dcdv]
        """
        n = int(np.ceil(self.interp.xrange))
        # Here is range of pixels to use in each dimension relative to ceil(u,v)
        _duv = np.arange(-n, n, dtype=int)
        # And here are flattened arrays of u, v displacement for whole footprint
        _du = np.ones( (2*n,2*n), dtype=int) * _duv
        _du = _du.flatten()
        _dv = np.ones( (2*n,2*n), dtype=int) * _duv[:,np.newaxis]
        _dv = _dv.flatten()

        # Get integer and fractional parts of u, v
        u_ceil = np.ceil(u).astype(int)
        v_ceil = np.ceil(v).astype(int)
        # Make arrays giving coordinates of grid points within footprint
        x = u_ceil[:,np.newaxis] + _du[np.newaxis,:]
        y = v_ceil[:,np.newaxis] + _dv[np.newaxis,:]

        # Make npts x (2*order) arrays holding 1d displacements
        # to be arguments of the 1d kernel functions
        argu = (u_ceil-u)[:,np.newaxis] + _duv
        argv = (v_ceil-v)[:,np.newaxis] + _duv
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
            return coeffs, x, y, dcdu, dcdv
        else:
            return coeffs, x, y

    def _kernel1d(self, u):
        """ Calculate the 1d interpolation kernel at each value in array u.

        :param u: 1d array of (u_dest-u_src) spanning the footprint of the kernel.

        :returns: interpolation kernel values at these grid points
        """
        # TODO: It would be nice to allow any GalSim.Interpolant for the interp, but
        #       we'd need to get the equivalent of this functionality into the public API
        #       of the galsim.Interpolant class.  Currently, there is nothing like this
        #       available in the python API, although the C++ layer does do this calculation.
        #       For now, we just implement this for Lanczos.

        # Normalize Lanczos to unit sum over kernel elements
        n = self.interp._n
        k = np.sinc(u) * np.sinc(u/n)
        return k / np.sum(k,axis=1)[:,np.newaxis]
