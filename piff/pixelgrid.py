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
from .util import hsm

class PixelGrid(Model):

    _method = 'no_pixel'

    """A PSF modeled as interpolation between a grid of points.

    The parameters of the model are the values at the grid points, although the sum of the
    values is constrained to be 1/scale**2, to give it unit total flux. The grid is in uv
    space, with the scale and size specified on construction.  Interpolation will always
    assume values of zero outside of grid.

    PixelGrid also needs to specify an interpolant to define how to values between grid points
    are determined from the pixelated model.  For now only Lanczos is implemented, but we
    plan to expand this to include all GalSim Interpolants.

    Stellar data is assumed either to be in flux units (with default sb=False), such that
    flux is defined as sum of pixel values; or in surface brightness units (sb=True), such
    that flux is (sum of pixels)*(pixel area).  Internally the sb convention is used, although
    the flux convention is more typical of input data.

    :param scale:       Pixel scale of the PSF model (in arcsec)
    :param size:        Number of pixels on each side of square grid.
    :param interp:      An Interpolant to be used [default: Lanczos(3); currently only
                        Lanczos(n) is implemented, for any n]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, scale, size, interp=None, centered=True, logger=None,
                 start_sigma=None, degenerate=None):
        # start_sigma and degenerate are for backwards compatibility.  Ignore.
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Building Pixel model with the following parameters:")
        logger.debug("scale = %s",scale)
        logger.debug("size = %s",size)
        logger.debug("interp = %s",interp)
        logger.debug("centered = %s",centered)

        self.scale = scale
        self.size = size
        self.pixel_area = self.scale*self.scale
        if interp is None: interp = Lanczos(3)
        elif isinstance(interp, basestring): interp = eval(interp)
        self.interp = interp
        self._centered = centered

        # We will limit the calculations to |u|, |v| <= maxuv
        self.maxuv = self.size/2. * self.scale

        # The origin of the model in image coordinates
        self._origin = (self.size//2, self.size//2)

        # These are the kwargs that can be serialized easily.
        self.kwargs = {
            'scale' : scale,
            'size' : size,
            'centered' : centered,
            'interp' : repr(self.interp),
        }

        if size <= 0:
            raise ValueError("Non-positive PixelGrid size {:d}".format(size))

        self._nparams = size*size
        logger.debug("nparams = %d",self._nparams)

    def _indexFromPsfxy(self, psfx, psfy):
        """ Turn arrays of coordinates of the PSF array into a single same-shape
        array of indices into a 1d parameter vector.  The index is <0 wherever
        the psf x,y values were outside the PSF mask.

        :param psfx:  array of integer x displacements from origin of the PSF grid
        :param psfy:  array of integer y displacements from origin of the PSF grid

        :returns: same shape array, filled with indices into 1d array
        """
        # Shift psfy, psfx to reference a 0-indexed array
        y = psfy + self._origin[0]
        x = psfx + self._origin[1]

        # Good pixels are where there is a valid index
        # Others are set to -1.
        ind = np.ones_like(psfx, dtype=int) * -1
        good = (0 <= y) & (y < self.size) & (0 <= x) & (x < self.size)
        indices = np.arange(self._nparams, dtype=int).reshape(self.size,self.size)
        ind[good] = indices[y[good], x[good]]
        return ind

    def initialize(self, star, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a star instance with the appropriate initial fit values
        """
        data, weight, u, v = star.data.getDataVector()
        # Start with the sum of pixels as initial estimate of flux.
        flux = np.sum(data)
        if self._centered:
            # Initial center is the centroid of the data.
            Ix = np.sum(data * u) / flux
            Iy = np.sum(data * v) / flux
            center = (Ix,Iy)
        else:
            # In this case, center is fixed.
            center = star.fit.center

        # Calculate the second moment to initialize an initial Gaussian profile.
        # hsm returns: flux, x, y, sigma, g1, g2, flag
        sigma = hsm(star)[3]

        # Create an initial parameter array using a Gaussian profile.
        u = np.arange( -self._origin[0], self.size-self._origin[0]) * self.scale
        v = np.arange( -self._origin[1], self.size-self._origin[1]) * self.scale
        rsq = (u*u)[:,np.newaxis] + (v*v)[np.newaxis,:]
        gauss = np.exp(-rsq / (2.* sigma**2))
        params = gauss.ravel()

        # Normalize to get unity flux
        params /= np.sum(params)*self.pixel_area

        starfit = StarFit(params, flux, center)
        return Star(star.data, starfit)

    def fit(self, star, logger=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated fit information
        """
        star1 = self.chisq(star)  # Get chisq Taylor expansion for linearized model

        # The chisq function calculates A and b where
        #
        #    chisq = chisq_0 + 2 bT A dp + dpT AT A dp
        #
        # is the linearized variation in chisq with respect to changes in the parameter values.
        # The minimum of this linearized functional form is
        #
        #    dp = (AT A)^-1 AT b
        #
        # This is just the least squares solution of
        #
        #    A dp = b
        #
        # Even if the solutionis degenerate, gelsy works fine using QRP decomposition.
        # And it's much faster than SVD.
        dparam = scipy.linalg.lstsq(star1.fit.A, star1.fit.b,
                                    check_finite=False, cond=1.e-6,
                                    lapack_driver='gelsy')[0]

        # Create new StarFit, update the chisq value.  Note no beta is returned as
        # the quadratic Taylor expansion was about the old parameters, not these.
        # TODO: Calculate params_var and set that as well.  params_var=np.diag(alpha) ??
        Adp = star1.fit.A.dot(dparam)
        new_chisq = star1.fit.chisq + Adp.dot(Adp) - 2 * Adp.dot(star1.fit.b)

        starfit2 = StarFit(star1.fit.params + dparam,
                           flux = star1.fit.flux,
                           center = star1.fit.center,
                           A = star1.fit.A,
                           chisq = new_chisq)

        star = Star(star1.data, starfit2)
        self.normalize(star)
        return star

    def chisq(self, star, logger=None):
        """Calculate dependence of chi^2 = -2 log L(D|p) on PSF parameters for single star.
        as a quadratic form chi^2 = dp^T AT A dp - 2 bT A dp + chisq,
        where dp is the *shift* from current parameter values.  Returned Star
        instance has the resultant A, b, chisq, flux, center) attributes,
        but params vector has not have been updated yet (could be degenerate).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated StarFit
        """
        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()

        # Subtract star.fit.center from u, v:
        u -= star.fit.center[0]
        v -= star.fit.center[1]

        # Only use pixels covered by the model.
        mask = (np.abs(u) <= self.maxuv) & (np.abs(v) <= self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]

        # Compute the full set of coefficients for each pixel in the data
        # The returned arrays here are Ndata x Ninterp.
        # Each column column corresponds to a different x,y value in the model that could
        # contribute information about the given data pixel.
        coeffs, psfx, psfy = self.interp_calculate(u/self.scale, v/self.scale)

        # Turn the (psfy,psfx) coordinates into an index into 1d parameter vector.
        index1d = self._indexFromPsfxy(psfx, psfy)
        # All invalid pixel references now have negative index; record and set to zero
        nopsf = index1d < 0
        alt_index1d = np.where(nopsf, 0, index1d)
        # And null the coefficients for such pixels
        # This just makes it easier to do the sums, since then the nopsf values won't contribute.
        coeffs = np.where(nopsf, 0., coeffs)

        # Multiply kernel (and derivs) by current PSF element values to get current estimates
        pvals = star.fit.params[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        scaled_flux = star.fit.flux * star.data.pixel_area
        resid = data - mod * scaled_flux

        # Now construct A, b, chisq that give chisq vs linearized model.
        #
        # We can say we are looking for a weighted least squares solution to the problem
        #
        #   A dp = b
        #
        # where b is the array of residuals, and A_ij = coeffs[i][k] iff alt_index1d[i][k] == j.
        #
        # The weights are dealt with in the standard way, by multiplying both A and b by sqrt(w).

        A = np.zeros((len(data), self._nparams), dtype=float)
        for i in range(len(data)):
            ii = index1d[i,:]
            cc = coeffs[i,:]
            # Select only those with ii >= 0
            cc = cc[ii>=0] * scaled_flux
            ii = ii[ii>=0]
            A[i,ii] = cc
        sw = np.sqrt(weight)
        Aw = A * sw[:,np.newaxis]
        bw = resid * sw
        chisq = np.sum(bw**2)
        dof = np.count_nonzero(weight)

        outfit = StarFit(star.fit.params,
                         flux = star.fit.flux,
                         center = star.fit.center,
                         chisq = chisq,
                         dof = dof,
                         A = Aw,
                         b = bw)

        return Star(star.data, outfit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      The fit parameters for a given star.

        :returns: a galsim.GSObject instance
        """
        im = galsim.Image(params.reshape(self.size,self.size), scale=self.scale)
        return galsim.InterpolatedImage(im, x_interpolant=self.interp,
                                        normalization='sb', use_true_center=False, flux=1.)

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
            if nparams2 == nparams1 + 3:  # pragma: no branch
                # Note: the only regression we test is with centroids free, so we always hit
                #       this branch.
                mask[origin[0]+1,origin[1]] = False
                mask[origin[0],origin[1]+1] = False
            temp[mask] = star.fit.params

            # Now populate the masked pixels
            delta_u = np.arange(-origin[0], self.size-origin[0])
            delta_v = np.arange(-origin[1], self.size-origin[1])
            u, v = np.meshgrid(delta_u, delta_v)
            if nparams2 == nparams1 + 3:  # pragma: no branch
                # Do off-origin pixels first so that the centroid is 0,0.
                temp[origin[0]+1, origin[1]] = -np.sum(v*temp)
                temp[origin[0], origin[1]+1] = -np.sum(u*temp)

            # Now the center from the total flux == 1
            temp[origin] = 1./self.pixel_area - np.sum(temp)

            star.fit.params = temp.flatten()

        # Normally this is all that is required.
        star.fit.params /= np.sum(star.fit.params)*self.pixel_area

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  DOF in the result assume only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance, with updated flux, center, chisq, dof
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

        # Make sure input is properly normalized
        self.normalize(star)
        scaled_flux = star.fit.flux * star.data.pixel_area
        center = star.fit.center

        # Calculate the current centroid of the model at the location of this star.
        # We'll shift the star's position to try to zero this out.
        delta_u = np.arange(-self._origin[0], self.size-self._origin[0])
        delta_v = np.arange(-self._origin[1], self.size-self._origin[1])
        u, v = np.meshgrid(delta_u, delta_v)
        temp = star.fit.params.reshape(self.size,self.size)
        params_cenu = np.sum(u*temp)/np.sum(temp)
        params_cenv = np.sum(v*temp)/np.sum(temp)

        # Start by getting all interpolation coefficients for all observed points
        data, weight, u, v = star.data.getDataVector()

        u -= center[0]
        v -= center[1]
        mask = (np.abs(u) <= self.maxuv) & (np.abs(v) <= self.maxuv)
        data = data[mask]
        weight = weight[mask]
        u = u[mask]
        v = v[mask]

        # Build the model and maybe also d(model)/dcenter
        # This tracks the same steps in chisq.
        # TODO: Make a helper function to consolidate the common code.
        if self._centered:
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
        if self._centered:
            dcdu = np.where(nopsf, 0., dcdu)
            dcdv = np.where(nopsf, 0., dcdv)

        # Multiply kernel (and derivs) by current PSF element values to get current estimates
        pvals = star.fit.params[alt_index1d]
        mod = np.sum(coeffs*pvals, axis=1)
        if self._centered:
            dmdu = scaled_flux * np.sum(dcdu*pvals, axis=1)
            dmdv = scaled_flux * np.sum(dcdv*pvals, axis=1)
            derivs = np.vstack((mod, dmdu, dmdv)).T
        else:
            # In this case, we're just making it a column vector.
            derivs = np.vstack((mod,)).T
        resid = data - mod*scaled_flux
        logger.debug("total pixels = %s, nopsf = %s",len(pvals),np.sum(nopsf))

        # Now construct the design matrix for this minimization
        #
        #    A x = b
        #
        # where x = [ dflux, duc, dvc ]^T or just [ dflux ] and b = resid.
        #
        # A[0] = d( mod * flux ) / dflux = mod
        # A[1] = d( mod * flux ) / duc   = flux * sum(dcdu * pvals, axis=1)
        # A[2] = d( mod * flux ) / dvc   = flux * sum(dcdv * pvals, axis=1)

        # For large matrices, it is generally better to solve this with QRP, but with this
        # small a matrix, it is faster and not any less stable to just compute AT A and AT b
        # and solve the equation
        #
        #    AT A x = AT b

        Atw = derivs.T * weight  # weighted least squares
        AtA = Atw.dot(derivs)
        Atb = Atw.dot(resid)
        x = np.linalg.solve(AtA, Atb)
        chisq = np.sum(resid**2 * weight)
        dchi = Atb.dot(x)
        logger.debug("chisq = %s - %s => %s",chisq,dchi,chisq-dchi)

        # update the flux (and center) of the star
        logger.debug("initial flux = %s",scaled_flux)
        scaled_flux += x[0]
        logger.debug("flux += %s => %s",x[0],scaled_flux)
        logger.debug("center = %s",center)
        if self._centered:
            center = (center[0]+x[1], center[1]+x[2])
            logger.debug("center += (%s,%s) => %s",x[1],x[2],center)

            # In addition to shifting to the best fit center location, also shift
            # by the centroid of the model itself, so the next next pass through the
            # fit will be closer to centered.  In practice, this converges pretty quickly.
            center = (center[0]+params_cenu*self.scale, center[1]+params_cenv*self.scale)
            logger.debug("params cen = %s,%s.  center => %s",params_cenu,params_cenv,center)

        dof = np.count_nonzero(weight)
        logger.debug("dchi, dof, do_center = %s, %s, %s", dchi, dof, self._centered)

        # Update to the expected new chisq value.
        chisq = chisq - dchi
        return Star(star.data, StarFit(star.fit.params,
                                       flux = scaled_flux / star.data.pixel_area,
                                       center = center,
                                       chisq = chisq,
                                       dof = dof,
                                       A = star.fit.A,
                                       b = star.fit.b))

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
