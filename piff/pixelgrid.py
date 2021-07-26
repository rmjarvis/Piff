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
    """A PSF modeled as interpolation between a grid of points.

    The parameters of the model are the values at the grid points, although the sum of the
    values is constrained to be 1/scale**2, to give it unit total flux. The grid is in uv
    space, with the scale and size specified on construction.  Interpolation will always
    assume values of zero outside of grid.

    PixelGrid also needs to specify an interpolant to define how to values between grid points
    are determined from the pixelated model.  Any galsim.Interpolant type is allowed.
    The default interpolant is galsim.Lanczos(3)

    :param scale:       Pixel scale of the PSF model (in arcsec)
    :param size:        Number of pixels on each side of square grid.
    :param interp:      An Interpolant to be used [default: Lanczos(3)]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """

    _method = 'no_pixel'
    _model_can_be_offset = True  # Indicate that in reflux, the centroid should also move by the
                                 # current centroid of the model.  This way on later iterations,
                                 # the model will be close to centered.

    def __init__(self, scale, size, interp=None, centered=True, logger=None):
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
        self.maxuv = (self.size+1)/2. * self.scale

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
        # (Skip w=0 pixels here.)
        mask = weight!=0
        flux = np.sum(data[mask])
        if self._centered:
            # Initial center is the centroid of the data.
            Ix = np.sum(data[mask] * u[mask]) / flux
            Iy = np.sum(data[mask] * v[mask]) / flux
            center = (Ix,Iy)
        else:
            # In this case, center is fixed.
            center = star.fit.center

        # Calculate the second moment to initialize an initial Gaussian profile.
        # hsm returns: flux, x, y, sigma, g1, g2, flag
        sigma = star.hsm[3]

        # Create an initial parameter array using a Gaussian profile.
        u = np.arange( -self._origin[0], self.size-self._origin[0]) * self.scale
        v = np.arange( -self._origin[1], self.size-self._origin[1]) * self.scale
        rsq = (u*u)[:,np.newaxis] + (v*v)[np.newaxis,:]
        gauss = np.exp(-rsq / (2.* sigma**2))
        params = gauss.ravel()

        # Normalize to get unity flux
        params /= np.sum(params)

        starfit = StarFit(params, flux, center)
        return Star(star.data, starfit)

    def fit(self, star, logger=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star instance with updated fit information
        """
        logger = galsim.config.LoggerWrapper(logger)
        star1 = self.chisq(star, logger=logger)  # Get chisq Taylor expansion for linearized model

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
        # Even if the solution is degenerate, gelsy works fine using QRP decomposition.
        # And it's much faster than SVD.
        dparam = scipy.linalg.lstsq(star1.fit.A, star1.fit.b,
                                    check_finite=False, cond=1.e-6,
                                    lapack_driver='gelsy')[0]
        logger.debug('dparam = %s',dparam)

        # Create new StarFit, update the chisq value.  Note no beta is returned as
        # the quadratic Taylor expansion was about the old parameters, not these.
        Adp = star1.fit.A.dot(dparam)
        new_chisq = star1.fit.chisq + Adp.dot(Adp) - 2 * Adp.dot(star1.fit.b)
        logger.debug('chisq = %s',new_chisq)

        # covariance of dp is C = (AT A)^-1
        # params_var = diag(C)
        try:
            params_var = np.diagonal(scipy.linalg.inv(star1.fit.A.T.dot(star1.fit.A)))
        except np.linalg.LinAlgError as e:
            # If we get an error, set the variance to "infinity".
            logger.info("Caught error %s making params_var.  Setting all to 1.e100",e)
            params_var = np.ones_like(dparam) * 1.e100

        starfit2 = StarFit(star1.fit.params + dparam,
                           flux = star1.fit.flux,
                           center = star1.fit.center,
                           params_var = params_var,
                           A = star1.fit.A,
                           chisq = new_chisq)

        star = Star(star1.data, starfit2)
        self.normalize(star)
        return star

    def chisq1(self, star, logger=None):
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
        scaled_flux = star.fit.flux * star.data.pixel_area / self.pixel_area
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
                         params_var = star.fit.params_var,
                         chisq = chisq,
                         dof = dof,
                         A = Aw,
                         b = bw)

        return Star(star.data, outfit)

    def chisq2(self, star, logger=None, convert_func=None):
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug('Start chisq function')
        logger.debug('initial params = %s',star.fit.params)

        data, weight, u, v = star.data.getDataVector()
        prof = self.getProfile(star.fit.params)._shift(*star.fit.center)
        logger.debug('prof.flux = %s',prof.flux)

        # My idea for doing composite functions is that at this point in the calculation, we
        # could apply a function to prof to convert it to the full effective PSF profile.
        # This function could be passed as an optional extra argument to the fit and chisq
        # functions.  (The default would just be `lambda prof:prof`.)
        # E.g. in a Sum context, this function could be
        #    convert_func = `lambda prof: galsim.Sum(prof, *other_profiles)`
        # Then at this point in the chisq function (and comparable places for other profiles),
        # we would call:
        #    prof = convert_func(prof)
        # The draw function below would then draw the full composite profile, not just the
        # PixelGrid profile.

        # To test the convert_func branches, uncomment this line.
        #convert_func = lambda x: x

        if convert_func is not None:
            prof = convert_func(prof)

        model_image = star.image.copy()
        if 0:
            # This is the straightforward implementation.
            # This step isn't too slow, but it will be below for the differential draws,
            # where the GalSim overhead in drawImage adds a lot of cruft.
            prof.drawImage(model_image, method=self._method, center=star.image_pos)
        else:
            # Fortunately, we make it possible (although less intuitive) to avoid most of the
            # overhead. The following is equivalent to the above drawImage call, but faster.
            offset = prof._adjust_offset(model_image.bounds, galsim.PositionD(0,0),
                                         star.image_pos, False)
            prof = star.data.local_wcs.profileToImage(prof, offset=offset)
            model_image._shift(-model_image.center)
            model_image.wcs = galsim.PixelScale(1.0)
            # Be careful here.  If prof was converted to something that isn't analytic in
            # real space, this will need to be _drawFFT.
            prof._drawReal(model_image)

        logger.debug('drawn flux = %s',model_image.array.sum())
        model = model_image.array.ravel() * star.fit.flux

        # Only use data points where model gives reasonable support
        u0, v0 = star.fit.center
        u -= u0
        v -= v0
        mask = (np.abs(u) <= self.maxuv) & (np.abs(v) <= self.maxuv) & (weight != 0)
        data = data[mask]
        model = model[mask]
        weight = weight[mask]
        resid = data - model

        # Calculate A = d(model)/dp
        # A[i,k] = d(model_i)/dp_k
        # b[i] = resid_i
        # Solution to A x = b will be the desired dparams
        A = np.empty((len(data), self._nparams), dtype=float)
        temp = model_image.copy()

        basis_profile, basis_scale, basis_shifts = self._getBasisProfile()
        dx, dy = basis_shifts

        # Get the inverse jacobian, so we can apply it below.
        # Faster than letting GalSim combine the two Transformation wrappers.
        jac = star.data.local_wcs.jacobian().inverse()
        jac1 = np.array(((jac.dudx, jac.dudy), (jac.dvdx, jac.dvdy)))

        if convert_func is not None:
            # In this case we need the basis_profile to have the right scale (rather than
            # incorporate it into the jacobian) so that convert_func will have the right size.
            basis_profile = basis_profile.dilate(basis_scale)
            # Find the net shift from the star.fit.center and the offset.
            u1 = jac1[0,0]*u0 + jac1[0,1]*v0 + offset.x
            v1 = jac1[1,0]*u0 + jac1[1,1]*v0 + offset.y
        else:
            # Convert u,v shifts into x,y shifts
            dx1 = dx + u0
            dy1 = dy + v0
            dx = jac1[0,0]*dx1 + jac1[0,1]*dy1 + offset.x
            dy = jac1[1,0]*dx1 + jac1[1,1]*dy1 + offset.y
            # Incorporate scale into jac for _drawReal call.
            jac2 = jac1 * self.scale

        for k, dxk, dyk in zip(range(len(dx)), dx,dy):

            if 0:
                # This is the straightforward implementation.
                # But it's slow due to lots of GalSim overhead (mostly in the drawImage function)
                basis_profile_k = basis_profile._shift(dxk,dyk)

                # At this point if this model is just a component in a larger full PSF profile,
                # we can apply a function to it to get to the net effective profile.
                if convert_func is not None:
                    basis_profile_k = convert_func(basis_profile_k)

                basis_profile_k = basis_profile_k_shift(u0,v0)
                basis_profile_k.drawImage(temp, method=self._method, center=star.image_pos)

            elif convert_func is not None:
                # This implementation removes most of the overhead, and is still relatively
                # straightforward.
                # The one wrinkle is that if the net profile is no longer analytic in real space,
                # we'll need a call to _drawFFT rather than _drawReal.  That would be a lot slower
                # I think, so we may want to limit convolutions in conjunction with PixelGrid.
                # (That's less of an issue for models with few paramters.)

                basis_profile_k = basis_profile._shift(dxk,dyk)
                basis_profile_k = convert_func(basis_profile_k)
                basis_profile_k._drawReal(temp, jac1, (u1, v1), 1.)

            else:
                # If we don't have a convert_func, then it's faster to combine the shifts with
                # the jacobian all in a numpy multiplication (above).
                basis_profile._drawReal(temp, jac2, (dxk,dyk), 1.)

            A[:,k] = temp.array.ravel()[mask]

        A *= star.fit.flux

        # But actually, do weighted least squares.
        # Multiply A and b by sqrt(weight).
        sw = np.sqrt(weight)
        A = A * sw[:,np.newaxis]
        b = resid * sw
        chisq = np.sum(resid**2 * weight)
        dof = np.count_nonzero(weight)
        logger.debug('chisq,dof = %s,%s',chisq,dof)

        outfit = StarFit(star.fit.params,
                         flux = star.fit.flux,
                         center = star.fit.center,
                         params_var = star.fit.params_var,
                         chisq = chisq,
                         dof = dof,
                         A = A,
                         b = b)

        return Star(star.data, outfit)

    chisq = chisq2

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      The fit parameters for a given star.

        :returns: a galsim.GSObject instance
        """
        im = galsim.Image(params.reshape(self.size,self.size), scale=self.scale)
        return galsim.InterpolatedImage(im, x_interpolant=self.interp,
                                        use_true_center=False, flux=1.)

    def _getBasisProfile(self):
        if not hasattr(self, '_basis_profile'):
            self._basis_profile = []
            # Note: Things are faster if the underlying InterpolatedImage has scale=1.
            # We apply the dilation and shifts separately in chisq2.
            im = galsim.Image(np.array([[1.]]), scale=1.)
            self._basis_profile = galsim.InterpolatedImage(im, x_interpolant=self.interp)

            uar = []
            var = []
            for i in range(self.size):
                v = (-self._origin[1] + i) * self.scale
                for j in range(self.size):
                    u = (-self._origin[0] + j) * self.scale
                    uar.append(u)
                    var.append(v)
            self._basis_shifts = np.array(uar), np.array(var)

        return self._basis_profile, self.scale, self._basis_shifts

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
            temp[origin] = 1. - np.sum(temp)

            star.fit.params = temp.flatten()

        # Normally this is all that is required.
        star.fit.params /= np.sum(star.fit.params)

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
        k = self.interp.xval(u.ravel()).reshape(u.shape)
        # Normalize to unit sum over kernel elements
        return k / np.sum(k,axis=1)[:,np.newaxis]
