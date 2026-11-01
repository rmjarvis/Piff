# Copyright (c) 2026 by Mike Jarvis and the other collaborators on GitHub at
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

import numpy as np
import galsim
from scipy.special import lambertw
from scipy.optimize import minimize_scalar

from .model import Model
from .star import Star


class FourierGaussian(galsim.GSObject):
    """A Gaussian profile defined in terms of its moments and only rendered in Fourier space.

    This enables a couple things over the normal galsim.Gaussian profile.

    1. It can handle singular positive-semidefinite moment matrices, which are valid kinds of
       blur. If the blur is exactly one-dimensional (or zero-dimensional), this class can handle
       it just fine.
    2. The derivatives of the Fourier image with respect to the moments are analytic, which
       simplifies the math for doing the fit, since we don't need finite-difference derivatives.

    :param moments:     The moment matrix as a 3-element array [ixx, iyy, ixy]
    :param derivative:  None means this is the regular profile with these moments
                        0 means this is the derivative of the profile with respect to ixx
                        1 means this is the derivative of the profile with respect to iyy
                        2 means this is the derivative of the profile with respect to ixy
                        (default: None)
    :param gsparams:    An optional GSParams instance to control behavior (default: None)
    """
    _has_hard_edges = False
    _is_analytic_x = False
    _is_analytic_k = True
    _centroid = galsim.PositionD(0, 0)
    _max_sb = float('inf')

    def __init__(self, moments, derivative=None, gsparams=None):
        self.moments = tuple(float(x) for x in moments)
        self.derivative = derivative
        self._gsparams = galsim.GSParams.check(gsparams)
        ixx, iyy, ixy = self.moments
        self._eigenvalues = np.linalg.eigvalsh([[ixx, ixy], [ixy, iyy]])
        self._flux = 1. if derivative is None else 0.
        self._is_axisymmetric = ixx == iyy and ixy == 0

    @property
    def _maxk(self):
        smallest = self._eigenvalues[0]
        if smallest <= 0:
            return 1.e100
        else:
            threshold = self.gsparams.maxk_threshold
            if self.derivative is None:
                radius = np.sqrt(-2*np.log(threshold))
            else:
                # Derivative profiles have zero net flux, so for these, we define
                # max_threshold to be relative to the peak absolute Fourier amplitude.
                # A moment derivative is a quadratic in wavevector times a Gaussian.
                # For covariance C and wavevector k, define the radius r as r^2 = k.T C k.
                # Past the derivative's peak, its absolute amplitude divided by
                # its peak amplitude has an upper bound of (r^2/2) exp(1-r^2/2).
                # Set this envelope equal to threshold and solve for r using Lambert W's
                # negative branch, which selects the outer, decreasing tail.
                radius = np.sqrt(-2*lambertw(-threshold/np.e, k=-1).real)
            # The narrowest covariance width gives a cutoff covering every k direction.
            return radius / np.sqrt(smallest)

    @property
    def _stepk(self):
        largest = self._eigenvalues[1]
        if largest == 0:
            return 1.e100
        else:
            threshold = self.gsparams.folding_threshold
            if self.derivative is None:
                radius = np.sqrt(-2*np.log(threshold))
            else:
                # Again, derivative profiles have zero net flux, so here we consider
                # the amount of omitted flux relative to the integrated absolute value
                # of the profile.
                # Choose a radius outside which the integral of |derivative| is at
                # most threshold times its integral over the whole plane. Work in
                # coordinates where the Gaussian has unit covariance, with radius r.
                # Bound the quadratic derivative factor by an angular envelope and
                # the whole-plane integral from below. The resulting tail fraction
                # is at most (r^2+4) exp(-r^2/2) / (4 phi(1)), where
                # phi(1) = exp(-1/2)/sqrt(2*pi) is the standard-normal density at 1.
                # Set this bound equal to threshold and solve for r with Lambert W's
                # negative branch; the shifted variable is r^2/2 + 2.
                argument = -np.sqrt(2/np.pi)*np.exp(-2.5)*threshold
                radius = np.sqrt(-2*lambertw(argument, k=-1).real-4)
            # Use the widest covariance width to enclose the tail in every direction,
            # retaining the ordinary Gaussian minimum of stepk_minimum_hlr half-light radii.
            radius = max(radius, self.gsparams.stepk_minimum_hlr * 1.1774100225154747)
            return np.pi / (radius * np.sqrt(largest))

    def _kval(self, kx, ky):
        ixx, iyy, ixy = self.moments
        kval = np.exp(-0.5*(ixx*kx*kx + iyy*ky*ky + 2*ixy*kx*ky))
        if self.derivative is not None:
            if self.derivative == 0:
                kval *= -0.5*kx*kx
            elif self.derivative == 1:
                kval *= -0.5*ky*ky
            else:
                kval *= -kx*ky
        return kval

    def _kValue(self, pos):
        return self._kval(pos.x, pos.y)

    def _drawKImage(self, image, jac=None):
        kx, ky = np.meshgrid(
            np.arange(image.bounds.xmin, image.bounds.xmax+1) * image.scale,
            np.arange(image.bounds.ymin, image.bounds.ymax+1) * image.scale)
        if jac is not None:
            kx, ky = jac[0, 0]*kx + jac[1, 0]*ky, jac[0, 1]*kx + jac[1, 1]*ky
        image.array[:, :] = self._kval(kx, ky)

    def __eq__(self, other):
        return (isinstance(other, FourierGaussian) and self.moments == other.moments
                and self.derivative == other.derivative and self.gsparams == other.gsparams)

    def __hash__(self):
        return hash((self.moments, self.derivative, self.gsparams))

    def __repr__(self):
        return ('piff.gaussian_blur.FourierGaussian('
                f'{self.moments!r}, derivative={self.derivative!r}, gsparams={self.gsparams!r})')


class GaussianBlur(Model):
    """A centered, unit-flux Gaussian blur parameterized by ``[Ixx, Iyy, Ixy]``.

    Unlike ``Gaussian``, which fits size and reduced ellipticity, GaussianBlur uses
    unweighted second moments to gracefully handle nearly degenerate ellipses.
    Moments are in arcsec**2. Positive-semidefinite covariances, including zero blur
    and exactly one-dimensional blur, are supported as convolution kernels.

    This is typically used as a component in a `Convolve` PSF to describe the blur that
    comes from some kind of telescope jitter. It would not likely be a useful model
    for a full PSF profile.

    :param init: Initial second moments ``[Ixx, Iyy, Ixy]``, or a scalar trace
                 ``Ixx + Iyy`` split equally between the two axes. [default: None,
                 which starts with zero blur, equivalent to a delta function.]
    :param logger: Optional logger. [default: None]
    """
    _type_name = 'GaussianBlur'
    _method = 'auto'
    _centered = True
    _model_can_be_offset = False
    nonlinear_chisq = True

    def __init__(self, init=None, logger=None):
        if init is None:
            self.init = None
        else:
            self.init = np.array(init, dtype=float)
            if self.init.ndim == 0:
                self.init = np.array([self.init/2, self.init/2, 0.])
            if self.init.shape != (3,) or not self.is_valid_params(self.init):
                raise ValueError('init must be a nonnegative trace or positive-semidefinite '
                                 '[Ixx, Iyy, Ixy]')
        self.kwargs = {'init': self.init}
        self.set_num(None)

    def is_valid_params(self, params):
        """Whether second moments define a positive-semidefinite covariance."""
        ixx, iyy, ixy = params
        return (np.all(np.isfinite(params)) and ixx >= 0 and iyy >= 0
                and ixx*iyy >= ixy*ixy)

    def constrain_params(self, params):
        """Project a proposed covariance onto the positive-semidefinite cone.

        The parameters stored on a star are generally unconstrained params returned from an
        interpolator. These may be unphysical. E.g. ixx,iyy < 0 or ixy^2 < ixx * iyy.
        This function projects the raw params to the nearest physical param vector by
        setting any negative eigenvalues of the covariance matrix to 0.
        """
        ixx, iyy, ixy = params
        values, vectors = np.linalg.eigh([[ixx, ixy], [ixy, iyy]])
        # Make any negative eignenvalues -> 0 to get a positive semi-definite matrix.
        # This is the nearest positive semi-definite matrix in Frobenius norm, fwiw.
        covariance = (vectors * np.maximum(values, 0.)) @ vectors.T
        # It's possible for numerical rounding to end up with slightly negative results.
        # The next bits just enforce ixx,iyy >= 0 and ixy^2 <= ixx iyy given the machine precision.
        ixx, iyy = max(covariance[0, 0], 0.), max(covariance[1, 1], 0.)
        # Round toward the physical side of the determinant boundary.
        limit = np.nextafter(np.sqrt(ixx*iyy), 0.)
        ixy = np.clip(covariance[0, 1], -limit, limit)
        return np.array([ixx, iyy, ixy])

    def initialize(self, star, logger=None, default_init=None):
        """Initialize a star to work with the current model."""
        if self.init is None:
            params = np.zeros(3)
        else:
            params = self.init.copy()
        return Star(star.data, star.fit.newParams(params, num=self._num))

    def getProfile(self, params):
        """Return a unit-flux FourierGaussian instance corresponding to the current model."""
        return FourierGaussian(self.constrain_params(params))

    def _draw(self, profile, star, convert_func, draw_method):
        profile = profile.shift(star.fit.center) * star.fit.flux
        if convert_func is not None:
            profile = convert_func(profile)
        image = galsim.Image(star.image.bounds, wcs=star.image.wcs, dtype=float)
        profile.drawImage(image=image, center=star.image_pos,
                          method=self._method if draw_method is None else draw_method)
        return image.array

    def residual(self, star, convert_func=None, draw_method=None):
        """Return weighted model-minus-data residuals at the current parameters."""
        profile = self.getProfile(star.fit.get_params(self._num))
        model = self._draw(profile, star, convert_func, draw_method)
        return ((model-star.image.array) * np.sqrt(star.weight.array)).ravel()

    def chisq(self, star, logger=None, convert_func=None, draw_method=None):
        """Build the quadratic objective using analytic Fourier covariance derivatives."""
        raw_params = star.fit.get_params(self._num)
        params = self.constrain_params(raw_params)
        mask = star.weight.array.ravel() != 0
        b = -self.residual(star, convert_func, draw_method)[mask]
        chisq = np.dot(b, b)
        sw = np.sqrt(star.weight.array).ravel()[mask]
        A = np.empty((len(b), 3))
        for index in range(3):
            profile = FourierGaussian(params, derivative=index)
            A[:, index] = self._draw(profile, star, convert_func, draw_method).ravel()[mask] * sw
        # Derivatives are evaluated at the physical covariance, but A dp = b describes
        # increments from the raw parameters retained on the star and by interpolation.
        # This means we need to update b to correspond to the step from the raw params
        # rather than the valid projected params.
        b += A @ (params - raw_params)
        return Star(star.data, star.fit.withNew(A=A, b=b, chisq=chisq, dof=len(b)))

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        """Minimize chi-squared along one projected quadratic update direction."""
        measured = self.chisq(star, logger, convert_func, draw_method)
        raw_params = star.fit.get_params(self._num)
        proposal = raw_params + np.linalg.lstsq(measured.fit.A, measured.fit.b, rcond=None)[0]
        proposal = self.constrain_params(proposal)

        # We don't necessarily want to take the full proposed step. The likelihood surface
        # is non-linear, so it could go past the minimum and back up. We consider the
        # segment between the current parameters and the full proposal and do a line search
        # to find the minimum chisq in this direction.
        # Note that we don't need to project on every proposed step, since the positive
        # semi-definite region is convex, so once the full step is valid, all proposed
        # fractions of this will remain valid.
        def line_chisq(fraction):
            trial = (1-fraction)*raw_params + fraction*proposal
            candidate = Star(star.data, star.fit.newParams(trial, num=self._num))
            resid = self.residual(candidate, convert_func, draw_method)
            return np.dot(resid, resid)

        optimum = minimize_scalar(line_chisq, bounds=(0., 1.), method='bounded')
        # scipy's minimize_scalar doesn't check the end points, so check fraction=0,1 explicitly
        # to find the best chisq value.
        fraction, chisq = 0., measured.fit.chisq
        full_chisq = line_chisq(1.)
        if full_chisq < chisq:
            fraction, chisq = 1., full_chisq
        if optimum.fun < chisq:
            fraction, chisq = optimum.x, optimum.fun
        if fraction == 0.:
            fitted = measured
        else:
            trial = (1-fraction)*raw_params + fraction*proposal
            fit = star.fit.newParams(trial, num=self._num, chisq=chisq, dof=measured.fit.dof)
            fitted = Star(star.data, fit)
        return fitted
