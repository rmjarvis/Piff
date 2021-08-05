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
.. module:: gsobject_model
"""

import numpy as np
import galsim
import scipy

from .model import Model
from .star import Star, StarFit
from .util import estimate_cov_from_jac


class GSObjectModel(Model):
    """ Model that takes a fiducial GalSim.GSObject and dilates, shifts, and shears it to get a
    good match to stars.

    :param gsobj:       GSObject to use as fiducial profile.
    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel when drawing?  [default: True]
    :param scipy_kwargs: Optional kwargs to pass to scipy.optimize.least_squares [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _model_can_be_offset = False

    def __init__(self, gsobj, fastfit=False, centered=True, include_pixel=True,
                 scipy_kwargs=None, logger=None):
        if isinstance(gsobj, str):
            gsobj = eval(gsobj)

        self.kwargs = {'gsobj':repr(gsobj),
                       'fastfit':fastfit,
                       'centered':centered,
                       'include_pixel':include_pixel}

        # Center and normalize the fiducial model.
        self.gsobj = gsobj.withFlux(1.0).shift(-gsobj.centroid)
        self._fastfit = fastfit
        self._centered = centered
        self._method = 'auto' if include_pixel else 'no_pixel'
        self._scipy_kwargs = scipy_kwargs if scipy_kwargs is not None else {}

        # Params are [du, dv], scale, g1, g2, i.e., transformation parameters that bring the
        # fiducial gsobject towards the data.
        if self._centered:
            self._nparams = 3
        else:
            self._nparams = 5

    def moment_fit(self, star, logger=None):
        """Estimate transformations needed to bring self.gsobj towards given star."""
        flux, cenu, cenv, size, g1, g2, flag = star.hsm
        if flag != 0:
            raise RuntimeError("Error initializing star fit values using hsm.")
        shape = galsim.Shear(g1=g1, g2=g2)

        ref_flux, ref_cenu, ref_cenv, ref_size, ref_g1, ref_g2, flag = self.draw(star).hsm
        ref_shape = galsim.Shear(g1=ref_g1, g2=ref_g2)
        if flag != 0:
            raise RuntimeError("Error calculating model moments for this star.")

        param_flux = star.fit.flux
        if star.fit.params is None:
            param_scale = 1
            param_g1 = param_g2 = param_du = param_dv = 0
        elif self._centered:
            param_scale, param_g1, param_g2 = star.fit.params
            param_du, param_dv = star.fit.center
        else:
            param_du, param_dv, param_scale, param_g1, param_g2 = star.fit.params
        param_shear = galsim.Shear(g1=param_g1, g2=param_g2)

        param_flux *= flux / ref_flux
        param_du += cenu - ref_cenu
        param_dv += cenv - ref_cenv
        param_scale *= size / ref_size
        param_shear += (shape - ref_shape)
        param_g1 = param_shear.g1
        param_g2 = param_shear.g2

        # Rough estimate of the variance, assuming noise is uniform.
        var_pix = 1./np.mean(star.weight.array)
        pixel_area = star.image.wcs.pixelArea(image_pos=star.image_pos)
        var_flux = 2*np.pi * var_pix * size**2 / pixel_area
        f = var_flux / flux**2
        var_cenx = f * (1+g1)**2 * size**2
        var_ceny = f * (1-g1)**2 * size**2
        # This estimate for var_size is not very close actually.  A better calculation would
        # require an integral of r^4.  For some plausible profiles, this is within about 20% or
        # so of the right answer, so not too terrible.
        var_size = f * size**2
        var_g = f

        var = np.zeros(6)
        var[0] = var_flux
        # We expect some fudge factors for this because of the non-linearity in the hsm fitter.
        # These are completely empirical that work ok for the default models we have available
        # for gsobj (Gaussian, Kolmogorov, Moffat).  Probably won't work well for a wider array
        # of user-provided gsobj parameters.
        var[1] = var_cenx * 4.8
        var[2] = var_ceny * 4.8
        var[3] = var_size * 4.8
        var[4] = var_g * 2.0
        var[5] = var_g * 2.0

        return param_flux, param_du, param_dv, param_scale, param_g1, param_g2, var

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with either  [ size, g1, g2 ]
                            or  [ cenu, cenv, size, g1, g2 ]
                            depending on if the center of the model is being forced to (0.0, 0.0)
                            or not.

        :returns: a galsim.GSObject instance
        """
        if params is None:
            return self.gsobj
        elif self._centered:
            scale, g1, g2 = params
            return self.gsobj.dilate(scale).shear(g1=g1, g2=g2)
        else:
            du, dv, scale, g1, g2 = params
            return self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

    def _resid(self, params, star, convert_func):
        """Residual function to use with least_squares.

        Essentially `chi` from `chisq`, but not summed over pixels yet.

        :param params:          A numpy array of model parameters.
        :param star:            A Star instance.
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component.

        :returns: `chi` as a flattened numpy array.
        """
        image, weight, image_pos = star.data.getImage()
        flux, du, dv, scale, g1, g2 = params

        # Make sure the shear is sane.
        g = g1 + 1j * g2
        if np.abs(g) >= 1.:
            # Return "infinity"
            return np.ones_like(image.array.ravel()) * 1.e300

        # We shear/dilate/shift the profile as follows.
        #    prof = self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv) * flux
        # However, it is a bit faster to do all these operations at once to avoid some superfluous
        # calculations that GalSim does for each of these steps when done separately.
        jac = galsim._Shear(g).getMatrix()
        jac[:,:] *= scale
        flux /= scale**2
        prof = galsim._Transform(self.gsobj, jac, offset=(du,dv), flux_ratio=flux)

        # Equivalent to galsim.Image(image, dtype=float), but without the sanity checks.
        model_image = galsim._Image(np.empty_like(image.array, dtype=float),
                                    image.bounds, image.wcs)

        if convert_func is not None:
            prof = convert_func(prof)

        prof.drawImage(model_image, method=self._method, center=image_pos)

        # Caculate sqrt(weight) * (model_image - image) in place for efficiency.
        model_image.array[:,:] -= image.array
        model_image.array[:,:] *= np.sqrt(weight.array)
        return model_image.array.ravel()

    def _get_params(self, star):
        """Generate an array of model parameters.

        :param star:         A Star from which to initialize parameter values.

        :returns: a numpy array
        """
        # Get initial parameter values.  Either use values currently in star.fit, or if those are
        # absent, run HSM to get initial values.
        if star.fit.params is None:
            flux, du, dv, scale, g1, g2, var = self.moment_fit(star)
        else:
            flux = star.fit.flux
            if self._centered:
                du, dv = star.fit.center
                scale, g1, g2 = star.fit.params
            else:
                du, dv, scale, g1, g2 = star.fit.params

        return np.array([flux, du, dv, scale, g1, g2])

    def least_squares_fit(self, star, logger=None, convert_func=None):
        """Fit parameters of the given star using least-squares minimization.

        :param star:            A Star to fit.
        :param logger:          A logger object for logging debug info. [default: None]
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component. [default: None]

        :returns: (flux, dx, dy, scale, g1, g2, flag)
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Start least_squares")
        params = self._get_params(star)

        results = scipy.optimize.least_squares(self._resid, params, args=(star,convert_func),
                                               **self._scipy_kwargs)
        if logger:
            logger.debug(results)
        if not results.success:
            raise RuntimeError("Error finding the full nonlinear solution")

        flux, du, dv, scale, g1, g2 = results.x
        var = np.diagonal(estimate_cov_from_jac(results.jac))

        return flux, du, dv, scale, g1, g2, var

    def fit(self, star, fastfit=None, logger=None, convert_func=None):
        """Fit the image either using HSM or least-squares minimization.

        If ``fastfit`` is True, then the galsim.hsm module will be used to estimate the
        transformation parameters that take the fiducial moments into the data moments.
        If ``fastfit`` is False, then the Levenberg-Marquardt minimization algorithm will be used
        instead.  The latter should generally be more accurate, but slower due to the need to
        iteratively propose model improvements.

        :param star:            A Star to fit.
        :param fastfit:         Use fast HSM moments to fit? [default: None, which means use
                                fitting mode specified in the constructor.]
        :param logger:          A logger object for logging debug info. [default: None]
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component. [default: None]

        :returns: a new Star with the fitted parameters in star.fit
        """
        if fastfit is None:
            fastfit = self._fastfit
        if convert_func is not None:
            # Can't do the moments fit technique if fitting using moments.
            # At least not as it is currently structured.  May be possible to convert if there
            # is a need, but it seems hard.
            fastfit = False

        if fastfit:
            flux, du, dv, scale, g1, g2, var = self.moment_fit(star, logger=logger)
        else:
            flux, du, dv, scale, g1, g2, var = self.least_squares_fit(star, logger=logger,
                                                                      convert_func=convert_func)

        # Make a StarFit object with these parameters
        if self._centered:
            params = np.array([ scale, g1, g2 ])
            center = (du, dv)
            params_var = var[3:]
        else:
            params = np.array([ du, dv, scale, g1, g2 ])
            center = (0.0, 0.0)
            params_var = var[1:]

        # Also need to compute chisq
        prof = self.getProfile(params) * flux
        model_image = star.image.copy()
        prof = prof.shift(center)

        if convert_func is not None:
            prof = convert_func(prof)

        prof.drawImage(model_image, method=self._method, center=star.image_pos)
        chisq = np.sum(star.weight.array * (star.image.array - model_image.array)**2)
        # Don't subtract number of parameters from dof, since we'll be interpolating, so
        # these parameters don't really apply to each star separately.
        # After refluxing, we may drop this by 1 or 3 if adjusting flux and/or centroid.
        dof = np.count_nonzero(star.weight.array)
        fit = StarFit(params, params_var=params_var, flux=flux, center=center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def initialize(self, star, logger=None):
        """Initialize the given star's fit parameters.

        :param star:  The Star to initialize.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new initialized Star.
        """
        if star.fit.params is None:
            if self._centered:
                params = np.array([ 1.0, 0.0, 0.0])
                params_var = np.array([ 0.0, 0.0, 0.0])
            else:
                params = np.array([ 0.0, 0.0, 1.0, 0.0, 0.0])
                params_var = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            fit = StarFit(params, flux=1.0, center=(0.0, 0.0), params_var=params_var)
            star = Star(star.data, fit)
            star = self.fit(star, fastfit=True)
        star = self.reflux(star, fit_center=False)
        return star


class Gaussian(GSObjectModel):
    """ Model PSFs as elliptical Gaussians.

    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel when drawing?  [default: True]
    :param scipy_kwargs: Optional kwargs to pass to scipy.optimize.least_squares [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, centered=True, include_pixel=True,
                 scipy_kwargs=None, logger=None):
        gsobj = galsim.Gaussian(sigma=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, scipy_kwargs, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Gaussian, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']


class Kolmogorov(GSObjectModel):
    """ Model PSFs as elliptical Kolmogorovs.

    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel when drawing?  [default: True]
    :param scipy_kwargs: Optional kwargs to pass to scipy.optimize.least_squares [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, centered=True, include_pixel=True,
                 scipy_kwargs=None, logger=None):
        gsobj = galsim.Kolmogorov(half_light_radius=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, scipy_kwargs, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Kolmogorov, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']


class Moffat(GSObjectModel):
    """ Model PSFs as elliptical Moffats.

    :param beta:        Moffat shape parameter.
    :param trunc:       Optional truncation radius at which profile drops to zero.  Measured in half
                        light radii.  [default: 0, indicating no truncation]
    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel when drawing?  [default: True]
    :param scipy_kwargs: Optional kwargs to pass to scipy.optimize.least_squares [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, beta, trunc=0., fastfit=False, centered=True, include_pixel=True,
                 scipy_kwargs=None, logger=None):
        gsobj = galsim.Moffat(half_light_radius=1.0, beta=beta, trunc=trunc)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, scipy_kwargs, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Moffat, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']
        # Need to add `beta` and `trunc` though.
        self.kwargs.update(dict(beta=beta, trunc=trunc))
