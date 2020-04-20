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

from .model import Model, ModelFitError
from .star import Star, StarFit, StarData
from .util import hsm


class GSObjectModel(Model):
    """ Model that takes a fiducial GalSim.GSObject and dilates, shifts, and shears it to get a
    good match to stars.

    :param gsobj:       GSObject to use as fiducial profile.
    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel?  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, gsobj, fastfit=False, centered=True, include_pixel=True,
                 logger=None):
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
        # Params are [du, dv], scale, g1, g2, i.e., transformation parameters that bring the
        # fiducial gsobject towards the data.
        if self._centered:
            self._nparams = 3
        else:
            self._nparams = 5

    def moment_fit(self, star, logger=None):
        """Estimate transformations needed to bring self.gsobj towards given star."""
        flux, cenu, cenv, size, g1, g2 = star.data.properties['hsm']
        shape = galsim.Shear(g1=g1, g2=g2)

        ref_flux, ref_cenu, ref_cenv, ref_size, ref_g1, ref_g2, flag = hsm(self.draw(star))
        ref_shape = galsim.Shear(g1=ref_g1, g2=ref_g2)
        if flag:
            raise ModelFitError("Error calculating model moments for this star.")

        param_flux = star.fit.flux
        if self._centered:
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

        return param_flux, param_du, param_dv, param_scale, param_g1, param_g2

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with either  [ size, g1, g2 ]
                            or  [ cenu, cenv, size, g1, g2 ]
                            depending on if the center of the model is being forced to (0.0, 0.0)
                            or not.

        :returns: a galsim.GSObject instance
        """
        if self._centered:
            scale, g1, g2 = params
            du, dv = (0.0, 0.0)
        else:
            du, dv, scale, g1, g2 = params
        return self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

    def _resid(self, params, star):
        """Residual function to use with least_squares.

        Essentially `chi` from `chisq`, but not summed over pixels yet.

        :param params:      A numpy array of model parameters.
        :param star:        A Star instance.

        :returns: `chi` as a flattened numpy array.
        """
        image, weight, image_pos = star.data.getImage()
        flux, du, dv, scale, g1, g2 = params

        # We shear/dilate/shift the profile as follows.
        #    prof = self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv) * flux
        # However, it is a bit faster to do all these operations at once to avoid some superfluous
        # calculations that GalSim does for each of these steps when done separately.
        jac = galsim._Shear(g1 + 1j*g2).getMatrix()
        jac[:,:] *= scale
        flux /= scale**2
        prof = galsim._Transform(self.gsobj, jac.ravel(), offset=galsim.PositionD(du,dv),
                                 flux_ratio=flux)

        # Equivalent to galsim.Image(image, dtype=float), but without the sanity checks.
        model_image = galsim._Image(np.empty_like(image.array, dtype=float),
                                    image.bounds, image.wcs)
        prof.drawImage(model_image, method=self._method,
                       offset=(image_pos - model_image.true_center))

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
            flux, du, dv, scale, g1, g2, flag = self.moment_fit(star)
            if flag != 0:
                raise RuntimeError("Error initializing star fit values using hsm.")
        else:
            flux = star.fit.flux
            if self._centered:
                du, dv = star.fit.center
                scale, g1, g2 = star.fit.params
            else:
                du, dv, scale, g1, g2 = star.fit.params

        return np.array([flux, du, dv, scale, g1, g2])

    def _minimize(self, params, star, logger=None):
        """Find the least-squares best fit solution.

        :param params: numpy array of initial guess
        :param star:   Star to fit.

        :returns: scipy.optimize.OptimizeResults instance containing fit results.
        """
        import scipy
        import time
        logger = galsim.config.LoggerWrapper(logger)
        t0 = time.time()
        logger.debug("Start least_squares")

        results = scipy.optimize.least_squares(self._resid, params, args=(star,))
        logger.debug("Done.  Elapsed time: {0}".format(time.time() - t0))
        return results

    def least_squares_fit(self, star, logger=None):
        """Fit parameters of the given star using least-squares minimization.

        :param star:    A Star to fit.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: (flux, dx, dy, scale, g1, g2, flag)
        """
        logger = galsim.config.LoggerWrapper(logger)
        params = self._get_params(star)
        results = self._minimize(params, star, logger=logger)
        if logger:
            logger.debug(results)
        flux, du, dv, scale, g1, g2 = results.x
        if not results.success:
            raise RuntimeError("Error finding the full nonlinear solution")

        try:
            params_var = np.diag(results.covar)
        except (ValueError, AttributeError) as e:
            logger.debug("Failed to get params_var")
            logger.debug("  -- Caught exception: %s",e)
            # results.covar is either None or does not exist
            params_var = np.zeros(6)

        return flux, du, dv, scale, g1, g2, params_var

    @staticmethod
    def with_hsm(star):
        if not hasattr(star.data.properties, 'hsm'):
            flux, cenu, cenv, size, g1, g2, flag = hsm(star)
            if flag != 0:
                raise RuntimeError("Error initializing star fit values using hsm.")
            sd = star.data.copy()
            sd.properties['hsm'] = flux, cenu, cenv, size, g1, g2
            return Star(sd, star.fit)
        return star

    def fit(self, star, fastfit=None, logger=None):
        """Fit the image either using HSM or least-squares minimization.

        If `fastfit` is True, then the galsim.hsm module will be used to estimate the transformation
        parameters that take the fiducial moments into the data moments.  If `fastfit` is False,
        then the Levenberg-Marquardt minimization algorithm will be used instead.  The latter should
        generally be more accurate, but slower due to the need to iteratively propose model
        improvements.

        :param star:    A Star to fit.
        :param fastfit: Use fast HSM moments to fit? [default: None, which means use fitting mode
                        specified in the constructor.]
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star with the fitted parameters in star.fit
        """
        if fastfit is None:
            fastfit = self._fastfit

        if not hasattr(star.data.properties, 'hsm'):
            star = self.initialize(star)

        if fastfit:
            flux, du, dv, scale, g1, g2 = self.moment_fit(star, logger=logger)
            var = np.zeros(6)
        else:
            flux, du, dv, scale, g1, g2, var = self.least_squares_fit(star, logger=logger)
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
        prof.shift(center).drawImage(model_image, method=self._method,
                                     offset=(star.image_pos - model_image.true_center))
        chisq = np.sum(star.weight.array * (star.image.array - model_image.array)**2)
        dof = np.count_nonzero(star.weight.array) - self._nparams
        fit = StarFit(params, params_var=params_var, flux=flux, center=center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def initialize(self, star, logger=None):
        """Initialize the given star's fit parameters.

        :param star:  The Star to initialize.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new initialized Star.
        """
        star = self.with_hsm(star)
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

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.  DOF in the result assume
        only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance, with updated flux, center, chisq, dof
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

        image, weight, u, v = star.data.getDataVector(include_zero_weight=True)
        model = self.draw(star).image.array.flatten()

        WIM = weight * image * model
        WMM = weight * model**2

        f_data = np.sum(WIM)
        f_model = np.sum(WMM)
        flux_ratio = f_data / f_model

        if fit_center and self._centered:
            ushift = np.sum(WIM * u)/f_data - np.sum(WMM * u)/f_model
            vshift = np.sum(WIM * v)/f_data - np.sum(WMM * v)/f_model
            new_center = (star.fit.center[0] + ushift,
                          star.fit.center[1] + vshift)
        else:
            new_center = star.fit.center

        # new_chisq ignores centroid shift, but close enough.
        new_chisq = np.sum(weight * (image - flux_ratio*model)**2)

        return Star(star.data, StarFit(star.fit.params,
                                       flux = star.fit.flux * flux_ratio,
                                       center = new_center,
                                       chisq = new_chisq,
                                       dof = np.count_nonzero(weight) - 1,
                                       params_var = star.fit.params_var))


class Gaussian(GSObjectModel):
    """ Model PSFs as elliptical Gaussians.

    :param fastfit:     Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param centered:    If True, PSF model centroid is forced to be (0,0), and the
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel: Include integration over pixel?  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, centered=True, include_pixel=True, logger=None):
        gsobj = galsim.Gaussian(sigma=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, logger)
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
    :param include_pixel: Include integration over pixel?  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, centered=True, include_pixel=True, logger=None):
        gsobj = galsim.Kolmogorov(half_light_radius=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, logger)
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
    :param include_pixel: Include integration over pixel?  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, beta, trunc=0., fastfit=False, centered=True, include_pixel=True,
                 logger=None):
        gsobj = galsim.Moffat(half_light_radius=1.0, beta=beta, trunc=trunc)
        GSObjectModel.__init__(self, gsobj, fastfit, centered, include_pixel, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Moffat, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']
        # Need to add `beta` and `trunc` though.
        self.kwargs.update(dict(beta=beta, trunc=trunc))
