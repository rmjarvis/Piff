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

from .model import Model, ModelFitError
from .star import Star, StarFit, StarData
from .util import hsm


class GSObjectModel(Model):
    """ Model that takes a fiducial GalSim.GSObject and dilates, shifts, and shears it to get a
    good match to stars.

    :param gsobj:    GSObject to use as fiducial profile.
    :param fastfit:  Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param force_model_center: If True, PSF model centroid is fixed at origin and
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel:   Include integration over pixel?  [default: True]
    :param logger:   A logger object for logging debug info. [default: None]
    """
    def __init__(self, gsobj, fastfit=False, force_model_center=True, include_pixel=True,
                 logger=None):
        if isinstance(gsobj, str):
            import galsim
            gsobj = eval(gsobj)

        self.kwargs = {'gsobj':repr(gsobj),
                       'fastfit':fastfit,
                       'force_model_center':force_model_center,
                       'include_pixel':include_pixel}

        self.min_scale = 0.1
        self.max_shear = 0.7

        # Center and normalize the fiducial model.
        self.gsobj = gsobj.withFlux(1.0).shift(-gsobj.centroid())
        self._fastfit = fastfit
        self._force_model_center = force_model_center
        self._method = 'auto' if include_pixel else 'no_pixel'
        # Params are [du, dv], scale, g1, g2, i.e., transformation parameters that bring the
        # fiducial gsobject towards the data.
        if self._force_model_center:
            self._nparams = 3
        else:
            self._nparams = 5

    def moment_fit(self, star, profile=None, logger=None):
        """Estimate transformations needed to bring self.gsobj towards given star."""
        import galsim
        flux, cenu, cenv, size, g1, g2 = star.data.properties['hsm']
        shape = galsim.Shear(g1=g1, g2=g2)

        ref_flux, ref_cenu, ref_cenv, ref_size, ref_g1, ref_g2, flag = hsm(self.draw(star, profile=profile))
        ref_shape = galsim.Shear(g1=ref_g1, g2=ref_g2)
        if logger:
            logger.debug("    hsm_flux = {0}".format(flux))
            logger.debug("    hsm_cenu = {0}".format(cenu))
            logger.debug("    hsm_cenv = {0}".format(cenv))
            logger.debug("    hsm_size = {0}".format(size))
            logger.debug("    hsm_g1 = {0}".format(g1))
            logger.debug("    hsm_g2 = {0}".format(g2))
            logger.debug("    fit params = {0}".format(star.fit.params))
            logger.debug("    ref_flux = {0}".format(ref_flux))
            logger.debug("    ref_cenu = {0}".format(ref_cenu))
            logger.debug("    ref_cenv = {0}".format(ref_cenv))
            logger.debug("    ref_size = {0}".format(ref_size))
            logger.debug("    ref_g1 = {0}".format(ref_g1))
            logger.debug("    ref_g2 = {0}".format(ref_g2))
            logger.debug("    flag = {0}".format(flag))
        if flag:
            raise ModelFitError("Error calculating model moments for this star.")

        param_flux = star.fit.flux
        if self._force_model_center:
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

        if logger:
            logger.debug("    flux = {0}".format(param_flux))
            logger.debug("    du = {0}".format(param_du))
            logger.debug("    dv = {0}".format(param_dv))
            logger.debug("    scale = {0}".format(param_scale))
            logger.debug("    g1 = {0}".format(param_g1))
            logger.debug("    g2 = {0}".format(param_g2))

        return param_flux, param_du, param_dv, param_scale, param_g1, param_g2

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with either  [ size, g1, g2 ]
                            or  [ cenu, cenv, size, g1, g2 ]
                            depending on if the center of the model is being forced to (0.0, 0.0)
                            or not.

        :returns: a galsim.GSObject instance
        """
        if self._force_model_center:
            scale, g1, g2 = params
            du, dv = (0.0, 0.0)
        else:
            du, dv, scale, g1, g2 = params
        return self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv)

    def draw(self, star, profile=None):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        prof = self.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        if profile:
            from galsim import Convolve
            prof = Convolve([profile, prof])
        image = star.image.copy()
        prof.drawImage(image, method=self._method, offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, star.fit)

    def _lmfit_resid(self, lmparams, star, profile=None):
        """Residual function to use with lmfit.  Essentially `chi` from `chisq`, but not summed
        over pixels yet.

        :param lmparams:  An lmfit.Parameters() instance.  The model.
        :param star:    A Star instance.  The data.

        :returns: `chi` as a flattened numpy array.
        """
        image, weight, image_pos = star.data.getImage()
        flux, du, dv, scale, g1, g2 = lmparams.valuesdict().values()
        # Fit du and dv regardless of force_model_center.  The difference is whether the fit
        # value is recorded (force_model_center=False) or discarded (force_model_center=True).
        prof = self.gsobj.dilate(scale).shear(g1=g1, g2=g2).shift(du, dv) * flux
        if profile:
            from galsim import Convolve
            prof = Convolve([profile, prof])
        model_image = image.copy()
        prof.drawImage(model_image, method=self._method,
                       offset=(image_pos - model_image.trueCenter()))
        return (np.sqrt(weight.array)*(model_image.array - image.array)).ravel()

    def _lmfit_params(self, star, vary_params=True, vary_flux=True, vary_center=True, profile=None):
        """Generate an lmfit.Parameters() instance from arguments.

        :param star:         A Star from which to initialize parameter values.
        :param vary_params:  Allow non-flux and non-center params to vary?
        :param vary_flux:    Allow flux to vary?
        :param vary_center:  Allow center to vary?

        :returns: lmfit.Parameters() instance.
        """
        import lmfit

        # Get initial parameter values.  Either use values currently in star.fit, or if those are
        # absent, run HSM to get initial values.
        if star.fit.params is None:
            flux, du, dv, scale, g1, g2, flag = self.moment_fit(star, profile=profile)
            if flag != 0:
                raise RuntimeError("Error initializing star fit values using hsm.")
        else:
            flux = star.fit.flux
            if self._force_model_center:
                du, dv = star.fit.center
                scale, g1, g2 = star.fit.params
            else:
                du, dv, scale, g1, g2 = star.fit.params

        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=flux, vary=vary_flux, min=0.0)
        params.add('du', value=du, vary=vary_center)
        params.add('dv', value=dv, vary=vary_center)
        params.add('scale', value=scale, vary=vary_params, min=self.min_scale)
        # Limits of +/- 0.7 is definitely a hack to avoid |g| > 1, but if the PSF is ever actually
        # this elliptical then we have more serious problems to worry about than hacky code!
        params.add('g1', value=g1, vary=vary_params, min=-self.max_shear, max=self.max_shear)
        params.add('g2', value=g2, vary=vary_params, min=-self.max_shear, max=self.max_shear)
        return params

    def _lmfit_minimize(self, params, star, profile=None, logger=None):
        """ Run lmfit.minimize with given lmfit.Parameters() and on given star data.

        :param params: lmfit.Parameters() instance (holds initial guess and which params to let
                       float or hold fixed).
        :param star:   Star to fit.

        :returns: lmfit.MinimizerResult instance containing fit results.
        """
        import lmfit
        if logger:
            import time
            t0 = time.time()
            logger.debug("Start lmfit minimize.")
        results = lmfit.minimize(self._lmfit_resid, params, args=(star,profile,))
        flux, du, dv, scale, g1, g2 = results.params.valuesdict().values()

        if logger:
            logger.debug("End lmfit minimize.  Elapsed time: {0}".format(time.time() - t0))
        return results

    def lmfit(self, star, profile=None, logger=None):
        """Fit parameters of the given star using lmfit (Levenberg-Marquardt minimization
        algorithm).

        :param star:    A Star to fit.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: (flux, dx, dy, scale, g1, g2, flag)
        """
        params = self._lmfit_params(star, profile=profile)
        results = self._lmfit_minimize(params, star, profile=profile, logger=logger)
        if logger:
            import lmfit
            logger.debug(lmfit.fit_report(results))
        flux, du, dv, scale, g1, g2 = results.params.valuesdict().values()
        if not results.success:
            raise RuntimeError("Error fitting with lmfit.")

        return flux, du, dv, scale, g1, g2

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

    def fit(self, star, fastfit=None, profile=None, logger=None):
        """Fit the image either using HSM or lmfit.

        If `fastfit` is True, then the galsim.hsm module will be used to estimate the transformation
        parameters that take the fiducial moments into the data moments.  If `fastfit` is False,
        then the Levenberg-Marquardt minimization algorithm will be used instead.  The latter should
        generally be more accurate, but slower due to the need to iteratively propose model
        improvements.

        :param star:    A Star to fit.
        :param fastfit: Use fast HSM moments to fit? [default: None, which means use fitting mode
                        specified in the constructor.]
        :param profile: A galsim profile which is convolved with gsobject's
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star with the fitted parameters in star.fit
        """
        if fastfit is None:
            fastfit = self._fastfit

        if not hasattr(star.data.properties, 'hsm'):
            star = self.initialize(star, logger=logger)

        if fastfit:
            flux, du, dv, scale, g1, g2 = self.moment_fit(star, profile=profile, logger=logger)
        else:
            flux, du, dv, scale, g1, g2 = self.lmfit(star, profile=profile, logger=logger)
        # Make a StarFit object with these parameters
        if self._force_model_center:
            params = np.array([ scale, g1, g2 ])
            center = (du, dv)
        else:
            params = np.array([ du, dv, scale, g1, g2 ])
            center = (0.0, 0.0)

        # Also need to compute chisq
        prof = self.getProfile(params) * flux
        model_image = star.image.copy()
        prof = prof.shift(center)
        if profile:
            # not sure if importing here is kosher
            from galsim import Convolve
            prof = Convolve([profile, prof])
        prof.drawImage(model_image, method=self._method,
                                     offset=(star.image_pos - model_image.trueCenter()))
        chisq = np.sum(star.weight.array * (star.image.array - model_image.array)**2)
        dof = np.count_nonzero(star.weight.array) - self._nparams
        fit = StarFit(params, flux=flux, center=center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def initialize(self, star, mask=True, logger=None):
        """Initialize the given star's fit parameters.

        :param star:  The Star to initialize.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new initialized Star.
        """
        star = self.with_hsm(star)
        if star.fit.params is None:
            if self._force_model_center:
                params = np.array([ 1.0, 0.0, 0.0])
            else:
                params = np.array([ 0.0, 0.0, 1.0, 0.0, 0.0])
            fit = StarFit(params, flux=1.0, center=(0.0, 0.0))
            star = Star(star.data, fit)
            star = self.fit(star, fastfit=True, logger=logger)
        star = self.reflux(star, fit_center=False, logger=logger)
        return star

    def reflux(self, star, fit_center=True, profile=None, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.  DOF in the result assume
        only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
        :param profile: A galsim profile which is convolved with gsobject's
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance, with updated flux, center, chisq, dof, worst
        """
        if logger:
            logger.debug("Reflux for star:")
            logger.debug("    flux = %s",star.fit.flux)
            logger.debug("    center = %s",star.fit.center)
            logger.debug("    props = %s",star.data.properties)
            logger.debug("    image = %s",star.data.image)
            #logger.debug("    image = %s",star.data.image.array)
            #logger.debug("    weight = %s",star.data.weight.array)
            logger.debug("    image center = %s",star.data.image(star.data.image.center()))
            logger.debug("    weight center = %s",star.data.weight(star.data.weight.center()))
        do_center = fit_center and self._force_model_center
        if do_center:
            params = self._lmfit_params(star, vary_params=False)
            results = self._lmfit_minimize(params, star, profile=profile, logger=logger)
            return Star(star.data, StarFit(star.fit.params,
                                           flux = results.params['flux'].value,
                                           center = (results.params['du'].value,
                                                     results.params['dv'].value),
                                           chisq = results.chisqr,
                                           dof = np.count_nonzero(star.data.weight.array) - 3,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))
        else:
            image, weight, image_pos = star.data.getImage()
            model_image = self.draw(star, profile=profile).image
            flux_ratio = (np.sum(weight.array * image.array * model_image.array)
                          / np.sum(weight.array * model_image.array**2))
            new_chisq = np.sum(weight.array * (image.array - flux_ratio*model_image.array)**2)
            return Star(star.data, StarFit(star.fit.params,
                                           flux = star.flux*flux_ratio,
                                           center = star.fit.center,
                                           chisq = new_chisq,
                                           dof = np.count_nonzero(weight.array) - 1,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))


class Gaussian(GSObjectModel):
    """ Model PSFs as elliptical Gaussians.

    :param fastfit:  Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param force_model_center: If True, PSF model centroid is fixed at origin and
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel:   Include integration over pixel?  [default: True]
    :param logger:   A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, force_model_center=True, include_pixel=True, logger=None):
        import galsim
        gsobj = galsim.Gaussian(sigma=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, force_model_center, include_pixel, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Gaussian, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']


class Kolmogorov(GSObjectModel):
    """ Model PSFs as elliptical Kolmogorovs.

    :param fastfit:  Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param force_model_center: If True, PSF model centroid is fixed at origin and
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel:   Include integration over pixel?  [default: True]
    :param logger:   A logger object for logging debug info. [default: None]
    """
    def __init__(self, fastfit=False, force_model_center=True, include_pixel=True, logger=None):
        import galsim
        gsobj = galsim.Kolmogorov(half_light_radius=1.0)
        GSObjectModel.__init__(self, gsobj, fastfit, force_model_center, include_pixel, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Kolmogorov, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']


class Moffat(GSObjectModel):
    """ Model PSFs as elliptical Moffats.

    :param beta:  Moffat shape parameter.
    :param trunc:  Optional truncation radius at which profile drops to zero.  Measured in half
                   light radii.  [default: 0, indicating no truncation]
    :param fastfit:  Use HSM moments for fitting.  Approximate, but fast.  [default: False]
    :param force_model_center: If True, PSF model centroid is fixed at origin and
                        PSF fitting will marginalize over stellar position.  If False, stellar
                        position is fixed at input value and the fitted PSF may be off-center.
                        [default: True]
    :param include_pixel:   Include integration over pixel?  [default: True]
    :param logger:   A logger object for logging debug info. [default: None]
    """
    def __init__(self, beta, trunc=0., fastfit=False, force_model_center=True, include_pixel=True,
                 logger=None):
        import galsim
        gsobj = galsim.Moffat(half_light_radius=1.0, beta=beta, trunc=trunc)
        GSObjectModel.__init__(self, gsobj, fastfit, force_model_center, include_pixel, logger)
        # We'd need self.kwargs['gsobj'] if we were reconstituting via the GSObjectModel
        # constructor, but since config['type'] for this will be Moffat, it gets reconstituted
        # here, where there is no `gsobj` argument.  So remove `gsobj` from kwargs.
        del self.kwargs['gsobj']
        # Need to add `beta` and `trunc` though.
        self.kwargs.update(dict(beta=beta, trunc=trunc))
