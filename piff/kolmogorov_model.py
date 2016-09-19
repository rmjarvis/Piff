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
.. module:: model_kolmogorov
"""

from __future__ import print_function
import numpy as np

from .model import Model
from .star import Star, StarFit, StarData

class Kolmogorov(Model):
    """An extremely simple PSF model that just considers the PSF as a sheared Kolmogorov with
    parameters for full-width-half-maximum `fwhm`, the applied shear `g1` and `g2`, and optionally,
    centroid parameters `cenu` and `cenv`.
    """
    def __init__(self, fastfit=False, force_model_center=True, logger=None):
        """A PSF modeled as a Kolmogorov profile.

        :param fastfit:  Use HSM moments for fitting.  Approximate, but fast.  [default: True]
        :param force_model_center: If True, PSF model centroid is fixed at origin and
                            PSF fitting will marginalize over stellar position.  If False, stellar
                            position is fixed at input value and the fitted PSF may be off-center.
                            [default: True]
        :param logger:  A logger object for logging debug info. [default: None]
        """
        self._fastfit = fastfit
        self._force_model_center = force_model_center
        self._nparams = 3 if force_model_center else 5  # Flux doesn't count as param.

        self.kwargs = {
            'fastfit' : fastfit,
            'force_model_center' : force_model_center
        }

    @staticmethod
    def hsmfit(star):
        """Compute the hsm moments for a given star.

        :param star:    A Star instance

        :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
        """
        import galsim
        image, weight, image_pos = star.data.getImage()
        mom = image.FindAdaptiveMom(weight=weight, strict=False)

        fwhm = mom.moments_sigma / 0.4519  # Magic number to convert sigma -> FWHM for a Kolmogorov
        shape = mom.observed_shape
        # These are in pixel coordinates.  Need to convert to world coords.
        jac = image.wcs.jacobian(image_pos=image_pos)
        scale, shear, theta, flip = jac.getDecomposition()
        # Fix sigma
        fwhm *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
        # Next the rotation
        shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
        # Finally the shear
        shape = shear + shape

        # Another magic number below to convert flux -> actual flux for Kolmogorov.
        flux = mom.moments_amp / 0.9053

        # center = image.wcs.toWorld(mom.moments_centroid - image_pos)
        center = image.wcs.toWorld(mom.moments_centroid) - image.wcs.toWorld(image_pos)
        flag = mom.moments_status

        return flux, center.x, center.y, fwhm, shape.g1, shape.g2, flag

    @staticmethod
    def _lmfit_resid(params, star):
        """Residual function to use with lmfit.  Essentially `chi` from `chisq`, but not summed
        over pixels yet.

        :param params:  An lmfit.Parameters() instance
        :param star:    A Star instance

        :returns: `chi` as a flattened numpy array
        """
        import galsim
        image, weight, image_pos = star.data.getImage()
        prof = galsim.Kolmogorov(fwhm=params['fwhm'].value)
        prof *= params['flux'].value
        prof = prof.shear(g1=params['g1'].value, g2=params['g2'].value)
        prof = prof.shift(params['cenu'].value, params['cenv'].value)
        #print(prof)
        model_image = image.copy()
        prof.drawImage(model_image, method='no_pixel',
                       offset=(image_pos - model_image.trueCenter()))
        return (weight.array*(model_image.array - image.array)).ravel()

    def _lmfit_params(self, star, vary_params=True, vary_flux=True, vary_center=True):
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
            flux, cenu, cenv, fwhm, g1, g2, flag = self.hsmfit(star)
            if flag != 0:
                raise RuntimeError("Error initializing star fit values using hsm.")
        else:
            flux = star.fit.flux
            if self._force_model_center:
                cenu, cenv = star.fit.center
                fwhm, g1, g2 = star.fit.params
            else:
                cenu, cenv, fwhm, g1, g2 = star.fit.params

        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=flux, vary=vary_flux)
        params.add('cenu', value=cenu, vary=vary_center)
        params.add('cenv', value=cenv, vary=vary_center)
        params.add('fwhm', value=fwhm, vary=vary_params)
        params.add('g1', value=g1, vary=vary_params)
        params.add('g2', value=g2, vary=vary_params)
        return params

    @classmethod
    def _lmfit_minimize(cls, params, star, logger=None):
        import lmfit
        if logger:
            import time
            t0 = time.time()
            logger.debug("Start lmfit minimize.")
        results = lmfit.minimize(cls._lmfit_resid, params, args=(star,))
        if logger:
            logger.debug("End lmfit minimize.  Elapsed time: {0}".format(time.time() - t0))
        return results

    def lmfit(self, star, logger=None):
        """Fit parameters of the given star using lmfit.

        :param star:    A Star instance
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: (flux, cenx, ceny, sigma, g1, g2, flag)
        """
        params = self._lmfit_params(star)
        results = self._lmfit_minimize(params, star, logger=logger)
        if logger:
            import lmfit
            logger.debug(lmfit.fit_report(results))
        return results.params.valuesdict().values() + [0 if results.success else 1]

    def fit(self, star, fastfit=False, logger=None):
        """Fit the image by ...

        :param star:    A Star instance
        :param fastfit: Use HSM moments to fit [default: False]
        :param logger:  A logger object for logging debug info. [default: None]

        :returns: a new Star with the fitted parameters in star.fit
        """
        if self._fastfit or fastfit:
            flux, cenu, cenv, fwhm, g1, g2, flag = self.hsmfit(star)
        else:
            flux, cenu, cenv, fwhm, g1, g2, flag = self.lmfit(star, logger=logger)

        # Make a StarFit object with these parameters
        if self._force_model_center:
            params = np.array([ fwhm, g1, g2 ])
            center = (cenu, cenv)
        else:
            params = np.array([ cenu, cenv, fwhm, g1, g2 ])
            center = (0.0, 0.0)

        # Also need to compute chisq
        prof = self.getProfile(params) * flux
        model_image = star.image.copy()
        prof.shift(center).drawImage(model_image, method='no_pixel',
                                     offset=(star.image_pos - model_image.trueCenter()))
        chisq = np.sum(star.weight.array * (star.image.array - model_image.array)**2)
        dof = np.count_nonzero(star.weight.array) - self._nparams

        fit = StarFit(params, flux=flux, center=center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with either [ fwhm, g1, g2 ]
                            or  [ cenu, cenv, fwhm, g1, g2 ]
                            depending on _force_model_center.

        :returns: a galsim.GSObject instance
        """
        import galsim
        if self._force_model_center:
            fwhm, g1, g2 = params
            cenu, cenv = (0.0, 0.0)
        else:
            cenu, cenv, fwhm, g1, g2 = params
        return galsim.Kolmogorov(fwhm=fwhm).shear(g1=g1, g2=g2).shift(cenu, cenv)

    def draw(self, star):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        prof = self.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        image = star.image.copy()
        prof.drawImage(image, method='no_pixel', offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, star.fit)

    def initialize(self, star, logger=None):
        if star.fit.params is None:
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
            results = self._lmfit_minimize(params, star, logger=logger)
            return Star(star.data, StarFit(star.fit.params,
                                           flux = results.params['flux'].value,
                                           center = (results.params['cenu'].value,
                                                     results.params['cenv'].value),
                                           chisq = results.chisqr,
                                           dof = np.count_nonzero(star.data.weight.array) - 3,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))
        else:
            image, weight, image_pos = star.data.getImage()
            model_image = self.draw(star).image
            new_flux = (np.sum(weight.array * image.array * model_image.array)
                        / np.sum(weight.array * model_image.array**2))
            new_chisq = np.sum(weight.array * (image.array - new_flux*model_image.array)**2)
            return Star(star.data, StarFit(star.fit.params,
                                           flux = new_flux,
                                           center = star.fit.center,
                                           chisq = new_chisq,
                                           dof = np.count_nonzero(weight.array) - 1,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))
