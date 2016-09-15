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
    """An extremely simple PSF model that just considers the PSF as a sheared Kolmogorov.
    """
    def __init__(self, logger=None, fastfit=True):
        self._fastfit = fastfit
        self.kwargs = {}

    @staticmethod
    def hsm(star):
        """Compute the hsm moments for a given star.

        :param star:    A Star instance

        :returns: (flux, cenx, ceny, sigma, g1, g2, flag)
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
        center = mom.moments_centroid
        flag = mom.moments_status

        return (flux, center.x, center.y, fwhm, shape.g1, shape.g2, flag)

    @staticmethod
    def lmfit_resid(params, star):
        """Residual function to use with lmfit.  Essentially `chi` from `chisq`.
        :param params:  An lmfit.Parameters() instance
        :param star:    A Star instance

        :returns: (flux, cenx, ceny, fwhm, g1, g2, flag)
        """
        import galsim
        image, weight, image_pos = star.data.getImage()
        prof = galsim.Kolmogorov(fwhm=params['fwhm'].value)
        prof *= params['flux'].value
        prof = prof.shear(g1=params['g1'].value, g2=params['g2'].value)
        # cenx and ceny are in image coords; need to convert
        prof = prof.shift(
            image.wcs.toWorld(
                galsim.PositionD(
                    params['cenx'].value - image.trueCenter().x,
                    params['ceny'].value - image.trueCenter().y)))
        model = prof.drawImage(image=image.copy(), method='no_pixel')
        return (weight.array*(model.array - image.array)).ravel()

    @staticmethod
    def lmfit_params(flux, cenx, ceny, fwhm, g1, g2,
                     vary_params=True, vary_flux=True, vary_center=True):
        """Generate an lmfit.Parameters() instance from arguments.

        :param flux:
        :param cenx:
        :param ceny:
        :param fwhm:
        :param g1:
        :param g2:
        :param vary_params:  Allow non-flux and non-center params to vary?
        :param vary_flux:  Allow flux to vary?
        :param vary_center:  Allow center to vary?

        :returns: lmfit.Parameters() instance.
        """
        import lmfit
        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=flux, vary=vary_flux)
        params.add('cenx', value=cenx, vary=vary_center)
        params.add('ceny', value=ceny, vary=vary_center)
        params.add('fwhm', value=fwhm, vary=vary_params)
        params.add('g1', value=g1, vary=vary_params)
        params.add('g2', value=g2, vary=vary_params)
        return params

    @classmethod
    def lmfit(cls, star):
        """Fit parameters of the given star using lmfit.

        :param star:    A Star instance

        :returns: (flux, cenx, ceny, sigma, g1, g2, flag)
        """
        import lmfit

        flux, cenx, ceny, fwhm, g1, g2, flag = Kolmogorov.hsm(star)
        params = cls.lmfit_params(flux, cenx, ceny, fwhm, g1, g2)

        import time
        t0 = time.time()
        print("Start lmfit.")
        results = lmfit.minimize(cls.lmfit_resid, params, args=(star,))
        print("End lmfit.  {0} sec elapsed.".format(time.time()-t0))
        return results.params.valuesdict().values() + [0 if results.success else 1]

    def initialize(self, star, mask=True, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param mask:    If True, set data.weight to zero at pixels that are outside
                        the range of the model. [default: True]
        :param logger:  A logger object for logging debug info. [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        # If implemented, update the flux to something close to right.
        if hasattr(self, 'reflux'):
            star = self.reflux(star, fit_center=False, logger=logger)
        else:
            star = star.withFlux(np.sum(star.data.image.array))

        # stars need to have initial fit params
        return self.fit(star)

    def fit(self, star, logger=None):
        """Fit the image by ...

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        import galsim

        if self._fastfit:
            flux, cenx, ceny, fwhm, g1, g2, flag = self.hsm(star)
        else:
            flux, cenx, ceny, fwhm, g1, g2, flag = self.lmfit(star, logger=logger)

        # Make a StarFit object with these parameters
        params = np.array([ fwhm, g1, g2 ])

        # Also need to compute chisq
        prof = self.getProfile(params) * flux
        center = galsim.PositionD(cenx, ceny)
        offset = center - star.image.trueCenter()
        model_image = prof.drawImage(star.image.copy(), method='no_pixel', offset=offset)
        chisq = np.std(star.image.array - model_image.array)
        dof = np.count_nonzero(star.weight.array) - 6

        fit = StarFit(params, flux=flux, center=center-star.image_pos, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with [ fwhm, g1, g2 ]

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = galsim.Kolmogorov(fwhm=params[0])
        prof = prof.shear(g1=params[1], g2=params[2])
        return prof

    def draw(self, star):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        import galsim
        prof = self.getProfile(star.fit.params)
        center = galsim.PositionD(*star.fit.center)
        offset = star.image_pos + center - star.image.trueCenter()
        image = prof.drawImage(star.image.copy(), method='no_pixel', offset=offset)
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, star.fit)

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

        if fit_center:
            import lmfit

            params = lmfit.Parameters()
            # Order is important for params!
            params.add('flux', value=star.fit.flux)
            params.add('cenx', value=star.fit.center[0])
            params.add('ceny', value=star.fit.center[1])
            params.add('fwhm', value=star.fit.params[0], vary=False)
            params.add('g1', value=star.fit.params[1], vary=False)
            params.add('g2', value=star.fit.params[2], vary=False)
            import time
            t0 = time.time()
            if logger:
                logger.debug("Start lmfit.")
            results = lmfit.minimize(self.lmfit_resid, params, args=(star,))
            if logger:
                logger.debug("End lmfit.  {0} sec elapsed.".format(time.time()-t0))
            return Star(star.data, StarFit(star.fit.params,
                                           flux = results.params['flux'].value,
                                           center = np.array([results.params['cenx'].value,
                                                              results.params['ceny'].value]),
                                           chisq = results.chisq,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))
        else:
            image, weight, image_pos = star.data.getImage()
            model_image = self.draw(star).image
            flux = np.sum(weight*image*model_image) / np.sum(weight*model_image*model_image)
            chisq = np.sum(weight * (image - flux*model_image)**2)
            return Star(star.data, StarFit(star.fit.params,
                                           flux = flux,
                                           center = star.fit.center,
                                           chisq = chisq,
                                           alpha = star.fit.alpha,
                                           beta = star.fit.beta))
