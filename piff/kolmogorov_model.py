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
    def __init__(self, logger=None):
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

        fwhm = mom.moments_sigma /0.4519  # Magic number to convert sigma -> FWHM for a Kolmogorov
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

    def fit(self, star):
        """Fit the image by ...

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        import galsim

        flux, cenx, ceny, fwhm, g1, g2, flag = self.hsm(star)

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
