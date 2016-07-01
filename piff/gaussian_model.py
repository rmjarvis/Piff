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
.. module:: model_gaussian
"""

from __future__ import print_function
import numpy as np

from .model import Model
from .star import Star, StarFit, StarData

class Gaussian(Model):
    """An extremely simple PSF model that just considers the PSF as a sheared Gaussian.
    """
    def __init__(self, logger=None):
        self.kwargs = {}

    def fit(self, star):
        """Fit the image by running the HSM adaptive moments code on the image and using
        the resulting moments as an estimate of the Gaussian size/shape.

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        import galsim
        image, weight, image_pos = star.data.getImage()
        mom = image.FindAdaptiveMom(weight=weight)

        sigma = mom.moments_sigma
        shape = mom.observed_shape
        # These are in pixel coordinates.  Need to convert to world coords.
        jac = image.wcs.jacobian(image_pos=image_pos)
        scale, shear, theta, flip = jac.getDecomposition()
        # Fix sigma
        sigma *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
        # Next the rotation
        shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
        # Finally the shear
        shape = shear + shape

        # Make a StarFit object with these parameters
        params = np.array([ sigma, shape.g1, shape.g2 ])

        flux = mom.moments_amp
        center = mom.moments_centroid

        # Also need to compute chisq
        prof = self.getProfile(params) * flux
        offset = center - image.trueCenter()
        model_image = prof.drawImage(image.copy(), method='no_pixel', offset=offset)
        chisq = np.std(image.array - model_image.array)
        dof = np.count_nonzero(weight.array) - 6

        fit = StarFit(params, flux=flux, center=center-image_pos, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with [ sigma, g1, g2 ]

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = galsim.Gaussian(sigma=params[0])
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
        offset = star.data.image_pos + center - star.data.image.trueCenter()
        image = prof.drawImage(star.data.image.copy(), method='no_pixel', offset=offset)
        data = StarData(image, star.data.image_pos, star.data.weight)
        return Star(data, star.fit)
