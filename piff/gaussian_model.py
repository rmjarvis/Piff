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
import numpy

from .model import Model

class Gaussian(Model):
    """An extremely simple PSF model that just considers the PSF as a sheared Gaussian.
    """
    def __init__(self):
        self.kwargs = {}


    def fitStar(self, star):
        """Fit the image by running the HSM adaptive moments code on the image and using
        the resulting moments as an estimate of the Gaussian size/shape.

        :param star:    A StarData instance

        :returns: self
        """
        import galsim
        image, weight, image_pos = star.getImage()
        mom = image.FindAdaptiveMom(weight=weight)
        self.sigma = mom.moments_sigma
        self.shape = mom.observed_shape
        # These are in pixel coordinates.  Need to convert to world coords.
        jac = image.wcs.jacobian(image_pos=image_pos)
        scale, shear, theta, flip = jac.getDecomposition()
        # Fix sigma
        self.sigma *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            self.shape = galsim.Shear(g1 = -self.shape.g1, g2 = self.shape.g2)
        # Next the rotation
        self.shape = galsim.Shear(g = self.shape.g, beta = self.shape.beta + theta)
        # Finally the shear
        self.shape = shear + self.shape

        return self

    def getProfile(self):
        """Get a version of the PSF model as a GalSim GSObject

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = galsim.Gaussian(sigma=self.sigma).shear(self.shape)
        return prof

    def drawImage(self, image, pos=None):
        """Draw the model on the given image.

        :param image:   A galsim.Image on which to draw the model.
        :param pos:     The position on the image at which to place the nominal center.
                        [default: None, which means to use the center of the image.]

        :returns: image
        """
        prof = self.getProfile()
        if pos is not None:
            offset = pos - image.trueCenter()
        else:
            offset = None
        return prof.drawImage(image, method='no_pixel', offset=offset)

    def getParameters(self):
        """Get the parameters of the model, to be used by the interpolator.

        :returns: a numpy array of the model parameters
        """
        return numpy.array([self.sigma, self.shape.g1, self.shape.g2])

    def setParameters(self, params):
        """Set the parameters of the model, typically provided by an interpolator.

        :param params:  A numpy array of the model parameters

        :returns: self
        """
        import galsim
        sigma, g1, g2 = params
        self.sigma = sigma
        self.shape = galsim.Shear(g1=g1,g2=g2)

        return self
