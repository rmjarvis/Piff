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

from .model import Model
import numpy
class Gaussian(Model):
    """An extremely simple PSF model that just considers the PSF as a sheared Gaussian.
    """
    def __init__(self):
        pass

    def fitImage(self, image, weight=None):
        """Fit the image by running the HSM adaptive moments code on the image and using
        the resulting moments as an estimate of the Gaussian size/shape.
        """
        import galsim
        mom = image.FindAdaptiveMom(weight=weight)
        self.sigma = mom.moments_sigma
        self.shape = mom.observed_shape
        # These are in pixel coordinates.  Need to convert to world coords.
        self.jac = image.wcs.jacobian(image_pos=image.center())
        scale, shear, theta, flip = self.jac.getDecomposition()
        # Fix sigma
        self.sigma *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            self.shape = galsim.Shear(g1 = -self.shape.g1, g2 = self.shape.g2)
        # Next the rotation
        self.shape = galsim.Shear(g = self.shape.g, beta = self.shape.beta + theta)
        # Finally the shear
        self.shape = shear + self.shape

    def getProfile(self):
        """Get a version of the PSF model as a GalSim GSObject

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = galsim.Gaussian(sigma=self.sigma).shear(self.shape)
        return prof
    
    def getParameters(self):
        return numpy.array([self.sigma, self.shape.g1, self.shape.g2])
