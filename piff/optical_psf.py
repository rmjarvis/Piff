
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
.. module:: optical_psf
"""

from __future__ import print_function

from .psf import PSF
from .interp import Interp
from .model import Model

class Optical(PSF):
    """A class that covers the optical portion of the PSF.

    The usual way to create a PSF is through one of the two factory functions::

        >>> optical = piff.Optical.build(images=images, pos=pos, model=model, interp=interp, ...)
        >>> optical = piff.Optical.read(file_name=file_name, ...)

    The first is used to build an Optical PSF model from the data.
    The second is used to read in an Optical PSF model from disk.

    NOTE: not sure about where the optics drawing goes. probably model? so hsm goes
    in a separate place to fit stuff
    """

    def __init__(self, model, interp):
        self.model = model
        self.interp = interp

    @classmethod
    def build(cls, stars, model, interp, logger=None):
        """The main driver function to build an Optical PSF model from data.

        :param stars:       A list of StarData instances.
        :param model:       A Model instance that defines how to model the individual PSFs
                            at the location of each star.
        :param interp:      An Interp instance that defines how to do the interpolation of the
                            data vectors (produced by model for each star).
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: an Optical PSF instance
        """
        if logger:
            logger.info("Start building Optical PSF using %s stars", len(stars))
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)

        # reduce stars to positions and model outputs

        # load up base zernikes for each position

        # solve the interpolation model

        # Return this as a PSF instance
        if logger:
            logger.debug("Done building PSF")
        return cls(model, interp)

class Zernike(Interp):
    # a class that takes in positions and returns zernikes from reference to file
    # and interp-specific corrections

