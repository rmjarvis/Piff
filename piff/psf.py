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
.. module:: psf
"""

from __future__ import print_function

class PSF(object):
    """A class that encapsulates the full interpolated PSF model.

    The usual way to create a PSF is through one of the two factory functions::

        >>> psf = piff.PSF.build(images=images, pos=pos, model=model, interp=interp, ...)
        >>> psf = piff.PSF.read(file_name=file_name, ...)

    The first is used to build a PSF model from the data.
    The second is used to read in a PSF model from disk.

    However, it is also possible to construct a PSF from a model instance and an interp instance.
    Any existing parameters in the model instance are ignored.  The model just determines how
    interpolated parameters are to be interpreted.
    """
    def __init__(self, model, interp):
        self.model = model
        self.interp = interp

    @classmethod
    def build(cls, stars, model, interp, logger=None):
        """The main driver function to build a PSF model from data.

        :param stars:       A list of StarData instances.
        :param model:       A Model instance that defines how to model the individual PSFs
                            at the location of each star.
        :param interp:      An Interp instance that defines how to do the interpolation of the
                            data vectors (produced by model for each star).
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        if logger:
            logger.info("Start building PSF using %s stars", len(stars))
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)

        # Fit the model to each star
        if logger:
            logger.debug("Making model vectors")
        vectors = [ model.fitStar(star).getParameters() for star in stars ]

        # Get the "positions" to use for interpolation.
        # (The "position" may include non-spatial information like color, but we use the position
        # nomenclature throughout to refer to the position in some arbitrary-dimensional space.)
        if logger:
            logger.debug("Getting star positions")
        pos = [ interp.getStarPosition(star) for star in stars ]

        # Use the interpolator to fit the parameter vectors
        if logger:
            logger.debug("Performing interpolation")
        interp.solve(pos, vectors)

        # Return this as a PSF instance
        if logger:
            logger.debug("Done building PSF")
        return cls(model, interp)

    def write(self, file_name, logger=None):
        """Write a PSF object to a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import fitsio
        if logger:
            logger.info("Writing PSF to file %s",file_name)

        with fitsio.FITS(file_name,'rw',clobber=True) as f:
            self.model.write(f, extname='model')
            self.interp.write(f, extname='interp')

    @classmethod
    def read(cls, file_name, logger=None):
        """Read a PSF object from a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        import fitsio
        import piff
        if logger:
            logger.info("Reading PSF from file %s",file_name)

        with fitsio.FITS(file_name,'r') as f:
            if logger:
                logger.debug('opened FITS file')
            model = piff.Model.read(f, 'model')
            if logger:
                logger.debug("model = %s",model)
            interp = piff.Interp.read(f, 'interp')
            if logger:
                logger.debug("interp = %s",interp)
        return cls(model,interp)

    def drawImage(self, position, offset=None, logger=None, **kwargs):
        """Generates star given position and galsim Image

        :param position:    Position we are interpolating to.
        :param offset:      Tuple; sett the nominal center of the image
                            relative to actual center of image [Default: None]
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim image
        """
        import galsim
        image = galsim.Image(32, 32)

        # interpolate params
        interpolated_params = self.interp.interpolate(position)
        # set the parameters on the psf model, which gives a new model instance
        interpolated_model = self.model.setParameters(interpolated_params)
        # pass offset pos
        if offset is not None:
            pos = galsim.PositionD(*offset) + image.trueCenter()
        else:
            pos = None
        # draw on the galsim image container
        interpolated_image = interpolated_model.drawImage(image, pos=pos)

        return interpolated_image
