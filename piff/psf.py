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

import numpy
import fitsio

from .star import Star, StarFit, StarData
from .model import Model
from .interp import Interp

class PSF(object):
    """A class that encapsulates the full interpolated PSF model.

    The usual way to create a PSF is through one of the two factory functions::

        >>> psf = piff.PSF.build(stars, wcs, model=model, interp=interp, ...)
        >>> psf = piff.PSF.read(file_name=file_name, ...)

    The first is used to build a PSF model from the data.
    The second is used to read in a PSF model from disk.

    However, it is also possible to construct a PSF directly from a Model instance and an Interp
    instance.  The model defines the functional form of the surface brightness profile, and the
    interp defines how the parameters of the mdoel vary across the field of view.

    The stars member is a list holding the Star instances for all measurement points being fit.
    At the end of a fit, it contains the PSF parameters at each star and information on what
    was clipped and how good the fit is.
    """
    def __init__(self, stars, wcs, model, interp, extra_interp_properties=()):
        self.stars = stars
        self.wcs = wcs
        self.model = model
        self.interp = interp
        self.extra_interp_properties = extra_interp_properties

    @classmethod
    def build(cls, stars, wcs, model, interp, logger=None):
        """The main driver function to build a PSF model from data.

        :param stars:       A list of Star instances.
        :param wcs:         A dict of WCS solutions indexed by chipnum.
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

        psf = cls(stars, wcs, model, interp)
        if logger:
            logger.info("Fitting PSF model")
        psf.fit(logger=logger)
        if logger:
            logger.debug("Done building PSF")
        return psf

    def fit(self, chisq_threshold=0.1, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param chisq_threshold:  Change in reduced chisq at which iteration will terminate.
                            [default: 0.1]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug("Initializing models")
        self.stars = [self.model.initialize(s, mask=True) for s in self.stars]

        if logger:
            logger.debug("Initializing interpolator")
        self.stars = self.interp.initialize(self.stars, logger=logger)

        # Begin iterations.  Very simple convergence criterion right now.
        # ??? Also will need to include outlier rejection here.
        # ??? Make these convergence constants program options???
        max_iterations = 30

        # For basis models, we can compute a quadratic form for chisq, and if we are using
        # a basis interpolator, then we can use it.  It's kind of ugly to query this, but
        # the double dispatch makes it tricky to implement this with class heirarchy, so for
        # now we just check if we have all the required parts to use the quadratic form
        quadratic_chisq = hasattr(self.model, 'chisq') and self.interp.degenerate_points

        oldchisq = 0.
        for iteration in range(max_iterations):
            if logger:
                logger.debug("Fitting stars, iteration %d", iteration)

            if quadratic_chisq:
                self.stars = [self.model.chisq(s) for s in self.stars]
            else:
                self.stars = [self.model.fit(s) for s in self.stars]

            if logger:
                logger.debug("Interpolator solving, iteration %d", iteration)
            self.interp.solve(self.stars, logger=logger)

            # Refit and recenter all stars, collect stats
            if logger:
                logger.debug("Re-fluxing stars, iteration %d", iteration)

            if hasattr(self.model, 'reflux'):
                self.stars = [self.model.reflux(self.interp.interpolate(s)) for s in self.stars]
            chisq = numpy.sum([s.fit.chisq for s in self.stars])
            dof   = numpy.sum([s.fit.dof for s in self.stars])
            if logger:
                logger.info('Iteration {:d} chisq= {:.2f} / {:d} dof'.format(iteration,
                                                                             chisq,dof))

            # ??? This is where we should do some outlier rejection.

            # ??? Very simple convergence test here:
            # Note, the lack of abs here means if chisq increases, we also stop.
            if oldchisq>0 and oldchisq-chisq < chisq_threshold*dof:
                break
            if logger and iteration+1 >= max_iterations:
                logger.warning('PSF fit did not converge')
            oldchisq = chisq

    def draw(self, x, y, chipnum=0, flux=1.0, offset=(0,0), stamp_size=48,
             logger=None, **kwargs):
        """Generates PSF image at a given location.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, stamp_size=48)

        However, if the PSF interpolation used extra properties for the interpolation
        (cf. psf.extra_interp_properties), you need to provide them as additional kwargs.

            >>> print(psf.extra_interp_properties)
            ('ri_color',)
            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, ri_color=0.23, stamp_size=48)

        :param x:           The image x position.
        :param y:           The image y position.
        :param chipnum:     Which chip to use for WCS information. [default: 0, which is
                            appropriate if only using a single chip]
        :param flux:        Flux of PSF to be drawn [default: 1.0]
        :param offset:      (dx,dy) tuple giving offset of stellar center relative
                            to star.data.image_pos [default: (0,0)]
        :param logger:      A logger object for logging debug info. [default: None]
        :param **kwargs:    Additional properties required for the interpolation.

        :returns:           A GalSim Image of the PSF
        """
        wcs = self.wcs[chipnum]
        properties = {}
        for key in self.extra_interp_properties:
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required"%key)
            properties = kwags.pop(key)
        if len(kwargs) != 0:
            raise TypeError("draw got an unexpecte keyword argument %r"%kwargs.keys()[0])

        star = Star.makeTarget(x=x, y=y, wcs=wcs, properties=properties,
                               stamp_size=stamp_size)
        if logger:
            logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        star = self.drawStar(star, flux=flux, offset=offset)
        return star.data.image

    def drawStar(self, star, flux=None, offset=(0,0)):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param flux:        Flux of PSF to be drawn [default: None, which will use the
                            existing flux value in the star object.
        :param offset:      (dx,dy) tuple giving offset of stellar center relative
                            to the center of the image [default: (0,0)]

        :returns:           Star instance with its image filled with rendered PSF
        """
        import galsim
        # The model is in sky coordinates, so figure out what (du,dv) corresponds to this offset.
        jac = star.data.image.wcs.jacobian(image_pos=galsim.PositionD(star.data.image.trueCenter()))
        dx, dy = offset
        du = jac.dudx * dx + jac.dudy * dy
        dv = jac.dvdx * dx + jac.dvdy * dy
        u = star.fit.center[0] + du
        v = star.fit.center[1] + dv
        # Update the flux, center.
        star = star.withFlux(flux=flux, center=(u,v))
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        # Render the image
        return self.model.draw(star)

    def write(self, file_name, logger=None):
        """Write a PSF object to a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        """
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
        if logger:
            logger.info("Reading PSF from file %s",file_name)

        with fitsio.FITS(file_name,'r') as f:
            if logger:
                logger.debug('opened FITS file')
            model = Model.read(f, 'model')
            if logger:
                logger.debug("model = %s",model)
            interp = Interp.read(f, 'interp')
            if logger:
                logger.debug("interp = %s",interp)
        return cls(None,None,model,interp)

