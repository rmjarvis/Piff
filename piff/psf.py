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

from .starfit import Star, StarFit

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

    The stars member is a list holding the Star instances for all measurement points being fit.
    At the end of a fit, it contains the PSF parameters at each star and information on what
    was clipped and how good the fit is.
    """
    def __init__(self, model, interp, stars=None):
        self.model = model
        self.interp = interp
        self.stars = stars

    @classmethod
    def build(cls, data, model, interp, logger=None):
        """The main driver function to build a PSF model from data.

        :param data:        A list of StarData instances.
        :param model:       A Model instance that defines how to model the individual PSFs
                            at the location of each star.
        :param interp:      An Interp instance that defines how to do the interpolation of the
                            data vectors (produced by model for each star).
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        if logger:
            logger.info("Start building PSF using %s stars", len(data))
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)

        psf = cls(model,interp)
        if logger:
            logger.info("Fitting PSF model")
        psf.fit(data, logger=logger)
        if logger:
            logger.debug("Done building PSF")
        return psf

    def fit(self, data, chisq_threshold=0.1, logger=None):
        """Fit interpolated PSF model to data using standard sequence of operations.

        :param data:     List of StarData instances holding images to be fit.
        :param chisq_threshold:  Change in reduced chisq at which iteration will terminate.
                         [default: 0.1]
        :param logger:   A logger object for logging debug info. [default: None]
        """

        import numpy

        if logger:
            logger.debug("Making Star structures")
        self.stars = [self.model.makeStar(s, mask=True) for s in data]  #?? mask optional?

        if logger:
            logger.debug("Initializing interpolator")
        self.interp.initialize(self.stars, logger=logger)

        # Before beginning iterative solutions, install the interpolator
        # state into the parameter vectors of all Stars
        self.stars = self.interp.interpolateList(self.stars)

        if hasattr(self.model, 'reflux'):
            if logger:
                logger.debug("Initializing fluxes")
            self.stars = [self.model.reflux(s, fit_center=False) for s in self.stars]

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

    def draw(self, data, flux=1., center=(0.,0.), logger=None):
        """Generates PSF image from interpolated model

        :param data:        StarData instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :param flux:        Flux of PSF to be drawn
        :param center:      (u,v) tuple giving position of stellar center relative
                            to data.image_pos [default: (0.,0.)]
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           Star instance with its image filled with rendered star
        """

        # Make a Star structure
        s = self.model.makeStar(data, flux=flux, center=center)
        # Interpolate parameters to this position/properties:
        s = self.interp.interpolate(s)
        # Render the image
        return self.model.draw(s)

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

