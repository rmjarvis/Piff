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
from .outliers import Outliers
from .psf import PSF

class SimplePSF(PSF):
    """A PSF class that uses a single model and interpolator.

    A SimplePSF is built from a Model and an Interp object.
    The model defines the functional form of the surface brightness profile, and the
    interpolator defines how the parameters of the model vary across the field of view.
    """
    def __init__(self, stars, wcs, pointing, model, interp,
                 outliers=None, extra_interp_properties=()):
        """
        :param stars:       A list of Star instances.
        :param wcs:         A dict of WCS solutions indexed by chipnum.
        :param pointing:    A galsim.CelestialCoord object giving the telescope pointing.
                            [Note: pointing should be None if the WCS is not a galsim.CelestialWCS]
        :param model:       A Model instance used for modeling the surface brightness profile.
        :param interp:      An Interp instance used to interpolate across the field of view.
        :param outliers:    Optionally, an Outliers instance used to remove outliers.
                            [default: None]
        """
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing
        self.model = model
        self.interp = interp
        self.outliers = outliers
        self.extra_interp_properties = extra_interp_properties
        self.kwargs = {
            # These are junk entries that will be overwritten.
            # TODO: Come up with a nicer mechanism for specifying items that can be overwritten
            #       in the _finish_read function.
            'model': 0,
            'interp': 0,
            'outliers': 0,
        }

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        import piff

        kwargs = {}
        kwargs.update(config_psf)

        for key in ['model', 'interp']:
            if key not in kwargs:
                raise ValueError("%s field is required in psf field for type=Simple"%key)

        # make a Model object to use for the individual stellar fitting
        model = piff.Model.process(kwargs.pop('model'), logger=logger)
        kwargs['model'] = model

        # make an Interp object to use for the interpolation
        interp = piff.Interp.process(kwargs.pop('interp'), logger=logger)
        kwargs['interp'] = interp

        if 'outliers' in kwargs:
            outliers = piff.Outliers.process(kwargs.pop('outliers'), logger=logger)
            kwargs['outliers'] = outliers

        return kwargs

    def fit(self, chisq_threshold=0.1, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param chisq_threshold: Change in reduced chisq at which iteration will terminate.
                                [default: 0.1]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug("Initializing models")
        self.stars = [self.model.initialize(s, mask=True) for s in self.stars]

        if logger:
            logger.debug("Initializing interpolator")
        self.stars = self.interp.initialize(self.stars, logger=logger)

        # Begin iterations.  Very simple convergence criterion right now.
        # TODO: Make these convergence constants configurable in config dict.
        max_iterations = 30

        # For basis models, we can compute a quadratic form for chisq, and if we are using
        # a basis interpolator, then we can use it.  It's kind of ugly to query this, but
        # the double dispatch makes it tricky to implement this with class heirarchy, so for
        # now we just check if we have all the required parts to use the quadratic form
        quadratic_chisq = hasattr(self.model, 'chisq') and self.interp.degenerate_points

        oldchisq = 0.
        nremoved = 0
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

            if self.outliers and (iteration > 0 or self.interp.degenerate_points):
                # Perform outlier rejection, but not on first iteration for degenerate solvers.
                self.stars, nremoved = self.outliers.removeOutliers(self.stars, logger=logger)
                if logger:
                    if nremoved == 0:
                        logger.debug("No outliers found")
                    else:
                        logger.debug("Removed %d outliers", nremoved)

            chisq = numpy.sum([s.fit.chisq for s in self.stars])
            dof   = numpy.sum([s.fit.dof for s in self.stars])
            if logger:
                logger.info('Iteration %d: chisq = %.2f / %d dof', iteration, chisq, dof)

            # Very simple convergence test here:
            # Note, the lack of abs here means if chisq increases, we also stop.
            # Also, don't quit if we removed any outliers.
            if (nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < chisq_threshold*dof):
                return
            oldchisq = chisq

        logger.warning('PSF fit did not converge')


    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        # Render the image
        return self.model.draw(star)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        self.model.write(fits, extname + '_model')
        if logger:
            logger.debug("Wrote the PSF model to extension %s",extname + '_model')
        self.interp.write(fits, extname + '_interp')
        if logger:
            logger.debug("Wrote the PSF interp to extension %s",extname + '_interp')
        if self.outliers:
            self.outliers.write(fits, extname + '_outliers')
            if logger:
                logger.debug("Wrote the PSF outliers to extension %s",extname + '_outliers')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        self.model = Model.read(fits, extname + '_model')
        self.interp = Interp.read(fits, extname + '_interp')
        if extname + '_outliers' in fits:
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            self.outliers = None

