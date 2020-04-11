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

import numpy as np
import galsim

from .model import Model, ModelFitError
from .interp import Interp
from .outliers import Outliers
from .psf import PSF
from .util import write_kwargs, read_kwargs

class SimplePSF(PSF):
    """A PSF class that uses a single model and interpolator.

    A SimplePSF is built from a Model and an Interp object.
    The model defines the functional form of the surface brightness profile, and the
    interpolator defines how the parameters of the model vary across the field of view.
    """
    def __init__(self, model, interp, outliers=None, extra_interp_properties=None,
                 chisq_thresh=0.1, max_iter=30):
        """
        :param model:       A Model instance used for modeling the surface brightness profile.
        :param interp:      An Interp instance used to interpolate across the field of view.
        :param outliers:    Optionally, an Outliers instance used to remove outliers.
                            [default: None]
        :param extra_interp_properties: A list of any extra properties that will be used for
                            the interpolation in addition to (u,v).
                            [default: None]
        :param chisq_thresh: Change in reduced chisq at which iteration will terminate.
                            [default: 0.1]
        :param max_iter:    Maximum number of iterations to try. [default: 30]
        """
        self.model = model
        self.interp = interp
        self.outliers = outliers
        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties
        self.chisq_thresh = chisq_thresh
        self.max_iter = max_iter
        self.kwargs = {
            # These are junk entries that will be overwritten.
            # TODO: Come up with a nicer mechanism for specifying items that can be overwritten
            #       in the _finish_read function.
            'model': 0,
            'interp': 0,
            'outliers': 0,
            'chisq_thresh': self.chisq_thresh,
            'max_iter': self.max_iter,
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
        kwargs.pop('type',None)

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

    def fit(self, stars, wcs, pointing, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing

        logger.debug("Initializing models")
        # model.initialize may fail
        nremoved = 0
        new_stars = []
        for s in self.stars:
            try:
                new_star = self.model.initialize(s, logger=logger)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed initializing star at %s. Excluding it.", s.image_pos)
                logger.warning("  -- Caught exception: %s",e)
                nremoved += 1
            else:
                new_stars.append(new_star)
        if nremoved == 0:
            logger.debug("No stars removed in initialize step")
        else:
            logger.info("Removed %d stars in initialize", nremoved)
        self.stars = new_stars

        logger.debug("Initializing interpolator")
        self.stars = self.interp.initialize(self.stars, logger=logger)

        # For basis models, we can compute a quadratic form for chisq, and if we are using
        # a basis interpolator, then we can use it.  It's kind of ugly to query this, but
        # the double dispatch makes it tricky to implement this with class heirarchy, so for
        # now we just check if we have all the required parts to use the quadratic form
        if hasattr(self.interp, 'degenerate_points'):
            quadratic_chisq = hasattr(self.model, 'chisq') and self.interp.degenerate_points
            degenerate_points = self.interp.degenerate_points
        else:
            quadratic_chisq = False
            degenerate_points = False

        # Begin iterations.  Very simple convergence criterion right now.
        oldchisq = 0.
        for iteration in range(self.max_iter):
            if len(self.stars) == 0:
                raise RuntimeError("No stars.  Cannot find PSF model.")

            logger.warning("Iteration %d: Fitting %d stars", iteration+1, len(self.stars))


            fit_fn = self.model.chisq if quadratic_chisq else self.model.fit

            nremoved = 0
            new_stars = []
            for s in self.stars:
                try:
                    new_star = fit_fn(s, logger=logger)
                except ModelFitError as e:  # pragma: no cover
                    logger.warning("Failed fitting star at %s.  Excluding it.", s.image_pos)
                    logger.warning("  -- Caught exception: %s", e)
                    nremoved += 1
                else:
                    new_stars.append(new_star)
            self.stars = new_stars

            logger.debug("             Calculating the interpolation")
            self.interp.solve(self.stars, logger=logger)

            # Refit and recenter all stars, collect stats
            logger.debug("             Re-fluxing stars")

            if hasattr(self.model, 'reflux'):
                new_stars = []
                signals = self.drawStarList(self.stars)
                signalized_stars = [s.addPoisson(signal) for s, signal in zip(self.stars, signals)]
                interpolated_stars = self.interp.interpolateList(signalized_stars)
                for s in interpolated_stars:
                    try:
                        new_star = self.model.reflux(s, logger=logger)
                    except Exception as e:  # pragma: no cover
                        logger.warning("Failed trying to reflux star at %s.  Excluding it.",
                                       s.image_pos)
                        logger.warning("  -- Caught exception: %s", e)
                        nremoved += 1
                    else:
                        new_stars.append(new_star)
                self.stars = new_stars

            if self.outliers and (iteration > 0 or not degenerate_points):
                # Perform outlier rejection, but not on first iteration for degenerate solvers.
                logger.debug("             Looking for outliers")
                self.stars, nremoved1 = self.outliers.removeOutliers(self.stars, logger=logger)
                if nremoved1 == 0:
                    logger.debug("             No outliers found")
                else:
                    logger.info("             Removed %d outliers", nremoved1)
                nremoved += nremoved1

            chisq = np.sum([s.fit.chisq for s in self.stars])
            dof   = np.sum([s.fit.dof for s in self.stars])
            logger.warning("             Total chisq = %.2f / %d dof", chisq, dof)

            # Save these so we can write them to the output file.
            self.chisq = chisq
            self.last_delta_chisq = oldchisq-chisq
            self.dof = dof
            self.nremoved = nremoved

            # Very simple convergence test here:
            # Note, the lack of abs here means if chisq increases, we also stop.
            # Also, don't quit if we removed any outliers.
            if (nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < self.chisq_thresh*dof):
                return
            oldchisq = chisq

        logger.warning("PSF fit did not converge.  Max iterations = %d reached.",self.max_iter)

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars. Takes advantage of
        interpolateList for significant speedup with some interpolators.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns:           List of Star instances with its image filled with
                            rendered PSF
        """
        stars_interpolated = self.interp.interpolateList(stars)
        stars_drawn = [self.model.draw(star, copy_image=copy_image) for star in stars_interpolated]
        return stars_drawn

    def drawStar(self, star, copy_image=True):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns:           Star instance with its image filled with rendered PSF
        """
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        self.model.normalize(star)
        # Render the image
        return self.model.draw(star, copy_image=copy_image)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        logger = galsim.config.LoggerWrapper(logger)
        if hasattr(self, 'chisq'):
            chisq_dict = {
                'chisq' : self.chisq,
                'last_delta_chisq' : self.last_delta_chisq,
                'dof' : self.dof,
                'nremoved' : self.nremoved,
            }
        else:
            chisq_dict = {}
        write_kwargs(fits, extname + '_chisq', chisq_dict)
        logger.debug("Wrote the chisq info to extension %s",extname + '_chisq')
        self.model.write(fits, extname + '_model')
        logger.debug("Wrote the PSF model to extension %s",extname + '_model')
        self.interp.write(fits, extname + '_interp')
        logger.debug("Wrote the PSF interp to extension %s",extname + '_interp')
        if self.outliers:
            self.outliers.write(fits, extname + '_outliers')
            logger.debug("Wrote the PSF outliers to extension %s",extname + '_outliers')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        chisq_dict = read_kwargs(fits, extname + '_chisq')
        for key in chisq_dict:
            setattr(self, key, chisq_dict[key])
        self.model = Model.read(fits, extname + '_model')
        self.interp = Interp.read(fits, extname + '_interp')
        if extname + '_outliers' in fits:
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            self.outliers = None
