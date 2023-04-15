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

import numpy as np
import galsim

from .model import Model
from .interp import Interp
from .outliers import Outliers
from .psf import PSF
from .util import write_kwargs, read_kwargs

class SimplePSF(PSF):
    """A PSF class that uses a single model and interpolator.

    A SimplePSF is built from a Model and an Interp object.
    The model defines the functional form of the surface brightness profile, and the
    interpolator defines how the parameters of the model vary across the field of view.

    :param model:       A Model instance used for modeling the surface brightness profile.
    :param interp:      An Interp instance used to interpolate across the field of view.
    :param outliers:    Optionally, an Outliers instance used to remove outliers.
                        [default: None]
    :param chisq_thresh: Change in reduced chisq at which iteration will terminate.
                        [default: 0.1]
    :param max_iter:    Maximum number of iterations to try. [default: 30]
    """
    _type_name = 'Simple'

    def __init__(self, model, interp, outliers=None, chisq_thresh=0.1, max_iter=30):
        self.model = model
        self.interp = interp
        self.outliers = outliers
        self.chisq_thresh = chisq_thresh
        self.max_iter = max_iter
        self.kwargs = {
            # model and interp are junk entries that will be overwritten.
            # TODO: Come up with a nicer mechanism for specifying items that can be overwritten
            #       in the _finish_read function.
            'model': 0,
            'interp': 0,
            'outliers': 0,
            'chisq_thresh': self.chisq_thresh,
            'max_iter': self.max_iter,
        }
        self.chisq = 0.
        self.last_delta_chisq = 0.
        self.dof = 0
        self.nremoved = 0
        self.niter = 0

    @property
    def interp_property_names(self):
        return self.interp.property_names

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        from .model import Model
        from .interp import Interp
        from .outliers import Outliers

        kwargs = {}
        kwargs.update(config_psf)
        kwargs.pop('type',None)

        for key in ['model', 'interp']:
            if key not in kwargs:  # pragma: no cover
                # This actually is covered, but for some reason, codecov thinks it isn't.
                raise ValueError("%s field is required in psf field for type=Simple"%key)

        # make a Model object to use for the individual stellar fitting
        model = Model.process(kwargs.pop('model'), logger=logger)
        kwargs['model'] = model

        # make an Interp object to use for the interpolation
        interp = Interp.process(kwargs.pop('interp'), logger=logger)
        kwargs['interp'] = interp

        if 'outliers' in kwargs:
            outliers = Outliers.process(kwargs.pop('outliers'), logger=logger)
            kwargs['outliers'] = outliers

        return kwargs

    def initialize(self, stars, logger):
        nremoved = 0

        logger.debug("Initializing models")
        # model.initialize may fail
        new_stars = []
        for s in stars:
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

        logger.debug("Initializing interpolator")
        stars = self.interp.initialize(new_stars, logger=logger)

        # For basis models, we can compute a quadratic form for chisq, and if we are using
        # a basis interpolator, then we can use it.  It's kind of ugly to query this, but
        # the double dispatch makes it tricky to implement this with class heirarchy, so for
        # now we just check if we have all the required parts to use the quadratic form
        if hasattr(self.interp, 'degenerate_points'):
            self.quadratic_chisq = hasattr(self.model, 'chisq') and self.interp.degenerate_points
            self.degenerate_points = self.interp.degenerate_points
        else:
            self.quadratic_chisq = False
            self.degenerate_points = False

        return stars, nremoved

    def single_iteration(self, use_stars, all_stars, logger, convert_func):

        # Perform the fit or compute design matrix as appropriate using just non-reserve stars
        fit_fn = self.model.chisq if self.quadratic_chisq else self.model.fit

        nremoved = 0  # For this iteration
        new_use_stars = []
        for star in use_stars:
            try:
                star = fit_fn(star, logger=logger, convert_func=convert_func)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed fitting star at %s.", star.image_pos)
                logger.warning("Excluding it from this iteration.")
                logger.warning("  -- Caught exception: %s", e)
                nremoved += 1
            else:
                new_use_stars.append(star)
        use_stars = new_use_stars

        # Perform the interpolation, again using just non-reserve stars
        logger.debug("             Calculating the interpolation")
        self.interp.solve(use_stars, logger=logger)

        # Note: From here forward, we are back to using all_stars, rather than use_stars.
        # We want to run the interpolation on everything and refit/recenter everything,
        # so reserve stars may get outlier rejected as well as non-reserve stars.
        all_stars = self.interp.interpolateList(all_stars)

        # Update estimated poisson noise
        signals = self.drawStarList(all_stars)
        all_stars = [s.addPoisson(signal) for s, signal in zip(all_stars, signals)]

        # Refit and recenter all stars, collect stats
        logger.debug("             Re-fluxing stars")
        new_stars = []
        for s in all_stars:
            try:
                new_star = self.model.reflux(s, logger=logger)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed trying to reflux star at %s.  Excluding it.",
                                s.image_pos)
                logger.warning("  -- Caught exception: %s", e)
                nremoved += 1
            else:
                new_stars.append(new_star)
        return new_stars, nremoved

    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        stars = self.interp.interpolateList(stars)
        for star in stars:
            self.model.normalize(star)
        return stars

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        star = self.interp.interpolate(star)
        self.model.normalize(star)
        return star

    def _drawStar(self, star, copy_image=True, center=None):
        return self.model.draw(star, copy_image=copy_image, center=center)

    def _getProfile(self, star, copy_image=True, center=None):
        prof = self.model.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        return prof, self.model._method

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        logger = galsim.config.LoggerWrapper(logger)
        chisq_dict = {
            'chisq' : self.chisq,
            'last_delta_chisq' : self.last_delta_chisq,
            'dof' : self.dof,
            'nremoved' : self.nremoved,
            'niter' : self.niter,
        }
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
