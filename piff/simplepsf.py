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

    Use type name "Simple" in a config field to use this psf type, or leave off the type
    name, as this is the default PSF type.

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
            # Use 0 here for things that will get overwritten in _finish_read.
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

        # Run this by default on construction.
        # If this is a component, it will be overwritten by a higher level composite class.
        self.set_num(None)

    def set_num(self, num):
        """If there are multiple components involved in the fit, set the number to use
        for this model.
        """
        from .model import Model
        from .interp import Interp
        self._num = num
        # Note: they might be 0 if this is part of a read process, and they haven't been
        # overwritten yet.
        if isinstance(self.model, Model):
            self.model.set_num(num)
        if isinstance(self.interp, Interp):
            self.interp.set_num(num)

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

    def initialize_params(self, stars, logger=None, default_init=None):
        nremoved = 0

        logger.debug("Initializing models")
        # model.initialize may fail
        new_stars = []
        for star in stars:
            try:
                star = self.model.initialize(star, logger=logger, default_init=default_init)
            except Exception as e:
                logger.warning("Failed initializing star at %s. Excluding it.", star.image_pos)
                logger.warning("  -- Caught exception: %s",e)
                nremoved += 1
                star = star.flag_if(True)
            new_stars.append(star)
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

    def single_iteration(self, stars, logger, convert_funcs, draw_method):

        # Perform the fit or compute design matrix as appropriate using just non-reserve stars
        fit_fn = self.model.chisq if self.quadratic_chisq else self.model.fit

        nremoved = 0  # For this iteration
        use_stars = []  # Just the stars we want to use for fitting.
        all_stars = []  # All the stars (with appropriate flags as necessary)
        for k, star in enumerate(stars):
            if not star.is_flagged and not star.is_reserve:
                try:
                    convert_func = None if convert_funcs is None else convert_funcs[k]
                    star = fit_fn(star, logger=logger, convert_func=convert_func,
                                  draw_method=draw_method)
                    use_stars.append(star)
                except Exception as e:
                    logger.warning("Failed fitting star at %s.", star.image_pos)
                    logger.warning("Excluding it from this iteration.")
                    logger.warning("  -- Caught exception: %s", e)
                    nremoved += 1
                    star = star.flag_if(True)
            all_stars.append(star)

        if len(use_stars) == 0:
            raise RuntimeError("No stars left to fit.  Cannot find PSF model.")

        # Perform the interpolation, again using just non-reserve stars
        logger.debug("             Calculating the interpolation")
        self.interp.solve(use_stars, logger=logger)

        # Propagate that solution to all the stars' parameters, including reserve stars.
        all_stars = self.interp.interpolateList(all_stars)

        return all_stars, nremoved

    @property
    def fit_center(self):
        """Whether to fit the center of the star in reflux.

        This is generally set in the model specifications.
        If all component models includes a shift, then this is False.
        Otherwise it is True.
        """
        return self.model._centered

    @property
    def include_model_centroid(self):
        """Whether a model that we want to center can have a non-zero centroid during iterations.
        """
        return self.model._centered and self.model._model_can_be_offset

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

    def _drawStar(self, star):
        return self.model.draw(star)

    def _getRawProfile(self, star):
        return self.model.getProfile(star.fit.get_params(self._num)), self.model._method

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
