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

from __future__ import print_function

import galsim
import fitsio
import numpy

from .psf import PSF
from .interp import Interp
from .model import Model
from .stardata import StarData
from .starfit import Star, StarFit
from .gaussian_model import Gaussian

class HyperPSF(PSF):
    """A class that can fit the hyperparameters associated with the models and interpolations based on moments.

    The usual way to create a PSF is through one of the two factory functions::

        >>> hyper = piff.HyperPSF.build(images=images, pos=pos, model=model, interp=interp, ...)
        >>> hyper = piff.HyperPSF.read(file_name=file_name, ...)

    The first is used to build an HyperPSF model from the data.
    The second is used to read in an HyperPSF model from disk.

    NOTE: not sure about where the optics drawing goes. probably model? so hsm goes
    in a separate place to fit stuff
    """

    def __init__(self, model, interp):
        """

        :param model:                       A Model instance that defines how to model the individual PSFs
                                            at the location of each star.
        :param interp:                      An Interp instance that defines how to do the interpolation of the
                                            data vectors (produced by model for each star).
        """
        self.model = model
        self.interp = interp

    @classmethod
    def build(cls, stars, model, interp, model_keys, interp_keys, model_comparer=Gaussian(), model_comparer_weights=None, fit_kwargs={}, logger=None):
        """The main driver function to build an HyperPSF model from data.

        :param stars:                       A list of StarData instances.
        :param model:                       A Model instance that defines how to model the individual PSFs
                                            at the location of each star.
        :param interp:                      An Interp instance that defines how to do the interpolation of the
                                            data vectors (produced by model for each star).
        :param model_keys, interp_keys:     A list that gives the attributes of model/interp that the params correspond to
        :param model_comparer:              Model that gets applied to both stars and model_stars to compare success of model
        :param model_comparer_weights:      Weights to apply to the params from model_comparer
        :param fit_kwargs:                  kwargs to pass to iminuit fitter
        :param logger:                      A logger object for logging debug info. [default: None]

        :returns: an HyperPSF instance

        http://iminuit.readthedocs.io/en/latest/api.html
        """
        if logger:
            logger.info("Start building HyperPSF using %s stars", len(stars))
            logger.debug("Model Comparer is %s", model_comparer)
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)
            logger.debug("Model params are %s", model_keys)
            logger.debug("Interp params are %s", interp_keys)

        psf = cls(model, interp)
        if logger:
            logger.info("Fitting PSF model")
        psf.fit(stars, model_keys, interp_keys, model_comparer, model_comparer_weights, logger=logger, **fit_kwargs)
        if logger:
            logger.debug("Done building PSF")
        return psf

    def fit(self, stars, model_keys, interp_keys, model_comparer, logger=None, **kwargs):
        """Fit the model!
        :param stars:                       A list of StarData instances.
        :param model:                       A Model instance that defines how to model the individual PSFs
                                            at the location of each star.
        :param interp:                      An Interp instance that defines how to do the interpolation of the
                                            data vectors (produced by model for each star).
        :param model_keys, interp_keys:     A list that gives the attributes of model/interp that the params correspond to
        :param model_comparer:              Model that gets applied to both stars and model_stars to compare success of model
        :param model_comparer_weights:      Weights to apply to the params from model_comparer
        :param logger:                      A logger object for logging debug info. [default: None]
        :param kwargs:                  kwargs to pass to iminuit fitter
        """

        # evaluate the stars with model_comparer
        stars_evaluated = [model_comparer.fit(star) for star in stars]

        # set up iminuit object

    def _fit_func(self, params, stars, interp_keys, model_keys, model_comparer, model_comparer_weights, logger=None):
        # update model and interp via params
        model_params = params[:len(model_keys)]
        model = {model_keys[i]: model_params[i]
                 for i in xrange(len(model_keys))}
        self.model.update(logger=logger, **model)

        interp_params = params[len(model_keys):]
        interp = {interp_keys[i]: interp_params[i]
                 for i in xrange(len(interp_keys))}
        self.interp.update(logger=logger, **interp)

        # interp
        stars_interp = self.interp.interpolateList(stars, logger=logger)

        # model
        stars_model = [self.model.draw(star) for star in stars_interp]

        # model_compare
        stars_compare = [model_comparer(star) for star in stars_model]

        # now compare the params of stars and stars_compare
        chi = numpy.array([stars_compare.fit.params - stars.fit.params])
        chisq_vec = numpy.sum(chi ** 2, axis=1)
        chisq = numpy.sum(chisq_vec * model_comparer_weights)
        return chisq

    def draw():
        pass
