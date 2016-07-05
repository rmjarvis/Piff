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

import numpy

from .psf import PSF

class HyperPSF(PSF):
    """A class that can fit the hyperparameters associated with the models and interpolations based on moments.

    The usual way to create a PSF is through one of the two factory functions::

        >>> hyper = piff.HyperPSF.build(images=images, pos=pos, model=model, interp=interp, ...)
        >>> hyper = piff.HyperPSF.read(file_name=file_name, ...)

    The first is used to build an HyperPSF model from the data.
    The second is used to read in an HyperPSF model from disk.
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

        self.n_iter = 0
        self._p_hist = []

    @classmethod
    def build(cls, stars, model, interp, logger=None, **fit_kwargs):
        """The main driver function to build an HyperPSF model from data.

        :param stars:                       A list of StarData instances.
        :param model:                       A Model instance that defines how to model the individual PSFs
                                            at the location of each star.
        :param interp:                      An Interp instance that defines how to do the interpolation of the
                                            data vectors (produced by model for each star).
        :param fit_kwargs:                  kwargs to pass to iminuit fitter
        :param logger:                      A logger object for logging debug info. [default: None]

        :returns: an HyperPSF instance

        http://iminuit.readthedocs.io/en/latest/api.html
        """
        if logger:
            logger.info("Start building HyperPSF using %s stars", len(stars))
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)

        psf = cls(model, interp)
        if logger:
            logger.info("Fitting PSF model")
        psf.fit(stars, logger=logger, **fit_kwargs)
        if logger:
            logger.debug("Done building PSF")
        return psf

    def fit(self, stars, model_keys=[], interp_keys=[], model_comparer=None, model_comparer_weights=None, model_init=None, interp_init=None, model_error=None, interp_error=None, model_limit=None, interp_limit=None, logger=None, skip_fit=False, **kwargs):
        """Fit the model!
        :param stars:                       A list of StarData instances.
        :param model:                       A Model instance that defines how to model the individual PSFs
                                            at the location of each star.
        :param interp:                      An Interp instance that defines how to do the interpolation of the
                                            data vectors (produced by model for each star).
        :param model_keys, interp_keys:     A list that gives the attributes of model/interp that the params correspond to
        :param model_comparer:              Model that gets applied to both stars and model_stars to compare success of model. If none is given, compare pixels.
        :param model_comparer_weights:      Weights to apply to the params from model_comparer
        :param model_init, interp_init:     A list that gives the initial values
        :param model_error, interp_error:   A list that gives the initial step sizes
        :param model_limit, interp_limit:   A list that gives the limits of the fit
        :param logger:                      A logger object for logging debug info. [default: None]
        :param skip_fit:                    If True, do not run migrad fit
        :param kwargs:                      kwargs to pass to iminuit fitter
        """
        if logger:
            logger.info("Start fitting HyperPSF using %s stars", len(stars))
            logger.debug("Model Comparer is %s", model_comparer)
            logger.debug("Model is %s", self.model)
            logger.debug("Interp is %s", self.interp)
            logger.debug("Model params are %s", model_keys)
            logger.debug("Interp params are %s", interp_keys)
        from iminuit import Minuit

        # set up interior args for running the fit function
        self._set_fit_func_kwargs(stars, model_keys, interp_keys, model_comparer, model_comparer_weights, model_init, interp_init, model_error, interp_error, model_limit, interp_limit)

        # set up iminuit object
        self.minuit_kwargs = {'throw_nan': False,
                              'pedantic': True,
                              'print_level': 0,
                              'errordef': 1,
                              }

        # update minuit_kwargs from set_fit_func_kwargs
        self.minuit_kwargs.update(self._fit_func_kwargs_minuit)

        # update minuit kwargs from the kwargs
        self.minuit_kwargs.update(kwargs)

        self._logger = logger
        # initialize the minuit object
        self._minuit = Minuit(self._fit_func, **self.minuit_kwargs)
        if not skip_fit:
            # run the fit and solve! This will update the interior parameters
            self._minuit.migrad()
        # these are the best fit parameters
        self._fitarg = self._minuit.fitarg
        # TODO: make sure we have updated to the best values

        # TODO: make a function to return the best_fit values

    def _set_fit_func_kwargs(self, stars, model_keys, interp_keys, model_comparer, model_comparer_weights, model_init, interp_init, model_error, interp_error, model_limit, interp_limit):
        # everything is _'d private because I don't want people to touch it!
        if model_comparer:
            self._stars = [model_comparer.fit(star) for star in stars]
        else:
            self._stars = stars
        self._model_keys = model_keys
        self._interp_keys = interp_keys
        self._model_comparer = model_comparer
        self._model_comparer_weights = model_comparer_weights
        self._model_init = model_init
        self._interp_init = interp_init
        self._model_error = model_error
        self._interp_error = interp_error
        self._model_limit = model_limit
        self._interp_limit = interp_limit

        # build the iminuit arguments
        self._fit_func_kwargs_minuit = {}
        Nmodel = len(self._model_keys)
        Ninterp = len(self._interp_keys)
        Nkeys = Nmodel + Ninterp
        # in principle I could make these as long as I wanted.
        if Nkeys > 50:
            raise Exception('Max of 50 variables! You have {0}!'.format(Nkeys))
        for i in xrange(Nkeys, 50):
            # fix unused parameters
            self._fit_func_kwargs_minuit['fix_p{0}'.format(i)] = True

        # set initial values
        if self._model_init:
            for i, val in enumerate(self._model_init):
                self._fit_func_kwargs_minuit['p{0}'.format(i)] = val
        if self._interp_init:
            for i, val in enumerate(self._interp_init):
                self._fit_func_kwargs_minuit['p{0}'.format(i + Nmodel)] = val
        # set rest
        for i in xrange(Nkeys, 50):
            self._fit_func_kwargs_minuit['p{0}'.format(i)] = 0

        # set initial step sizes
        if self._model_error:
            for i, val in enumerate(self._model_error):
                self._fit_func_kwargs_minuit['error_p{0}'.format(i)] = val
        if self._interp_error:
            for i, val in enumerate(self._interp_error):
                self._fit_func_kwargs_minuit['error_p{0}'.format(i + Nmodel)] = val

        # set limits
        if self._model_limit:
            for i, val in enumerate(self._model_limit):
                self._fit_func_kwargs_minuit['limit_p{0}'.format(i)] = val
        if self._interp_limit:
            for i, val in enumerate(self._interp_limit):
                self._fit_func_kwargs_minuit['limit_p{0}'.format(i + Nmodel)] = val

        # run initial solves for model and interp
        self.model.solve(self._stars)
        self.interp.solve(self._stars)

    # forgive me oh lord this python sin
    def _fit_func(self, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                  p10, p11, p12, p13, p14, p15, p16, p17, p18, p19,
                  p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
                  p30, p31, p32, p33, p34, p35, p36, p37, p38, p39,
                  p40, p41, p42, p43, p44, p45, p46, p47, p48, p49,
                  ):
        # convert p to params
        params = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                  p10, p11, p12, p13, p14, p15, p16, p17, p18, p19,
                  p20, p21, p22, p23, p24, p25, p26, p27, p28, p29,
                  p30, p31, p32, p33, p34, p35, p36, p37, p38, p39,
                  p40, p41, p42, p43, p44, p45, p46, p47, p48, p49,]

        return self._fit_func_interior(params)

    def _fit_func_interior(self, params):
        # update model and interp via params
        if len(self._model_keys) > 0:
            model_params = params[:len(self._model_keys)]
            model = dict(zip(self._model_keys, model_params))
            self.model.update(logger=self._logger, **model)
            self.model.solve(self._stars)

        if len(self._interp_keys) > 0:
            interp_params = params[len(self._model_keys):len(self._model_keys) + len(self._interp_keys)]
            interp = dict(zip(self._interp_keys, interp_params))
            self.interp.update(logger=self._logger, **interp)
            self.interp.solve(self._stars)

        # interp
        stars_interp = self.interp.interpolateList(self._stars, logger=self._logger)

        # model
        stars_model = [self.model.draw(star) for star in stars_interp]

        # if given model_comparer, use to evaluate, else compare drawn images
        if self._model_comparer:
            # model_compare
            stars_compare = [self._model_comparer.fit(star) for star in stars_model]

            # now compare the params of stars and stars_compare
            chisq_vec = numpy.array([numpy.square(stars_compare[i].fit.params - self._stars[i].fit.params)
                                     for i in xrange(len(stars_compare))])
            chisq = numpy.sum(chisq_vec * self._model_comparer_weights)
        else:
            chisqs = []
            for star, star_model in zip(self._stars, stars_model):
                image, weight, pos = star.data.getImage()
                model_image = star_model.data.getImage()[0]
                # square of the difference
                chisq = numpy.sum(numpy.square(((image - model_image) * weight).array))
                chisqs.append(chisq)
            chisq = numpy.sum(chisqs)


        if (self.n_iter % 10 == 0) and (self.minuit_kwargs['print_level'] > 0):
            self._logger.debug(self.n_iter)
            if len(self._model_keys) > 0:
                self._logger.debug(model)
            if len(self._interp_keys) > 0:
                self._logger.debug(interp)
            self._logger.debug(chisq)
            self._p_hist.append([params, chisq])
        self.n_iter += 1

        return chisq

    def __getitem__(self, key):
        """Get best fit value of parameter of the fit.

        This is a key that is in self._model_keys and self._interp_keys

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        # figure out which parameter it is
        if key in self._model_keys and key in self._interp_keys:
            raise Exception('Key {0} is in both the model and interpolator?'.format(key))
        elif key in self._model_keys:
            index = self._model_keys.index(key)
        elif key in self._interp_keys:
            index = self._interp_keys.index(key) + len(self._model_keys)
        else:
            raise Exception('Key {0} is not a valid parameter'.format(key))
        pkey = 'p{0}'.format(index)
        return self._minuit.fitarg[pkey]

    def draw(self, star):
        star = self.interp.interpolate(star)
        star = self.model.draw(star)
        return star
