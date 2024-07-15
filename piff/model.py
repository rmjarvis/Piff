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
.. module:: model
"""

import numpy as np
import galsim

from .util import write_kwargs, read_kwargs
from .star import Star


class Model(object):
    """The base class for modeling a single PSF (i.e. no interpolation yet)

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    # This class-level dict will store all the valid model types.
    # Each subclass should set a cls._type_name, which is the name that should
    # appear in a config dict.  These will be the keys of valid_model_types.
    # The values in this dict will be the Model sub-classes.
    valid_model_types = {}

    @classmethod
    def process(cls, config_model, logger=None):
        """Parse the model field of the config dict.

        :param config_model:    The configuration dict for the model field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a Model instance
        """
        # Get the class to use for the model
        if 'type' not in config_model:
            raise ValueError("config['model'] has no type field")

        model_type = config_model['type']
        if model_type not in Model.valid_model_types:
            raise ValueError("type %s is not a valid model type. "%model_type +
                             "Expecting one of %s"%list(Model.valid_model_types.keys()))

        model_class = Model.valid_model_types[model_type]

        # Read any other kwargs in the model field
        kwargs = model_class.parseKwargs(config_model, logger)

        # Build model object
        model = model_class(**kwargs)

        return model

    def set_num(self, num):
        """If there are multiple components involved in the fit, set the number to use
        for this model.
        """
        self._num = num

    @classmethod
    def __init_subclass__(cls):
        # Classes that don't want to register a type name can either not define _type_name
        # or set it to None.
        if hasattr(cls, '_type_name') and cls._type_name is not None:
            if cls._type_name in Model.valid_model_types:
                raise ValueError('Model type %s already registered'%cls._type_name +
                                 'Maybe you subclassed and forgot to set _type_name?')
            Model.valid_model_types[cls._type_name] = cls

    @classmethod
    def parseKwargs(cls, config_model, logger=None):
        """Parse the model field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_model:    The model field of the configuration dict, config['model']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_model)
        kwargs.pop('type', None)
        kwargs['logger'] = logger
        return kwargs

    def initialize(self, star, logger=None, default_init=None):
        """Initialize a star to work with the current model.

        :param star:            A Star instance with the raw data.
        :param logger:          A logger object for logging debug info. [default: None]
        :param default_init:    The default initilization method if the user doesn't specify one.
                                [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        raise NotImplementedError("Derived classes must define the initialize function")

    def normalize(self, star):
        """Make sure star.fit.params are normalized properly.

        Note: This modifies the input star in place.
        """
        # This is by default a no op.  Some models may need to do something to noramlize the
        # parameter values in star.fit.
        pass

    def fit(self, star, convert_func=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).
        The returned star.fit.alpha will be inverse covariance of solution if
        it is estimated, else is None.

        :param star:            A Star instance
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component. [default: None]

        :returns:      New Star instance with updated fit information
        """
        raise NotImplementedError("Derived classes must define the fit function")

    def draw(self, star, copy_image=True):
        """Draw the model on the given image.

        :param star:        A Star instance with the fitted parameters to use for drawing and a
                            data field that acts as a template image for the drawn model.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        params = star.fit.get_params(self._num)
        prof = self.getProfile(params).shift(star.fit.center) * star.fit.flux
        if copy_image:
            image = star.image.copy()
        else:
            image = star.image
        prof.drawImage(image, method=self._method, center=star.image_pos)
        return Star(star.data.withNew(image=image), star.fit)

    def write(self, fits, extname):
        """Write a Model to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int)

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        # First write the basic kwargs that works for all Model classes
        model_type = self._type_name
        write_kwargs(fits, extname, dict(self.kwargs, type=model_type))

        # Now do any class-specific steps.
        self._finish_write(fits, extname)

    def _finish_write(self, fits, extname):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Model classes need to write extra information to the
        fits file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension
        """
        pass

    @classmethod
    def read(cls, fits, extname):
        """Read a Model from a FITS file.

        Note: the returned Model will not have its parameters set.  This just initializes a fresh
        model that can be used to interpret interpolated vectors.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the model information.

        :returns: a model built with a information in the FITS file.
        """
        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        model_type = fits[extname].read()['type']
        assert len(model_type) == 1
        try:
            model_type = str(model_type[0].decode())
        except AttributeError:
            # fitsio 1.0 returns strings
            model_type = model_type[0]

        # Check that model_type is a valid Model type.
        if model_type not in Model.valid_model_types:
            raise ValueError("model type %s is not a valid Piff Model"%model_type)
        model_cls = Model.valid_model_types[model_type]

        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type',None)
        if 'force_model_center' in kwargs: # pragma: no cover
            # old version of this parameter name.
            kwargs['centered'] = kwargs.pop('force_model_center')
        model_cls._fix_kwargs(kwargs)
        model = model_cls(**kwargs)
        model._finish_read(fits, extname)
        return model

    @classmethod
    def _fix_kwargs(cls, kwargs):
        """Fix the kwargs read in from an input file.

        This is intended to make it easier to preserve backwards compatibility if a class
        has changed something about the kwargs, this provides a way for old parameter names
        or defaults to be updated for a newer version of Piff than the one that wrong them.

        Usually, this is a no op.

        :param kwargs:  The old kwargs read in from a previous version Piff output file.
        """
        pass

    def _finish_read(self, fits, extname):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Model classes need to read extra information from the
        fits file.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        pass
