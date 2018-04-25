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

from __future__ import print_function
import numpy as np

from .util import write_kwargs, read_kwargs


# Raise this if there's a failure in the Model.fit() method.
class ModelFitError(Exception):
    pass


class Model(object):
    """The base class for modeling a single PSF (i.e. no interpolation yet)

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def process(cls, config_model, logger=None):
        """Parse the model field of the config dict.

        :param config_model:    The configuration dict for the model field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a Model instance
        """
        import piff

        if 'type' not in config_model:
            raise ValueError("config['model'] has no type field")

        # Get the class to use for the model
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        model_class = getattr(piff, config_model['type'])

        # Read any other kwargs in the model field
        kwargs = model_class.parseKwargs(config_model, logger)

        # Build model object
        model = model_class(**kwargs)

        return model

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
        return kwargs

    def initialize(self, star, mask=True, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param mask:    If True, set data.weight to zero at pixels that are outside
                        the range of the model. [default: True]
        :param logger:  A logger object for logging debug info. [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        # If implemented, update the flux to something close to right.
        if hasattr(self, 'reflux'):
            star = self.reflux(star, fit_center=False, logger=logger)
        else:
            star = star.withFlux(np.sum(star.data.image.array))
        return star

    def fit(self, star):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).
        The returned star.fit.alpha will be inverse covariance of solution if
        it is estimated, else is None.

        :param star:   A Star instance

        :returns:      New Star instance with updated fit information
        """
        raise NotImplementedError("Derived classes must define the fit function")

    def draw(self, star, copy_image=True):
        """Create new Star instance that has star.data filled with a rendering
        of the PSF specified by the current StarFit parameters, flux, and center.
        Coordinate mapping of the current StarData is assumed.

        :param star:   A Star instance
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns:      New Star instance with rendered PSF in StarData
        """
        raise NotImplementedError("Derived classes must define the draw function")

    def write(self, fits, extname):
        """Write a Model to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int)

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        # First write the basic kwargs that works for all Model classes
        model_type = self.__class__.__name__
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
        import piff

        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        model_type = fits[extname].read()['type']
        assert len(model_type) == 1
        model_type = str(model_type[0].decode())

        # Check that model_type is a valid Model type.
        model_classes = piff.util.get_all_subclasses(piff.Model)
        valid_model_types = dict([ (c.__name__, c) for c in model_classes ])
        if model_type not in valid_model_types:
            raise ValueError("model type %s is not a valid Piff Model"%model_type)
        model_cls = valid_model_types[model_type]

        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type',None)
        model = model_cls(**kwargs)
        model._finish_read(fits, extname)
        return model

    def _finish_read(self, fits, extname):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Model classes need to read extra information from the
        fits file.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        pass
