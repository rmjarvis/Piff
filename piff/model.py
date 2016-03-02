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

def process_model(config, logger=None):
    """Parse the model field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a Model instance
    """
    import piff

    if logger is None:
        logger = config.setup_logger(verbosity=0)

    if 'model' not in config:
        raise ValueError("config dict has no model field")
    config_model = config['model']

    if 'type' not in config_model:
        raise ValueError("config['model'] has no type field")

    # Get the class to use for the model
    # Not sure if this is what we'll always want, but it would be simple if we can make it work.
    model_class = eval('piff.' + config_model.pop('type'))

    # Read any other kwargs in the model field
    kwargs = model_class.parseKwargs(config_model)

    # Build model object
    model = model_class(**kwargs)

    return model

class Model(object):
    """The base class for modeling a single PSF (i.e. no interpolation yet)

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def parseKwargs(cls, config_model):
        """Parse the model field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_model:    The model field of the configuration dict, config['model']

        :returns: a kwargs dict to pass to the initializer
        """
        return config_model

    def fitStar(self, star):
        """Fit the model parameters to the data for a single star.

        :param star:    A StarData instance

        :returns: self (for convenience of stringing together operations)
        """
        raise NotImplemented("Derived classes must define the fitImage function")

    def drawImage(self, image, pos=None):
        """Draw the model on the given image.

        :param image:   A galsim.Image on which to draw the model.
        :param pos:     The position on the image at which to place the nominal center.
                        [default: None, which means to use the center of the image.]

        :returns: image
        """
        raise NotImplemented("Derived classes must define the getProfile function")

    def getParameters(self):
        """Get the parameters of the model, to be used by the interpolator.

        :returns: a numpy array of the model parameters
        """
        raise NotImplemented("Derived classes must define the getParameters function")

    def setParameters(self, params):
        """Set the parameters of the model, typically provided by an interpolator.

        :param params:  A numpy array of the model parameters

        :returns: self
        """
        raise NotImplemented("Derived classes must define the setParameters function")
