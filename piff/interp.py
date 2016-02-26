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
.. module:: interp
"""

from __future__ import print_function

def process_interp(config, logger):
    """Parse the interp field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info.

    :returns: an Interp instance
    """
    import piff

    if 'interp' not in config:
        raise ValueError("config dict has no interp field")
    config_interp = config['interp']

    if 'type' not in config_interp:
        raise ValueError("config['interp'] has no type field")

    # Get the class to use for the interp
    # Not sure if this is what we'll always want, but it would be simple if we can make it work.
    interp_class = eval('piff.' + config_interp.pop('type'))

    # Read any other kwargs in the interp field
    kwargs = interp_class.parseKwargs(config_interp)

    # Build interp object
    interp = interp_class(**kwargs)

    return interp

class Interp(object):
    """The base class for interpolating a set of data vectors across the field of view.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def parseKwargs(cls, config_interp):
        """Parse the interp field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_interp:   The interp field of the configuration dict, config['interp']

        :returns: a kwargs dict to pass to the initializer
        """
        return config_interp

    def fitData(self, data, pos):
        """Fit for the interpolation coefficients given some data.

        :param data:        A list of lists of data vectors (numpy arrays) for each star
        :param pos:         A list of lists of positions of the stars
        """
        raise NotImplemented("Derived classes must define the fitData function")

    def getParameters(self):
        """After fitting, get the parameter describing the shape of the PSF variation across the field

        :returns:  A sequence of parameters depending on the interpolation model
        """
        raise NotImplemented("Derived classes must define the fitData function")

    def interpolate(self, image_num, pos):
        """Perform the interpolation to find the interpolated data vector at some position in
        some image.

        :param image_num:   The index of the image in the original list of data vectors.
        :param pos:         The position to which to interpolate.

        :returns: the data vector (a numpy array) interpolated to the given position.
        """
        raise NotImplemented("Derived classes must define the interpolate function")
