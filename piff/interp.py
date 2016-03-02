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
import numpy

def process_interp(config, logger=None):
    """Parse the interp field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: an Interp instance
    """
    import piff

    if logger is None:
        logger = config.setup_logger(verbosity=0)

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

    In general, the interpolator is agnostic as to the meaning of the parameter vectors.
    These parameter vectors are passed as simple numpy arrays.  They are imbued meaning by
    a Model instance.  Thus, the same interpolators may be used with many different Model
    types.

    The principal ways that interpolators will differ are

    1. Which properties of the star are used for ther interpolation
    2. What functional form (or algorithm) is used to interpolate between measurements.

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

    def getStarPosition(self, star):
        """Extract the appropriate information out of a StarData object to create some sort
        of position object.

        The base class implementation returns the field position (u,v) as numpy.array.
        The returned object will only be used by the Interp instance, so it may define this to
        be whatever type is appropriate for its interpolation scheme.

        :param star:        A StarData instances from which to extract the relevant properties.

        :returns: some kind of Position object; in the base class, a numpy.array instance.
        """
        return numpy.array([ star['u'], star['v'] ])

    def getTargetPosition(self, image_pos, wcs, pointing, properties):
        """Get an appropriate position to use for an interpolation target.

        The base class implementation returns the field position corresponding to a given
        image position.

        :param image_pos:   The position in chip coordinates to use as the target location.
        :param wcs:         The wcs to use to connect image coordinates with sky coordinates.
        :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                            exposure.  This is required if wcs is a CelestialWCS, but should
                            be None if wcs is a EuclideanWCS. [default: None]
        :param properties:  A dict containing other properties that the interpolator needs to
                            perform the interpolation. [default: None]

        :returns: the same kind of Position object that getStarPosition returns.
        """
        field_pos = piff.StarData.calculateFieldPos(image_pos, wcs, pointing)
        return numpy.array([ field_pos.x, field_pos.y ])

    def solve(self, pos, vectors, logger=None):
        """Solve for the interpolation coefficients given some data.

        :param pos:         A list of positions to use for the interpolation.
        :param vectors:     A list of parameter vectors (numpy arrays) for each star.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplemented("Derived classes must define the solve function")

    def interpolate(self, pos, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param pos:         The position to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the parameter vector (a numpy array) interpolated to the given position.
        """
        raise NotImplemented("Derived classes must define the interpolate function")

    def interpolateList(self, pos_list, logger=None):
        """Perform the interpolation for a list of positions.

        The base class just calls interpolate(pos) for each position in the list, but in many
        cases, this may be more efficiently done with a matrix operation, so we make it
        available for derived classes to override.

        :param pos_list:    A list (or numpy.array) of positions to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a numpy array of parameter vectors for the given positions.
        """
        return numpy.array([ self.interpolate(pos) for pos in pos_list ])
