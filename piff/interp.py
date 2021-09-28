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

import numpy as np
from .util import write_kwargs, read_kwargs

class Interp(object):
    """The base class for interpolating a set of data vectors across the field of view.

    In general, the interpolator is agnostic as to the meaning of the parameter vectors.
    These parameter vectors are passed as simple numpy arrays.  They are imbued meaning by
    a Model instance.  Thus, the same interpolators may be used with many different Model
    types.

    The principal ways that interpolators will differ are:

    1. Which properties of the star are used for their interpolation.
    2. What functional form (or algorithm) is used to interpolate between measurements.
    3. Whether the interpolator assumes each sample has a non-degenerate parameter fit, vs
       getting a differential quadratic form for chisq from each sample.

    The answer to #3 is given in a boolean property degenerate_points.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def process(cls, config_interp, logger=None):
        """Parse the interp field of the config dict.

        :param config_interp:   The configuration dict for the interp field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: an Interp instance
        """
        import piff

        if 'type' not in config_interp:
            raise ValueError("config['interp'] has no type field")

        # Get the class to use for the interpolator
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        interp_class = getattr(piff, config_interp['type'])

        # Read any other kwargs in the interp field
        kwargs = interp_class.parseKwargs(config_interp, logger)

        # Build interp object
        interp = interp_class(**kwargs)

        return interp

    @classmethod
    def parseKwargs(cls, config_interp, logger=None):
        """Parse the interp field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_interp:   The interpolator field of the configuration dict, config['interp']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_interp)
        kwargs.pop('type',None)
        return kwargs

    def getProperties(self, star):
        """Extract the appropriate properties to use as the independent variables for the
        interpolation.

        The base class implementation returns the field position (u,v) as a 1d numpy array.

        :param star:    A Star instance from which to extract the properties to use.

        :returns:       A numpy vector of these properties.
        """
        return np.array([ star.data['u'], star.data['v'] ])

    @property
    def property_names(self):
        """List of properties used by this interpolant.
        """
        return ('u', 'v')

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        The nature of the initialization is specific to the derived classes.

        The base class implentation calls interpolateList, which will set the stars to have
        the right type object in its star.fit.params attribute.

        :param stars:       A list of Star instances to use to initialize.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new list of Star instances
        """
        return self.interpolateList(stars)

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data.

        :param stars:       A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplementedError("Derived classes must define the solve method.")

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance holding the interpolated parameters
        """
        raise NotImplementedError("Derived classes must define the interpolate method.")

    def interpolateList(self, stars, logger=None):
        """Perform the interpolation for a list of stars.

        The base class just calls interpolate(star) for each star in the list, but in many
        cases, this may be more efficiently done with a matrix operation, so we make it
        available for derived classes to override.

        :param stars:       A list of Star instances to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """
        return [ self.interpolate(star) for star in stars ]

    def write(self, fits, extname):
        """Write an Interp to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int).

        However, the derived class will need to implement _finish_write to write the solution
        parameters to a binary table.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the interpolator information.
        """
        # First write the basic kwargs that works for all Interp classes
        interp_type = self.__class__.__name__
        write_kwargs(fits, extname, dict(self.kwargs, type=interp_type))

        # Now do the class-specific steps.  Typically, this will write out the solution parameters.
        self._finish_write(fits, extname)

    def _finish_write(self, fits, extname):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, but this will probably always be
        overridden by the derived class.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension
        """
        raise NotImplementedError("Derived classes must define the _finish_write method.")

    @classmethod
    def read(cls, fits, extname):
        """Read an Interp from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the interpolator information.

        :returns: an interpolator built with a information in the FITS file.
        """
        import piff

        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        assert 'type' in fits[extname].read().dtype.names
        # interp_type = fits[extname].read_column('type')
        interp_type = fits[extname].read()['type']
        assert len(interp_type) == 1
        try:
            interp_type = str(interp_type[0].decode())
        except AttributeError:
            # fitsio 1.0 returns strings
            interp_type = interp_type[0]

        # Check that interp_type is a valid Interp type.
        interp_classes = piff.util.get_all_subclasses(piff.Interp)
        valid_interp_types = dict([ (kls.__name__, kls) for kls in interp_classes ])
        if interp_type not in valid_interp_types:
            raise ValueError("interpolator type %s is not a valid Piff Interpolator"%interp_type)
        interp_cls = valid_interp_types[interp_type]

        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type',None)
        interp = interp_cls(**kwargs)
        interp._finish_read(fits, extname)
        return interp

    def _finish_read(self, fits, extname):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, but this will probably always be
        overridden by the derived class.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        raise NotImplementedError("Derived classes must define the _finish_read method.")
