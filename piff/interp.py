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
    # This class-level dict will store all the valid interp types.
    # Each subclass should set a cls._type_name, which is the name that should
    # appear in a config dict.  These will be the keys of valid_interp_types.
    # The values in this dict will be the Interp sub-classes.
    valid_interp_types = {}

    @classmethod
    def process(cls, config_interp, logger=None):
        """Parse the interp field of the config dict.

        :param config_interp:   The configuration dict for the interp field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: an Interp instance
        """
        # Get the class to use for the interpolator
        if 'type' not in config_interp:
            raise ValueError("config['interp'] has no type field")

        interp_type = config_interp['type']
        if interp_type not in Interp.valid_interp_types:
            raise ValueError("type %s is not a valid interp type. "%interp_type +
                             "Expecting one of %s"%list(Interp.valid_interp_types.keys()))

        interp_class = Interp.valid_interp_types[interp_type]

        # Read any other kwargs in the interp field
        kwargs = interp_class.parseKwargs(config_interp, logger)

        # Build interp object
        interp = interp_class(**kwargs)

        return interp

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
            if cls._type_name in Interp.valid_interp_types:
                raise ValueError('Interpolation type %s already registered'%cls._type_name +
                                 'Maybe you subclassed and forgot to set _type_name?')
            Interp.valid_interp_types[cls._type_name] = cls

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
        kwargs['logger'] = logger
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

    def write(self, writer, name):
        """Write an Interp via a writer object.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implementation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int).

        However, the derived class will need to implement _finish_write to write the solution
        parameters to a binary table.

        :param writer:      A writer object that encapsulates the serialization format.
        :param name:        A name to associate with this interpolator in the serialized output.
        """
        # First write the basic kwargs that works for all Interp classes
        interp_type = self.__class__._type_name
        writer.write_struct(name, dict(self.kwargs, type=interp_type))

        # Now do the class-specific steps.  Typically, this will write out the solution parameters.
        with writer.nested(name) as w:
            self._finish_write(w)

    def _finish_write(self, writer):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, but this will probably always be
        overridden by the derived class.

        :param writer:      A writer object that encapsulates the serialization format.
        """
        raise NotImplementedError("Derived classes must define the _finish_write method.")

    @classmethod
    def read(cls, reader, name):
        """Read an Interp via a reader object.

        :param reader:      A reader object that encapsulates the serialization format.
        :param name:        Name associated with this interpolator in the serialized output.

        :returns: an interpolator built from serialized information.
        """
        kwargs = reader.read_struct(name)
        assert kwargs is not None
        assert 'type' in kwargs
        interp_type = kwargs.pop('type')

        # Check that interp_type is a valid Interp type.
        if interp_type not in Interp.valid_interp_types:
            raise ValueError("interp type %s is not a valid Piff Interpolation"%interp_type)
        interp_cls = Interp.valid_interp_types[interp_type]

        interp = interp_cls(**kwargs)
        with reader.nested(name) as r:
            interp._finish_read(r)
        return interp

    def _finish_read(self, reader):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, but this will probably always be
        overridden by the derived class.

        :param reader:      A reader object that encapsulates the serialization format.
        """
        raise NotImplementedError("Derived classes must define the _finish_read method.")
