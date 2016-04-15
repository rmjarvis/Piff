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

def process_interpolator(config, logger=None):
    """Parse the interpolator field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: an Interpolator instance
    """
    import piff

    if logger is None:
        logger = config.setup_logger(verbosity=0)

    if 'interpolator' not in config:
        raise ValueError("config dict has no interpolator field")
    config_interpolator = config['interpolator']

    if 'type' not in config_interpolator:
        raise ValueError("config['interpolator'] has no type field")

    # Get the class to use for the interpolator
    # Not sure if this is what we'll always want, but it would be simple if we can make it work.
    interpolator_class = getattr(piff, config_interpolator.pop('type'))

    # Read any other kwargs in the interpolator field
    kwargs = interpolator_class.parseKwargs(config_interpolator)

    # Build interpolator object
    interpolator = interpolator_class(**kwargs)

    return interpolator

class Interpolator(object):
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
    def parseKwargs(cls, config_interpolator):
        """Parse the interpolator field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_interpolator:   The interpolator field of the configuration dict, config['interpolator']

        :returns: a kwargs dict to pass to the initializer
        """
        return config_interpolator

    def getStarPosition(self, sdata):
        """Extract the appropriate information out of a StarData object to create some sort
        of position object.

        The base class implementation returns the field position (u,v) as numpy.array.
        The returned object will only be used by the Interpolator instance, so it may define this to
        be whatever type is appropriate for its interpolation scheme.

        :param sdata:        A StarData instances from which to extract the relevant properties.

        :returns: some kind of Position object; in the base class, a numpy.array instance.
        """
        return numpy.array([ sdata['u'], sdata['v'] ])

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
        import piff

        field_pos = piff.StarData.calculateFieldPos(image_pos, wcs, pointing)
        return numpy.array([ field_pos.x, field_pos.y ])

    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given some data.

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplemented("Derived classes must define the solve function")

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        raise NotImplemented("Derived classes must define the interpolate function")

    def interpolateList(self, star_list, logger=None):
        """Perform the interpolation for a list of stars.

        The base class just calls interpolate(star) for each star in the list, but in many
        cases, this may be more efficiently done with a matrix operation, so we make it
        available for derived classes to override.

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """
        return numpy.array([ self.interpolate(star) for star in star_list ])

    def write(self, fits, extname):
        """Write an Interpolator to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int).

        However, the derived class will need to implement writeSolution to write the solution
        parameters to a binary table.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the interpolator information.
        """
        # TODO: The I/O routines for Model and Interpolator share a lot of code.  Probably could move
        #       a lot of it into utility functions that both of them call.

        # First write the basic kwargs
        # Start with 'type', since that always needs to be in the table.
        interpolator_type = self.__class__.__name__
        cols = [ [interpolator_type] ]
        dtypes = [ ('type', str, len(interpolator_type) ) ]
        for key, value in self.kwargs.items():
            t = type(value)
            dt = numpy.dtype(t) # just used to categorize the type into int, float, str
            if dt.kind in numpy.typecodes['AllInteger']:
                i = int(value)
                dtypes.append( (key, int) )
                cols.append([i])
            elif dt.kind in numpy.typecodes['AllFloat']:
                f = float(value)
                dtypes.append( (key, float) )
                cols.append([f])
            else:
                s = str(value)
                dtypes.append( (key, str, len(s)) )
                cols.append([s])
        data = numpy.array(zip(*cols), dtype=dtypes)
        fits.write_table(data, extname=extname)

        # Now write the solution parameters
        self.writeSolution(fits, extname + '_solution')

    def writeSolution(self, fits, extname):
        """Write the solution parameters to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the solution.
        """
        raise NotImplemented("Derived classes must define the writeSolution function")

    @classmethod
    def readKwargs(cls, fits, extname):
        """Read the kwargs from the data in a FITS binary table.

        The base class implementation just reads each value in the table and uses the column
        name for the name of the kwarg.  However, derived classes may want to do something more
        sophisticated.  Also, they may want to read other extensions from the fits file
        besides just extname.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interpolator information.

        :returns: a kwargs dict to use to initialize the interpolator
        """
        cols = fits[extname].get_colnames()
        # Remove 'type'
        assert 'type' in cols
        cols = [ col for col in cols if col != 'type' ]

        data = fits[extname].read()
        assert len(data) == 1
        kwargs = dict([ (col, data[col][0]) for col in cols ])
        return kwargs

    @classmethod
    def read(cls, fits, extname):
        """Read an Interpolator from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the interpolator information.

        :returns: an interpolator built with a information in the FITS file.
        """
        import piff

        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        assert 'type' in fits[extname].read().dtype.names
        # interpolator_type = fits[extname].read_column('type')
        interpolator_type = fits[extname].read()['type']
        assert len(interpolator_type) == 1
        interpolator_type = interpolator_type[0]

        # Check that interpolator_type is a valid Interpolator type.
        valid_interpolator_types = dict([ (cls.__name__, cls) for cls in piff.Interpolator.__subclasses__() ])
        if interpolator_type not in valid_interpolator_types:
            raise ValueError("interpolator type %s is not a valid Piff Interpolator")
        interpolator_cls = valid_interpolator_types[interpolator_type]

        kwargs = interpolator_cls.readKwargs(fits, extname)
        interpolator = interpolator_cls(**kwargs)
        interpolator.readSolution(fits, extname + '_solution')
        return interpolator

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interpolator information.
        """
        raise NotImplemented("Derived classes must define the readSolution function")

