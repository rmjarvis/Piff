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
import numpy

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
    model_class = getattr(piff, config_model.pop('type'))

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

    def write(self, fits, extname):
        """Write a Model to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int)

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        # Start with 'type', since that always needs to be in the table.
        model_type = self.__class__.__name__
        cols = [ [model_type] ]
        dtypes = [ ('type', str, len(model_type) ) ]
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

    @classmethod
    def readKwargs(cls, fits, extname):
        """Read the kwargs from the data in a FITS binary table.

        The base class implementation just reads each value in the table and uses the column
        name for the name of the kwarg.  However, derived classes may want to do something more
        sophisticated.  Also, they may want to read other extensions from the fits file
        besides just extname.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the model information.

        :returns: a kwargs dict to use to initialize the model
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
        assert 'type' in fits[extname].read().dtype.names
        #model_type = fits[extname].read_column('type')  # This isn't working for me...
        model_type = fits[extname].read()['type']
        assert len(model_type) == 1
        model_type = model_type[0]

        # Check that model_type is a valid Model type.
        valid_model_types = dict([ (cls.__name__, cls) for cls in piff.Model.__subclasses__() ])
        if model_type not in valid_model_types:
            raise ValueError("model type %s is not a valid Piff Model")
        model_cls = valid_model_types[model_type]

        kwargs = model_cls.readKwargs(fits, extname)
        model = model_cls(**kwargs)
        return model

    @classmethod
    def readKwargs(cls, fits, extname):
        """Read the kwargs from the data in a FITS binary table.

        The base class implementation just reads each value in the table and uses the column
        name for the name of the kwarg.  However, derived classes may want to do something more
        sophisticated.  Also, they may want to read other extensions from the fits file
        besides just extname.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the model information.

        :returns: a kwargs dict to use to initialize the model
        """
        cols = fits[extname].get_colnames()
        # Remove 'type'
        assert 'type' in cols
        cols = [ col for col in cols if col != 'type' ]

        data = fits[extname].read()
        assert len(data) == 1
        kwargs = dict([ (col, data[col][0]) for col in cols ])
        return kwargs

