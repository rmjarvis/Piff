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

class Star(object):
    """Structure that combines the data for a star with the results of fitting it.

    We will expect it to have the following attributes:
    data:   the StarData pixel data
    params: parameters of the PSF that apply to this star
    flux:   flux of the star (PSF is normalized to unit flux)
    center: (u,v) tuple giving position of stellar center (interaction with StarData??)

    The attributes will be needed if the interpolator is using the chisq expansion instead of
    just parameter vectors:
    
    alpha, beta, gamma: matrix, vector, scalar giving Taylor expansion of chisq wrt params

    Also can hold intermediate data useful to the Model that one wants to save during
    iteration of the fit.

    """
    def __init__(self, data, params, flux=1., center=(0.,0.)):
        """Constructor for base version of Star

        :param data:   A StarData instance
        :param params: An array holding PSF parameters for this star.  Size of array should be set at construction.
        :param flux:   Estimated flux for this star
        :param center: Estimated or fixed center position (u,v) of this star relative to
        the origin of the system returned by StarData.
        """
        
        self.data = data
        self.params = params
        self.flux = flux
        self.center = center
        self.alpha = None
        self.beta = None
        self.gamma = None
        return
    
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

    def makeStar(self, data, flux=1., center=(0.,0.)):
        """Create a Star instance that this Model can manipulate, include any setup needed
        before fitting.

        :param data:    A StarData instance
        :param flux:    Initial estimate of stellar flux
        :param center:  Initial estimate of stellar center in world coord system

        :returns: Star instance
        """
        raise NotImplemented("Derived classes must define the makeStar function")

    def fit(self, star):
        """Fit the Model to the star's data and save results in its params attribute.
        Also updates the star's flux (and center if requested).  

        :param star:   A Star instance

        :returns: None 
        """
        raise NotImplemented("Derived classes must define the fit function")

    def reflux(self, star):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.

        :param star:   A Star instance

        :returns: chi-squared, dof of the fit to the data. ??
        """
        raise NotImplemented("Derived classes must define the reflux function")

    def chisq(self, star):
        """Calculate dependence of chi^2 = -2 log L(D|p) on PSF parameters for single star.
        as a quadratic form chi^2 = dp^T*alpha*dp - 2*beta*dp + gamma,
        where dp is the *shift* from current parameter values.  Marginalization over
        flux (and, optionally, center) should be done by this routine. Results are saved in
        alpha,beta,gamma,flux, (center) attributes of Star.

        :param star:   A Star instance

        :returns: None
        """
        raise NotImplemented("Derived classes must define the chisq function")

    def draw(self, star):
        """Fill the star's pixel data array with a rendering of the PSF specified by
        its current parameters.

        :param star:   A Star instance

        :returns: None
        """
        raise NotImplemented("Derived classes must define the draw function")

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

