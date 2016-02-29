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
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: config
"""

from __future__ import print_function

def setup_logger(verbosity=1, log_file=None):
    """Build a logger object to use for logging progress

    Note: This will update the verbosity if a previous call to setup_logger used a different
    value for verbose.  However, it will not update the handler to use a different log_file
    or switch between using a log_file and stdout.

    :param verbosity:   A number from 0-3 giving the level of verbosity to use. [default: 1]
    :param log_file:    A file name to which to output the logging information. [default: None]

    :returns: a logging.Logger instance
    """
    import logging
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[verbosity]
                                                                                                        # Setup logging to go to sys.stdout or (if requested) to an output file
    logger = logging.getLogger('piff')
    if len(logger.handlers) == 0:  # only add handler once!
        if log_file is None:
            handle = logging.StreamHandler()
        else:
            handle = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')  # Simple text output
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    logger.setLevel(logging_level)

    return logger

def parse_variables(config, variables, logger):
    """Parse configuration variables and add them to the config dict

    The command line variables should be specified as key=value.
    The key string can include dots, such as interp.order=2, which means to set::

        config['interp']['order'] = 2

    :param config:      The configuration dict to which to write the key,value pairs.
    :param varaibles:   A list of (typically command line) variables to parse.
    :param logger:      A logger object for logging debug info.
    """
    import yaml
    for v in variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise ValueError('Improper variable specificationi: %s.  Use field.item=value.'%v)
        key, value = v.split('=',1)
        try:
            # Use YAML parser to evaluate the string in case it is a list for instance.
            value = yaml.load(value)
        except:
            logger.debug('Unable to parse %s.  Treating it as a string.'%value)
        config[key] = value


def read_config(file_name):
    """Read a configuration dict from a file.

    :param file_name:   The file name from which the configuration dict should be read.
    """
    import yaml
    with open(file_name) as fin:
        config = yaml.load(fin.read())
    return config

def build_psf(images, stars, model, interp, optics,logger):
    """The main workhorse, which build the PSF.

    :param images: A list of full exposure images
    :param stars: A list of lists of galsim.PositionD objects with star positions.
    :param model: An instance of a Model subclass
    :param interp: An instance of an Interp subclass
    :param optics: An instance of an Optics subclass (maybe None)
    :param logger: A python logging.Logger object
    """
    import galsim
    #Get the star cutout images out if the full images
    #Fit the model to each of them
    parameters = []
    print("Will be analyzing {} images and {} stars".format(len(images), len(stars)))
    for (image, star_positions) in zip(images, stars):
        #We will for the moment just separately analyze each image.
        #So these are the parameters of all the stars for this image
        image_parameters = []
        for star_position in star_positions:

            #Get the cutout for a particular star
            #Box size chosen arbitrarily
            xs = int(star_position.x)
            ys = int(star_position.y)
            bounds = galsim.BoundsI(xs-16,xs+16,ys-16,ys+16)
            cutout = image[bounds]
        
            #Fit this star image
            model.fitImage(cutout)

            #Get the fitted parameters
            params = model.getParameters()
            image_parameters.append(params)

        #Use the interpolator to fit this model
        interp.fitData(image_parameters, star_positions)
        
        #accumulate the parameters for output
        parameters.append(interp.getParameters())

    return parameters

def piffify(config, logger):
    """Build a Piff model according to the specifications in a config dict.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress
    """
    import piff

    for key in ['input', 'output', 'model', 'interp']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)

    # read in the input images
    images, stars = piff.process_input(config, logger)

    # make a Model object to use for the individual stellar fitting
    model = piff.process_model(config, logger)

    # make an Interp object to use for the interpolation
    interp = piff.process_interp(config, logger)

    # if given, make a Optics object to use as the prior information about the optics.
    if 'optics' in config:
        optics = piff.process_optics(config, logger)
    else:
        optics = None

    # build the PSF model
    psf = build_psf(images=images, stars=stars, model=model, interp=interp, optics=optics,
                    logger=logger)

    # write it out to a file
    output = piff.process_output(config, logger)
    output.write(psf)

