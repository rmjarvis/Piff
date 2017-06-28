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
.. module:: config
"""

from __future__ import print_function

import yaml
import galsim

def setup_logger(verbose=1, log_file=None):
    """Build a logger object to use for logging progress

    Note: This will update the verbosity if a previous call to setup_logger used a different
    value for verbose.  However, it will not update the handler to use a different log_file
    or switch between using a log_file and stdout.

    :param verbose:     A number from 0-3 giving the level of verbosity to use. [default: 1]
    :param log_file:    A file name to which to output the logging information. [default: None]

    :returns: a logging.Logger instance
    """
    import logging
    # Parse the integer verbosity level from the command line args into a logging_level string
    logging_levels = { 0: logging.CRITICAL,
                       1: logging.WARNING,
                       2: logging.INFO,
                       3: logging.DEBUG }
    logging_level = logging_levels[verbose]

    # Setup logging to go to sys.stdout or (if requested) to an output file
    logger = logging.getLogger('piff')
    logger.handlers = []  # Remove any existing handlers
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
    # Note: This is basically a copy of the GalSim function ParseVariables in the galsim.py script.
    new_params = {}
    for v in variables:
        logger.debug('Parsing additional variable: %s',v)
        if '=' not in v:
            raise ValueError('Improper variable specificationi: %s.  Use field.item=value.'%v)
        key, value = v.split('=',1)
        try:
            # Use YAML parser to evaluate the string in case it is a list for instance.
            value = yaml.load(value)
        except yaml.YAMLError as e:  # pragma: no cover
            logger.warning('Caught %r',e)
            logger.warning('Unable to parse %s.  Treating it as a string.',value)
        new_params[key] = value
    galsim.config.UpdateConfig(config, new_params)

def read_config(file_name):
    """Read a configuration dict from a file.

    :param file_name:   The file name from which the configuration dict should be read.
    """
    with open(file_name) as fin:
        config = yaml.load(fin.read())
    return config


def piffify(config, logger=None):
    """Build a Piff model according to the specifications in a config dict.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]
    """
    import piff
    import copy

    # Make a copy to make sure we don't change the original.
    config = copy.deepcopy(config)

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = piff.setup_logger(verbose=verbose)

    for key in ['input', 'output', 'psf']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)

    # Import extra modules if requested
    if 'modules' in config:
        galsim.config.ImportModules(config)

    # read in the input images
    stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)

    psf = piff.PSF.process(config['psf'], logger=logger)
    psf.fit(stars, wcs, pointing, logger=logger)

    # write it out to a file
    output = piff.Output.process(config['output'], logger=logger)
    output.write(psf, logger=logger)

