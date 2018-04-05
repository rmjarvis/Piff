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
import os
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


def process(config, logger=None):
    """Build a Piff model according to the specifications in a config dict.

    Note: This is almost the same as the piffify function/executable.  The difference is
          that it returns the resulting psf, rather than writing it to a file.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]

    :returns: the psf model
    """
    from .input import Input
    from .psf import PSF

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = setup_logger(verbose=verbose)

    for key in ['input', 'psf']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)

    # Import extra modules if requested
    if 'modules' in config:
        galsim.config.ImportModules(config)

    # read in the input images
    stars, wcs, pointing = Input.process(config['input'], logger=logger)

    psf = PSF.process(config['psf'], logger=logger)
    psf.fit(stars, wcs, pointing, logger=logger)

    return psf

def piffify(config, logger=None):
    """Build a Piff model according to the specifications in a config dict.

    This includes writing the model to disk according to the output field.
    If you would rather get the psf object in return, see the process function.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]
    """
    from .output import Output

    for key in ['input', 'output', 'psf']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)

    psf = process(config, logger)

    # write it out to a file
    output = Output.process(config['output'], logger=logger)
    output.write(psf, logger=logger)

def plotify(config, logger=None):
    """Take a Piff model, load in images, and execute output.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]
    """
    from .input import Input
    from .psf import PSF
    from .output import Output

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = setup_logger(verbose=verbose)

    for key in ['input', 'output']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)
    for key in ['file_name']:
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output"%key)

    # Import extra modules if requested
    if 'modules' in config:
        galsim.config.ImportModules(config)

    # read in the input images
    stars, wcs, pointing = Input.process(config['input'], logger=logger)

    # load psf by looking at output file
    file_name = config['output']['file_name']
    if 'dir' in config['output']:
        file_name = os.path.join(config['output']['dir'], file_name)
    logger.info("Looking for PSF at %s", file_name)
    psf = PSF.read(file_name, logger=logger)

    # we don't want to rewrite the PSF to disk, so jump straight to the stats_list
    output = Output.process(config['output'], logger=logger)
    logger.debug("stats_list = %s",output.stats_list)
    for stats in output.stats_list:
        stats.compute(psf,stars,logger=logger)
        stats.write(logger=logger)

def meanify(config, logger=None):
    """Take Piff output(s), build an average of the FoV, and wirte output average.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]
    """
    from .input import Input
    from .psf import PSF
    from .star import Star
    from .output import Output
    import glob
    import numpy as np
    import fitsio

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = setup_logger(verbose=verbose)

    for key in ['input', 'output']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)
    for key in ['file_name']:
        if key not in config['input']:
            raise ValueError("%s field is required in config dict output"%key)

    for key in ['file_name']:
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output"%key)

    if 'dir' in config['input']:
        dir = config['input']['dir']
    else:
        dir = None

    if 'binning' in config['input']:
        if type(config['input']['binning']) is not int:
            ValueError("type of binning should be integer (%s now)"%type(config['input']['binning']))
        else:
            binning = config['input']['binning']
    else:
        binning = 50

    if isinstance(config['input']['file_name'], list):
        psf_list = config['input']['file_name']
        if len(psf_list) == 0:
            raise ValueError("file_name may not be an empty list")
    elif isinstance(config['input']['file_name'], str):
        file_name = config['input']['file_name']
        if dir is not None:
            file_name = os.path.join(dir, file_name)
        psf_list = sorted(glob.glob(file_name))
        if len(psf_list) == 0:
            raise ValueError("No files found corresponding to "+config['file_name'])
    elif not isinstance(config['file_name'], dict):
        raise ValueError("file_name should be either a dict or a string")

    if psf_list is not None:
        logger.debug('psf_list = %s',psf_list)
        npsfs = len(psf_list)
        logger.debug('npsfs = %d',npsfs)
        config['input']['file_name'] = psf_list

    file_name_in = config['input']['file_name']
    logger.info("Looking for PSF at %s", file_name_in)
    if 'dir' in config['output']:
        file_name_out = os.path.join(config['output']['dir'], file_name)
    file_name_out = config['output']['file_name']

    def _getcoord(star):
        return np.array([star.data[key] for key in ['u', 'v']])

    coords = []
    params = []

    for f in file_name_in:
        hdu = fitsio.FITS(f) 
        stars = Star.read(hdu, 'psf_stars')
        for s in stars:
            coords.append(_getcoord(s))
            params.append(s.fit.params)

    coords = np.array(coords)
    params = np.array(params)

    lu_min, lu_max = np.min(coords[:,0]), np.max(coords[:,0])
    lv_min, lv_max = np.min(coords[:,1]), np.max(coords[:,1])

    binning = [np.linspace(lu_min, lu_max, binning),np.linspace(lv_min, lv_max, binning)]
    counts, u0, v0 = np.histogram2d(coords[:,0], coords[:,1], bins=binning)
    counts = counts.T
    counts = counts.reshape(-1)

    params0 = np.zeros((len(counts),len(params[0])))

    for i in range(len(params[0])):
        average = np.histogram2d(coords[:,0], coords[:,1], bins=binning, weights=params[:,i])[0].T
        average = average.reshape(-1)
        average[average!=0] /= counts[average!=0]
        params0[:,i] = average

    # get center of each bin 
    u0 = u0[:-1] + (u0[1] - u0[0])/2.
    v0 = v0[:-1] + (v0[1] - v0[0])/2.
    u0, v0 = np.meshgrid(u0, v0)

    coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T

    dtypes = [('COORDS0', coords0.dtype, coords0.shape),
              ('PARAMS0', params0.dtype, params0.shape),
          ]
    data = np.empty(1, dtype=dtypes)

    data['COORDS0'] = coords0
    data['PARAMS0'] = params0

    with fitsio.FITS(file_name_out,'rw',clobber=True) as f:
        f.write_table(data, extname='average_solution')
