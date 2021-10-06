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
            value = yaml.safe_load(value)
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
        config = yaml.safe_load(fin.read())
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
    from .select import Select
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

    # We used to allow a bunch of items in config['input'], which now belong in config['select']
    # For the 1.x series, allow the old API, but give a warning.
    select_keys = set(Select.base_keys)
    user_input_keys = set(config['input'].keys())
    depr_keys = select_keys & user_input_keys
    if len(depr_keys) > 0:
        logger.error("WARNING: Items %r should now be in the 'select' field of the config file.",
                     sorted(depr_keys))
        if 'select' not in config:
            config['select'] = {}
        for key in depr_keys:
            config['select'][key] = config['input'].pop(key)

    # read in the input images
    objects, wcs, pointing = Input.process(config['input'], logger=logger)
    stars = Select.process(config.get('select',{}), objects, logger=logger)

    psf = PSF.process(config['psf'], logger=logger)
    psf.fit(stars, wcs, pointing, logger=logger)

    # Attach these for reference
    psf.initial_objects = objects
    psf.initial_stars = stars

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
    import piff
    from .psf import PSF
    from .output import Output
    from .star import Star

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

    # load psf by looking at output file
    file_name = config['output']['file_name']
    if 'dir' in config['output']:
        file_name = os.path.join(config['output']['dir'], file_name)
    logger.info("Looking for PSF at %s", file_name)
    psf = PSF.read(file_name, logger=logger)

    # The stars we care about for plotify are psf.stars, not what would be made from scratch by
    # processing the input and select fields.  For one, there may be a random component to that
    # process, which would be different this time.  But also, some input objects might have been
    # removed as outliers, so we don't want to include them here.
    # However, the psf.stars do not have the images loaded (to save space in the output file
    # so it's not enormous).  So we need to load in the images for the stats.
    input_handler_class = getattr(piff, 'Input' + config['input'].get('type','Files'))
    input_handler = input_handler_class(config['input'], logger)
    stars = input_handler.load_images(psf.stars, logger=logger)

    # We don't want to rewrite the PSF to disk, so jump straight to the stats_list
    output = Output.process(config['output'], logger=logger)
    logger.debug("stats_list = %s",output.stats_list)
    for stats in output.stats_list:
        stats.compute(psf,stars,logger=logger)
        stats.write(logger=logger)

def meanify(config, logger=None):
    """Take Piff output(s), build an average of the FoV, and write output average.

    :param config:      The configuration file that defines how to build the model
    :param logger:      A logger object for logging progress. [default: None]
    """
    from .star import Star
    import glob
    import numpy as np
    from scipy.stats import binned_statistic_2d
    import fitsio

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = setup_logger(verbose=verbose)

    for key in ['output', 'hyper']:
        if key not in config:
            raise ValueError("%s field is required in config dict"%key)
    for key in ['file_name']:
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output"%key)

    for key in ['file_name']:
        if key not in config['hyper']:
            raise ValueError("%s field is required in config dict hyper"%key)

    if 'dir' in config['output']:
        dir = config['output']['dir']
    else:
        dir = None

    if 'bin_spacing' in config['hyper']:
        bin_spacing = config['hyper']['bin_spacing'] #in arcsec
    else:
        bin_spacing = 120. #default bin_spacing: 120 arcsec

    if 'statistic' in config['hyper']:
        if config['hyper']['statistic'] not in ['mean', 'median']:
            raise ValueError("%s is not a suported statistic (only mean and median are currently "
                             "suported)"%config['hyper']['statistic'])
        else:
            stat_used = config['hyper']['statistic']
    else:
        stat_used = 'mean' #default statistics: arithmetic mean over each bin

    if 'params_fitted' in config['hyper']:
        if type(config['hyper']['params_fitted']) != list:
            raise TypeError('must give a list of index for params_fitted')
        else:
            params_fitted = config['hyper']['params_fitted']
    else:
        params_fitted = None

    if isinstance(config['output']['file_name'], list):
        psf_list = config['output']['file_name']
        if len(psf_list) == 0:
            raise ValueError("file_name may not be an empty list")
    elif isinstance(config['output']['file_name'], str):
        file_name = config['output']['file_name']
        if dir is not None:
            file_name = os.path.join(dir, file_name)
        psf_list = sorted(glob.glob(file_name))
        if len(psf_list) == 0:
            raise ValueError("No files found corresponding to "+config['output']['file_name'])
    else:
        raise ValueError("file_name should be either a list or a string")

    logger.debug('psf_list = %s',psf_list)
    npsfs = len(psf_list)
    logger.debug('npsfs = %d',npsfs)
    config['output']['file_name'] = psf_list

    file_name_in = config['output']['file_name']
    logger.info("Looking for PSF at %s", file_name_in)

    file_name_out = config['hyper']['file_name']
    if 'dir' in config['hyper']:
        file_name_out = os.path.join(config['hyper']['dir'], file_name_out)

    coords = []
    params = []

    for fi, f in enumerate(file_name_in):
        logger.debug('Loading file {0} of {1}'.format(fi, len(file_name_in)))
        fits = fitsio.FITS(f)
        coord, param = Star.read_coords_params(fits, 'psf_stars')
        fits.close()

        coords.append(coord)
        params.append(param)

    params = np.concatenate(params, axis=0)
    coords = np.concatenate(coords, axis=0)
    logger.info('Computing average for {0} params with {1} stars'.format(len(params[0]), len(coords)))

    if params_fitted is None:
        params_fitted = range(len(params[0]))

    lu_min, lu_max = np.min(coords[:,0]), np.max(coords[:,0])
    lv_min, lv_max = np.min(coords[:,1]), np.max(coords[:,1])

    nbin_u = int((lu_max - lu_min) / bin_spacing)
    nbin_v = int((lv_max - lv_min) / bin_spacing)
    binning = [np.linspace(lu_min, lu_max, nbin_u), np.linspace(lv_min, lv_max, nbin_v)]
    nbinning = (len(binning[0]) - 1) * (len(binning[1]) - 1)
    params0 = np.zeros((nbinning, len(params[0])))
    Filter = np.array([True]*nbinning)

    for i in range(len(params[0])):
        if i in params_fitted:
            average, u0, v0, bin_target = binned_statistic_2d(coords[:,0], coords[:,1],
                                                              params[:,i], bins=binning,
                                                              statistic=stat_used)
            average = average.T
            average = average.reshape(-1)
            Filter &= np.isfinite(average).reshape(-1)
            params0[:,i] = average

    # get center of each bin
    u0 = u0[:-1] + (u0[1] - u0[0])/2.
    v0 = v0[:-1] + (v0[1] - v0[0])/2.
    u0, v0 = np.meshgrid(u0, v0)

    coords0 = np.array([u0.reshape(-1), v0.reshape(-1)]).T

    # remove any entries with nan (counts == 0 and non finite value in
    # the 2D statistic computation)
    coords0 = coords0[Filter]
    params0 = params0[Filter]

    dtypes = [('COORDS0', coords0.dtype, coords0.shape),
              ('PARAMS0', params0.dtype, params0.shape),
          ]
    data = np.empty(1, dtype=dtypes)

    data['COORDS0'] = coords0
    data['PARAMS0'] = params0

    logger.info('Writing average solution to {0}'.format(file_name_out))
    with fitsio.FITS(file_name_out,'rw',clobber=True) as f:
        f.write_table(data, extname='average_solution')
