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
.. module:: output
"""

import os
import galsim

from .util import ensure_dir

class Output(object):
    """The base class for handling the output for writing a Piff model.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """

    @classmethod
    def process(cls, config_output, logger=None):
        """Parse the output field of the config dict.

        :param config_output:   The configuration dict for the output field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: an Output handler
        """
        import piff

        # Get the class to use for handling the output data
        # Default type is 'File'
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        output_handler_class = getattr(piff, 'Output' + config_output.pop('type','File'))

        # Read any other kwargs in the output field
        kwargs = output_handler_class.parseKwargs(config_output,logger=logger)

        # Build handler object
        output_handler = output_handler_class(**kwargs)

        return output_handler

    @classmethod
    def parseKwargs(cls, config_output, logger=None):
        """Parse the output field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_output:   The output field of the configuration dict, config['output']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = config_output.copy()
        return kwargs

    def write(self, psf, logger=None):
        """Write a PSF object to the output file.

        :param psf:         A piff.PSF instance
        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplementedError("Derived classes must define the write function")

    def read(self, logger=None):
        """Read a PSF object that was written to an output file back in.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a piff.PSF instance
        """
        raise NotImplementedError("Derived classes must define the read function")


# Note: I'm having a hard time imagining what other kinds of output handlers we'd want
#       here, so this whole idea of an Output base class might be overkill.  For now, I'm
#       keeping the code for writing and reading PSF objects to a file in the PSF class,
#       so this class is really bare-bones, just farming out the work to PSF.
class OutputFile(Output):
    """An Output handler that just writes to a FITS file.

    This is the only Output handler we have, so it doesn't need to be specified by name
    with a ``type`` field.

    It includes specification of both the output file name as well as potentially some
    statistics to output as well.

    :param file_name:   The file name to write the data to.
    :param dir:         Optionally specify a directory for this file. [default: None]
    :param stats_list:  Optionally a list of Stats instances to also output. [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, file_name, dir=None, stats_list=None, logger=None):
        self.file_name = file_name
        if stats_list is not None:
            self.stats_list = stats_list
        else:
            # Make it an empty list if it was None to make some of the later code easier.
            self.stats_list = []

        # Apply the directory name to all file names.
        if dir is not None:
            self.file_name = os.path.join(dir, self.file_name)
            for stats in self.stats_list:
                stats.file_name = os.path.join(dir, stats.file_name)

    @classmethod
    def parseKwargs(cls, config_output, logger=None):
        """Parse the output field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_output:   The output field of the configuration dict, config['output']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        import piff
        kwargs = config_output.copy()
        if 'stats' in config_output:
            stats = piff.Stats.process(kwargs.pop('stats'), logger=logger)
            kwargs['stats_list'] = stats
        return kwargs

    def write(self, psf, logger=None):
        """Write a PSF object to the output file.

        :param psf:         A piff.PSF instance
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        ensure_dir(self.file_name)
        psf.write(self.file_name, logger=logger)

        logger.debug("stats_list = %s",self.stats_list)
        for stats in self.stats_list:
            stats.compute(psf,psf.stars,logger=logger)
            stats.write(logger=logger)
