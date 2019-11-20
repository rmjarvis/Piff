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
.. module:: psf
"""

from __future__ import print_function

import numpy as np
import copy
import galsim
import logging
from io import StringIO

from .psf import PSF
from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

# Used by SingleChipPSF.fit
def single_chip_run(chipnum, single_psf, stars, wcs, pointing, logger):
    if isinstance(logger, int):
        # In multiprocessing, we cannot pass in the logger, so log to a string and then
        # return that back at the end to be logged by the parent process.
        logger1 = logging.getLogger('logtostring')
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        logger1.addHandler(handler)
        logger1.setLevel(logger) # Input logger in this case is the level to use.
        logger1 = galsim.config.LoggerWrapper(logger1)
    else:
        logger1 = galsim.config.LoggerWrapper(logger)
    # Make a copy of single_psf for each chip
    psf_chip = copy.deepcopy(single_psf)

    # Break the list of stars up into a list for each chip
    stars_chip = [ s for s in stars if s['chipnum'] == chipnum ]
    wcs_chip = { chipnum : wcs[chipnum] }

    # Run the psf_chip fit function using this stars and wcs (and the same pointing)
    logger1.warning("Building solution for chip %s with %d stars", chipnum, len(stars_chip))
    psf_chip.fit(stars_chip, wcs_chip, pointing, logger=logger1)
    if isinstance(logger, int):
        handler.flush()
        buf.flush()
        return chipnum, psf_chip, buf.getvalue()
    else:
        return chipnum, psf_chip, None


class SingleChipPSF(PSF):
    """A PSF class that uses a separate PSF solution for each chip
    """
    def __init__(self, single_psf, nproc=1, extra_interp_properties=None):
        """
        :param single_psf:  A PSF instance to use for the PSF solution on each chip.
                            (This will be turned into nchips copies of the provided object.)
        :param nproc:       How many multiprocessing processes to use for running multiple
                            chips at once. [default: 1]
        :param extra_interp_properties:     A list of any extra properties that will be used for
                                            the interpolation in addition to (u,v).
                                            [default: None]
        """
        self.single_psf = single_psf
        self.nproc = nproc
        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'single_psf': 0,
            'nproc' : nproc,
        }

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        import piff

        config_psf = config_psf.copy()  # Don't alter the original dict.
        config_psf.pop('type', None)
        nproc = config_psf.pop('nproc', 1)

        # If there is a "single_type" specified, call that the type for now.
        if 'single_type' in config_psf:
            config_psf['type'] = config_psf.pop('single_type')

        # Now the regular PSF process function can process the dict.
        single_psf = piff.PSF.process(config_psf, logger)

        return { 'single_psf' : single_psf, 'nproc' : nproc }

    def fit(self, stars, wcs, pointing, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        from itertools import repeat
        from multiprocessing import Pool

        logger = galsim.config.LoggerWrapper(logger)
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing
        self.psf_by_chip = {}

        nproc = galsim.config.util.UpdateNProc(self.nproc, len(wcs), {}, logger)

        def log_output(output):
            chipnum, psf, log = output
            self.psf_by_chip[chipnum] = psf
            if log is not None:
                logger.info(log)

        chipnums = list(wcs.keys())
        if nproc == 1:
            for chipnum in chipnums:
                args = (chipnum, self.single_psf, stars, wcs, pointing, logger)
                output = single_chip_run(*args)
                log_output(output)
        else:
            with Pool(nproc) as pool:
                results = []
                for chipnum in chipnums:
                    args = (chipnum, self.single_psf, stars, wcs, pointing, logger.logger.level)
                    result = pool.apply_async(single_chip_run, args=args, callback=log_output)
                    results.append(result)
                [result.get() for result in results]
                pool.close()
                pool.join()

        # update stars from psf outlier rejection
        self.stars = [ star for chipnum in chipnums for star in self.psf_by_chip[chipnum].stars ]

    def drawStar(self, star, copy_image=True):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns:           Star instance with its image filled with rendered PSF
        """
        if 'chipnum' not in star.data.properties:
            raise ValueError("SingleChip drawStar requires the star to have a chipnum property")
        chipnum = star['chipnum']
        return self.psf_by_chip[chipnum].drawStar(star, copy_image=copy_image)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # Write the colnums to an extension.
        chipnums = list(self.psf_by_chip.keys())
        dt = make_dtype('chipnums', chipnums[0])
        chipnums = [ adjust_value(c,dt) for c in chipnums ]
        cols = [ chipnums ]
        dtypes = [ dt ]
        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname + '_chipnums')

        # Add _1, _2, etc. to the extname for the psf model of each chip.
        for chipnum in self.psf_by_chip:
            self.psf_by_chip[chipnum]._write(fits, extname + '_%s'%chipnum, logger)

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        chipnums = fits[extname + '_chipnums'].read()['chipnums']
        self.psf_by_chip = {}
        for chipnum in chipnums:
            self.psf_by_chip[chipnum] = PSF._read(fits, extname + '_%s'%chipnum, logger)
