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

import numpy as np
import copy
import galsim

from .psf import PSF
from .util import write_kwargs, read_kwargs, make_dtype, adjust_value, run_multi

# Used by SingleChipPSF.fit
def single_chip_run(chipnum, single_psf, stars, wcs, pointing, convert_func, logger):
    # Make a copy of single_psf for each chip
    psf_chip = copy.deepcopy(single_psf)

    # Break the list of stars up into a list for each chip
    stars_chip = [ s for s in stars if s['chipnum'] == chipnum ]
    wcs_chip = { chipnum : wcs[chipnum] }

    # Run the psf_chip fit function using this stars and wcs (and the same pointing)
    logger.warning("Building solution for chip %s with %d stars", chipnum, len(stars_chip))
    psf_chip.fit(stars_chip, wcs_chip, pointing, logger=logger, convert_func=convert_func)

    return psf_chip

class SingleChipPSF(PSF):
    """A PSF class that uses a separate PSF solution for each chip

    :param single_psf:  A PSF instance to use for the PSF solution on each chip.
                        (This will be turned into nchips copies of the provided object.)
    :param nproc:       How many multiprocessing processes to use for running multiple
                        chips at once. [default: 1]
    """
    def __init__(self, single_psf, nproc=1):
        self.single_psf = single_psf
        self.nproc = nproc

        self.kwargs = {
            'single_psf': 0,
            'nproc' : nproc,
        }

    @property
    def interp_property_names(self):
        return self.single_psf.interp_property_names

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
        config_psf['type'] = config_psf.pop('single_type', 'Simple')

        # Now the regular PSF process function can process the dict.
        single_psf = piff.PSF.process(config_psf, logger)

        return { 'single_psf' : single_psf, 'nproc' : nproc }

    def fit(self, stars, wcs, pointing, logger=None, convert_func=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing
        self.psf_by_chip = {}

        chipnums = list(wcs.keys())
        args = [(chipnum, self.single_psf, stars, wcs, pointing, convert_func)
                for chipnum in chipnums]

        output = run_multi(single_chip_run, self.nproc, raise_except=False,
                           args=args, logger=logger)

        for chipnum, psf in zip(chipnums, output):
            self.psf_by_chip[chipnum] = psf

        # If any chips failed their solution, remove them.
        if any([self.psf_by_chip[c] is None for c in chipnums]):
            logger.warning("Solutions failed for chipnums: %s",
                           [c for c in chipnums if self.psf_by_chip[c] is None])
            logger.warning("Removing these chips from the output")
            chipnums = [c for c in chipnums if self.psf_by_chip[c] is not None]

        # update stars from psf outlier rejection
        self.stars = [ star for chipnum in chipnums for star in self.psf_by_chip[chipnum].stars ]

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        if 'chipnum' not in star.data.properties:
            raise ValueError("SingleChip requires the star to have a chipnum property")
        chipnum = star['chipnum']
        return self.psf_by_chip[chipnum].interpolateStar(star)

    def _drawStar(self, star, copy_image=True, center=None):
        if 'chipnum' not in star.data.properties:
            raise ValueError("SingleChip requires the star to have a chipnum property")
        chipnum = star['chipnum']
        return self.psf_by_chip[chipnum].drawStar(star, copy_image=copy_image, center=center)

    def _getProfile(self, star):
        chipnum = star['chipnum']
        return self.psf_by_chip[chipnum]._getProfile(star)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # Write the colnums to an extension.
        chipnums = list(self.psf_by_chip.keys())
        chipnums = [c for c in chipnums if self.psf_by_chip[c] is not None]
        dt = make_dtype('chipnums', chipnums[0])
        chipnums = [ adjust_value(c,dt) for c in chipnums ]
        cols = [ chipnums ]
        dtypes = [ dt ]
        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname + '_chipnums')

        # Add _1, _2, etc. to the extname for the psf model of each chip.
        for chipnum in chipnums:
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
        self.single_psf = self.psf_by_chip[chipnums[0]]
