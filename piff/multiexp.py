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

from .psf import PSF
from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

class SingleExpPSF(PSF):
    """A PSF class that uses a separate PSF solution for each chip
    """
    def __init__(self, single_psf, extra_interp_properties=None):
        """
        :param single_psf:  A PSF instance to use for the PSF solution on each chip.
                            (This will be turned into nchips copies of the provided object.)
        :param extra_interp_properties:     A list of any extra properties that will be used for
                                            the interpolation in addition to (u,v).
                                            [default: None]
        """
        self.single_psf = single_psf
        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'single_psf': 0,
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

        # If there is a "single_type" specified, call that the type for now.
        if 'single_type' in config_psf:
            config_psf['type'] = config_psf.pop('single_type')

        # Now the regular PSF process function can process the dict.
        single_psf = piff.PSF.process(config_psf, logger)

        return { 'single_psf' : single_psf }

    def fit(self, stars, wcs, pointing, exposures, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing
        self.psf_by_exp = {}
        for exp in exposures:
            self.psf_by_chip = {}
            for chipnum in wcs[exp]:
                # Make a copy of single_psf for each chip
                psf_chip = copy.deepcopy(self.single_psf)
                self.psf_by_chip[chipnum] = psf_chip

                # Break the list of stars up into a list for each chip
                stars_chip = [ s for s in stars if s['expnum'] == exp and s['chipnum'] == chipnum]

                wcs_chip = { chipnum : wcs[exp][chipnum] }

                # Run the psf_chip fit function using this stars and wcs (and the same pointing)
                if logger:
                    logger.info("Building solution for chip %s with %d stars",
                                chipnum, len(stars_chip))
                psf_chip.fit(stars_chip, wcs_chip, pointing, logger=logger)
            self.psf_by_exp[exp] = self.psf_by_chip


        #import pdb; pdb.set_trace()
        self.stars = [star for exp in self.psf_by_exp for chip in self.psf_by_exp[exp] for star in self.psf_by_exp[exp][chip].stars]


    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        chipnum = star['chipnum']
        self.psf_by_chip = self.psf_by_exp[star['expnum']]
        return self.psf_by_chip[chipnum].drawStar(star)

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # Write the colnums to an extension.
        for expr in self.psf_by_exp:
            self.psf_by_chip = self.psf_by_exp[expr]
            chipnums = self.psf_by_chip.keys()
            dt = make_dtype('chipnums', chipnums[0])
            chipnums = [ adjust_value(c,dt) for c in chipnums ]
            cols = [ chipnums ]
            dtypes = [ dt ]
            data = np.array(zip(*cols), dtype=dtypes)
            fits.write_table(data, extname=expr + extname + '_chipnums')

        # Add _1, _2, etc. to the extname for the psf model of each chip.
        for expr in self.psf_by_exp:
            self.psf_by_chip = self.psf_by_exp[expr]
            for chipnum in self.psf_by_chip:
                self.psf_by_chip[chipnum]._write(fits, expr + extname + '_%s'%chipnum, logger)

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        chipnums = fits[extname + '_chipnums'].read()['chipnums']
        self.psf_by_chip = {}
        for chipnum in chipnums:
            self.psf_by_chip[chipnum] = PSF._read(fits, expr + extname + '_%s'%chipnum, logger)

