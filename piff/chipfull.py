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

class ChipFullPSF(PSF):
    """A PSF class that uses two PSF solutions -- one for each chip, one for the full.
    """
    def __init__(self, single_psf_chip, single_psf_full,
                 extra_interp_properties=None):
        """
        :param single_psf_chip:     A PSF instance to use for the PSF solution
                                    on each chip.  (This will be turned into
                                    nchips copies of the provided object.)
        :param single_psf_full:     A PSF instance to use for the PSF solution
                                    on the full field.
        :param extra_interp_properties:     A list of any extra properties that
                                            will be used for the interpolation
                                            in addition to (u,v).
                                            [default: None]
        """
        self.single_psf_chip = single_psf_chip
        self.single_psf_full = single_psf_full
        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'single_psf_chip': 0,
            'single_psf_full': 0,
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

        kwargs = {}
        for psf_kind in ['chip', 'full']:
            config = config_psf.pop(psf_kind)
            kwargs['single_psf_{0}'.format(psf_kind)] = piff.PSF.process(config, logger)

        return kwargs

    def fit(self, stars, wcs, pointing,
            chisq_threshold=0.1, max_iterations=30, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param chisq_threshold: Change in reduced chisq at which iteration will terminate.
                                [default: 0.1]
        :param max_iterations:  Maximum number of iterations to try. [default: 30]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing
        self.psf_by_chip = {}

        # TODO: redo this part of the fit!
        # for chipnum in wcs:
        #     # Make a copy of single_psf for each chip
        #     psf_chip = copy.deepcopy(self.single_psf)
        #     self.psf_by_chip[chipnum] = psf_chip

        #     # Break the list of stars up into a list for each chip
        #     stars_chip = [ s for s in stars if s['chipnum'] == chipnum ]
        #     wcs_chip = { chipnum : wcs[chipnum] }

        #     # Run the psf_chip fit function using this stars and wcs (and the same pointing)
        #     if logger:
        #         logger.warning("Building solution for chip %s with %d stars",
        #                        chipnum, len(stars_chip))
        #     psf_chip.fit(stars_chip, wcs_chip, pointing, logger=logger)
        # # update stars from psf outlier rejection
        # self.stars = [ star for chipnum in wcs for star in self.psf_by_chip[chipnum].stars ]

        """
        The awkward thing is that model.fit(Image) would presumably be for Image_full / Image_other_psf, but you can't really do that.
        """

        # initialize full and chip

        # iterate:

        # fit full

        # update stars from outlier rejection

        # fit chip

        # update stars from outlier rejection

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # first draw the wide field piece
        full_star = self.psf_full.drawStar(star)

        # now draw the chip piece
        chipnum = star['chipnum']
        chip_star = self.psf_by_chip[chipnum].drawStar(star)

        # now convolve them
        return self.convolve(full_star, chip_star)

    def convolve(self, star1, star2):
        """Given two stars, combine together via convolution

        :param star1, star2:    Star instances for our two stars

        :returns:               Star instance with image filled by convolution
                                of the two rendered PSFs
        """
        # create empty star by copying star1

        # combine star1 and star2 fit params

        # convolve star1 and star2 image

        pass


    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # Write the colnums to an extension.
        chipnums = self.psf_by_chip.keys()
        dt = make_dtype('chipnums', chipnums[0])
        chipnums = [ adjust_value(c,dt) for c in chipnums ]
        cols = [ chipnums ]
        dtypes = [ dt ]
        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname + '_chipnums')

        # Add _1, _2, etc. to the extname for the psf model of each chip.
        for chipnum in self.psf_by_chip:
            self.psf_by_chip[chipnum]._write(fits, extname + '_%s'%chipnum, logger)

        # repeat for the full psf
        self.psf_full._write(fits, extname + '_full', logger)

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

        # repeat for the full psf
        self.psf_full = PSF._read(fits, extname + '_full', logger)

