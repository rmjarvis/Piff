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

from .psf import PSF
from .star import Star, StarFit, StarData
from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

class CompoundPSF(PSF):
    """A PSF class that uses two PSF solutions.
    """
    def __init__(self, **psfs):
        """
        :param psfs:     PSF instances to use, labelled by 'psf_0', 'psf_1', etc (for order in operation)
        """

        self._npsfs = len(psfs)
        self.kwargs = {}
        self.psfs = []
        for i in range(self._npsfs):
            psf_key = 'psf_{0}'.format(i)
            self.kwargs[psf_key] = 0
            self.psfs.append(psfs[psf_key])

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
        # find the psf_kinds
        psf_kinds = [key for key in config_psf.keys() if 'psf_' == key[:4]]
        for psf_kind in psf_kinds:
            config = config_psf.pop(psf_kind)
            kwargs[psf_kind] = piff.PSF.process(config, logger)

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
        import galsim

        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing

        # iterate:
        self.stars = stars
        oldchisq = 0.
        for iteration in range(max_iterations):
            if logger:
                logger.warning("Iteration %d of compound: Fitting %d stars", iteration+1, len(self.stars))

            nremoved = 0
            for i, psf_i in enumerate(self.psfs):
                # collect other profiles
                profiles = []
                for star in self.stars:
                    profiles_star = []
                    for j, psf_j in enumerate(self.psfs):
                        # on the first passthrough, later PSFs haven't been
                        # initialized, so skip them
                        if iteration == 0 and j > i:
                            continue
                        if i == j:
                            continue
                        profiles_star.append(psf_j.getProfile(star))
                    if len(profiles_star) > 0:
                        profiles.append(galsim.Convolve(profiles_star))

                # fit
                psf_i.fit(self.stars, wcs, pointing, profiles=profiles, logger=logger)

                # update stars from outlier rejection
                nremoved += len(self.stars) - len(psf_i.stars)
                self.stars = psf_i.stars

            chisq = np.sum([s.fit.chisq for s in self.stars])
            dof   = np.sum([s.fit.dof for s in self.stars])
            if logger:
                logger.warn("             Total chisq = %.2f / %d dof", chisq, dof)

            # Very simple convergence test here:
            # Note, the lack of abs here means if chisq increases, we also stop.
            # Also don't quit if a substantial fraction of stars were removed. Basically, having 1 removed out of 1k is not a good reason to keep iterating...
            if (nremoved / len(self.stars) < 0.01) and (oldchisq > 0) and (oldchisq-chisq < chisq_threshold*dof):
                return
            oldchisq = chisq

        logger.warning("PSF fit did not converge.  Max iterations = %d reached.",max_iterations)

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        import galsim
        params = []
        profs = []
        for psf_i in self.psfs:
            profs.append(psf_i.getProfile(star))
            params.append(psf_i.drawStar(star).fit.params)
        params = np.hstack(params)

        # draw star
        prof = galsim.Convolve(profs)
        image = star.image.copy()
        prof.drawImage(image, method='auto', offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, StarFit(params))

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        for i, psf_i in enumerate(self.psfs):
            psf_i._write(fits, extname + '_{0}'.format(i), logger)

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        self.psfs = []
        for i in range(self._npsfs):
            self.psfs.append(PSF._read(fits, extname + '_{0}'.format(i), logger))
