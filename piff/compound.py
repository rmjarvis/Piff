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
from .star import Star, StarFit, StarData
from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

# TODO: in principal, can be arbitrary number of PSFs
class CompoundPSF(PSF):
    """A PSF class that uses two PSF solutions -- one for each chip, one for the full.
    """
    def __init__(self, psf_1, psf_2,
                 extra_interp_properties=None):
        """
        :param psf_1:     A PSF instance to use for the PSF solution on each
                          chip.  (This will be turned into nchips copies of the
                          provided object.)
        :param psf_2:     A PSF instance to use for the PSF solution on the
                          full field.
        :param extra_interp_properties:     A list of any extra properties that
                                            will be used for the interpolation
                                            in addition to (u,v).
                                            [default: None]
        """
        self.psfs = [psf_1, psf_2]
        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'psf_1': 0,
            'psf_2': 0,
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
        for psf_kind in ['1', '2']:
            config = config_psf.pop(psf_kind)
            kwargs['psf_{0}'.format(psf_kind)] = piff.PSF.process(config, logger)

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
                logger.warning("Iteration %d: Fitting %d stars", iteration+1, len(self.stars))

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
                if len(profiles) == 0:
                    profiles = None

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
            # Also, don't quit if we removed any outliers.
            if (nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < chisq_threshold*dof):
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
        # TODO: is there a way to collect fit params without drawStar?
        params = []
        profs = []
        for psf_i in self.psfs:
            star = psf_i.drawStar(star)
            profs.append(psf_i.getProfile(star))
            params.append(star.fit.params)
        params = np.hstack(params)

        # draw star
        prof = galsim.Convolve(profs)
        image = star.image.copy()
        prof.drawImage(image, method=self._method, offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, StarFit(params))

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        for i, psf_i in enumerate(self.psfs):
            psf_i.write(fits, extname + '_{0}'.format(i), logger)

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # TODO: need to specify how many psfs in advance!
        self.psfs = []
        for i in range(2):
            self.psfs.append(PSF._read(fits, extname + '_{0}'.format(i), logger))
