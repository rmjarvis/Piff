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
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

import numpy as np
import warnings
import galsim

# This file include both the SizeMagSelect selection type and the SizeMagStats stat type.

from .stats import Stats
from .select import Select

class SizeMagStats(Stats):
    """Statistics class that plots a size magnitude diagram to check the quality of
    the star selection.
    """

    def __init__(self, file_name=None, zeropoint=30, logger=None):
        """
        :param file_name:       Name of the file to output to. [default: None]
        :param zeropoint:       Zeropoint to use = the magnitude of flux=1. [default: 30]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.file_name = file_name
        self.zeropoint = zeropoint

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # measure everything with hsm
        logger.info("Measuring Star and Model Shapes")
        _, star_shapes, psf_shapes = self.measureShapes(psf, stars, logger=logger)

        if hasattr(psf, 'initial_stars'):
            # remove initial objects that ended up being used.
            init_pos = [s.image_pos for s in psf.initial_stars]
            initial_objects = [s for s in psf.initial_objects if s.image_pos not in init_pos]
            star_pos = [s.image_pos for s in stars]
            initial_stars = [s for s in psf.initial_stars if s.image_pos not in star_pos]
            _, obj_shapes, _ = self.measureShapes(None, initial_objects, logger=logger)
            _, init_shapes, _ = self.measureShapes(None, initial_stars, logger=logger)
        else:
            # if didn't make psf using process, then inital fields will be absent.
            # Just make them empty arrays.
            obj_shapes = np.empty(shape=(0,7))
            init_shapes = np.empty(shape=(0,7))

        # Pull out the sizes and fluxes
        flag_star = star_shapes[:, 6]
        mask = flag_star == 0
        self.f_star = star_shapes[mask, 0]
        self.T_star = star_shapes[mask, 3]
        flag_psf = psf_shapes[:, 6]
        mask = flag_psf == 0
        self.f_psf = psf_shapes[mask, 0]
        self.T_psf = psf_shapes[mask, 3]
        flag_obj = obj_shapes[:, 6]
        mask = flag_obj == 0
        self.f_obj = obj_shapes[mask, 0]
        self.T_obj = obj_shapes[mask, 3]
        flag_init = init_shapes[:, 6]
        mask = flag_init == 0
        self.f_init = init_shapes[mask, 0]
        self.T_init = init_shapes[mask, 3]

        # Calculate the magnitudes
        self.m_star = self.zeropoint - 2.5 * np.log10(self.f_star)
        self.m_psf = self.zeropoint - 2.5 * np.log10(self.f_psf)
        self.m_init = self.zeropoint - 2.5 * np.log10(self.f_init)
        self.m_obj = self.zeropoint - 2.5 * np.log10(self.f_obj)

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:  Any additional kwargs go into the matplotlib scatter() function.

        :returns: fig, ax
        """
        from matplotlib.figure import Figure
        logger = galsim.config.LoggerWrapper(logger)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)

        xmin = np.floor(np.min(self.m_star))
        xmin = np.min(self.m_init, initial=xmin)  # Do it this way in case m_init is empty.
        xmax = np.ceil(np.max(self.m_star))
        xmax = np.max(self.m_obj, initial=xmax)   # Likewise, m_obj might be empty.
        ymax = max(np.median(self.T_star)*2, np.max(self.T_star)*1.01)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel('Magnitude (ZP=%s)'%self.zeropoint, fontsize=15)
        ax.set_ylabel(r'$T = 2\sigma^2$ (arcsec${}^2$)', fontsize=15)

        # This looks better if we plot all the objects a bit at a time, rather than all of one type
        # than all of the next, etc.  This makes it easier to see if there are some galaxies in
        # the star area or vice versa.  We implement this by sorting all of them in magnitude
        # and then plotting the indices mod 20.
        group_obj = np.ones_like(self.m_obj) * 1
        group_init = np.ones_like(self.m_init) * 2
        group_star = np.ones_like(self.m_star) * 3
        group_psf = np.ones_like(self.m_psf) * 4

        g = np.concatenate([group_obj, group_init, group_star, group_psf])
        m = np.concatenate([self.m_obj, self.m_init, self.m_star, self.m_psf])
        T = np.concatenate([self.T_obj, self.T_init, self.T_star, self.T_psf])

        index = np.argsort(m)
        n = np.arange(len(m))

        for i in range(20):
            s = index[n%20==i]

            obj = ax.scatter(m[s][g[s] == 1], T[s][g[s] == 1],
                             color='black', marker='o', s=2., **kwargs)
            init = ax.scatter(m[s][g[s] == 2], T[s][g[s] == 2],
                              color='magenta', marker='o', s=8., **kwargs)
            psf = ax.scatter(m[s][g[s] == 3], T[s][g[s] == 3],
                             marker='o', s=50.,
                             facecolors='none', edgecolors='cyan', **kwargs)
            star = ax.scatter(m[s][g[s] == 4], T[s][g[s] == 4],
                              color='green', marker='*', s=40., **kwargs)

        ax.legend([obj, init, star, psf],
              ['Detected Object', 'Candidate Star', 'PSF Star', 'PSF Model'],
              loc='lower left', frameon=True, fontsize=15)

        return fig, ax


class SmallBrightSelect(Select):
    """A Select handler that picks stars by looking for a locus of small, bright objects that
    are all about the same size.

    This is a very crude selection.  It is typically used to provide an initial selection to
    the SizeMag selector.
    """
    def __init__(self, config, logger=None):
        """
        Parse the config dict (Normally the 'where' field in the overall configuration dict).

        The following parameters are tunable in the config field, but they all have reasonable
        default values:

            :bright_fraction:   The fraction of the detected objects to consider "bright" and
                                thus available for consideration for star selection.
                                [default: 0.2]
            :small_fraction:    The fraction of the bright objects to consider "small".
                                [default: 0.2]
            :locus_fraction:    The fraction of the initial small, bright objects to look for
                                having all nearly the same size.  These constitude the first
                                extremely crude selection of high-confidence stars. [default: 0.5]
            :max_spread:        The maximum spread in log(T) to allow for the final set of stars.
                                From the initial high-confidence stars, we expand the range in
                                size to try to capture the full stellar locus based on the
                                inter-quartile range of the candidate stars.  This parameter sets
                                a maximum range to allow for this locus to make sure it doesn't
                                erroneously expand to include everything. [default: 0.1]

        :param config:      The configuration dict used to define the above parameters.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        super(SmallBrightSelect, self).__init__(config, logger)

        opt = {
            'bright_fraction': float,
            'small_fraction': float,
            'locus_fraction': float,
            'max_spread': float,
        }
        params = galsim.config.GetAllParams(config, config, opt=opt, ignore=Select.base_keys)[0]
        self.bright_fraction = params.get('bright_fraction', 0.2)
        self.small_fraction = params.get('small_fraction', 0.2)
        self.locus_fraction = params.get('locus_fraction', 0.5)
        self.max_spread = params.get('max_spread', 0.1)

    def selectStars(self, objects, logger=None):
        """Select which of the input objects should be considered stars.

        :param objects:     A list of input objects to be considered as potential stars.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = galsim.config.LoggerWrapper(logger)

        logger.info("Selecting small/bright objects as stars")

        logger.debug("Initial count = %s", len(objects))

        # Get size, flux from hsm
        obj_shapes = np.array([ obj.hsm for obj in objects ])
        flag_obj = obj_shapes[:, 6]
        f_obj = obj_shapes[:, 0]
        T_obj = obj_shapes[:, 3]

        # Getting rid of the flags will mess with the indexing, so keep track of the original
        # index numbers.
        mask = flag_obj == 0
        orig_index = np.arange(len(objects))[mask]

        # Work in log/log space.
        # log(f) is basically a magnitude with different spacing.
        # size of stars is constant, so log(T) doesn't matter that much, but it means that
        # the width of the locus in the size direction is really a fractional width.  This is
        # nice because it gets rid of any scaling issues due to units or pixel size, etc.
        logf = np.log(f_obj[mask])
        logT = np.log(T_obj[mask])
        logger.debug("After removing flags count = %s", len(logf))

        # Pick out brightest 20% (or bright_fraction if given)
        i20 = int(np.ceil(len(logf) * self.bright_fraction))
        sort_index = np.argpartition(-logf, i20)
        bright_logf = logf[sort_index[:i20]]
        bright_logT = logT[sort_index[:i20]]
        logger.debug("Bright objects:")
        logger.debug("logf = %s", bright_logf)
        logger.debug("logT = %s", bright_logT)

        # Now take smallest 20% of these (or small_fraction if given)
        i20 = int(np.ceil(len(bright_logf) * self.small_fraction))
        sort_index = np.argpartition(bright_logT, i20)
        bright_small_logf = bright_logf[sort_index[:i20]]
        bright_small_logT = bright_logT[sort_index[:i20]]
        logger.debug("Bright/small objects:")
        logger.debug("logf = %s", bright_small_logf)
        logger.debug("logT = %s", bright_small_logT)

        # Sort these by size
        sort_index = np.argsort(bright_small_logT)
        bright_small_logf = bright_small_logf[sort_index]
        bright_small_logT = bright_small_logT[sort_index]

        # Find the "half" with the smallest range in size
        half_len = int(np.ceil(len(bright_small_logT) * self.locus_fraction))
        delta_T = bright_small_logT[-half_len-1:] - bright_small_logT[:half_len+1]
        imin = np.argmin(delta_T[:-half_len])
        star_logT = bright_small_logT[imin:imin+half_len]
        logger.info("Initial bright/small selection includes %d objects",half_len)

        # Expand this to include all stars that are within twice the interquarile range of
        # these candidate stars.  Keep doing so until we converge on a good set of stars.
        old_select = None  # Force at least 2 iterations.
        while True:
            logger.debug("Sizes of candidate stars = %s", np.exp(star_logT))
            med = np.median(star_logT)
            logger.info("Median size = %s", np.exp(med))
            q25, q75 = np.percentile(star_logT, [25,75])
            iqr = q75 - q25
            logger.debug("Range of initial star logT size = %s, %s",
                        np.min(star_logT), np.max(star_logT))
            logger.debug("IQR = %s",iqr)
            iqr = min(iqr,self.max_spread/4)
            select = (logT > med - 2*iqr) & (logT < med + 2*iqr) & (logf > np.min(bright_logf))
            new_count = np.sum(select)
            # Break out when we stop adding more stars.
            if np.array_equal(select, old_select):
                break
            old_select = select
            logger.info("Expand this to include %d selected stars",new_count)
            star_logT = logT[select]

        # Get the initial indexes of these objects
        select_index = orig_index[select]
        logger.debug("select_index = %s",select_index)
        stars = [objects[i] for i in select_index]
        logger.debug("sizes of stars = %s",[s.hsm[3] for s in stars])
        logger.debug("fluxs of stars = %s",[s.hsm[0] for s in stars])

        return stars
