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

    :param file_name:       Name of the file to output to. [default: None]
    :param zeropoint:       Zeropoint to use = the magnitude of flux=1. [default: 30]
    :param logger:          A logger object for logging debug info. [default: None]
    """
    def __init__(self, file_name=None, zeropoint=30, logger=None):
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

        if len(self.m_star) == 0:
            xmin = np.floor(np.min(self.m_obj, initial=0))
            xmax = np.ceil(np.max(self.m_obj, initial=100))
            ymax = np.median(self.T_obj)*2 if len(self.T_obj) > 0 else 1.0
        else:
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
            star = ax.scatter(m[s][g[s] == 3], T[s][g[s] == 3],
                              color='green', marker='*', s=40., **kwargs)
            psf = ax.scatter(m[s][g[s] == 4], T[s][g[s] == 4],
                             marker='o', s=50.,
                             facecolors='none', edgecolors='cyan', **kwargs)

        ax.legend([obj, init, star, psf],
              ['Detected Object', 'Candidate Star', 'PSF Star', 'PSF Model'],
              loc='lower left', frameon=True, fontsize=15)

        return fig, ax


class SmallBrightSelect(Select):
    """A Select handler that picks stars by looking for a locus of small, bright objects that
    are all about the same size.

    This is a very crude selection.  It is typically used to provide an initial selection to
    the SizeMag selector.

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
                        (Normally the 'select' field in the overall configuration dict).
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, config, logger=None):
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

        logger.warning("Selecting small/bright objects as stars")

        logger.debug("Initial count = %s", len(objects))

        # The algorithm cannot recover from having fewer than 2 input objects.
        if len(objects) < 2:
            logger.warning("%s input object%s. Cannot find bright/small stellar locus",
                           "Only 1" if len(objects) == 1 else "No",
                           "" if len(objects) == 1 else "s")
            return []

        # Get size, flux from hsm
        obj_shapes = np.array([ obj.hsm for obj in objects ])
        flag_obj = obj_shapes[:, 6]
        f_obj = obj_shapes[:, 0]
        T_obj = 2*obj_shapes[:, 3]**2

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
        i20 = int(np.floor(len(logf) * self.bright_fraction))
        i20 = max(i20, 1)  # Need at least 2
        i20 = min(i20, len(logf)-1)  # sanity check if the user inputs bright_fraction >= 1
        sort_index = np.argpartition(-logf, i20)
        bright_logf = logf[sort_index[:i20+1]]
        bright_logT = logT[sort_index[:i20+1]]
        logger.debug("Bright objects:")
        logger.debug("logf = %s", bright_logf)
        logger.debug("logT = %s", bright_logT)

        # Now take smallest 20% of these (or small_fraction if given)
        i20 = int(np.floor(len(bright_logT) * self.small_fraction))
        i20 = max(i20, 1)  # Need at least 2
        i20 = min(i20, len(bright_logT)-1)  # sanity check if the user inputs small_fraction >= 1
        sort_index = np.argpartition(bright_logT, i20)
        bright_small_logf = bright_logf[sort_index[:i20+1]]
        bright_small_logT = bright_logT[sort_index[:i20+1]]
        logger.debug("Bright/small objects:")
        logger.debug("logf = %s", bright_small_logf)
        logger.debug("logT = %s", bright_small_logT)

        # Sort these by size
        sort_index = np.argsort(bright_small_logT)
        bright_small_logf = bright_small_logf[sort_index]
        bright_small_logT = bright_small_logT[sort_index]

        # Find the "half" with the smallest range in size
        half_len = int(np.floor(len(bright_small_logT) * self.locus_fraction))
        half_len = max(half_len, 1)  # Need at least 2, but half_len is n-1
        half_len = min(half_len, len(bright_small_logT)-1)  # And at most all of them.
        logger.debug("half_len = %s", half_len)
        delta_T = bright_small_logT[half_len:] - bright_small_logT[:-half_len]
        logger.debug("delta_T = %s", delta_T)
        imin = np.argmin(delta_T)
        logger.debug("imin = %s", imin)
        star_logT = bright_small_logT[imin:imin+half_len+1]
        logger.info("Initial bright/small selection includes %d objects",half_len+1)

        # Expand this to include all stars that are within twice the interquarile range of
        # these candidate stars.  Keep doing so until we converge on a good set of stars.
        old_select = None  # Force at least 2 iterations
        for it in range(10):  # (and at most 10)
            if len(star_logT) == 0:
                # This will give an error when taking the median, so bail out here.
                logger.warning("Failed to find bright/small stellar locus.")
                break
            logger.debug("Iteration %d",it)
            logger.debug("Sizes of candidate stars = %s", np.exp(star_logT))
            med = np.median(star_logT)
            logger.info("Median size = %s", np.exp(med))
            q25, q75 = np.percentile(star_logT, [25,75])
            iqr = q75 - q25
            logger.debug("Range of star logT size = %s, %s", np.min(star_logT), np.max(star_logT))
            logger.debug("IQR = %s",iqr)
            iqr = max(iqr,0.01)  # Make sure we don't get too tight an initial grouping
            iqr = min(iqr,self.max_spread/4)
            logger.debug("IQR => %s",iqr)
            select = (logT >= med - 2*iqr) & (logT <= med + 2*iqr) & (logf >= np.min(bright_logf))
            new_count = np.sum(select)
            # Break out when we stop adding more stars.
            if np.array_equal(select, old_select):
                break
            old_select = select
            logger.info("Expand this to include %d selected stars",new_count)
            star_logT = logT[select]
        else:
            logger.info("Max iter = 10 reached.  Stop updating based on median/IQR.")

        # Get the initial indexes of these objects
        select_index = orig_index[select]
        logger.debug("select_index = %s",select_index)
        stars = [objects[i] for i in select_index]
        logger.debug("sizes of stars = %s",[2*s.hsm[3]**2 for s in stars])
        logger.debug("fluxs of stars = %s",[s.hsm[0] for s in stars])
        logger.warning("Bright/small selection found %d likely stars",len(stars))

        return stars

class SizeMagSelect(Select):
    """A Select handler that picks stars by finding where the stellar locus in the size-magnitude
    diagram starts to blend into the galaxy locus.

    By default, the initial selection uses the SmallBright selector, but you can provide
    any other selection algorithm for the initial stars.

    The following parameters are tunable in the config field, but they all have reasonable
    default values:

        :fit_order:     The order of the polynomial fit of log(size) vs (u,v) across the
                        field of view.  This is used to subtract off the gross size variation
                        across the focal plane, which tends to tighten up the stellar locus.
                        [default: 2]
        :purity:        The maximum ratio of objects at the minimum between the stellar locus
                        and the galaxy locus to the number of stars found. [default: 0.01]
        :num_iter:      How many iterations to repeat the processing of building a histogram
                        looking for when it merges with the galaxy locus. [default: 3]

    In addition to these, you may specify the algorithm for making the initial selection
    in an ``inital_select`` dict, which should specify the selection algorithm to use.
    The default is to use the SmallBright selection algorithm with default parameters.
    But you could specify non-default parameters for that as follows (e.g.)::

        select:
            type: SizeMag

            initial_select:
                type: SmallBright
                bright_fraction: 0.3
                small_fraction: 0.4
                locus_fraction: 0.3

            fit_order: 3

    Or, you may want to start with stars according to some other selection algorith.
    E.g. to use some column(s) from the input catalog::

        select:
            type: SizeMag

            initial_select:
                type: Properties
                where: (CLASS_STAR > 0.9) & (MAX_AUTO < 16)

    :param config:      The configuration dict used to define the above parameters.
                        (Normally the 'select' field in the overall configuration dict)
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, config, logger=None):
        super(SizeMagSelect, self).__init__(config, logger)

        opt = {
            'fit_order': int,
            'purity': float,
            'num_iter': int,
        }
        ignore = Select.base_keys + ['initial_select']

        params = galsim.config.GetAllParams(config, config, opt=opt, ignore=ignore)[0]
        self.fit_order = params.get('fit_order', 2)
        self.purity = params.get('purity', 0.01)
        self.num_iter = params.get('num_iter', 3)

        if 'initial_select' in config:
            self.initial_select = config['initial_select']
        else:
            self.initial_select = {'type': 'SmallBright'}

    def selectStars(self, objects, logger=None):
        """Select which of the input objects should be considered stars.

        :param objects:     A list of input objects to be considered as potential stars.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = galsim.config.LoggerWrapper(logger)

        logger.warning("Selecting stars according to locus in size-magnitude diagram")

        stars = Select.process(self.initial_select, objects, logger=logger, select_only=True)

        logger.debug("N objects = %s", len(objects))
        logger.debug("N initial stars = %s", len(stars))

        # Get size, flux from hsm
        obj_shapes = np.array([ obj.hsm for obj in objects ])
        flag_obj = obj_shapes[:, 6]
        f_obj = obj_shapes[:, 0]
        T_obj = 2*obj_shapes[:, 3]**2
        u_obj = np.array([ obj.u for obj in objects ])
        v_obj = np.array([ obj.v for obj in objects ])

        # Getting rid of the flags will mess with the indexing, so keep track of the original
        # index numbers.
        mask = flag_obj == 0
        orig_index = np.arange(len(objects))[mask]

        # Work in log/log space.
        # log(f) is basically a magnitude with different spacing.
        # size of stars is constant, so log(T) doesn't matter that much, but it means that
        # the width of the locus in the size direction is really a fractional width.  This is
        # nice because it gets rid of any scaling issues due to units or pixel size, etc.
        logf_obj = np.log(f_obj[mask])
        logT_obj = np.log(T_obj[mask])
        u_obj = u_obj[mask]
        v_obj = v_obj[mask]
        logger.debug("After removing flags count = %s", len(logf_obj))

        # Sort the objects by brightness (brightest first)
        sort_index = np.argsort(-logf_obj)
        logf_obj = logf_obj[sort_index]
        logT_obj = logT_obj[sort_index]
        u_obj = u_obj[sort_index]
        v_obj = v_obj[sort_index]
        orig_index = orig_index[sort_index]

        # Get the size, flux of the initial candidate stars
        star_shapes = np.array([ star.hsm for star in stars ])
        mask = star_shapes[:, 6] == 0
        logf_star = np.log(star_shapes[mask, 0])
        logT_star = np.log(2*star_shapes[mask, 3]**2)
        u_star = np.array([ star.u for star in stars ])[mask]
        v_star = np.array([ star.v for star in stars ])[mask]
        logger.debug("logf_star = %s",logf_star)
        logger.debug("logT_star = %s",logT_star)

        # Do 3 passes of this because as we add more stars, the fit may become better.
        for i_iter in range(self.num_iter):
            logger.debug("Start iter %d/%d", i_iter, self.num_iter)
            logger.debug("Nstars = %s",len(logT_star))
            logger.debug("Mean logT of stars = %s, std = %s",
                         np.mean(logT_star), np.std(logT_star))

            # Clip outliers so they don't pull the fit.
            q25, q75 = np.percentile(logT_star, [25,75])
            iqr = q75 - q25
            iqr = max(iqr,0.01)  # Make sure we don't get too tight an initial grouping
            good = np.abs(logT_star - np.median(logT_star)) < 2*iqr
            logf_star = logf_star[good]
            logT_star = logT_star[good]
            u_star = u_star[good]
            v_star = v_star[good]
            logger.debug("After clipping 3sigma outliers, N = %s, mean logT = %s, std = %s",
                         len(logT_star), np.mean(logT_star), np.std(logT_star))

            if len(u_star) < (self.fit_order+1)*(self.fit_order+2)//2:
                logger.warning("Too few candidate stars (%d) to use fit_order=%d.",
                               len(u_star), self.fit_order)
                logger.warning("Cannot find stellar locus.")
                return []

            # Fit a polynomial logT(u,v) and subtract it off.
            fn = self.fit_2d_polynomial(u_star, v_star, logT_star, self.fit_order)
            logT_star -= fn(u_star, v_star)
            logger.debug("After subtract 2d polynomial fit logT(u,v), mean logT = %s, std = %s",
                         np.mean(logT_star), np.std(logT_star))
            sigma = np.std(logT_star)
            sigma = max(sigma, 0.01)  # Don't let sigma be 0 in case all logT are equal here.

            # Now build up a histogram in logT (after also subtracting the polynomial fit)
            # Start with brightest objects and slowly go fainter until we see the stellar
            # peak start to merge with the galaxies.  This will define our minimum logf for stars.
            # We don't need to keep the whole range of size.  Just go from 0 (where the stars
            # are now) up to 10 sigma.
            logT_fit = logT_obj - fn(u_obj, v_obj)
            logT_fit_shift = logT_fit + sigma/2.  # Add half sigma, so 0 bin is centered at logT=0.
            use = (logT_fit_shift >= 0) & (logT_fit_shift < 10 * sigma)
            logT = logT_fit_shift[use]
            logf = logf_obj[use]
            hist = np.zeros(10, dtype=int)
            hist_index = (np.floor(logT/sigma)).astype(int)
            assert np.all(hist_index >= 0)
            assert np.all(hist_index < len(hist))

            for i in range(len(logT)):
                hist[hist_index[i]] += 1
                # Find the first valley to the right of the peak at 0.
                # This is defined as locations where the count increases.
                # At first, valley may be index=1, in which case, keep going.
                valleys = np.where(np.diff(hist) > 0)[0]
                if len(valleys) > 0 and valleys[0] > 1:
                    valley = valleys[0]
                    logger.debug("hist = %s, valley = %s",hist, valley)
                    if hist[valley] > self.purity * hist[0]:
                        logger.debug("Value is %s, which is too high (cf. %s)",
                                     hist[valley], self.purity * hist[0])
                        break
            else:
                # If never find a valley (e.g. if all stars or all galaxies are much brighter
                # than the stars being considered), then use the first 0 as the "valley".
                valley = np.argmin(hist)
                # NB. i (used below) is left as the last index in the loop in this case.

            logger.debug('Final hist = %s',hist)
            logger.debug('Added %d objects',i)

            # When we broke out of that loop (if ever), the last object added gives us our
            # flux limit for star selection.
            # The location of the minimum gives us our allowed spread in size.
            # And we make it symmetric, picking the same spread on the small side of the peak.
            half_range = valley * sigma
            min_logf = logf[i]
            logger.debug('Last logf was %s',min_logf)
            logger.debug('valley is at %d sigma = %f', valley, half_range)

            select = (logT_fit >= -half_range) & (logT_fit <= half_range) & (logf_obj >= min_logf)

            # Set up arrays for next iteration
            logf_star = logf_obj[select]
            logT_star = logT_obj[select]
            u_star = u_obj[select]
            v_star = v_obj[select]
            logger.warning("SizeMag iteration %d => N stars = %d", i_iter, len(logf_star))
            logger.warning("Mean logT of stars = %s, std = %s", np.mean(logT_star), np.std(logT_star))

        select_index = orig_index[select]
        logger.debug("select_index = %s",select_index)
        stars = [objects[i] for i in select_index]
        logger.debug("sizes of stars = %s",[2*s.hsm[3]**2 for s in stars])
        logger.debug("fluxs of stars = %s",[s.hsm[0] for s in stars])
        logger.warning("SizeMag selection found %d likely stars",len(stars))

        return stars

    @staticmethod
    def fit_2d_polynomial(x, y, z, order):
        """Fit z = f(x,y) as a 2d polynomial function

        Returns a function object f.
        """
        # I seriously don't know why this isn't a first-level numpy function.
        # It required some sleuthing to find all the numpy primitives required, but once
        # I found them, it's almost trivial to put them together.

        from numpy.polynomial import chebyshev as cheby
        A = cheby.chebvander2d(x, y, (order,order))
        coeff = np.linalg.lstsq(A, z, rcond=None)[0].reshape(order+1,order+1)
        fn = lambda x,y: cheby.chebval2d(x,y,coeff)
        return fn
