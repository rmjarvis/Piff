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
.. module:: input
"""

import numpy as np
import scipy
import glob
import os
import galsim

from .util import run_multi, calculateSNR
from .star import Star, StarData
from .input import InputFiles

class Select(object):
    """The base class for selecting which stars to use for characterizing the PSF.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    # Parameters that derived classes should ignore if they appear in the config dict
    # (since they are handled by the base class).
    base_keys= ['min_snr', 'max_snr', 'hsm_size_reject', 'max_pixel_cut', 'reject_where',
                'max_edge_frac', 'stamp_center_size', 'max_mask_pixels',
                'reserve_frac', 'seed']

    def __init__(self, config, logger=None):
        # Read the optional parameters that are used by the base class.
        self.min_snr = config.get('min_snr', None)
        self.max_snr = config.get('max_snr', 100)
        self.max_edge_frac = config.get('max_edge_frac', None)
        self.stamp_center_size = config.get('stamp_center_size', 13)
        self.max_mask_pixels = config.get('max_mask_pixels', None)
        self.hsm_size_reject = config.get('hsm_size_reject', 0.)
        self.max_pixel_cut = config.get('max_pixel_cut', None)
        self.reject_where = config.get('reject_where', None)
        self.reserve_frac = config.get('reserve_frac', 0.)
        self.rng = np.random.default_rng(config.get('seed', None))

        if self.hsm_size_reject == 1:
            # Enable True to be equivalent to 10.  True comes in as 1.0, which would be a
            # silly value to use, so it shouldn't be a problem to turn 1.0 -> 10.0.
            self.hsm_size_reject = 10.

    @classmethod
    def process(cls, config_select, objects, logger=None, select_only=False):
        """Parse the select field of the config dict.

        This stage handles three somewhat separate actions:

        1. Select which objects in the input catalog are likely to be stars.
        2. Reject stars according to a number of possible criteria.
        3. Reserve some fraction of the remaining stars to not use for PSF fitting.

        The first step is handled by the derived classes.  There are a number of possible
        algorithms for doing that.  The default select type (Flag) selects objects that
        are not flagged, or if no flag property is specified, then uses all input objects.

        The second and third steps are common to all types and are handled by the base class.

        The following parameters are relevant to steps 2 and 3 and are allowed for all
        select types:

            :min_snr:       The minimum S/N ratio to use.  If an input star is too faint, it is
                            removed from the input list of PSF stars.
            :max_snr:       The maximum S/N ratio to allow for any given star.  If an input star
                            is too bright, it can have too large an influence on the interpolation,
                            so this parameter limits the effective S/N of any single star.
                            Basically, it adds noise to bright stars to lower their S/N down to
                            this value.  [default: 100]
            :max_edge_frac: Cutoff on the fraction of the flux comming from pixels on the edges of
                            the postage stamp. [default: None]
            :stamp_center_size: Distance from center of postage stamp (in pixels) to consider as
                            defining the edge of the stamp for the purpose of the max_edge_fact cut.
                            The default value of 13 is most of the radius of a 32x32 stamp size.
                            If you change stamp_size, you should consider what makes sense here.
                            [default 13].
            :max_mask_pixels: If given, reject stars with more than this many masked pixels
                            (i.e. those with w=0). [default: None]
            :hsm_size_reject: Whether to reject stars with a very different hsm-measured size than
                            the other stars in the input catalog.  (Used to reject objects with
                            neighbors or other junk in the postage stamp.) [default: False]
                            If this is a float value, it gives the number of inter-quartile-ranges
                            to use for rejection relative to the median.  hsm_size_reject=True
                            is equivalent to hsm_size_reject=10.
            :reject_where:  Reject stars based on an arbitrary eval string using variables that
                            are properties of each star (usually input using property_cols).
                            It should evaluate to a bool for a single star or an array of bool
                            if the variables are arrays of property values for all the stars.
                            [default: None]
            :reserve_frac:  Reserve a fraction of the stars from the PSF calculations, so they
                            can serve as fair points for diagnostic testing.  These stars will
                            not be used to constrain the PSF model, but the output files will
                            contain the reserve stars, flagged as such.  Generally 0.2 is a
                            good choice if you are going to use this. [default: 0.]
            :seed:          A seed to use for numpy.random.default_rng, if desired. [default: None]

        .. note::

            The max_snr parameter is not actually a "selection" parameter.  It doesn't change
            what stars are used.  Rather, it adjusts the relative weight that is given to the
            brightest stars (so that they don't dominate the fit).

        :param config_select:   The configuration dict.
        :param objects:         A list of Star instances, which are at this point all potential
                                objects to consider as possible stars.
        :param logger:          A logger object for logging debug info. [default: None]
        :param select_only:     Whether to stop after the primary selection step. [default: False]

        :returns: stars, the subset of objects which are to be considered stars
        """
        import piff

        # Get the class to use for handling the selection
        # Default type is 'Files'
        select_handler_class = getattr(piff, config_select.get('type','Flag') + 'Select')

        # Build handler object
        select_handler = select_handler_class(config_select)

        # Creat a list of Star objects
        stars = select_handler.selectStars(objects, logger)

        if len(stars) == 0:
            raise RuntimeError("No stars were selected.")

        if select_only:
            return stars

        # Reject bad stars
        stars = select_handler.rejectStars(stars, logger)

        if len(stars) == 0:
            raise RuntimeError("All stars were rejected.")

        # Mark the reserve stars
        select_handler.reserveStars(stars, logger)

        return stars

    def selectStars(self, objects, logger=None):
        """Select which of the input objects should be considered stars.

        :param objects:     A list of input objects to be considered as potential stars.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        raise NotImplementedError("Derived classes must define the selectStars function")

    def reserveStars(self, stars, logger=None):
        """Mark some of the stars as reserve stars.

        This operates on the star list in place, adding the property ``is_reserve``
        to each star (only if some stars are being reserved).

        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        if self.reserve_frac == 0:
            return

        logger = galsim.config.LoggerWrapper(logger)

        # Apply the reserve separately on each ccd, so they each reserve 20% of their stars
        # (or whatever fraction).  We wouldn't want to accidentally reserve all the stars on
        # one of the ccds by accident, for instance.
        chipnums = np.unique(list(s['chipnum'] for s in stars))
        all_stars = [ [s for s in stars if s['chipnum'] == chipnum] for chipnum in chipnums]
        nreserve_all = 0
        for chip_stars in all_stars:
            # Mark a fraction of the stars as reserve stars
            nreserve = int(self.reserve_frac * len(chip_stars))  # round down
            nreserve_all += nreserve
            logger.info("Reserve %s of %s (reserve_frac=%s) input stars on chip %s",
                        nreserve, len(stars), self.reserve_frac, chip_stars[0]['chipnum'])
            reserve_list = self.rng.choice(len(chip_stars), nreserve, replace=False)
            for i, star in enumerate(chip_stars):
                star.data.properties['is_reserve'] = i in reserve_list
        logger.warning("Reserved %s of %s total stars", nreserve_all, len(stars))

    def rejectStars(self, stars, logger=None):
        """Reject some nominal stars that may not be good exemplars of the PSF.

        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: The subset of the input list that passed the rejection cuts.
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.info('start rejectStars: %s',len(stars))

        if self.max_edge_frac is not None and len(stars) > 0:
            stamp_size = stars[0].image.array.shape[0]
            cen = (stamp_size-1.)/2.            # index at center of array.  May be half-integral.
            i,j = np.ogrid[0:stamp_size,0:stamp_size]
            edge_mask = (i-cen)**2 + (j-cen)**2 > self.stamp_center_size**2
        else:
            edge_mask = None

        good_stars = []
        for star in stars:
            # Here we remove stars that have been at least partially covered by a mask
            # and thus have weight exactly 0 in at least a certain number of pixels of their
            # postage stamp
            if self.max_mask_pixels is not None:
                n_masked = np.prod(star.weight.array.shape) - np.count_nonzero(star.weight.array)
                if n_masked >= self.max_mask_pixels:
                    logger.info("Star at position %f,%f has %i masked pixels, ",
                                star.image_pos.x, star.image_pos.y, n_masked)
                    logger.info("Skipping this star.")
                    continue

            # Check the snr and limit it if appropriate
            snr = calculateSNR(star.image, star.weight)
            logger.debug("SNR = %f",snr)
            if self.min_snr is not None and snr < self.min_snr:
                logger.info("Skipping star at position %f,%f with snr=%f.",
                            star.image_pos.x, star.image_pos.y, snr)
                continue
            if self.max_snr > 0 and snr > self.max_snr:
                factor = (self.max_snr / snr)**2
                logger.debug("Scaling noise by factor of %f to achieve snr=%f",
                             factor, self.max_snr)
                star.data.weight *= factor
                snr = self.max_snr
                logger.debug("SNR => %f",snr)
            star.data.properties['snr'] = snr

            # Reject stars with lots of flux near the edge of the stamp.
            if self.max_edge_frac is not None and self.max_edge_frac < 1:
                flux = np.sum(star.image.array)
                try:
                    flux_extra = np.sum(star.image.array[edge_mask])
                    flux_frac = flux_extra / flux
                except IndexError:
                    logger.info("Star at position %f,%f overlaps the edge of the image and "+
                                "max_edge_frac cut is set.",
                                star.image_pos.x, star.image_pos.y)
                    logger.info("Skipping this star.")
                    continue
                if flux_frac > self.max_edge_frac:
                    logger.info("Star at position %f,%f fraction of flux near edge of stamp "+
                                "exceeds cut: %f > %f",
                                star.image_pos.x, star.image_pos.y,
                                flux_frac, self.max_edge_frac)
                    logger.info("Skipping this star.")
                    continue

            if self.reject_where is not None:
                # Use the eval_where function of PropertiesSelect
                reject = PropertiesSelect.eval_where([star], self.reject_where, logger=logger)
                if reject:
                    logger.info("Skipping star at position %f,%f due to reject_where",
                                star.image_pos.x, star.image_pos.y)
                    logger.debug("reject_where string: %s",self.reject_where)
                    logger.debug("star properties = %s",star.data.properties)
                    continue

            # Add Poisson noise now.  It's not a rejection step, but it's something we want
            # to do to all the stars at the start, so they have the right noise level.
            # We didn't do it earlier for efficiency reasons, in case the full set of objects
            # included lots of non-stars.
            star = star.addPoisson()

            good_stars.append(star)

        # Calculate the hsm size for each star and throw out extreme outliers.
        if self.hsm_size_reject != 0:
            sigma = [star.hsm[3] for star in good_stars]
            med_sigma = np.median(sigma)
            iqr_sigma = scipy.stats.iqr(sigma)
            logger.debug("Doing hsm sigma rejection.")
            while np.max(np.abs(sigma - med_sigma)) > self.hsm_size_reject * iqr_sigma:
                logger.debug("median = %s, iqr = %s, max_diff = %s",
                                med_sigma, iqr_sigma, np.max(np.abs(sigma-med_sigma)))
                k = np.argmax(np.abs(sigma-med_sigma))
                logger.debug("remove k=%d: sigma = %s, pos = %s",k,sigma[k],good_stars[k].image_pos)
                del sigma[k]
                del good_stars[k]
                med_sigma = np.median(sigma)
                iqr_sigma = scipy.stats.iqr(sigma)

        # Reject based on a maximum pixel value, being careful to not induce a bias that smaller
        # stars of a given flux will have higher max pixel values.
        if self.max_pixel_cut is not None:
            # find median max_pixel/flux ratio, and use to set flux cut equivalent
            # on average to max_pixel_cut
            max_pixel = np.array([np.max(star.data.image.array) for star in good_stars])
            # If subtracted sky, need to add it back.
            max_pixel += np.array([star.data.properties.get('sky',0) for star in good_stars])
            flux = np.array([star.hsm[0] for star in good_stars])
            # Low S/N stars have a max_pixel that is more due to noise than signal.
            # Exclude S/N < 40 and also use a median of the bright ones to be more robust
            # to noisy max pixel values.
            bright = np.where([(star.data.properties['snr'] > 40) for star in good_stars])[0]
            logger.debug("Num bright = %s",len(bright))
            if len(bright) > 0:
                ratio = np.median(max_pixel[bright] / flux[bright])
                flux_cut = self.max_pixel_cut / ratio
                logger.debug("max_pixel/flux ratio = %.4f, flux_cut is %.1f", ratio, flux_cut)
                logger.info("Rejected %d stars for having a flux > %.1f, "
                            "which implies max_pixel > ~%s",
                            np.sum(flux>=flux_cut), flux_cut, self.max_pixel_cut)
                good_stars = [s for f,s in zip(flux,good_stars) if f < flux_cut]

        logger.warning("Rejected a total of %d stars out of %s total candidates",
                       len(stars) - len(good_stars), len(stars))
        return good_stars


class FlagSelect(Select):
    """An Select handler that picks stars according to a flag column in the input catalog.

    The Flag type uses the following parameters, all optional.

        :flag_name:     The name of the flag property (typically the column name in the
                        input file) to use for selecting stars. [default: None]
        :use_flag:      The flag indicating which items to use. [default: None]
                        Items with flag & use_flag != 0 will be used.
        :skip_flag:     The flag indicating which items not to use. [default: -1]
                        Items with flag & skip_flag != 0 will be skipped.

    The default behavior if flag_name is not given is to consier all input objects as stars.
    If flag_name is given, but the others are not, then it selects all objects with flag=0.
    Otherwise, it will select according to the prescriptions given above.

    :param config:      The configuration dict used to define the above parameters.
                        (Normally the 'select' field in the overall configuration dict).
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, config, logger=None):
        super(FlagSelect, self).__init__(config, logger)

        opt = {
            'flag_name': str,
            'skip_flag': int,
            'use_flag': int,
        }
        params = galsim.config.GetAllParams(config, config, opt=opt, ignore=Select.base_keys)[0]
        self.flag_name = params.get('flag_name', None)
        self.skip_flag = params.get('skip_flag', -1)
        self.use_flag = params.get('use_flag', None)

    def selectStars(self, objects, logger=None):
        """Select which of the input objects should be considered stars.

        :param objects:     A list of input objects to be considered as potential stars.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = galsim.config.LoggerWrapper(logger)

        if self.flag_name is None:
            logger.info("Using all input objects as stars")
            return objects

        logger.info("Selecting stars according to flag %r", self.flag_name)
        try:
            flag_array = np.array([obj[self.flag_name] for obj in objects])
        except KeyError:
            raise ValueError("flag_name = {} is invalid.".format(self.flag_name))

        # Follow the same logic as we used in InputFiles for selecting on an overall flag.
        if self.use_flag is not None:
            select = InputFiles._flag_select(flag_array, self.use_flag) != 0
            if self.skip_flag != -1:
                select &= InputFiles._flag_select(flag_array, self.skip_flag) == 0
        else:
            select = InputFiles._flag_select(flag_array, self.skip_flag) == 0

        stars = [obj for use, obj in zip(select, objects) if use]
        logger.info("Seleced %d stars from %d total candidates.", len(stars), len(objects))
        return stars

class PropertiesSelect(Select):
    """A Select handler that picks stars according to any property or combination of properties
    in the input catalog.

    Parse the config dict (Normally the 'where' field in the overall configuration dict).

    The Properties type uses the following parameter, which is required.

        :where:     A string to be evaluated, which is allowed to use any properties of
                    the stars as variables.  It should evaluate to a bool for a single object
                    or an array of bool if the variables are arrays of property values for all
                    the objects.

    :param config:      The configuration dict used to define the above parameters.
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, config, logger=None):
        super(PropertiesSelect, self).__init__(config, logger)

        req = { 'where': str }
        params = galsim.config.GetAllParams(config, config, req=req, ignore=Select.base_keys)[0]
        self.where = params['where']

    @classmethod
    def eval_where(cls, objects, where, logger=None):
        """Perform the evaluation of a "where" string using the properties of objects.

        Used by both PropertiesSelect and the reject_where option.
        """
        logger = galsim.config.LoggerWrapper(logger)
        # Build appropriate locals and globals for the eval statement.
        gdict = globals().copy()
        # Import some likely packages in case needed.
        exec('import numpy', gdict)
        exec('import numpy as np', gdict)
        exec('import math', gdict)

        ldict = {}
        for prop_name in objects[0].data.properties.keys():
            ldict[prop_name] = np.array([obj[prop_name] for obj in objects])

        try:
            select = eval(where, gdict, ldict)
        except Exception as e:
            logger.info("Caught exception trying to evaluate where string")
            logger.info("%r",e)
            logger.info("Trying slower non-numpy array method")
            select = []
            for obj in objects:
                ldict = {}
                for prop_name in obj.data.properties.keys():
                    ldict[prop_name] = obj[prop_name]
                select.append(eval(where, gdict, ldict))
        return select

    def selectStars(self, objects, logger=None):
        """Select which of the input objects should be considered stars.

        :param objects:     A list of input objects to be considered as potential stars.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.info("Selecting stars according to %r", self.where)

        select = self.eval_where(objects, self.where)
        logger.debug("select = %s",select)

        stars = [obj for use, obj in zip(select, objects) if use]
        logger.info("Seleced %d stars from %d total candidates.", len(stars), len(objects))
        return stars
