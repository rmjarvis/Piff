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

from __future__ import print_function
from past.builtins import basestring
import numpy as np
import glob
import os


class Input(object):
    """The base class for handling inputs for building a Piff model.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """

    @classmethod
    def process(cls, config_input, logger=None):
        """Parse the input field of the config dict.

        :param config_input:    The configuration dict.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: stars, wcs, pointing

        stars is a list of Star instances with the initial data.
        wcs is a dict of WCS solutions indexed by chipnum.
        pointing is either a galsim.CelestialCoord or None.
        """
        import piff

        # Get the class to use for handling the input data
        # Default type is 'Files'
        input_handler_class = getattr(piff, 'Input' + config_input.get('type','Files'))

        # Build handler object
        input_handler = input_handler_class(config_input, logger)

        # Creat a lit of StarData objects
        stars = input_handler.makeStars(logger)

        # Get the wcs for all the input chips
        wcs = input_handler.getWCS(logger)

        # Get the pointing (the coordinate center of the field of view)
        pointing = input_handler.getPointing(logger)

        return stars, wcs, input_handler.pointing

    def makeStars(self, logger=None):
        """Process the input images and star data, cutting out stamps for each star along with
        other relevant information.

        The base class implementation expects the derived class to have appropriately set the
        following attributes:

            :stamp_size:    The size of the postage stamp to use for the cutouts
            :x_col:         The name of the column in the catalogs to use for the x position.
            :y_col:         The name of the column in the catalogs to use for the y position.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        import galsim
        import piff
        logger = galsim.config.LoggerWrapper(logger)

        stars = []
        if len(self.chipnums) == 1:
            logger.debug("Making star list")
        else:
            logger.debug("Making star list from %d catalogs", len(self.chipnums))
        for image_num in range(len(self.chipnums)):
            image = self.images[image_num]
            wt = self.weight[image_num]
            image_pos = self.image_pos[image_num]
            sky = self.sky[image_num]
            gain = self.gain[image_num]
            chipnum = self.chipnums[image_num]
            logger.info("Processing catalog %s with %d stars",chipnum,len(image_pos))
            nstars_in_image = 0
            for k in range(len(image_pos)):
                x = image_pos[k].x
                y = image_pos[k].y
                icen = int(x+0.5)
                jcen = int(y+0.5)
                half_size = self.stamp_size // 2
                bounds = galsim.BoundsI(icen+half_size-self.stamp_size+1, icen+half_size,
                                        jcen+half_size-self.stamp_size+1, jcen+half_size)
                if not image.bounds.includes(bounds):
                    bounds = bounds & image.bounds
                    if not bounds.isDefined():
                        logger.warning("Star at position %f,%f is off the edge of the image.  "
                                       "Skipping this star.", x, y)
                        continue
                    if self.use_partial:
                        logger.info("Star at position %f,%f overlaps the edge of the image.  "
                                    "Using smaller than the full stamp size: %s", x, y, bounds)
                    else:
                        logger.warning("Star at position %f,%f overlaps the edge of the image.  "
                                       "Skipping this star.", x, y)
                        continue
                stamp = image[bounds]
                props = { 'chipnum' : chipnum,
                          'gain' : gain[k] }
                if sky is not None:
                    logger.debug("Subtracting off sky = %f", sky[k])
                    logger.debug("Median pixel value = %f", np.median(stamp.array))
                    stamp = stamp - sky[k]  # Don't change the original!
                    props['sky'] = sky[k]
                wt_stamp = wt[bounds]

                # if a star is totally masked, then don't add it!
                if np.all(wt_stamp.array == 0):
                    logger.warning("Star at position %f,%f is completely masked."%(x,y))
                    logger.warning("Skipping this star.")
                    continue

                # Check the snr and limit it if appropriate
                snr = self.calculateSNR(stamp, wt_stamp)
                logger.debug("SNR = %f",snr)
                if self.min_snr is not None and snr < self.min_snr:
                    logger.info("Skipping star at position %f,%f with snr=%f."%(x,y,snr))
                    continue
                if self.max_snr > 0 and snr > self.max_snr:
                    factor = (self.max_snr / snr)**2
                    logger.debug("Scaling noise by factor of %f to achieve snr=%f",
                                 factor, self.max_snr)
                    wt_stamp = wt_stamp * factor
                    snr = self.max_snr
                props['snr'] = snr

                pos = galsim.PositionD(x,y)
                data = piff.StarData(stamp, pos, weight=wt_stamp, pointing=self.pointing,
                                     properties=props)
                star = piff.Star(data, None)
                g = gain[k]
                if g is not None:
                    logger.debug("Adding Poisson noise to weight map according to gain=%f",g)
                    star = star.addPoisson(gain=g)
                stars.append(star)
                nstars_in_image += 1
        logger.warning("Read a total of %d stars from %d image%s",len(stars),len(self.images),
                       "s" if len(self.images) > 1 else "")

        return stars

    @staticmethod
    def calculateSNR(image, weight):
        """Calculate the signal-to-noise of a given image.

        :param image:       The stamp image for a star
        :param weight:      The weight image for a star
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the SNR value.
        """
        # The S/N value that we use will be the weighted total flux where the weight function
        # is the star's profile itself.  This is the maximum S/N value that any flux measurement
        # can possibly produce, which will be closer to an in-practice S/N than using all the
        # pixels equally.
        #
        # F = Sum_i w_i I_i^2
        # var(F) = Sum_i w_i^2 I_i^2 var(I_i)
        #        = Sum_i w_i I_i^2             <--- Assumes var(I_i) = 1/w_i
        #
        # S/N = F / sqrt(var(F))
        I = image.array
        w = weight.array
        mask = np.isfinite(I) & np.isfinite(w)
        flux = (w[mask]*I[mask]**2).sum(dtype=float)
        if flux <= 0.:
            return 0.
        else:
            return flux**0.5

    def getWCS(self, logger=None):
        """Get the WCS solutions for all the chips in the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a dict of WCS solutions (galsim.BaseWCS instances) indexed by chipnum
        """
        return { chipnum : im.wcs for im, chipnum in zip(self.images, self.chipnums) }

    def getPointing(self, logger=None):
        """Get the pointing coordinate of the (noinal) center of the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim.CelestialCoord of the pointing direction.
        """
        return self.pointing


class InputFiles(Input):
    """An Input handler than just takes a list of image files and catalog files.
    """
    def __init__(self, config, logger=None):
        """
        Parse the input config dict (Normally the 'input' field in the overall configuration dict).

        The two required fields in the input dict are:

            :image_file_name:   The file name(s) of the input image(s).
            :cat_file_name:     The file name(s) of the input catalog(s).

        There are a number of ways to specify these file names.

        1. A string giving a single file name.  e.g.::

                image_file_name: image.fits
                cat_file_name: input_cat.fits

        2. A list of several file names.  e.g.::

                image_file_name: [image_00.fits, image_01.fits, image_02.fits]
                cat_file_name: [input_cat_00.fits, input_cat_01.fits, input_cat_02.fits]

        3. A string that glob can recognize to list several file names.  e.g.::

                image_file_name: image_*.fits
                cat_file_name: input_cat_*.fits

        4. A dict parseable as a string value according to the GalSim configuration parsing types.
           In this case, you also must specify nimages to say how many file names to generate
           in this way.  e.g.::

                nimages: 20
                image_file_name:
                    type: FormattedStr
                    format: image_%03d_%02d.fits.fz
                    items:
                        - { type : Sequence, first: 0, repeat: 4 }  # Exposure number
                        - { type : Sequence, first: 1, last: 4 }    # Chip number
                cat_file_name:
                    type: Eval
                    str: "image_file_name.replace('image','input_cat')"
                    simage_file_name: '@input.image_file_name'

           See the description of the GalSim config parser for more details about the various
           types that are valid here.

                `https://github.com/GalSim-developers/GalSim/wiki/Config-Values`_

        There are many other optional parameters, which help govern how the input files are
        read or interporeted:

            :chipnum:       The id number of this chip used to reference this image [default:
                            image_num]

            :image_hdu:     The hdu to use in the image files. [default: None, which means use
                            either 0 or 1 as typical given the compression sceme of the file]
            :weight_hdu:    The hdu to use for weight images. [default: None, which means a weight
                            image with all 1's will be automatically created]
            :badpix_hdu:    The hdu to use for badpix images. Pixels with badpix != 0 will be given
                            weight == 0. [default: None]
            :noise:         Rather than a weight image, provide the noise variance in the image.
                            (Useful for simulations where this is a known value.) [default: None]

            :cat_hdu:       The hdu to use in the catalog files. [default: 1]
            :x_col:         The name of the X column in the input catalogs. [default: 'x']
            :y_col:         The name of the Y column in the input catalogs. [default: 'y']
            :ra_col:        (Alternative to x_col, y_col) The name of a right ascension column in
                            the input catalogs.  Will use the WCS to find (x,y) [default: None]
            :dec_col:       (Alternative to x_col, y_col) The name of a declination column in
                            the input catalogs.  Will use the WCS to find (x,y) [default: None]
            :flag_col:      The name of a flag column in the input catalogs. [default: None]
                            By default, this will skip any objects with flag != 0, but see
                            skip_flag and use_flag for other possible meanings for how the
                            flag column can be used to select stars.
            :skip_flag:     The flag indicating which items to not use. [default: -1]
                            Items with flag & skip_flag != 0 will be skipped.
            :use_flag:      The flag indicating which items to use. [default: None]
                            Items with flag & use_flag == 0 will be skipped.
            :sky_col:       The name of a column with sky values. [default: None]
            :gain_col:      The name of a column with gain values. [default: None]
            :sky:           The sky level to subtract from the image values. [default: None]
                            Note: It is an error to specify both sky and sky_col. If both are None,
                            no sky level will be subtracted off.
            :gain:          The gain to use for adding Poisson noise to the weight map.  [default:
                            None] It is an error for both gain and gain_col to be specified.
                            If both are None, then no additional noise will be added to account
                            for the Poisson noise from the galaxy flux.
            :min_snr:       The minimum S/N ratio to use.  If an input star is too faint, it is
                            removed from the input list of PSF stars.
            :max_snr:       The maximum S/N ratio to allow for any given star.  If an input star
                            is too bright, it can have too large an influence on the interpolation,
                            so this parameter limits the effective S/N of any single star.
                            Basically, it adds noise to bright stars to lower their S/N down to
                            this value.  [default: 100]
            :use_partial:   Whether to use stars whose postage stamps are only partially on the
                            full image.  [default: False]
            :nstars:        Stop reading the input file at this many stars.  (This is applied
                            separately to each input catalog.)  [default: None]

            :wcs:           Normally, the wcs is automatically read in when reading the image.
                            However, this parameter allows you to optionally provide a different
                            WCS.  It should be defined using the same style as a wcs object
                            in GalSim config files. [defulat: None]

        The above values are parsed separately for each input image/catalog.  In addition, there
        are a couple other parameters that are just parsed once:

            :stamp_size:    The size of the postage stamps to use for the cutouts.  Note: some
                            stamps may be smaller than this if the star is near a chip boundary.
                            [default: 32]
            :ra, dec:       The RA, Dec of the telescope pointing. [default: None; See
                            :setPointing: for details about how this can be specified]


        :param config:      The configuration dict used to define the above parameters.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import galsim
        import copy
        logger = galsim.config.LoggerWrapper(logger)

        req = { 'image_file_name': str,
                'cat_file_name': str,
              }
        opt = {
                'dir' : str,
                'chipnum' : int,
                'x_col' : str,
                'y_col' : str,
                'ra_col' : str,
                'dec_col' : str,
                'ra_units' : str,
                'dec_units' : str,
                'sky_col' : str,
                'gain_col' : str,
                'flag_col' : str,
                'skip_flag' : int,
                'use_flag' : int,
                'image_hdu' : int,
                'weight_hdu' : int,
                'badpix_hdu' : int,
                'cat_hdu' : int,
                'invert_weight' : bool,
                'remove_signal_from_weight' : bool,
                'stamp_size' : int,
                'gain' : str,
                'min_snr' : float,
                'max_snr' : float,
                'use_partial' : bool,
                'sky' : str,
                'noise' : float,
                'nstars' : int,
              }
        ignore = [ 'nimages', 'ra', 'dec', 'wcs' ]  # These are parsed separately

        # We're going to change the config dict a bit. Make a copy so we don't mess up the
        # user's original dict (in case they care).
        config = copy.deepcopy(config)

        # In GalSim, the base dict holds additional parameters that may be of use.
        # Here, we just make a dict with a few values that could be relevant.
        base = { 'input' : config,
                 'index_key' : 'image_num',
               }

        # Convert options 2 and 3 above into option 4.  (1 is also parseable by GalSim's config.)
        nimages = None
        image_list = None
        cat_list = None
        dir = None

        if 'nimages' in config:
            nimages = galsim.config.ParseValue(config, 'nimages', base, int)[0]
            if nimages < 1:
                raise ValueError('input.nimages must be >= 1')

        # Deal with dir here, since sometimes we need to have it already atteched for glob
        # to work.
        if 'dir' in config:
            dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
            del config['dir']

        if 'image_file_name' not in config:
            raise AttributeError('Attribute image_file_name is required')
        elif isinstance(config['image_file_name'], list):
            image_list = config['image_file_name']
            if len(image_list) == 0:
                raise ValueError("image_file_name may not be an empty list")
            if dir is not None:
                image_list = [os.path.join(dir, n) for n in image_list]
        elif isinstance(config['image_file_name'], basestring):
            image_file_name = config['image_file_name']
            if dir is not None:
                image_file_name = os.path.join(dir, image_file_name)
            image_list = sorted(glob.glob(image_file_name))
            if len(image_list) == 0:
                raise ValueError("No files found corresponding to "+config['image_file_name'])
        elif not isinstance(config['image_file_name'], dict):
            raise ValueError("image_file_name should be either a dict or a string")

        if image_list is not None:
            logger.debug('image_list = %s',image_list)
            if nimages is not None and nimages != len(image_list):
                raise ValueError("nimages = %s doesn't match length of image_file_name list (%d)"%(
                        config['nimages'], len(image_list)))
            nimages = len(image_list)
            logger.debug('nimages = %d',nimages)
            config['image_file_name'] = {
                'type' : 'List',
                'items' : image_list
            }

        if 'cat_file_name' not in config:
            raise AttributeError('Attribute cat_file_name is required')
        elif isinstance(config['cat_file_name'], list):
            cat_list = config['cat_file_name']
            if len(cat_list) == 0:
                raise ValueError("cat_file_name may not be an empty list")
            if dir is not None:
                cat_list = [os.path.join(dir, n) for n in cat_list]
        elif isinstance(config['cat_file_name'], basestring):
            cat_file_name = config['cat_file_name']
            if dir is not None:
                cat_file_name = os.path.join(dir, cat_file_name)
            cat_list = sorted(glob.glob(cat_file_name))
            if len(cat_list) == 0:
                raise ValueError("No files found corresponding to "+config['cat_file_name'])
        elif not isinstance(config['cat_file_name'], dict):
            raise ValueError("cat_file_name should be either a dict or a string")

        if cat_list is not None:
            logger.debug('cat_list = %s',cat_list)
            if nimages is not None and nimages != len(cat_list):
                raise ValueError("nimages = %s doesn't match length of cat_file_name list (%d)"%(
                        nimages, len(cat_list)))
            nimages = len(cat_list)
            logger.debug('nimages = %d',nimages)
            config['cat_file_name'] = {
                'type' : 'List',
                'items' : cat_list
            }

        if nimages is None:
            raise ValueError('input.nimages is required if not using a list or simple string for ' +
                             'file names')

        self.chipnums = list(range(nimages))
        self.stamp_size = int(config.get('stamp_size', 32))
        self.images = []
        self.weight = []
        self.image_pos = []
        self.sky = []
        self.gain = []
        self.image_file_name = []
        self.cat_file_name = []

        logger.info("Reading in %d images",nimages)
        for image_num in range(nimages):

            # This changes for each input image.
            base['image_num'] = image_num

            logger.debug("config = %s", config)
            params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)[0]
            logger.debug("image_num = %d: params = %s", image_num, params)

            # Update the chipnum if not just using image_num
            if 'chipnum' in params:
                self.chipnums[image_num] = params['chipnum']

            # Read the image
            image_file_name = params['image_file_name']
            image_hdu = params.get('image_hdu', None)
            weight_hdu = params.get('weight_hdu', None)
            badpix_hdu = params.get('badpix_hdu', None)
            noise = params.get('noise', None)

            image, weight = self.readImage(
                    image_file_name, image_hdu, weight_hdu, badpix_hdu, noise, logger)

            if config.get('invert_weight', False):
                weight.invertSelf()

            # Update the wcs if necessary
            if 'wcs' in config:
                wcs = galsim.config.BuildWCS(config, 'wcs', base, logger)
                image.wcs = wcs

            self.image_file_name.append(image_file_name)
            self.images.append(image)

            # Read the catalog
            cat_file_name = params['cat_file_name']
            cat_hdu = params.get('cat_hdu', None)
            x_col = params.get('x_col', 'x')
            y_col = params.get('y_col', 'y')
            ra_col = params.get('ra_col', None)
            dec_col = params.get('dec_col', None)
            ra_units = params.get('ra_units', 'deg')
            dec_units = params.get('dec_units', 'deg')
            flag_col = params.get('flag_col', None)
            skip_flag = params.get('skip_flag', -1)
            use_flag = params.get('use_flag', None)
            sky_col = params.get('sky_col', None)
            gain_col = params.get('gain_col', None)
            sky = params.get('sky', None)
            gain = params.get('gain', None)
            nstars = params.get('nstars', None)

            image_pos, sky, gain = self.readStarCatalog(
                    cat_file_name, cat_hdu, x_col, y_col,
                    ra_col, dec_col, ra_units, dec_units, image.wcs,
                    flag_col, skip_flag, use_flag, sky_col, gain_col,
                    sky, gain, nstars, image_file_name, logger)
            # Check for objects well off the edge.  We won't use them.
            big_bounds = image.bounds.expand(self.stamp_size)
            image_pos = [ pos for pos in image_pos if big_bounds.includes(pos) ]

            if config.get('remove_signal_from_weight', False):
                # Subtract off the mean sky, since this isn't part of the "signal" we want to
                # remove from the weights.
                if sky is None:
                    signal = image
                else:
                    signal = image - np.mean(sky)
                # For the gain, either all are None or all are values.
                if gain[0] is None:
                    # If None, then we want to estimate the gain from the weight image.
                    weight, g = self._removeSignalFromWeight(signal, weight)
                    gain = [g for _ in gain]
                    logger.warning("Empirically determined gain = %f",g)
                else:
                    # If given, use the mean gain when removing the signal.
                    # This isn't quite right, but hopefully the gain won't vary too much for
                    # different objects, so it should be close.
                    weight, _ = self._removeSignalFromWeight(signal, weight, gain=np.mean(gain))
                logger.info("Removed signal from weight image.")

            self.weight.append(weight)
            self.cat_file_name.append(cat_file_name)
            self.image_pos.append(image_pos)
            self.sky.append(sky)
            self.gain.append(gain)

        self.min_snr = config.get('min_snr', None)
        self.max_snr = config.get('max_snr', 100)
        self.use_partial = config.get('use_partial', False)

        # Finally, set the pointing coordinate.
        ra = config.get('ra',None)
        dec = config.get('dec',None)
        self.setPointing(ra, dec, logger)


    @staticmethod
    def _removeSignalFromWeight(image, weight, gain=None):
        """Remove the image signal from the weight map.

        :param image:   The image to use as the signal
        :param weight:  The weight image.
        :param gain:    Optionally, the gain to use as the proportionality relation.
                        If gain is None, then it will be estimated automatically and returned.
                        [default: None]

        :returns: newweight, gain
        """
        signal = image.array
        variance = 1./weight.array

        use = (weight.array != 0.) & np.isfinite(signal)

        if gain is None:
            fit = np.polyfit(signal[use].flatten(), variance[use].flatten(), deg=1)
            gain = 1./fit[0]  # fit is [ 1/gain, sky_var ]

        variance[use] -= signal[use] / gain

        newweight = weight.copy()
        newweight.array[use] = 1. / variance[use]
        return newweight, gain


    def readImage(self, image_file_name, image_hdu, weight_hdu, badpix_hdu, noise, logger):
        """Read in the image and weight map (or make one if no weight information is given

        :param image_file_name: The name of the file to read.
        :param image_hdu:       The hdu of the main image.
        :param weight_hdu:      The hdu of the weight image (if any).
        :param badpix_hdu:      The hdu of the bad pixel mask (if any).
        :param noise:           A constant noise value to use in lieu of a weight map.
        :param logger:          A logger object for logging debug info.

        :returns: image, weight
        """
        import galsim
        # Read in the image
        logger.warning("Reading image file %s",image_file_name)
        image = galsim.fits.read(image_file_name, hdu=image_hdu)

        # Either read in the weight image, or build a dummy one
        if weight_hdu is not None:
            logger.info("Reading weight image from hdu %d.", weight_hdu)
            weight = galsim.fits.read(image_file_name, hdu=weight_hdu)
            if np.all(weight.array == 0):
                logger.error("According to the weight mask in %s, all pixels have zero weight!",
                             image_file_name)
            if np.any(weight.array < 0):
                logger.error("Warning: weight map has invalid negative-valued pixels. "+
                             "Taking them to be 0.0")
                weight.array[weight.array < 0] = 0.
        elif noise is not None:
            logger.debug("Making uniform weight image based on noise variance = %f", noise)
            weight = galsim.ImageF(image.bounds, init_value=1./noise)
        else:
            logger.debug("Making trivial (wt==1) weight image")
            weight = galsim.ImageF(image.bounds, init_value=1)

        # If requested, set wt=0 for any bad pixels
        if badpix_hdu is not None:
            logger.info("Reading badpix image from hdu %d.", badpix_hdu)
            badpix = galsim.fits.read(image_file_name, hdu=badpix_hdu)
            # The badpix image may be offset by 32768 from the true value.
            # If so, subtract it off.
            if np.any(badpix.array > 32767):  # pragma: no cover
                logger.debug('min(badpix) = %s',np.min(badpix.array))
                logger.debug('max(badpix) = %s',np.max(badpix.array))
                logger.debug("subtracting 32768 from all values in badpix image")
                badpix -= 32768
            if np.any(badpix.array < -32767):  # pragma: no cover
                logger.debug('min(badpix) = %s',np.min(badpix.array))
                logger.debug('max(badpix) = %s',np.max(badpix.array))
                logger.debug("adding 32768 to all values in badpix image")
                badpix += 32768
            if np.all(badpix.array != 0):  # pragma: no cover
                logger.error("According to the bad pixel array in %s, all pixels are masked!",
                             image_file_name)
            weight.array[badpix.array != 0] = 0
        return image, weight

    @staticmethod
    def _flag_select(col, flag):
        if len(col.shape) == 1:
            # Then just treat this as a straightforward bitmask.
            return col & flag
        else:
            # Then treat this as an array of bools rather than a bitmask
            mask = np.zeros(col.shape[0], dtype=bool)
            for bit in range(col.shape[1]):  # pragma: no branch
                if flag % 2 == 1:
                    mask |= col[:,bit]
                flag = flag // 2
                if flag == 0: break
            return mask

    def readStarCatalog(self, cat_file_name, cat_hdu, x_col, y_col,
                        ra_col, dec_col, ra_units, dec_units, wcs,
                        flag_col, skip_flag, use_flag, sky_col, gain_col,
                        sky, gain, nstars, image_file_name, logger):
        """Read in the star catalogs and return lists of positions for each star in each image.

        :param cat_file_name:   The name of the catalog file to read in.
        :param cat_hdu:         The hdu to use.
        :param x_col:           The name of the column with x values.
        :param y_col:           The name of the column with y values.
        :param ra_col:          The name of a column with RA values.
        :param dec_col:         The name of a column with Dec values.
        :param ra_units:        The units of the ra column.
        :param dec_units:       The units of the dec column.
        :param wcs:             The WCS to use to convert from Ra,Dec -> x,y.
        :param flag_col:        The name of a column with flag values.
        :param skip_flag:       The flag indicating which items to not use. [default: -1]
                                Items with flag & skip_flag != 0 will be skipped.
        :param use_flag:        The flag indicating which items to use. [default: None]
                                Items with flag & use_flag == 0 will be skipped.
        :param sky_col:         A column with sky (background) levels.
        :param gain_col:        A column with gain values.
        :param sky:             Either a float value for the sky to use for all objects or a str
                                keyword to read a value from the FITS header.
        :param gain:            Either a float value for the gain to use for all objects or a str
                                keyword to read a value from the FITS header.
        :param nstars:          Optionally a maximum number of stars to use.
        :param image_file_name: The image file name in case needed for header values.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: lists image_pos, sky, gain
        """
        import fitsio
        import galsim

        # Read in the star catalog
        logger.warning("Reading star catalog %s.",cat_file_name)
        cat = fitsio.read(cat_file_name, cat_hdu)

        if flag_col is not None:
            if flag_col not in cat.dtype.names:
                raise ValueError("flag_col = %s is not a column in %s"%(flag_col,cat_file_name))
            col = cat[flag_col]
            if len(col.shape) == 2:
                logger.warning("Flag col (%s) is multidimensional.  Treating as an array of bool",
                               flag_col)
            if use_flag is not None:
                # Remove any objects with flag & use_flag == 0
                mask = self._flag_select(col, use_flag) == 0
                logger.info("Removing objects with flag (col %s) & %d == 0",flag_col,use_flag)
                if skip_flag != -1:
                    mask |= self._flag_select(col, skip_flag) != 0
                    logger.info("Removing objects with flag (col %s) & %d != 0",flag_col,skip_flag)
            else:
                # Remove any objects with flag & skip_flag != 0
                mask = self._flag_select(col, skip_flag) != 0
                if skip_flag == -1:
                    logger.info("Removing objects with flag (col %s) != 0",flag_col)
                else:
                    logger.info("Removing objects with flag (col %s) & %d != 0",flag_col,skip_flag)
            cat = cat[mask == 0]

        # Limit to nstars objects
        if nstars is not None and nstars < len(cat):
            logger.info("Limiting to %d stars for %s",nstars,cat_file_name)
            cat = cat[:nstars]

        # Make the list of positions:
        if ra_col is not None or dec_col is not None:
            if ra_col is None or dec_col is None:
                raise ValueError("ra_col and dec_col are both required if one is provided.")
            ra_values = cat[ra_col]
            dec_values = cat[dec_col]
            ra_units = galsim.AngleUnit.from_name(ra_units)
            dec_units = galsim.AngleUnit.from_name(dec_units)
            def safe_to_image(wcs, ra, dec):
                try:
                    return wcs.toImage(galsim.CelestialCoord(ra*ra_units, dec*dec_units))
                except galsim.GalSimError:  # pragma: no cover
                    # If the ra,dec is way off the image, this might fail to converge.
                    # In this case return something clearly not on an image so it gets
                    # excluded during the bounds check.
                    return galsim.PositionD(1.e99, 1.e99)
            image_pos = [ safe_to_image(wcs,ra,dec) for ra,dec in zip(ra_values, dec_values) ]
        else:
            if x_col not in cat.dtype.names:
                raise ValueError("x_col = %s is not a column in %s"%(x_col,cat_file_name))
            if y_col not in cat.dtype.names:
                raise ValueError("y_col = %s is not a column in %s"%(y_col,cat_file_name))
            x_values = cat[x_col]
            y_values = cat[y_col]
            image_pos = [ galsim.PositionD(x,y) for x,y in zip(x_values, y_values) ]

        # Make the list of sky values:
        if sky_col is not None:
            if sky is not None:
                raise ValueError("Cannot provide both sky_col and sky.")
            if sky_col not in cat.dtype.names:
                raise ValueError("sky_col = %s is not a column in %s"%(sky_col,cat_file_name))
            sky = cat[sky_col]
        elif sky is not None:
            try:
                sky = float(sky)
            except ValueError:
                fits = fitsio.FITS(image_file_name)
                hdu = 1 if image_file_name.endswith('.fz') else 0
                header = fits[hdu].read_header()
                if sky not in header:
                    raise KeyError("Key %s not found in FITS header"%sky)
                sky = float(header[sky])
            sky = np.array([sky]*len(cat), dtype=float)
        else:
            sky = None

        # Make the list of gain values:
        # TODO: SV and Y1 DES images have two gain values, GAINA, GAINB.  It would be nice if we
        #       could get the right one properly.  OTOH, Y3+ will be in electrons, so gain=1 will
        #       the right value for all images.  So maybe not worth worrying about.
        if gain_col is not None:
            if gain is not None:
                raise ValueError("Cannot provide both gain_col and gain.")
            if gain_col not in cat.dtype.names:
                raise ValueError("gain_col = %s is not a column in %s"%(gain_col,cat_file_name))
            gain = cat[gain_col]
        elif gain is not None:
            try:
                gain = float(gain)
            except ValueError:
                fits = fitsio.FITS(image_file_name)
                hdu = 1 if image_file_name.endswith('.fz') else 0
                header = fits[hdu].read_header()
                if gain not in header:
                    raise KeyError("Key %s not found in FITS header"%gain)
                gain = float(header[gain])
            gain = np.array([gain]*len(cat), dtype=float)
        else:
            gain = [None] * len(cat)

        return image_pos, sky, gain

    def setPointing(self, ra, dec, logger=None):
        """Set the pointing attribute based on the input ra, dec (given in the initializer)

        There are a number of ways the pointing can be specified.
        Even this is probably not sufficiently generic for all applications, but it's a start.

        1. numerical values (in Hours, Degrees respective) for ra, dec
        2. hh:mm:ss.ssss, dd:mm:ss.ssss strings giving hours/degrees, minutes, seconds for each
        3. FITS header key words to read to get the ra, dec values
        4. None, which will attempt to find the spatial center of all the input images using the
           midpoint of the min/max ra and dec values of the image corners according to their
           individual WCS functions. [Not implemented currently.]
        """
        import fitsio
        import galsim
        logger = galsim.config.LoggerWrapper(logger)

        if (ra is None) != (dec is None):
            raise ValueError("Only one of ra, dec was specified")

        if ra is None:
            if self.images[0].wcs.isCelestial():
                if len(self.images) == 1:
                    # Here we can just use the image center.
                    im = self.images[0]
                    self.pointing = im.wcs.toWorld(im.true_center)
                    logger.info("Setting pointing to image center: %.3f h, %.3f d",
                                self.pointing.ra / galsim.hours,
                                self.pointing.dec / galsim.degrees)
                else:
                    # Use the mean of all the image centers
                    plist = [im.wcs.toWorld(im.true_center) for im in self.images]
                    # Do this in x,y,z coords, not ra, dec so we don't mess up near ra=0.
                    xlist, ylist, zlist = zip(*[p.get_xyz() for p in plist])
                    x = np.mean(xlist)
                    y = np.mean(ylist)
                    z = np.mean(zlist)
                    self.pointing = galsim.CelestialCoord.from_xyz(x,y,z)
                    logger.info("Setting pointing to mean of image centers: %.3f h, %.3f d",
                                self.pointing.ra / galsim.hours,
                                self.pointing.dec / galsim.degrees)
            else:
                self.pointing = None
        elif type(ra) in [float, int]:
            ra = float(ra) * galsim.hours
            dec = float(dec) * galsim.degrees
            self.pointing = galsim.CelestialCoord(ra,dec)
            logger.info("Setting pointing to: %.3f h, %.3f d",
                        self.pointing.ra / galsim.hours,
                        self.pointing.dec / galsim.degrees)
        elif str(ra) != ra:
            raise ValueError("Unable to parse input ra: %s"%ra)
        elif str(dec) != dec:
            raise ValueError("Unable to parse input dec: %s"%dec)
        elif ':' in ra and ':' in dec:
            ra = galsim.Angle.from_hms(ra)
            dec = galsim.Angle.from_dms(dec)
            self.pointing = galsim.CelestialCoord(ra,dec)
            logger.info("Setting pointing to: %.3f h, %.3f d",
                        self.pointing.ra / galsim.hours,
                        self.pointing.dec / galsim.degrees)
        else:
            file_name = self.image_file_name[0]
            if len(self.chipnums) == 1:
                logger.info("Setting pointing from keywords %s, %s", ra, dec)
            else:
                logger.info("Setting pointing from keywords %s, %s in %s", ra, dec, file_name)
            fits = fitsio.FITS(file_name)
            hdu = 1 if file_name.endswith('.fz') else 0
            header = fits[hdu].read_header()
            if ra not in header:
                raise KeyError("Key %s not found in FITS header"%ra)
            ra = header[ra]
            if dec not in header:
                raise KeyError("Key %s not found in FITS header"%dec)
            dec = header[dec]
            # Recurse to do further parsing.
            self.setPointing(ra, dec, logger)
