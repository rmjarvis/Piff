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
import glob
import numpy

def process_input(config, logger=None):
    """Parse the input field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a list of Star instances with the initial data.
    """
    import piff

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = piff.setup_logger(verbose=verbose)

    if 'input' not in config:
        raise ValueError("config dict has no input field")
    config_input = config['input']

    # Get the class to use for handling the input data
    # Default type is 'Files'
    # Not sure if this is what we'll always want, but it would be simple if we can make it work.
    input_handler_class = getattr(piff, 'Input' + config_input.pop('type','Files'))

    # Read any other kwargs in the input field
    kwargs = input_handler_class.parseKwargs(config_input)

    # Build handler object
    input_handler = input_handler_class(**kwargs)

    # read the image data
    input_handler.readImages(logger)

    # read the input catalogs
    input_handler.readStarCatalogs(logger)

    # Figure out the pointing
    input_handler.setPointing(logger)

    # Creat a lit of StarData objects
    stars = input_handler.makeStars(logger)

    # Maybe add poisson noise to the weights
    stars = input_handler.addPoisson(stars, logger)

    return stars


class InputHandler(object):
    """The base class for handling inputs for building a Piff model.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def parseKwargs(cls, config_input):
        """Parse the input field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_input:    The input field of the configuration dict, config['input']

        :returns: a kwargs dict to pass to the initializer
        """
        return config_input

    def readImages(self, logger=None):
        """Read in the images from whatever the input source is.

        After this call, self.images will be a list of galsim.Image instances with the full data
        images, and self.weight will be the corresponding weight images.

        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplemented("Derived classes must define the readImages function")

    def readStarCatalogs(self, logger=None):
        """Read in the star catalogs.

        After this call, self.cats will be a list of catalogs (numpy.ndarray instances).

        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplemented("Derived classes must define the readStarCatalogs function")

    def setPointing(self, logger=None):
        """Set the pointing attribute based on the input ra, dec (given in the initializer)

        There are a number of ways the pointing can be specified.
        Even this is probably not sufficiently generic for all applications, but it's a start.

        1. numerical values (in Hours, Degrees respective) for ra, dec
        2. hh:mm:ss.ssss, dd:mm:ss.ssss strings giving hours/degrees, minutes, seconds for each
        3. FITS header key words to read to get the ra, dec values
        4. None, which will attempt to find the spatial center of all the input images using the
           midpoint of the min/max ra and dec values of the image corners according to their
           individual WCS functions. [Not implemented currently.]

        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplemented("Derived classes must define the setPointing function")

    def setGain(self, logger=None):
        """Set the gain value according to the input gain (given in the initializer)

        There are two ways the gain can be specified.

        1. numerical value
        2. FITS header key word to read to get the gain

        TODO: SV and Y1 DES images have two gain values, GAINA, GAINB.  It would be nice if we
              could get the right one properly.  OTOH, Y3+ will be in electrons, so gain=1 will
              the right value for all images.  So maybe not worth worrying about.
        """
        raise NotImplemented("Derived classes must define the setPointing function")

    def addPoisson(self, stars, logger=None):
        """If the input parameters included a gain, then add Poisson noise to the weights
        according to the flux in the image.

        :param stars:       The list of stars to update.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the new list of stars.
        """
        import piff
        if self.gain is None:
            return stars
        self.setGain(logger)
        if logger:
            logger.info("Adding Poisson noise according to gain=%f",self.gain)
        stars = [piff.Star(s.data.addPoisson(gain=self.gain), s.fit) for s in stars]
        return stars

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

        stars = []
        if logger:
            logger.info("Making star list from %d catalog(s)", len(self.cats))
        for image,wt,cat in zip(self.images, self.weight, self.cats):
            if logger:
                logger.debug("Processing catalog with %d stars",len(cat))
            for k in range(len(cat)):
                x = cat[self.x_col][k]
                y = cat[self.y_col][k]
                icen = int(x+0.5)
                jcen = int(y+0.5)
                half_size = self.stamp_size // 2
                bounds = galsim.BoundsI(icen+half_size-self.stamp_size+1, icen+half_size,
                                        jcen+half_size-self.stamp_size+1, jcen+half_size)
                stamp = image[bounds]
                props = {}
                if self.sky_col is not None:
                    sky = cat[self.sky_col][k]
                    stamp = stamp - sky  # Don't change the original!
                    props['sky'] = sky
                wt_stamp = wt[bounds]
                pos = galsim.PositionD(x,y)
                data = piff.StarData(stamp, pos, weight=wt_stamp, pointing=self.pointing,
                                     properties=props)
                stars.append(piff.Star(data, None))

        return stars


class InputFiles(InputHandler):
    """An InputHandler than just takes a list of image files and catalog files.

    :param images:      Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                        (e.g. ["file1.fits", "file2.fits"]) listing the image files to read.
    :param cats:        Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                        (e.g. ["file1.fits", "file2.fits"]) listing the catalog files to read.
    :param x_col:       The name of the X column in the input catalogs. [default: 'x']
    :param y_col:       The name of the Y column in the input catalogs. [default: 'y']
    :param sky_col:     The name of a column with sky values to subtract from the image data.
                        [default: None, which means don't do any sky subtraction]
    :param flag_col:    The name of a flag column in the input catalogs.  Anything with flag != 0
                        is removed from the catalogs. [default: None]
    :param use_col:     The name of a use column in the input catalogs.  Anything with use == 0
                        is removed from the catalogs. [default: None]
    :param image_hdu:   The hdu to use in the image files. [default: None, which means use either
                        0 or 1 as typical given the compression sceme of the file]
    :param weight_hdu:  The hdu to use for weight images. [default: None, which means a weight
                        image with all 1's will be automatically created]
    :param badpix_hdu:  The hdu to use for badpix images. Pixels with badpix != 0 will be given
                        weight == 0. [default: None]
    :param cat_hdu:     The hdu to use in the catalgo files. [default: 1]
    :param stamp_size:  The size of the postage stamps to use for the cutouts.  Note: some
                        stamps may be smaller than this if the star is near a chip boundary.
                        [default: 32]
    :param ra, dec:     The RA, Dec of the telescope pointing. [default: None; See :setPointing:
                        for details about how this can be specified]
    :param gain:        The gain to use for adding Poisson noise to the weight map. [default: None]
    """
    def __init__(self, images, cats,
                 x_col='x', y_col='y', sky_col=None, flag_col=None, use_col=None,
                 image_hdu=None, weight_hdu=None, badpix_hdu=None, cat_hdu=1,
                 stamp_size=32, ra=None, dec=None, gain=None):

        if isinstance(images, basestring):
            self.image_files = glob.glob(images)
            if len(self.image_files) == 0:
                raise ValueError("No such files: %s"%images)
        else:
            self.image_files = images
        if isinstance(cats, basestring):
            self.cat_files = glob.glob(cats)
            if len(self.image_files) == 0:
                raise ValueError("No such files: %s"%cats)
        else:
            self.cat_files = cats
        self.x_col = x_col
        self.y_col = y_col
        self.sky_col = sky_col
        self.flag_col = flag_col
        self.use_col = use_col
        self.image_hdu = image_hdu
        self.weight_hdu = weight_hdu
        self.badpix_hdu = badpix_hdu
        self.cat_hdu = cat_hdu
        self.stamp_size = stamp_size
        self.ra = ra
        self.dec = dec
        self.gain = gain
        self.pointing = None

    def readImages(self, logger=None):
        """Read in the images from the input files and return them.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of galsim.Image instances
        """
        import galsim

        # Read in the images from the files
        if logger:
            logger.info("Reading image files %s",self.image_files)
        self.images = [ galsim.fits.read(fname, hdu=self.image_hdu) for fname in self.image_files ]

        # Either read in the weight image, or build a dummy one
        if self.weight_hdu is None:
            if logger:
                logger.debug("Making trivial (wt==1) weight images")
            self.weight = [ galsim.ImageI(im.bounds, init_value=1) for im in self.images ]
        else:
            if logger:
                logger.info("Reading weight images from hdu %d.",self.weight_hdu)
            self.weight = [ galsim.fits.read(fname, hdu=self.weight_hdu)
                            for fname in self.image_files ]
            for wt in self.weight:
                if numpy.all(wt.array == 0):
                    logger.error("According to the weight mask in %s, all pixels have zero weight!",
                                 fname)

        # If requested, set wt=0 for any bad pixels
        if self.badpix_hdu is not None:
            if logger:
                logger.info("Reading badpix images from hdu %d.",self.badpix_hdu)
            for fname, wt in zip(self.image_files, self.weight):
                badpix = galsim.fits.read(fname, hdu=self.badpix_hdu)
                # The badpix image may be offset by 32768 from the true value.
                # If so, subtract it off.
                if numpy.any(badpix.array > 32767):
                    if logger:
                        logger.debug('min(badpix) = %s',numpy.min(badpix.array))
                        logger.debug('max(badpix) = %s',numpy.max(badpix.array))
                        logger.info("subtracting 32768 from all values in badpix image")
                    badpix -= 32768
                if numpy.any(badpix.array < -32767):
                    if logger:
                        logger.debug('min(badpix) = %s',numpy.min(badpix.array))
                        logger.debug('max(badpix) = %s',numpy.max(badpix.array))
                        logger.info("adding 32768 to all values in badpix image")
                    badpix += 32768
                # Also, convert to int16, in case it isn't by default.
                badpix = galsim.ImageS(badpix)
                if numpy.all(badpix.array != 0):
                    logger.error("According to the bad pixel array in %s, all pixels are masked!",
                                 fname)
                wt.array[badpix.array != 0] = 0

    def readStarCatalogs(self, logger=None):
        """Read in the star catalogs and return lists of positions for each star in each image.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of lists of galsim.PositionD instances
        """
        import fitsio
        import galsim

        # Read in the star catalogs from the files
        if logger:
            logger.info("Reading star catalogs %s.",self.cat_files)
        self.cats = [ fitsio.read(fname,self.cat_hdu) for fname in self.cat_files ]

        # Remove any objects with flag != 0
        if self.flag_col is not None:
            if logger:
                logger.info("Removing objects with %s != 0",self.flag_col)
            self.cats = [ cat[cat[self.flag_col]==0] for cat in self.cats ]

        # Remove any objects with use == 0
        if self.use_col is not None:
            if logger:
                logger.info("Removing objects with %s == 0",self.use_col)
            self.cats = [ cat[cat[self.use_col]!=0] for cat in self.cats ]

    def setPointing(self, logger=None):
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

        ra = self.ra
        dec = self.dec
        if (ra is None) != (dec is None):
            raise ValueErro("Only one of ra, dec was specified")

        if ra is None:
            if self.images[0].wcs.isCelestial():
                if len(self.images) == 1:
                    # Here we can just use the image center.
                    im = self.images[0]
                    self.pointing = im.wcs.toWorld(im.trueCenter())
                    if logger:
                        logger.info("Setting pointing to image center: %.3f h, %.3f d",
                                    self.pointing.ra / galsim.hours,
                                    self.pointing.dec / galsim.degrees)
                else:
                    raise NotImplemented("The automatic pointing calculation is not implemented yet.")
            else:
                self.pointing = None
        elif type(ra) in [float, int]:
            ra = float(ra) * galsim.hours
            dec = float(dec) * galsim.degrees
            self.pointing = galsim.CelestialCoord(ra,dec)
            if logger:
                logger.info("Setting pointing to: %.3f h, %.3f d",
                            self.pointing.ra / galsim.hours,
                            self.pointing.dec / galsim.degrees)
        elif str(ra) != ra:
            raise ValueError("Unable to parse input ra: %s"%ra)
        elif str(dec) != dec:
            raise ValueError("Unable to parse input dec: %s"%dec)
        elif ':' in ra and ':' in dec:
            ra = galsim.HMS_Angle(ra)
            dec = galsim.DMS_Angle(dec)
            self.pointing = galsim.CelestialCoord(ra,dec)
            if logger:
                logger.info("Setting pointing to: %.3f h, %.3f d",
                            self.pointing.ra / galsim.hours,
                            self.pointing.dec / galsim.degrees)
        else:
            file_name = self.image_files[0]
            if logger:
                logger.info("Setting pointing from keywords %s, %s in %s", ra, dec, file_name)
            fits = fitsio.FITS(file_name)
            hdu = 1 if file_name.endswith('.fz') else 0
            header = fits[hdu].read_header()
            self.ra = header[ra]
            self.dec = header[dec]
            # Recurse to do further parsing.
            self.setPointing(logger)

    def setGain(self, logger=None):
        """Set the gain value according to the input gain (given in the initializer)

        There are two ways the gain can be specified.

        1. numerical value
        2. FITS header key word to read to get the gain

        TODO: SV and Y1 DES images have two gain values, GAINA, GAINB.  It would be nice if we
              could get the right one properly.  OTOH, Y3+ will be in electrons, so gain=1 will
              the right value for all images.  So maybe not worth worrying about.
        """
        import fitsio

        if self.gain is None:
            return

        elif type(self.gain) in [float, int]:
            self.gain = float(self.gain)
        elif str(self.gain) != self.gain:
            raise ValueError("Unable to parse input gain: %s"%self.gain)
        else:
            file_name = self.image_files[0]
            if logger:
                logger.info("Setting gain from keyword %s in %s", self.gain, file_name)
            fits = fitsio.FITS(file_name)
            hdu = 1 if file_name.endswith('.fz') else 0
            header = fits[hdu].read_header()
            self.gain = float(header[self.gain])

