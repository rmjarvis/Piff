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

        :returns: stars, wcs:   A list of Star instances with the initial data, and
                                a dict of WCS solutions indexed by chipnum.
        """
        import piff

        # Get the class to use for handling the input data
        # Default type is 'Files'
        input_handler_class = getattr(piff, 'Input' + config_input.pop('type','Files'))

        # Read any other kwargs in the input field
        kwargs = input_handler_class.parseKwargs(config_input, logger)

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

        # Get the wcs for all the input chips
        wcs = input_handler.getWCS(logger)

        return stars, wcs, input_handler.pointing

    @classmethod
    def parseKwargs(cls, config_input, logger=None):
        """Parse the input field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_input:    The input field of the configuration dict, config['input']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = config_input.copy()
        kwargs['logger'] = logger
        return kwargs

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
            logger.info("Adding Poisson noise to weight map according to gain=%f",self.gain)
        stars = [s.addPoisson(gain=self.gain) for s in stars]
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
            if len(self.cats) == 1:
                logger.debug("Making star list from catalog %s", self.cat_files[0])
            else:
                logger.debug("Making star list from %d catalogs", len(self.cats))
        for image,wt,cat,fname in zip(self.images, self.weight, self.cats, self.cat_files):
            if logger:
                logger.debug("Processing catalog %s with %d stars",fname,len(cat))
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

    def getWCS(self, logger=None):
        """Get the WCS solutions for all the chips in the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns:   A dict of WCS solutions (galsim.BaseWCS instances) indexed by chipnum
        """
        wcs_list = [im.wcs for im in self.images]
        return dict(zip(self.chipnums, wcs_list))


class InputFiles(Input):
    """An Input handler than just takes a list of image files and catalog files.
    """
    def __init__(self, images, cats, chipnums=None,
                 dir=None, image_dir=None, cat_dir=None,
                 x_col='x', y_col='y', sky_col=None, flag_col=None, use_col=None,
                 image_hdu=None, weight_hdu=None, badpix_hdu=None, cat_hdu=1,
                 stamp_size=32, ra=None, dec=None, gain=None, logger=None):
        """
        There are a number of ways to specify the input files (parameters `images` and `cats`):

        1. If you only have a single image/catalog, you may just give the file name directly
           as a single string.
        2. For multiple images, you may specify a list of strings listing all the file names.
        3. You may specify a string with ``{chipnum}`` which will be filled in by the chipnum
           values given in the `chipnums` parameter using ``s.format(chipnum=chipnum)``.
        4. You may specify a string with ``%s`` (or perhaps ``%02d``, etc.) which will be filled
           in by the chipnum values given in the `chipnums` parameter using ``s % chipnum``.
        5. You may specify a string that ``glob.glob(s)`` will understand and convert into a
           list of file names.  Caveat: ``glob`` returns the files in native directory order
           (cf. ``ls -f``).  This can thus be different for the images and catalogs if they
           were written to disk out of order.  Therefore, we sort the list returned by
           ``glob.glob(s)``.  Typically, this will result in the image file names and catalog
           file names matching up correctly, but it is the users responsibility to ensure
           that this is the case.

        The `chipnums` parameter specifies chip "numbers" which are really just any identifying
        number or string that is different for each chip in the exposure.  Typically, these are
        numbers, but they don't have to be if you have some other way of identifying the chips.

        There are a number of ways that the chipnums may be specified:

        1. A single number or string.
        2. A list of numbers or strings.
        3. A string that can be ``eval``ed to yield the appropriate list.  e.g.
           `[ c for c in range(1,63) if c is not 61 ]`
        4. None, in which case range(len(images)) will be used.  In this case options 3,4 above
           for the images and cats parameters are not allowed.

        :param images:      Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                            (e.g. ["file1.fits", "file2.fits"]) listing the image files to read.
                            See above for ways that this parameter may be specified.
        :param cats:        Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                            (e.g. ["file1.fits", "file2.fits"]) listing the catalog files to read.
                            See above for ways that this parameter may be specified.
        :param chipnums:    A list of "chip numbers" to use as the names of each image.  These may
                            be integers or strings and don't have to be sequential.
                            See above for ways that this parameter may be specified.
                            [default: None, which will use range(len(images))]
        :param dir:         Optionally specify the directory these files are in. [default: None]
        :param image_dir:   Optionally specify the directory of the image files. [default: dir]
        :param cat_dir:     Optionally specify the directory of the cat files. [default: dir]
        :param x_col:       The name of the X column in the input catalogs. [default: 'x']
        :param y_col:       The name of the Y column in the input catalogs. [default: 'y']
        :param sky_col:     The name of a column with sky values to subtract from the image data.
                            [default: None, which means don't do any sky subtraction]
        :param flag_col:    The name of a flag column in the input catalogs.  Anything with
                            flag != 0 is removed from the catalogs. [default: None]
        :param use_col:     The name of a use column in the input catalogs.  Anything with
                            use == 0 is removed from the catalogs. [default: None]
        :param image_hdu:   The hdu to use in the image files. [default: None, which means use
                            either 0 or 1 as typical given the compression sceme of the file]
        :param weight_hdu:  The hdu to use for weight images. [default: None, which means a weight
                            image with all 1's will be automatically created]
        :param badpix_hdu:  The hdu to use for badpix images. Pixels with badpix != 0 will be given
                            weight == 0. [default: None]
        :param cat_hdu:     The hdu to use in the catalgo files. [default: 1]
        :param stamp_size:  The size of the postage stamps to use for the cutouts.  Note: some
                            stamps may be smaller than this if the star is near a chip boundary.
                            [default: 32]
        :param ra, dec:     The RA, Dec of the telescope pointing. [default: None; See
                            :setPointing: for details about how this can be specified]
        :param gain:        The gain to use for adding Poisson noise to the weight map.
                            [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        if image_dir is None: image_dir = dir
        if cat_dir is None: cat_dir = dir

        # Try to eval chipnums that come in as a string.
        if isinstance(chipnums, basestring):
            try:
                chipnums = eval(chipnums)
            except Exception as e:
                if logger:
                    logger.debug("The string %r could not be eval-ed.",chipnums)
                    logger.debug("Exception raised: %s",e)
                    logger.debug("Assuming for now that this is ok, but if it was supposed to ",
                                 "be evaled, there might be a problem later.")

        # Not all combinations of errors are properly diagnosed, since we may not know yet whether
        # a string value of images implies 1 or more images.  But we do our best.
        if isinstance(chipnums, str):
            if isinstance(images, str) or len(images) == 1:
                self.chipnums = [ chipnums ]
            else:
                raise ValueError("Invalid chipnums = %s with multiple images",chipnums)
        elif isinstance(chipnums, int):
            if isinstance(images, str) or len(images) == 1:
                self.chipnums = [ chipnums ]
            else:
                raise ValueError("Invalid chipnums = %s with multiple images",chipnums)
        elif chipnums is None or isinstance(chipnums, list):
            self.chipnums = chipnums
        else:
            raise ValueError("Invalid chipnums = %s",chipnums)
        if logger:
            logger.debug("chipnums = %s",self.chipnums)

        # Parse the images and cats parameters.
        self.image_files = self._get_file_list(images, image_dir, self.chipnums, logger)
        if logger:
            logger.debug("image files = %s",self.image_files)

        self.cat_files = self._get_file_list(cats, cat_dir, self.chipnums, logger)
        if logger:
            logger.debug("cat files = %s",self.cat_files)

        # Finally, if chipnums is None, we can make it the default list.
        if self.chipnums is None:
            self.chipnums = range(len(self.image_files))
            if logger:
                logger.debug("Using default chipnums: %s",self.chipnums)

        # Check that the number of images, cats, chips are equal.
        if len(self.image_files) != len(self.cat_files):
            raise ValueError("Number of images (%d) and catalogs (%d) do not match."%(
                             len(self.image_files), len(self.cat_files)))
        if len(self.image_files) != len(self.chipnums):
            raise ValueError("Number of images (%d) and chipnums (%d) do not match."%(
                             len(self.image_files), len(self.chipnums)))

        # Other parameters are just saved for use later.
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

    def _get_file_list(self, s, d, chipnums, logger):
        if isinstance(s, basestring):
            # First try "{chipnum}" style formatting.
            if (chipnums is not None) and ('{' in s) and ('}' in s):
                try:
                    file_list = [ s.format(chipnum=c) for c in chipnums ]
                except Exception as e:
                    # There are a number of kinds of exceptions this could raise: ValueError,
                    # IndexError, and KeyError at least.  Easier to just catch Exception.
                    if logger:
                        logger.error("Trying %r.format(chipnum=chipnum) failed", s)
                else:
                    if logger:
                        logger.debug("Successfully used %r.format(chipnum=c) to parse string.",s)
                    if d is not None:
                        file_list = [ os.path.join(d, f) for f in file_list ]
                    return file_list

                # We told them to use {chipnum} but check if they did something like {} instead.
                try:
                    file_list = [ s.format(c) for c in chipnums ]
                except Exception as e2:
                    if logger:
                        logger.error("Trying %r.format(chipnum) failed", s)
                    # If this also failed, raise the original exception.
                    raise e
                else:
                    if logger:
                        logger.debug("Successfully used s.format(chipnum=c) to parse string.")
                    if d is not None:
                        file_list = [ os.path.join(d, f) for f in file_list ]
                    return file_list

            # Next try "%d" style formatting.
            if (chipnums is not None) and ('%' in s):
                try:
                    file_list = [ s % c for c in chipnums ]
                except Exception as e:
                    if logger:
                        logger.error("Trying %r %% chipnum failed", s)
                    raise e
                else:
                    if logger:
                        logger.debug("Successfully used %r %% chipnum to parse string.",s)
                    if d is not None:
                        file_list = [ os.path.join(d, f) for f in file_list ]
                    return file_list

            # Finally, try glob, which will also work for a single file name.
            try:
                # For glob, we need to join d before running glob.
                if d is not None:
                    s = os.path.join(d, s)
                file_list = sorted(glob.glob(s))
            except Exception as e:
                if logger:
                    logger.error("Trying glob.glob(%r) failed", s)
                raise e
            else:
                if len(file_list) == 0:
                    raise ValueError("No such files: %r"%s)
                return file_list

        elif not isinstance(s, list):
            raise ValueError("%r is not a list or a string",s)

        else:
            file_list = s
            if d is not None:
                file_list = [ os.path.join(d, f) for f in file_list ]
            return file_list



    def readImages(self, logger=None):
        """Read in the images from the input files and return them.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of galsim.Image instances
        """
        import galsim

        # Read in the images from the files
        self.images = []
        for fname in self.image_files:
            if logger:
                logger.info("Reading image file %s",fname)
            self.images.append(galsim.fits.read(fname, hdu=self.image_hdu))

        # Either read in the weight image, or build a dummy one
        if len(self.images) == 1:
            plural = ''
        else:
            plural = 's'
        if self.weight_hdu is None:
            if logger:
                logger.debug("Making trivial (wt==1) weight image%s", plural)
            self.weight = [ galsim.ImageI(im.bounds, init_value=1) for im in self.images ]
        else:
            if logger:
                logger.info("Reading weight image%s from hdu %d.", plural, self.weight_hdu)
            self.weight = [ galsim.fits.read(fname, hdu=self.weight_hdu)
                            for fname in self.image_files ]
            for wt in self.weight:
                if numpy.all(wt.array == 0):
                    logger.error("According to the weight mask in %s, all pixels have zero weight!",
                                 fname)

        # If requested, set wt=0 for any bad pixels
        if self.badpix_hdu is not None:
            if logger:
                logger.info("Reading badpix image%s from hdu %d.", plural, self.badpix_hdu)
            for fname, wt in zip(self.image_files, self.weight):
                badpix = galsim.fits.read(fname, hdu=self.badpix_hdu)
                # The badpix image may be offset by 32768 from the true value.
                # If so, subtract it off.
                if numpy.any(badpix.array > 32767):
                    if logger:
                        logger.debug('min(badpix) = %s',numpy.min(badpix.array))
                        logger.debug('max(badpix) = %s',numpy.max(badpix.array))
                        logger.debug("subtracting 32768 from all values in badpix image")
                    badpix -= 32768
                if numpy.any(badpix.array < -32767):
                    if logger:
                        logger.debug('min(badpix) = %s',numpy.min(badpix.array))
                        logger.debug('max(badpix) = %s',numpy.max(badpix.array))
                        logger.debug("adding 32768 to all values in badpix image")
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
        self.cats = []
        for fname in self.cat_files:
            if logger:
                logger.info("Reading star catalog %s.",fname)
            self.cats.append(fitsio.read(fname,self.cat_hdu))

        # Remove any objects with flag != 0
        if self.flag_col is not None:
            if logger:
                logger.info("Removing objects with flag (col %s) != 0",self.flag_col)
            self.cats = [ cat[cat[self.flag_col]==0] for cat in self.cats ]

        # Remove any objects with use == 0
        if self.use_col is not None:
            if logger:
                logger.info("Removing objects with use (col %s) == 0",self.use_col)
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
                if len(self.image_files) == 1:
                    logger.info("Setting pointing from keywords %s, %s", ra, dec)
                else:
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
            if logger:
                logger.info("Setting gain = %f", self.gain)
            self.gain = float(self.gain)
        elif str(self.gain) != self.gain:
            raise ValueError("Unable to parse input gain: %s"%self.gain)
        else:
            file_name = self.image_files[0]
            if logger:
                if len(self.image_files) == 1:
                    logger.info("Setting gain from keyword %s", self.gain)
                else:
                    logger.info("Setting gain from keyword %s in %s", self.gain, file_name)
            fits = fitsio.FITS(file_name)
            hdu = 1 if file_name.endswith('.fz') else 0
            header = fits[hdu].read_header()
            self.gain = float(header[self.gain])

