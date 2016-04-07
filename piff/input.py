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

def process_input(config, logger=None):
    """Parse the input field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a list of StarData instances
    """
    import piff

    if logger is None:
        logger = config.setup_logger(verbosity=0)

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

    # Creat a lit of StarData objects
    stars = input_handler.makeStarData(logger)

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

    def makeStarData(self, logger=None):
        """Process the input images and star data, cutting out stamps for each star along with
        other relevant information.

        The base class implementation expects the derived class to have appropriately set the
        following attributes:

            :stamp_size:    The size of the postage stamp to use for the cutouts
            :x_col:         The name of the column in the catalogs to use for the x position.
            :y_col:         The name of the column in the catalogs to use for the y position.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of StarData instances
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
                icen = int(x)
                jcen = int(y)
                half_size = self.stamp_size // 2
                bounds = galsim.BoundsI(icen-half_size+1, icen+half_size,
                                        jcen-half_size+1, jcen+half_size)
                stamp = image[bounds]
                wt_stamp = wt[bounds]
                pos = galsim.PositionD(x,y)
                stars.append(piff.StarData(stamp, pos, weight=wt_stamp))

        return stars

class InputFiles(InputHandler):
    """An InputHandler than just takes a list of image files and catalog files.

    :param images:      Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                        (e.g. ["file1.fits", "file2.fits"]) listing the image files to read.
    :param cats:        Either a string (e.g. ``some_dir/*.fits.fz``) or a list of strings
                        (e.g. ["file1.fits", "file2.fits"]) listing the catalog files to read.
    :param x_col:       The name of the X column in the input catalogs. [default: 'x']
    :param y_col:       The name of the Y column in the input catalogs. [default: 'y']
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
    """
    def __init__(self, images, cats,
                 x_col='x', y_col='y', flag_col=None, use_col=None,
                 image_hdu=None, weight_hdu=None, badpix_hdu=None, cat_hdu=1,
                 stamp_size=32):

        if isinstance(images, basestring):
            self.image_files = glob.glob(images)
        else:
            self.image_files = images
        if isinstance(cats, basestring):
            self.cat_files = glob.glob(cats)
        else:
            self.cat_files = cats
        self.x_col = x_col
        self.y_col = y_col
        self.flag_col = flag_col
        self.use_col = use_col
        self.image_hdu = image_hdu
        self.weight_hdu = weight_hdu
        self.badpix_hdu = badpix_hdu
        self.cat_hdu = cat_hdu
        self.stamp_size = stamp_size

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

        # If requested, set wt=0 for any bad pixels
        if self.badpix_hdu is not None:
            if logger:
                logger.info("Reading badpix images from hdu %d.",self.badpix_hdu)
            for fname, wt in zip(self.image_files, self.weight):
                badpix = galsim.fits.read(fname, hdu=self.badpix_hdu)
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
        self.cats = [ fitsio.read(fname) for fname in self.cat_files ]

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

