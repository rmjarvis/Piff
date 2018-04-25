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
import fitsio
import galsim

from .star import Star, StarData
from .util import write_kwargs, read_kwargs

class PSF(object):
    """The base class for describing a PSF model across a field of view.

    The usual way to create a PSF is through one of the two factory functions::

        >>> psf = piff.PSF.process(config, logger)
        >>> psf = piff.PSF.read(file_name, logger)

    The first is used to build a PSF model from the data according to a config dict.
    The second is used to read in a PSF model from disk.
    """
    @classmethod
    def process(cls, config_psf, logger=None):
        """Process the config dict and return a PSF instance.

        As the PSF class is an abstract base class, the returned type will in fact be some
        subclass of PSF according to the contents of the config dict.

        The provided config dict is typically the 'psf' field in the base config dict in
        a YAML file, although for compound PSF types, it may be the field for just one of
        several components.

        This function merely creates a "blank" PSF object.  It does not actually do any
        part of the solution yet.  Typically this will be followed by fit:

            >>> psf = piff.PSF.process(config['psf'])
            >>> stars, wcs, pointing = piff.Input.process(config['input'])
            >>> psf.fit(stars, wcs, pointing)

        at which point, the ``psf`` instance would have a solution to the PSF model.

        :param config_psf:  A dict specifying what type of PSF to build along with the
                            appropriate kwargs for building it.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance of the appropriate type.
        """
        import piff
        import yaml

        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Parsing PSF based on config dict:")
        logger.debug(yaml.dump(config_psf, default_flow_style=False))

        # Get the class to use for the PSF
        psf_type = config_psf.get('type', 'Simple') + 'PSF'
        logger.debug("PSF type is %s",psf_type)
        cls = getattr(piff, psf_type)

        # Read any other kwargs in the psf field
        kwargs = cls.parseKwargs(config_psf, logger)

        # Build PSF object
        logger.info("Building %s",psf_type)
        psf = cls(**kwargs)
        logger.debug("Done building PSF")

        return psf

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_psf)
        kwargs.pop('type',None)
        return kwargs

    def draw(self, x, y, chipnum=0, flux=1.0, offset=(0,0), stamp_size=48, image=None,
             logger=None, **kwargs):
        """Draws an image of the PSF at a given location.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, stamp_size=48)

        However, if the PSF interpolation used extra properties for the interpolation
        (cf. psf.extra_interp_properties), you need to provide them as additional kwargs.

            >>> print(psf.extra_interp_properties)
            ('ri_color',)
            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, ri_color=0.23, stamp_size=48)

        Normally, the image is constructed automatically based on stamp_size, in which case
        the WCS will be taken to be the local Jacobian at this location on the original image.
        However, if you provide your own image using the :image: argument, then whatever WCS
        is present in that image will be respected.  E.g. if you want an image of the PSF in
        sky coordinates rather than image coordinates, you can provide an image with just a
        pixel scale for the WCS.

        :param x:           The image x position.
        :param y:           The image y position.
        :param chipnum:     Which chip to use for WCS information. [default: 0, which is
                            appropriate if only using a single chip]
        :param flux:        Flux of PSF to be drawn [default: 1.0]
        :param offset:      (dx,dy) tuple giving offset of stellar center relative
                            to star.data.image_pos [default: (0,0)]
        :param stamp_size:  The size of the image to construct if no image is provided.
                            [default: 48]
        :param image:       An existing image on which to draw, if desired. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :param **kwargs:    Additional properties required for the interpolation.

        :returns:           A GalSim Image of the PSF
        """
        import galsim
        logger = galsim.config.LoggerWrapper(logger)
        properties = {'chipnum' : chipnum}
        for key in self.extra_interp_properties:
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required"%key)
            properties = kwags.pop(key)
        if len(kwargs) != 0:
            raise TypeError("draw got an unexpecte keyword argument %r"%kwargs.keys()[0])

        image_pos = galsim.PositionD(x,y)
        world_pos = StarData.calculateFieldPos(image_pos, self.wcs[chipnum], self.pointing,
                                               properties)
        u,v = world_pos.x, world_pos.y

        # We always use the correct wcs above for world_pos, but in makeTarget, we allow the
        # user's image.wcs to override.
        if image is None or image.wcs is None:
            wcs = self.wcs[chipnum]
        else:
            wcs = image.wcs

        star = Star.makeTarget(x=x, y=y, u=u, v=v, wcs=wcs, properties=properties,
                               stamp_size=stamp_size, image=image, pointing=self.pointing)
        logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        # Adjust the flux, center
        center = star.offset_to_center(offset)
        star = star.withFlux(flux, center)

        # if a user specifies an image, then we want to preserve that image, so
        # copy_image = False. If a user doesn't specify an image, then it
        # doesn't matter if we overwrite it, so use copy_image=False
        copy_image = False
        # Draw the star and return the image
        star = self.drawStar(star, copy_image=copy_image)
        return star.data.image

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns:           List of Star instances with its image filled with
                            rendered PSF
        """
        return [self.drawStar(star, copy_image=copy_image) for star in stars]

    def write(self, file_name, logger=None):
        """Write a PSF object to a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Writing PSF to file %s",file_name)

        with fitsio.FITS(file_name,'rw',clobber=True) as f:
            self._write(f, 'psf', logger)

    def _write(self, fits, extname, logger=None):
        """This is the function that actually does the work for the write function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        psf_type = self.__class__.__name__
        write_kwargs(fits, extname, dict(self.kwargs, type=psf_type))
        logger.info("Wrote the basic PSF information to extname %s", extname)
        Star.write(self.stars, fits, extname=extname + '_stars')
        logger.info("Wrote the PSF stars to extname %s", extname + '_stars')
        self.writeWCS(fits, extname=extname + '_wcs', logger=logger)
        logger.info("Wrote the PSF WCS to extname %s", extname + '_wcs')
        self._finish_write(fits, extname=extname, logger=logger)

    @classmethod
    def read(cls, file_name, logger=None):
        """Read a PSF object from a file.

        :param file_name:   The name of the file to read.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Reading PSF from file %s",file_name)

        with fitsio.FITS(file_name,'r') as f:
            logger.debug('opened FITS file')
            return cls._read(f, 'psf', logger)

    @classmethod
    def _read(cls, fits, extname, logger):
        """This is the function that actually does the work for the read function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        import piff

        # First get the PSF class from the 'psf' extension
        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        psf_type = fits[extname].read()['type']
        assert len(psf_type) == 1
        psf_type = str(psf_type[0].decode())

        # Check that this is a valid PSF type
        psf_classes = piff.util.get_all_subclasses(piff.PSF)
        valid_psf_types = dict([ (c.__name__, c) for c in psf_classes ])
        if psf_type not in valid_psf_types:
            raise ValueError("psf type %s is not a valid Piff PSF"%psf_type)
        psf_cls = valid_psf_types[psf_type]

        # Read the stars, wcs, pointing values
        stars = Star.read(fits, extname + '_stars')
        logger.debug("stars = %s",stars)
        wcs, pointing = cls.readWCS(fits, extname + '_wcs', logger=logger)
        logger.debug("wcs = %s, pointing = %s",wcs,pointing)

        # Get any other kwargs we need for this PSF type
        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type',None)

        # Make the PSF instance
        psf = psf_cls(**kwargs)
        psf.stars = stars
        psf.wcs = wcs
        psf.pointing = pointing

        # Just in case the class needs to do something else at the end.
        psf._finish_read(fits, extname, logger)

        return psf

    def _finish_read(self, fits, extname, logger):
        """Finish up the read process

        In the base class, this is a no op, but for classes that need to do something else at
        the end of the read process, this hook is available to be overridden.

        (E.g. composite psf classes might copy the stars, wcs, pointing information into the
        various components.)

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        pass


    def writeWCS(self, fits, extname, logger):
        """Write the WCS information to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write to
        :param logger:      A logger object for logging debug info.
        """
        import galsim
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        logger = galsim.config.LoggerWrapper(logger)

        # Start with the chipnums, which may be int or str type.
        # Assume they are all the same type at least.
        chipnums = list(self.wcs.keys())
        cols = [ chipnums ]
        if np.dtype(type(chipnums[0])).kind in np.typecodes['AllInteger']:
            dtypes = [ ('chipnums', int) ]
        else:
            # coerce to string, just in case it's something else.
            chipnums = [ str(c) for c in chipnums ]
            max_len = np.max([ len(c) for c in chipnums ])
            dtypes = [ ('chipnums', bytes, max_len) ]

        # GalSim WCS objects can be serialized via pickle
        wcs_str = [ base64.b64encode(pickle.dumps(w)) for w in self.wcs.values() ]
        max_len = np.max([ len(s) for s in wcs_str ])
        # Some GalSim WCS serializations are rather long.  In particular, the Pixmappy one
        # is longer than the maximum length allowed for a column in a fits table (28799).
        # So split it into chunks of size 2**14 (mildly less than this maximum).
        chunk_size = 2**14
        nchunks = max_len // chunk_size + 1
        cols.append( [nchunks]*len(chipnums) )
        dtypes.append( ('nchunks', int) )
        if nchunks > 1:
            logger.debug('Using %d chunks for the wcs pickle string',nchunks)

        # Update to size of chunk we actually need.
        chunk_size = (max_len + nchunks - 1) // nchunks

        chunks = [ [ s[i:i+chunk_size] for i in range(0, max_len, chunk_size) ] for s in wcs_str ]
        cols.extend(zip(*chunks))
        dtypes.extend( ('wcs_str_%04d'%i, bytes, chunk_size) for i in range(nchunks) )

        if self.pointing is not None:
            # Currently, there is only one pointing for all the chips, but write it out
            # for each row anyway.
            dtypes.extend( (('ra', float), ('dec', float)) )
            ra = [self.pointing.ra / galsim.hours] * len(chipnums)
            dec = [self.pointing.dec / galsim.degrees] * len(chipnums)
            cols.extend( (ra, dec) )

        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname)

    @classmethod
    def readWCS(cls, fits, extname, logger):
        """Read the WCS information from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to read from
        :param logger:      A logger object for logging debug info.

        :returns: wcs, pointing where wcs is a dict of galsim.BaseWCS instances and
                                      pointing is a galsim.CelestialCoord instance
        """
        import galsim
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        assert extname in fits
        assert 'chipnums' in fits[extname].get_colnames()
        assert 'nchunks' in fits[extname].get_colnames()

        data = fits[extname].read()

        chipnums = data['chipnums']
        nchunks = data['nchunks']
        nchunks = nchunks[0]  # These are all equal, so just take first one.

        wcs_keys = [ 'wcs_str_%04d'%i for i in range(nchunks) ]
        wcs_str = [ data[key] for key in wcs_keys ] # Get all wcs_str columns
        wcs_str = [ b''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each
        wcs_str = [ base64.b64decode(s) for s in wcs_str ] # Convert back from b64 encoding
        wcs_list = [ pickle.loads(s) for s in wcs_str ]  # Convert back into wcs objects

        wcs = dict(zip(chipnums, wcs_list))

        if 'ra' in fits[extname].get_colnames():
            ra = data['ra']
            dec = data['dec']
            pointing = galsim.CelestialCoord(ra[0] * galsim.hours, dec[0] * galsim.degrees)
        else:
            pointing = None

        return wcs, pointing

# Make a global function, piff.read, as an alias for piff.PSF.read, since that's the main thing
# users will want to do as their starting point for using a piff file.
def read(file_name, logger=None):
    """Read a Piff PSF object from a file.

    :param file_name:   The name of the file to read.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a piff.PSF instance
    """
    return PSF.read(file_name, logger=logger)
