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

import numpy as np
import fitsio
import galsim
import sys

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
    def parseKwargs(cls, config_psf, logger=None):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        raise NotImplementedError("Derived classes must define the parseKwargs function")

    def draw(self, x, y, chipnum=0, flux=1.0, center=None, offset=None, stamp_size=48,
             image=None, logger=None, **kwargs):
        r"""Draws an image of the PSF at a given location.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, stamp_size=48)

        However, if the PSF interpolation used extra properties for the interpolation
        (cf. psf.interp_property_names), you need to provide them as additional kwargs.

            >>> print(psf.interp_property_names)
            ('u','v','ri_color')
            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, ri_color=0.23, stamp_size=48)

        Normally, the image is constructed automatically based on stamp_size, in which case
        the WCS will be taken to be the local Jacobian at this location on the original image.
        However, if you provide your own image using the :image: argument, then whatever WCS
        is present in that image will be respected.  E.g. if you want an image of the PSF in
        sky coordinates rather than image coordinates, you can provide an image with just a
        pixel scale for the WCS.

        When drawing the PSF, there are a few options regarding how the profile will be
        centered on the image.

        1. The default behavior (``center==None``) is to draw the profile centered at the same
           (x,y) as you requested for the location of the PSF in the original image coordinates.
           The returned image will not (normally) be as large as the full image -- it will just be
           a postage stamp centered roughly around (x,y).  The image.bounds give the bounding box
           of this stamp, and within this, the PSF will be centered at position (x,y).
        2. If you want to center the profile at some other arbitrary position, you may provide
           a ``center`` parameter, which should be a tuple (xc,yc) giving the location at which
           you want the PSF to be centered.  The bounding box will still be around the nominal
           (x,y) position, so this should only be used for small adjustments to the (x,y) value
           if you want it centered at a slightly different location.
        3. If you provide your own image with the ``image`` parameter, then you may set the
           ``center`` to any location in this box (or technically off it -- it doesn't check that
           the center is actually inside the bounding box).  This may be useful if you want to draw
           on an image with origin at (0,0) or (1,1) and just put the PSF at the location you want.
        4. If you want the PSf centered exactly in the center of the image, then you can use
           ``center=True``.  This will work for either an automatically built image or one
           that you provide.
        5. With any of the above options you may additionally supply an ``offset`` parameter, which
           will apply a slight offset to the calculated center.  This is probably only useful in
           conjunction with the default ``center=None`` or ``center=True``.

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [default: 0, which is
                            appropriate if only using a single chip]
        :param flux:        Flux of PSF to be drawn [default: 1.0]
        :param center:      (xc,yc) tuple giving the location on the image where you want the
                            nominal center of the profile to be drawn.  Also allowed is the
                            value center=True to place in the center of the image.
                            [default: None, which means draw at the position (x,y) of the PSF.]
        :param offset:      Optional (dx,dy) tuple giving an additional offset relative to the
                            center. [default: None]
        :param stamp_size:  The size of the image to construct if no image is provided.
                            [default: 48]
        :param image:       An existing image on which to draw, if desired. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           A GalSim Image of the PSF
        """
        logger = galsim.config.LoggerWrapper(logger)

        prof, method = self.get_profile(x,y,chipnum=chipnum, flux=flux, logger=logger, **kwargs)

        logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        # Make the image if necessary
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)
            # Make the center of the image (close to) the image_pos
            xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
            ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))
            image.setCenter(xcen, ycen)

        # If no wcs is given, use the original wcs
        if image.wcs is None:
            image.wcs = self.wcs[chipnum]

        # Handle the input center
        if center is None:
            center = (x, y)
        elif center is True:
            center = image.true_center
            center = (center.x, center.y)
        elif not isinstance(center, tuple):
            raise ValueError("Invalid center parameter: %r. Must be tuple or None or True"%(
                             center))

        # Handle offset if given
        if offset is not None:
            center = (center[0] + offset[0], center[1] + offset[1])

        prof.drawImage(image, method=method, center=center)

        return image

    def get_profile(self, x, y, chipnum=0, flux=1.0, logger=None, **kwargs):
        r"""Get the PSF profile at the given position as a GalSim GSObject.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0)

        The first return value, prof, is the GSObject describing the PSF profile.
        The second one, method, is the method parameter that should be used when drawing the
        profile using ``prof.drawImage(..., method=method)``.  This may be either 'no_pixel'
        or 'auto' depending on whether the PSF model already includes the pixel response or not.
        Some underlying models includ the pixel response, and some don't, so this difference needs
        to be accounted for properly when drawing.  This method is also appropriate if you first
        convolve the PSF by some other (e.g. galaxy) profile and then draw that.

        If the PSF interpolation used extra properties for the interpolation (cf.
        psf.extra_interp_properties), you need to provide them as additional kwargs.

            >>> print(psf.extra_interp_properties)
            ('ri_color',)
            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0, ri_color=0.23)

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [default: 0, which is
                            appropriate if only using a single chip]
        :param flux:        Flux of PSF model [default: 1.0]
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           (profile, method)
                            profile = A GalSim GSObject of the PSF
                            method = either 'no_pixel' or 'auto' indicating which method to use
                            when drawing the profile on an image.
        """
        logger = galsim.config.LoggerWrapper(logger)

        properties = {'chipnum' : chipnum}
        for key in self.interp_property_names:
            if key in ['x','y','u','v']: continue
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required"%key)
            properties[key] = kwargs.pop(key)
        if len(kwargs) != 0:
            raise TypeError("Unexpected keyword argument(s) %r"%list(kwargs.keys())[0])

        image_pos = galsim.PositionD(x,y)
        field_pos = StarData.calculateFieldPos(image_pos, self.wcs[chipnum], self.pointing,
                                               properties)
        u,v = field_pos.x, field_pos.y

        wcs = self.wcs[chipnum]

        star = Star.makeTarget(x=x, y=y, u=u, v=v, wcs=wcs, properties=properties,
                               pointing=self.pointing)
        logger.debug("Getting PSF profile at (%s,%s) on chip %s", x, y, chipnum)

        # Interpolate and adjust the flux of the star.
        star = self.interpolateStar(star).withFlux(flux)

        # The last step is implementd in the derived classes.
        prof, method = self._getProfile(star)
        return prof, method

    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        return [self.interpolateStar(star) for star in stars]

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        raise NotImplementedError("Derived classes must define the interpolateStar function")

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars. Takes advantage of
        interpolateList for significant speedup with some interpolators.

        .. note::

            If the stars already have the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> stars = psf.interpolateList(stars)

            before running `drawStarList`.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]

        :returns:           List of Star instances with its image filled with
                            rendered PSF
        """
        if any(star.fit is None or star.fit.params is None for star in stars):
            stars = self.interpolateStarList(stars)
        return [self._drawStar(star, copy_image=copy_image) for star in stars]

    def drawStar(self, star, copy_image=True, center=None):
        """Generate PSF image for a given star.

        .. note::

            If the star already has the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> star = psf.interpolateList(star)

            before running `drawStar`.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]
        :param center:      An optional tuple (x,y) location for where to center the drawn profile
                            in the image. [default: None, which draws at the star's location.]

        :returns:           Star instance with its image filled with rendered PSF
        """
        # Interpolate parameters to this position/properties (if not already done):
        if star.fit is None or star.fit.params is None:
            star = self.interpolateStar(star)
        # Render the image
        return self._drawStar(star, copy_image=copy_image, center=center)

    def _drawStar(self, star, copy_image=True, center=None):
        # Derived classes may choose to override any of the above functions
        # But they have to at least override this one and interpolateStar to implement
        # their actual PSF model.
        raise NotImplementedError("Derived classes must define the _drawStar function")

    def _getProfile(self, star):
        raise NotImplementedError("Derived classes must define the _getProfile function")

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
        try:
            psf_type = str(psf_type[0].decode())
        except AttributeError:
            # fitsio 1.0 returns strings
            psf_type = psf_type[0]

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

    def writeWCS(self, fits, extname, logger):
        """Write the WCS information to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write to
        :param logger:      A logger object for logging debug info.
        """
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        logger = galsim.config.LoggerWrapper(logger)

        # Start with the chipnums
        chipnums = list(self.wcs.keys())
        cols = [ chipnums ]
        dtypes = [ ('chipnums', int) ]

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
        try:
            wcs_str = [ b''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each
        except TypeError:  # pragma: no cover
            # fitsio 1.0 returns strings
            wcs_str = [ ''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each

        wcs_str = [ base64.b64decode(s) for s in wcs_str ] # Convert back from b64 encoding
        # Convert back into wcs objects
        try:
            wcs_list = [ pickle.loads(s, encoding='bytes') for s in wcs_str ]
        except Exception:
            # If the file was written by py2, the bytes encoding might raise here,
            # or it might not until we try to use it.
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]

        wcs = dict(zip(chipnums, wcs_list))

        try:
            # If this doesn't work, then the file was probably written by py2, not py3
            repr(wcs)
        except Exception:
            logger.info('Failed to decode wcs with bytes encoding.')
            logger.info('Retry with encoding="latin1" in case file written with python 2.')
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]
            wcs = dict(zip(chipnums, wcs_list))
            repr(wcs)

        # Work-around for a change in the GalSim API with 2.0
        # If the piff file was written with pre-2.0 GalSim, this fixes it.
        for key in wcs:
            w = wcs[key]
            if hasattr(w, '_origin') and  isinstance(w._origin, galsim._galsim.PositionD):
                w._origin = galsim.PositionD(w._origin)

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
