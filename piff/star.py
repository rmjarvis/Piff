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
.. module:: stardata
"""

from __future__ import print_function
import numpy as np
import copy
import galsim

class Star(object):
    """Information about a "star", which may be either a real star or an interpolated star
    at some target location.

    The Star object is the fundamental way that Piff keeps track of information connected to
    a particular observation location.  This includes both the input data (postage stamp image,
    position on the detector, position in the field of view, weight image, etc.) and derived
    information connected to whatever Model is being used


    Stars are not normally constructed directly by the user.  They are built by various
    Piff functions such as:

        stars = input_handler.makeStars(logger)
        stars, wcs = piff.Input.process(config['input'])
        target_star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, ...)

    However, a star can be constructed directly from a StarData instance and a StarFit instance.
    The former keeps track of the data about the star (either as observed or the model at a
    location) and the latter keeps track of fitted parameters related to fitting the data to
    a model.

        star = piff.Star(star_data, star_fit)

    Stars have an immutable design, so any functions that change either the data or the fitted
    parameters return a new object and  don't change the original.  e.g.

        star = psf.drawStar(star)
        star = star.reflux()
        star = star.addPoisson(gain=gain)

    Stars have the following attributes:

        star.data       The component StarData object
        star.fit        The component StarFit object

    and the following read-only properties:

        star.image      The image of the star
        star.weight     The weight map connected to the image data
        star.image_pos  The position of the star in image coordinates, aka (x,y)
        star.field_pos  The position of the star in field coordinates, aka (u,v)
        star.x          The x position of the star in image coordinates
        star.y          The y position of the star in image coordinates
        star.u          The u position of the star in field coordinates
        star.v          The v position of the star in field coordinates
        star.chipnum    The chip number where this star was observed (or would be observed)
        star.flux       The flux of the object
        star.center     The nominal center of the object (not necessarily the centroid)
    """
    def __init__(self, data, fit):
        """Constructor for Star instance.

        :param data: A StarData instance (invariant)
        :param fit:  A StarFit instance (invariant)
        """
        self.data = data
        if fit is None:
            fit = StarFit(None, flux=1.0, center=(0.,0.))
        self.fit = fit

    def withFlux(self, flux=None, center=None):
        """Update the flux and/or center values

        :param flux:    The new flux.  [default: None, which means keep the existing value.]
        :param center:  The new center.  [default: None, which means keep the existing value.]
        """
        fit = self.fit.copy()
        if flux is not None:
            fit.flux = flux
        if center is not None:
            fit.center = center
        return Star(self.data, fit)

    def __getitem__(self, key):
        """Get a property of the star.

        This may be one of the values in the properties dict that was given when the data object
        was initialized, or one of 'x', 'y', 'u', 'v', where x,y are the position in image
        coordinates and u,v are the position in field coordinates.

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        # This is a StarData method.  Just pass the request on to it.
        return self.data[key]

    # Some properties that pass through to the data attribute to make using them easier.
    @property
    def image(self):
        return self.data.image

    @property
    def weight(self):
        return self.data.weight

    @property
    def image_pos(self):
        return self.data.image_pos

    @property
    def field_pos(self):
        return self.data.field_pos

    @property
    def x(self):
        return self.data.image_pos.x

    @property
    def y(self):
        return self.data.image_pos.y

    @property
    def u(self):
        return self.data.field_pos.x

    @property
    def v(self):
        return self.data.field_pos.y

    @property
    def chipnum(self):
        if 'chipnum' in self.data.properties:
            return self.data.properties['chipnum']
        else:
            return 0

    @property
    def flux(self):
        return self.fit.flux

    @property
    def center(self):
        return self.fit.center

    @classmethod
    def makeTarget(cls, x=None, y=None, u=None, v=None, properties={}, wcs=None, scale=None,
                   stamp_size=48, image=None, pointing=None, flux=1.0, **kwargs):
        """
        Make a target Star object with the requested properties.

        The image will be blank (all zeros), and the properties field will match the given
        input properties.

        The input properties must have either 'x' and 'y' or 'u' and 'v'.  The other pair will
        be calculated from the wcs if one is provided, or you may pass in both sets of coordinates
        and leave out the wcs

        :param x:           The image x position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param y:           The image y position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param u:           The image u position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param v:           The image v position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param properties:  The requested properties for the target star, including any other
                            requested properties besides x,y,u,v.  You may also provide x,y,u,v
                            in this dict rather than explicitly as kwargs. [default: None]
        :param wcs:         The requested WCS.  [optional]
        :param scale:       If wcs is None, you may instead provide a pixel scale. [default: None]
        :param stamp_size:  The size in each direction of the (blank) image. [default: 48]
        :param image:       An existing image to use instead of making a new one, if desired.
                            [default: None; this overrides stamp_size]
        :param pointing:    The pointing direction to use. [default: None]
        :param flux:        The flux of the target star. [default: 1]
        :param **kwargs:    Additional properties can also be given as keyword arguments if that
                            is more convenient than populating the properties dict.

        :returns:   A Star instance
        """
        # Check that input parameters are valid
        for param in ['x', 'y', 'u', 'v']:
            if eval(param) is not None and param in properties:
                raise AttributeError("%s may not be given both as a kwarg and in properties"%param)
        properties = properties.copy()  # So we can modify it and not mess up the caller.
        x = properties.pop('x', x)
        y = properties.pop('y', y)
        u = properties.pop('u', u)
        v = properties.pop('v', v)
        properties.update(kwargs)  # Add any extra kwargs into properties
        if (x is None) != (y is None):
            raise AttributeError("Either x and y must both be given, or neither.")
        if (u is None) != (v is None):
            raise AttributeError("Either u and v must both be given, or neither.")
        if x is None and u is None:
            raise AttributeError("Some kind of position must be given.")
        if wcs is not None and scale is not None:
            raise AttributeError("Scale is invalid when also providing wcs.")

        # Figure out what the wcs should be if not provided
        if wcs is None:
            if scale is None:
                scale = 1.
            wcs = galsim.PixelScale(scale)

        # Make the blank image
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)

        # Figure out the image_pos
        if x is None:
            field_pos = galsim.PositionD(float(u),float(v))
            image_pos = wcs.toImage(field_pos)
            x = image_pos.x
            y = image_pos.y
        else:
            image_pos = galsim.PositionD(float(x),float(y))

        # Make wcs locally accurate affine transformation
        if x is not None:
            if u is not None:
                field_pos = galsim.PositionD(float(u),float(v))
                wcs = wcs.local(image_pos).withOrigin(image_pos, field_pos)
            else:
                field_pos = None

        # Make the center of the image (close to) the image_pos
        image.setCenter(int(x+0.5), int(y+0.5))
        if image.wcs is None:
            image.wcs = wcs

        # Build the StarData instance
        data = StarData(image, image_pos, field_pos=field_pos, properties=properties, 
                        pointing=pointing)
        fit = StarFit(None, flux=flux, center=(0.,0.))
        return cls(data, fit)

    @classmethod
    def write(self, stars, fits, extname):
        """Write a list of stars to a FITS file.

        :param stars:       A list of stars to write
        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write to
        """
        import galsim
        # TODO This doesn't write everything out.  Probably want image as an optional I/O.

        cols = []
        dtypes = []

        # Start with the data properties
        prop_keys = list(stars[0].data.properties)
        # Do the position ones first
        for key in [ 'x', 'y', 'u', 'v' ]:
            dtypes.append( (key, float) )
            cols.append( [ s.data.properties[key] for s in stars ] )
            prop_keys.remove(key)
        # Add any remaining properties
        for key in prop_keys:
            if hasattr(stars[0].data.properties[key], '__iter__'):
                dtypes.append( (key, float, len(stars[0].data.properties[key])))
                cols.append( [ s.data.properties[key] for s in stars ])
            else:
                dtypes.append( (key, float) )
                cols.append( [ s.data.properties[key] for s in stars ] )

        # Add the local WCS values
        dtypes.extend( [('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float) ] )
        cols.append( [s.data.local_wcs.jacobian().dudx for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dudy for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dvdx for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dvdy for s in stars] )

        # Add the bounds
        dtypes.extend( [('xmin', int), ('xmax', int), ('ymin', int), ('ymax', int) ] )
        cols.append( [s.data.image.bounds.xmin for s in stars] )
        cols.append( [s.data.image.bounds.xmax for s in stars] )
        cols.append( [s.data.image.bounds.ymin for s in stars] )
        cols.append( [s.data.image.bounds.ymax for s in stars] )

        # Now the easy parts of fit:
        dtypes.extend( [ ('flux', float), ('center', float, 2), ('chisq', float) ] )
        cols.append( [ s.fit.flux for s in stars ] )
        cols.append( [ s.fit.center for s in stars ] )
        cols.append( [ s.fit.chisq for s in stars ] )

        # params might not be set, so check if it is None
        if stars[0].fit.params is not None:
            dtypes.append( ('params', float, len(stars[0].fit.params)) )
            cols.append( [ s.fit.params for s in stars ] )

        # If pointing is set, write that
        if stars[0].data.pointing is not None:
            dtypes.extend( [('point_ra', float), ('point_dec', float)] )
            cols.append( [s.data.pointing.ra / galsim.degrees for s in stars ] )
            cols.append( [s.data.pointing.dec / galsim.degrees for s in stars ] )

        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname)

    @classmethod
    def read(cls, fits, extname):
        """Read stars from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to read from

        :returns: a list of Star instances
        """
        import galsim
        assert extname in fits
        colnames = fits[extname].get_colnames()

        for key in ['x', 'y', 'u', 'v',
                    'dudx', 'dudy', 'dvdx', 'dvdy',
                    'xmin', 'xmax', 'ymin', 'ymax',
                    'flux', 'center', 'chisq']:
            assert key in colnames
            colnames.remove(key)

        data = fits[extname].read()
        x_list = data['x']
        y_list = data['y']
        u_list = data['u']
        v_list = data['v']
        dudx = data['dudx']
        dudy = data['dudy']
        dvdx = data['dvdx']
        dvdy = data['dvdy']
        xmin = data['xmin']
        xmax = data['xmax']
        ymin = data['ymin']
        ymax = data['ymax']
        flux = data['flux']
        center = data['center']
        chisq = data['chisq']

        if 'params' in colnames:
            params = data['params']
            colnames.remove('params')
        else:
            params = [ None ] * len(data)

        if 'point_ra' in colnames:
            pointing_list = [ galsim.CelestialCoord(row['point_ra'] * galsim.degrees,
                                                    row['point_dec'] * galsim.degrees)
                              for row in data ]
            colnames.remove('point_ra')
            colnames.remove('point_dec')
        else:
            pointing_list = [ None ] * len(data)

        fit_list = [ StarFit(p, flux=f, center=c, chisq=x)
                     for (p,f,c,x) in zip(params, flux, center, chisq) ]

        # The rest of the columns are the data properties
        prop_list = [ { c : row[c] for c in colnames } for row in data ]

        wcs_list = [ galsim.JacobianWCS(*jac) for jac in zip(dudx,dudy,dvdx,dvdy) ]
        pos_list = [ galsim.PositionD(*pos) for pos in zip(x_list,y_list) ]
        wpos_list = [ galsim.PositionD(*pos) for pos in zip(u_list,v_list) ]
        wcs_list = [ w.withOrigin(p, wp) for w,p,wp in zip(wcs_list, pos_list, wpos_list) ]
        bounds_list = [ galsim.BoundsI(*b) for b in zip(xmin,xmax,ymin,ymax) ]
        image_list = [ galsim.Image(bounds=b, wcs=w) for b,w in zip(bounds_list, wcs_list) ]
        weight_list = [ galsim.Image(bounds=b, wcs=w) for b,w in zip(bounds_list, wcs_list) ]
        data_list = [ StarData(im, pos, weight=w, properties=prop, pointing=point)
                      for im,pos,w,prop,point in zip(image_list, pos_list, weight_list,
                                                     prop_list, pointing_list) ]

        stars = [ Star(d,f) for (d,f) in zip(data_list, fit_list) ]
        return stars

    @staticmethod
    def load_images(stars, file_name, pointing=None,
                    image_hdu=None, weight_hdu=None, badpix_hdu=None, sky=None,
                    logger=None):
        """Load the image data into a list of Stars.

        We don't store the image data for Stars when we write them to a file, since that
        would take up a lot of space and is usually not desired.  However, we do store the
        bounds in the original image where the star was cutout, so if you want to load back in
        the original data from the image file, you can do so with this function.

        :param stars:           A list of Star instances.
        :param file_name:       The file with the image data for these stars.
        :param pointing:        The pointing direction to use. [default: None]
        :param image_hdu:       The hdu to use for the main image. [default: None, which means
                                either 0 or 1 as appropriate according to the compression.]
        :param weight_hdu:      The hdu to use for the weight image. [default: None]
        :param badpix_hdu:      The hdu to use for the bad pixel mask. [default: None]
        :param sky:             Optional sky image or float value to subtract from the main
                                image. [default: None]
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a new list of Stars with the images information loaded.
        """
        import galsim
        # TODO: This is largely copied from InputHandler.readImages.
        #       This should probably be refactored a bit to avoid the duplicated code.
        logger = galsim.config.LoggerWrapper(logger)
        logger.info("Loading image information from file %s",file_name)
        image = galsim.fits.read(file_name, hdu=image_hdu)

        if sky is not None:
            image = image - sky

        # Either read in the weight image, or build a dummy one
        if weight_hdu is None:
            logger.debug("Making trivial (wt==1) weight image")
            weight = galsim.ImageI(image.bounds, init_value=1)
        else:
            logger.info("Reading weight image from hdu %d.", weight_hdu)
            weight = galsim.fits.read(file_name, hdu=weight_hdu)
            if np.all(weight.array == 0):
                logger.error("According to the weight mask in %s, all pixels have zero weight!",
                             file_name)

        # If requested, set wt=0 for any bad pixels
        if badpix_hdu is not None:
            logger.info("Reading badpix image from hdu %d.", badpix_hdu)
            badpix = galsim.fits.read(file_name, hdu=badpix_hdu)
            # The badpix image may be offset by 32768 from the true value.
            # If so, subtract it off.
            if np.any(badpix.array > 32767):
                logger.debug('min(badpix) = %s',np.min(badpix.array))
                logger.debug('max(badpix) = %s',np.max(badpix.array))
                logger.debug("subtracting 32768 from all values in badpix image")
                badpix -= 32768
            if np.any(badpix.array < -32767):
                logger.debug('min(badpix) = %s',np.min(badpix.array))
                logger.debug('max(badpix) = %s',np.max(badpix.array))
                logger.debug("adding 32768 to all values in badpix image")
                badpix += 32768
            # Also, convert to int16, in case it isn't by default.
            badpix = galsim.ImageS(badpix)
            if np.all(badpix.array != 0):
                logger.error("According to the bad pixel array in %s, all pixels are masked!",
                                file_name)
            weight.array[badpix.array != 0] = 0

        stars = [ Star(data = StarData(image=image[star.data.image.bounds],
                                       image_pos=star.data.image_pos,
                                       weight=weight[star.data.image.bounds],
                                       pointing= (pointing if pointing is not None
                                                  else star.data.pointing),
                                       values_are_sb=star.data.values_are_sb,
                                       properties=star.data.properties,
                                       _xyuv_set=True),
                       fit = star.fit)
                  for star in stars ]
        return stars


    def offset_to_center(self, offset):
        """A utility routine to convert from an offset in image coordinates to the corresponding
        center position in focal plane coordinates on the postage stamp image.

        :param offset:      A tuple (dx,dy) in image coordinates

        :returns:           The corresponding (du,dv) in focal plane coordinates.
        """
        import galsim
        # The model is in sky coordinates, so figure out what (u,v) corresponds to this offset.
        jac = self.data.image.wcs.jacobian(self.data.image.true_center)
        dx, dy = offset
        du = jac.dudx * dx + jac.dudy * dy
        dv = jac.dvdx * dx + jac.dvdy * dy
        return (du,dv)

    def center_to_offset(self, center):
        """A utility routine to convert from a center position in focal plane coordinates to the
        corresponding offset in image coordinates on the postage stamp image.

        :param center:      A tuple (u,v) in focal plane coordinates

        :returns:           The corresponding (dx,dy) in image coordinates.
        """
        import galsim
        jac = self.data.image.wcs.jacobian(self.data.image.true_center).inverse()
        du, dv = center
        # The names (u,v) and (x,y) are reversed for jac, since we've taken its inverse,
        # so this looks a little confusing.  e.g. jac.dudx is really (dx/du), etc.
        dx = jac.dudx * du + jac.dudy * dv
        dy = jac.dvdx * du + jac.dvdy * dv
        return (dx,dy)

    def addPoisson(self, signal=None, gain=None):
        """Return new Star with the weight values altered to reflect
        Poisson shot noise from a signal source, e.g. when the weight
        only contains variance from background and read noise.

        :param signal:  The signal (as a Star instance) from which the Poisson variance is
                        extracted.  If None, the data image is used.  All signals are
                        clipped from below at zero.
        :param gain:    The gain, in e per ADU, assumed in calculating new weights.  If None
                        is given, then the 'gain' property is used, else defaults gain=1.

        :returns: a new Star instance with updated weight array.
        """
        # The functionality is implemented as a StarData method.
        # So just pass this task on to that and recast the return value as a Star instance.
        return Star(self.data.addPoisson(signal=signal, gain=gain), self.fit)


class StarData(object):
    """A class that encapsulates all the relevant information about an observed star.

    **Class intended to be immutable once returned from the method that creates it.**

    This includes:
      - a postage stamp of the imaging data
      - the weight map for these pixels (zero weight means the pixel is masked or otherwise bad)
      - whether the pixel values represent flux or surface brightness
      - the pixel area on the sky
      - the position of the star on the image
      - the position of the star in the full field-of-view (local tangent plane projection)
      - possibly extra information, such as colors

    A StarData object can render the pixel data in one of two ways:

      - getImage() returns the pixels as a galsim.Image.
      - getDataVector() returns the pixels as a numpy array.

    Different use cases may prefer the data in one of these forms or the other.

    A StarData object also must have these two properties:
      :property values_are_sb: True (False) if pixel values are in surface
      brightness (flux) units.
      :property pixel_area:    Solid angle on sky subtended by each pixel,
      in units of the uv system.

    The other information is stored in a dict.  This dict will include at least the following
    items:

      - x
      - y
      - u
      - v

    And anything else you want to store.  e.g.

      - chipnum
      - ra
      - dec
      - color_ri
      - color_iz
      - gain

    Any of these values may be used for interpolation.  The most typical choices would be either
    (x,y) or (u,v).  The choice of what values to use is made by the interpolator.  All such
    values should be either int or float values.

    Note that the position given for the star does not have to be a proper centroid.  It is
    rather just some position in image coordinates to use as the origin of the model.

    :param image:       A postage stamp image that includes the star
    :param image_pos:   The position in image coordinates to use as the "center" of the star.
                        Note: this does not have to be the centroid or anything specific about the
                        star.  It is merely the image position of the (0,0) coordinate for the
                        model's internal coordinate system.
    :param weight:      The corresponding weight map for that image. [default: None]
    :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                        exposure.  It does not have to be the exact center of the field of view.
                        But it should be the same for all StarData objects in the exposure.
                        This is required if image.wcs is a CelestialWCS, but should be None
                        if image.wcs is a EuclideanWCS. [default: None]
    :param field_pos:   Optionally provide the field_pos directly, rather than calculating it from
                        the wcs and a pointing. [default: None]
    :param properties:  A dict containing other properties about the star that might be of
                        interest. [default: None]
    :param values_are_sb: True if pixel data give surface brightness, False if they're flux
                        [default: False]
    :param orig_weight: The original weight map prior to any additional Poisson variance being
                        added.  [default: None, which means use orig_weight=weight]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, image, image_pos, weight=None, pointing=None, field_pos=None,
                 properties=None, values_are_sb=False, orig_weight=None, logger=None,
                 _xyuv_set=False):
        import galsim
        # Save all of these as attributes.
        self.image = image
        self.image_pos = image_pos
        self.values_are_sb = values_are_sb
        # Make sure we have a local wcs in case the provided image is more complex.
        self.local_wcs = image.wcs.local(image_pos)

        if weight is None:
            self.weight = galsim.Image(image.bounds, init_value=1, wcs=image.wcs, dtype=float)
        elif type(weight) in [int, float]:
            self.weight = galsim.Image(image.bounds, init_value=weight, wcs=image.wcs, dtype=float)
        elif isinstance(weight, galsim.Image):
            # Work-around for bug in GalSim 1.3
            self.weight = galsim.Image(weight, dtype=float, wcs=weight.wcs)
        else:
            self.weight = galsim.Image(weight, dtype=float)

        if orig_weight is None:
            self.orig_weight = self.weight
        else:
            self.orig_weight = orig_weight

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties

        self.pointing = pointing
        if field_pos is None:
            self.field_pos = self.calculateFieldPos(image_pos, image.wcs, pointing, self.properties)
        else:
            self.field_pos = field_pos
        self.pixel_area = self.local_wcs.pixelArea()

        # Make sure the user didn't provide their own x,y,u,v in properties.
        for key in ['x', 'y', 'u', 'v']:
            if properties is not None and key in properties and not _xyuv_set:
                raise AttributeError("Cannot provide property %s in properties dict."%key)

        self.properties['x'] = self.image_pos.x
        self.properties['y'] = self.image_pos.y
        self.properties['u'] = self.field_pos.x
        self.properties['v'] = self.field_pos.y

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def calculateFieldPos(image_pos, wcs, pointing, properties=None):
        """
        Convert from image coordinates to field coordinates.

        :param image_pos:   The position in image coordinates, as a galsim.PositionD instance.
        :param wcs:         The wcs to use to connect image coordinates with sky coordinates:
        :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                            exposure.  This is required if image.wcs is a CelestialWCS, but should
                            be None if image.wcs is a EuclideanWCS. [default: None]
        :param properties:  If properties is provided, and the wcs is a CelestialWCS, then add
                            'ra', 'dec' properties (in hours, degrees respectively) based on the
                            sky coordinates. [default: None]

        :returns: a galsim.PositionD instance representing the position in field coordinates.
        """
        import galsim
        # Calculate the field_pos, the position in the fov coordinates
        if wcs.isCelestial():
            if pointing is None:
                raise AttributeError("If the image uses a CelestialWCS then pointing is required.")
            sky_pos = wcs.toWorld(image_pos)
            if properties is not None:
                if 'ra' not in properties:
                    properties['ra'] = sky_pos.ra / galsim.hours
                if 'dec' not in properties:
                    properties['dec'] = sky_pos.dec / galsim.degrees
            if galsim.__version__ >= '2.0':
                u, v = pointing.project(sky_pos)
                return galsim.PositionD(u/galsim.arcsec, v/galsim.arcsec)
            else:
                return pointing.project(sky_pos)
        else:
            return wcs.toWorld(image_pos)

    def __getitem__(self, key):
        """Get a property of the star.

        This may be one of the values in the properties dict that was given when the object
        was initialized, or one of 'x', 'y', 'u', 'v', where x,y are the position in image
        coordinates and u,v are the position in field coordinates.

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        return self.properties[key]

    def getImage(self):
        """Get the pixel data as a galsim.Image.

        Also returns the weight image and the origin position (the position in x,y coordinates
        to use as the origin of the PSF model).

        :returns: image, weight, image_pos
        """
        return self.image, self.weight, self.image_pos

    def getDataVector(self, include_zero_weight=False):
        """Get the pixel data as a numpy array.

        Also returns the weight values and the local u,v coordinates of the pixels.
        Any pixels with zero weight (e.g. from masking in the original image) will not be
        included in the returned arrays.

        :param include_zero_weight: Should points with zero weight be included? [default: False]

        :returns: data_vector, weight_vector, u_vector, v_vector
        """
        nx = self.image.bounds.xmax - self.image.bounds.xmin + 1
        ny = self.image.bounds.ymax - self.image.bounds.ymin + 1
        # Alternatively, this also works.  Just need to remember that numpy arrays use y,x indexing.
        #ny, nx = self.image.array.shape

        # Image coordinates of pixels relative to nominal center
        xvals = np.arange(self.image.bounds.xmin, self.image.bounds.xmax+1, dtype=float)
        yvals = np.arange(self.image.bounds.ymin, self.image.bounds.ymax+1, dtype=float)
        x,y = np.meshgrid(xvals, yvals)
        x -= self.image_pos.x
        y -= self.image_pos.y

        # Convert to u,v coords
        u = self.local_wcs._u(x,y)
        v = self.local_wcs._v(x,y)

        # Get flat versions of everything
        u = u.flatten()
        v = v.flatten()
        pix = self.image.array.flatten()
        wt = self.weight.array.flatten()

        # Which pixels do we want to return?
        if include_zero_weight:
            return pix, wt, u, v
        else:
            mask = wt != 0.
            return pix[mask], wt[mask], u[mask], v[mask]

    def setData(self, data, include_zero_weight=False):
        """Return new StarData with data values replaced by elements of provided 1d array.
        The array should match the ordering of the one that is produced by getDataVector().

        :param data:                A 1d numpy array with new values for the image data.
        :param include_zero_weight: If True, the data array includes all pixels.
                                    If False, it only includes the pixels with weight > 0.
                                    [default: False]

        :returns:    New StarData structure
        """
        # ??? Do we need a way to fill in pixels that have zero weight and
        # don't get passed out by getDataVector()???

        newimage = self.image.copy()
        if include_zero_weight:
            newimage.array[:,:] = data.reshape(newimage.array.shape)
        else:
            ignore = self.weight.array==0.
            newimage.array[ignore] = 0.
            newimage.array[~ignore] = data

        return StarData(image=newimage,
                        image_pos=self.image_pos,
                        weight=self.weight,
                        orig_weight=self.orig_weight,
                        pointing=self.pointing,
                        field_pos=self.field_pos,
                        values_are_sb=self.values_are_sb,
                        properties=self.properties,
                        _xyuv_set=True)

    def addPoisson(self, signal=None, gain=None):
        """Return new StarData with the weight values altered to reflect
        Poisson shot noise from a signal source, e.g. when the weight
        only contains variance from background and read noise.

        :param signal:  The signal (as a Star instance) from which the Poisson variance is
                        extracted.  If None, the data image is used.  All signals are
                        clipped from below at zero.
        :param gain:    The gain, in e per ADU, assumed in calculating new weights.  If None
                        is given, then the 'gain' property is used, else defaults gain=1.

        :returns: a new StarData instance with updated weight array.
        """
        import galsim

        # Get the gain.  None both here and in properties, means don't add any variance.
        if gain is None:
            gain = self.properties.get('gain',None)
        if gain is None:
            return self

        # Mark the pixels that are not already worthless
        use = self.orig_weight.array!=0.

        # Get the signal data
        if signal is None:
            signal = self
        variance = signal.image.array
        if variance.shape != self.orig_weight.array.shape:
            raise ValueError('In addPoisson, signal has wrong shape: %s != %s'%(
                    variance.shape, self.orig_weight.array.shape))

        # clip variance
        use &= (variance >= 0)

        # Add to weight
        # Note: use the original weight here, not the current weight, since this may have already
        # had some Poisson noise added to the weight.
        newweight = self.orig_weight.copy()
        newweight.array[use] = 1. / (1./self.orig_weight.array[use] + variance[use] / gain)

        # Return new object
        return StarData(image=self.image,
                        image_pos=self.image_pos,
                        weight=newweight,
                        orig_weight=self.orig_weight,
                        pointing=self.pointing,
                        values_are_sb=self.values_are_sb,
                        properties=dict(self.properties, gain=gain),
                        _xyuv_set = True)

    def maskPixels(self, mask):
        """Return new StarData with weight nulled at pixels marked as False in the mask.
        Note that this cannot un-mask any previously nulled pixels.

        :param mask:      Boolean array with False marked in pixels that should henceforth
                          be ignored in fitting.
                          If this is a 2d array it is assumed to match the weight image.
                          If it is a 1d array, it is assumed to match the vectors returned by
                          getDataVector().  If None, the self.image is used, clipped from below

        :returns:         A new StarData instance with updated weight array.
        """

        # Mark the pixels that are not already worthless
        use = self.weight.array!=0.

        # Get the signal data
        if len(mask.shape)==2:
            m = mask[use]
        else:
            # Insert 1d vector into currently valid pixels
            m = mask

        # Zero appropriate weight pixels in new copy
        weight = self.weight.copy()
        weight.array[use] = np.where(m, self.weight.array[use], 0.)

        if self.orig_weight != self.weight:
            orig_weight = self.orig_weight.copy()
            orig_weight.array[use] = np.where(m, self.orig_weight.array[use], 0.)
        else:
            orig_weight = weight

        # Return new object
        return StarData(image=self.image,
                        image_pos=self.image_pos,
                        weight=weight,
                        orig_weight=orig_weight,
                        pointing=self.pointing,
                        values_are_sb=self.values_are_sb,
                        properties=self.properties,
                        _xyuv_set=True)


class StarFit(object):
    """Class to hold the results of fitting a Model to some StarData, or specify
    the PSF interpolated to an unmeasured location.

    **Class is intended to be invariant once created.**

    This class can be extended
    to carry information of use to a given Model instance (such as intermediate
    results), but interpolators will be looking for some subset of these properties:

    :params:      numpy vector of parameters of the PSF that apply to this star
    :params_var:  numpy array of variance error parameters of the PSF
    :flux:        flux of the star
    :center:      (u,v) tuple giving position of stellar center (relative
                  to data.image_pos)
    :chisq:       Chi-squared of  fit to the data (if any) with current params
    :dof:         Degrees of freedom in the fit (will depend on whether last fit had
                  parameters free or just the flux/center).
    :alpha, beta: matrix, vector, giving Taylor expansion of chisq wrt params about
                  their current values. The alpha matrix also is the inverse covariance
                  matrix of the params.

    The params and alpha,beta,chisq are assumed to be marginalized over flux (and over center,
    if it is free to vary).
    """
    def __init__(self, params, flux=1., center=(0.,0.), params_var=None, alpha=None, beta=None,
                 chisq=None, dof=None, worst_chisq=None):
        """Constructor for base version of StarFit

        :param params: A 1d numpy array holding estimated PSF parameters
        :param params_var: A 1d numpy array holding estimates PSF variance error parameters
        :param flux:   Estimated flux for this star
        :param center: Estimated or fixed center position (u,v) of this star relative to
                       the StarData.image_pos reference point.
        :param alpha:  Quadratic dependence of chi-squared on params about current values
        :param beta:   Linear dependence of chi-squared on params about current values
        :param chisq:  chi-squared value at current parameters.
        :param worst_chisq:  highest chi-squared in any single pixel, after reflux()
        """
        # center might be a galsim.PositionD.  That's fine, but we'll convert to a tuple here.
        try:
            center = (center.x, center.y)
        except AttributeError:
            pass

        self.params = params
        self.params_var = params_var
        self.flux = flux
        self.center = center
        self.alpha = alpha
        self.beta = beta
        self.chisq = chisq
        self.dof = dof
        self.worst_chisq = worst_chisq
        return

    def newParams(self, p):
        """Return new StarFit that has the array p installed as new parameters.

        :param params:  A 1d array holding new parameters; must match size of current ones

        :returns:  New StarFit object with altered parameters.  All chisq-related parameters
                   are set to None since they are no longer valid.
        """
        npp = np.array(p)
        if self.params is not None and npp.shape != self.params.shape:
            raise TypeError('new StarFit parameters do not match dimensions of old ones')
        return StarFit(npp, flux=self.flux, center=self.center)

    def copy(self):
        return StarFit(self.params, self.flux, self.center, self.alpha, self.beta,
                       self.chisq, self.dof, self.worst_chisq)

    def __getitem__(self, key):
        """Get a property of the star fit.

        Looks at params to get the property

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        return self.params[key]
