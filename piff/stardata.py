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
import numpy

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
    (x,y) or (u,v).  The choice of what values to use is made by the interpolator.

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
    :param properties:  A dict containing other properties about the star that might be of 
                        interest. [default: None]
    :param values_are_sb: True if pixel data give surface brightness, False if they're flux
                          [default: False]
    """
    def __init__(self, image, image_pos, weight=None, pointing=None, values_are_sb=False,
                 properties=None, logger=None):
        import galsim
        # Save all of these as attributes.
        self.image = image
        self.image_pos = image_pos
        self.values_are_sb = values_are_sb
        # Make sure we have a local wcs in case the provided image is more complex.
        self.local_wcs = image.wcs.local(image_pos)

        if weight is None:
            self.weight = galsim.Image(image.bounds, init_value=1, wcs=image.wcs)
        elif type(weight) is float:
            self.weight = galsim.Image(image.bounds, init_value=weight, wcs=image.wcs)
        else:
            self.weight = weight

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties

        self.pointing = pointing
        self.field_pos = self.calculateFieldPos(image_pos, image.wcs, pointing, self.properties)
        self.pixel_area = self._calculate_pixel_area()

        # Make sure the user didn't provide their own x,y,u,v in properties.
        for key in ['x', 'y', 'u', 'v']:
            if properties is not None and key in properties:
                raise AttributeError("Cannot provide property %s in properties dict."%key)

        self.properties['x'] = self.image_pos.x
        self.properties['y'] = self.image_pos.y
        self.properties['u'] = self.field_pos.x
        self.properties['v'] = self.field_pos.y

    @classmethod
    def makeTarget(cls, x=None, y=None, u=None, v=None, properties={}, wcs=None, scale=None,
                   stamp_size=48):
        """
        Make a target StarData object with the requested properties.

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
        :param wcs:         The requested WCS.  [optional; invalid if both (x,y) and (u,v) are
                            given.]
        :param scale:       If wcs is None, you may instead provide a pixel scale. [default: None]
        :param stamp_size:  The size in each direction of the (blank) image. [default: 48]
        
        :returns:   A StarData instance
        """
        import galsim
        # Check than input parameters are valid
        for param in ['x', 'y', 'u', 'v']:
            if eval(param) is not None and param in properties:
                raise AttributeError("%s may not be given both as a kwarg and in properties"%param)
        properties = properties.copy()  # So we can modify it and not mess up the caller.
        x = properties.pop('x', x)
        y = properties.pop('y', y)
        u = properties.pop('u', u)
        v = properties.pop('v', v)
        if (x is None) != (y is None):
            raise AttributeError("Eitehr x and y must both be given, or neither.")
        if (u is None) != (v is None):
            raise AttributeError("Eitehr u and v must both be given, or neither.")
        if x is None and u is None:
            raise AttributeError("Some kind of position must be given")
        if x is not None and u is not None and wcs is not None:
            raise AttributeError("wcs may not be given along with (x,y) and (u,v)")
        if wcs is not None and scale is not None:
            raise AttributeError("scale is invalid when also providing wcs")
        if wcs is not None and wcs.isCelestial():
            raise AttributeError("A CelestialWCS is not allowed")

        # Figure out what the wcs should be if not provided
        if wcs is None:
            if scale is None:
                scale = 1.
            wcs = galsim.PixelScale(scale)
            if x is not None and u is not None:
                wcs = wcs.withOrigin(galsim.PositionD(x,y), galsim.PositionD(u,v))

        # Make the blank image
        image = galsim.Image(stamp_size, stamp_size, wcs=wcs)

        # Figure out the image_pos
        if x is None:
            image_pos = wcs.toImage(galsim.PositionD(u,v))
        else:
            image_pos = galsim.PositionD(x,y)

        # Make the center of the iamge (close to) the image_pos
        image.setCenter(int(x)+1, int(y)+1)

        # Build the StarDat instance
        return cls(image, image_pos, properties=properties)


    @staticmethod
    def calculateFieldPos(image_pos, wcs, pointing, properties=None):
        """
        Convert from image coordinates to field coordinates.

        :param image_pos:   The position in image coordinates.
        :param wcs:         The wcs to use to connect image coordinates with sky coordinates:
        :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                            exposure.  This is required if image.wcs is a CelestialWCS, but should
                            be None if image.wcs is a EuclideanWCS. [default: None]
        :param properties:  If properties is provided, and the wcs is a CelestialWCS, then add
                            'ra', 'dec' properties based on the sky coordinates. [default: None]
        """
        # Calculate the field_pos, the position in the fov coordinates
        if pointing is None:
            if wcs.isCelestial():
                raise AttributeError("If the image uses a CelestialWCS then pointing is required.")
            return wcs.toWorld(image_pos)
        else:
            if not wcs.isCelestial():
                raise AttributeError("Cannot provide pointing unless the image uses a CelestialWCS")
            sky_pos = wcs.toWorld(image_pos)
            if properties is not None:
                if 'ra' not in properties:
                    properties['ra'] = sky_pos.ra
                if 'dec' not in properties:
                    properties['dec'] = sky_pos.dec
            return pointing.project(sky_pos)

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

    def _calculate_pixel_area(self):
        """Calculate uv-plane pixel area from a finite difference

        :returns: pixel area
        """
        dpix = 5.
        x = numpy.array([-dpix,+dpix,0.,0.])
        y = numpy.array([0.,0.,-dpix,+dpix])
        # Convert to u,v coords
        u = self.local_wcs._u(x,y)
        v = self.local_wcs._v(x,y)
        dudx = (u[1]-u[0])/(2*dpix)
        dudy = (u[3]-u[2])/(2*dpix)
        dvdx = (v[1]-v[0])/(2*dpix)
        dvdy = (v[3]-v[2])/(2*dpix)
        return numpy.abs(dudx*dvdy - dudy*dvdx)

    def getDataVector(self):
        """Get the pixel data as a numpy array. 
        
        Also returns the weight values and the local u,v coordinates of the pixels.
        Any pixels with zero weight (e.g. from masking in the original image) will not be
        included in the returned arrays.

        :returns: data_vector, weight_vector, u_vector, v_vector
        """
        nx = self.image.bounds.xmax - self.image.bounds.xmin + 1
        ny = self.image.bounds.ymax - self.image.bounds.ymin + 1
        # Alternatively, this also works.  Just need to remember that numpy arrays use y,x indexing.
        #ny, nx = self.image.array.shape

        # Image coordinates of pixels relative to nominal center
        xvals = numpy.arange(self.image.bounds.xmin, self.image.bounds.xmax+1, dtype=float)
        yvals = numpy.arange(self.image.bounds.ymin, self.image.bounds.ymax+1, dtype=float)
        x,y = numpy.meshgrid(xvals, yvals)
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
        mask = wt != 0.

        return pix[mask], wt[mask], u[mask], v[mask]

    def setData(self, data):
        """Return new StarData with data values replaced by elements of provided 1d array.
        The array should match the ordering of the one that is produced by getDataVector().

        :param data: A 1d numpy array with new values for the image data.
        
        :returns:    New StarData structure
        """
        # ??? Do we need a way to fill in pixels that have zero weight and
        # don't get passed out by getDataVector()???

        newimage = self.image.copy()
        ignore = self.weight.array==0.
        newimage.array[ignore] = 0.
        newimage.array[numpy.logical_not(ignore)] = data
        
        props = self.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            props.pop(key,None)
        return StarData(image=newimage,
                        image_pos=self.image_pos,
                        weight=self.weight,
                        pointing=self.pointing,
                        values_are_sb=self.values_are_sb,
                        properties=props)

    def addPoisson(self, signal=None, gain=None):
        """Return new StarData with the weight values altered to reflect
        Poisson shot noise from a signal source, e.g. when the weight
        only contains variance from background and read noise.

        :param signal:    The signal (in ADU) from which the Poisson variance is extracted.
                          If this is a 2d array or Image it is assumed to match the weight image.
                          If it is a 1d array, it is assumed to match the vectors returned by
                          getDataVector().  If None, the self.image is used.  All signals are
                          clipped from below at zero.
        :param gain:      The gain, in e per ADU, assumed in calculating new weights.  If None
                          is given, then the 'gain' property is used, else defaults gain=1.

        :returns:         A new StarData instance with updated weight array.
        """

        # Mark the pixels that are not already worthless
        use = self.weight.array!=0.
        
        # Get the signal data
        if signal is None:
            variance = self.image.array[use]
        elif isinstance(signal, galsim.image.Image):
            variance = signal.array[use]
        elif len(signal.shape)==2:
            variance = signal[use]
        else: 
            # Insert 1d vector into currently valid pixels
            variance = signal
            
        # Scale by gain
        if gain is None:
            try:
                g = self.properties['gain']
            except KeyError:
                g = 1.
        else:
            g = gain

        # Add to weight
        newweight = self.weight.copy()
        newweight.array[use] = 1. / (1./self.weight.array[use] + variance / g)

        # Return new object
        props = self.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            props.pop(key,None)
        return StarData(image=self.image,
                        image_pos=self.image_pos,
                        weight=newweight,
                        pointing=self.pointing,
                        values_are_sb=self.values_are_sb,
                        properties=props)

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
        newweight = self.weight.copy()
        newweight.array[use] = numpy.where(m, self.weight.array[use], 0.)

        # Return new object
        props = self.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            props.pop(key,None)
        return StarData(image=self.image,
                        image_pos=self.image_pos,
                        weight=newweight,
                        pointing=self.pointing,
                        values_are_sb=self.values_are_sb,
                        properties=props)
        
