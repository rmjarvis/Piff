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

    This includes:
      - a postage stamp of the imaging data
      - the weight map for these pixels (zero weight means the pixel is masked or otherwise bad)
      - the position of the star on the image
      - the position of the star in the full field-of-view (local tangent plane projection)
      - possibly extra information, such as colors

    A StarData object can render the pixel data in one of two ways:

      - getImage() returns the pixels as a galsim.Image.
      - getDataVector() returns the pixels as a numpy array.

    Different use cases may prefer the data in one of these forms or the other.

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
    """
    def __init__(self, image, image_pos, weight=None, pointing=None, properties=None,
                 logger=None):
        import galsim
        # Save all of these as attributes.
        self.image = image
        self.image_pos = image_pos
        # Make sure we have a local wcs in case the provided image is more complex.
        self.local_wcs = image.wcs.local(image_pos)

        if weight is None:
            self.weight = galsim.Image(numpy.ones_like(image.array))
        else:
            self.weight = weight

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties

        self.pointing = pointing
        self.field_pos = self.calculateFieldPos(image_pos, image.wcs, pointing, self.properties)

        # Make sure the user didn't provide their own x,y,u,v in properties.
        for key in ['x', 'y', 'u', 'v']:
            if properties is not None and key in properties:
                raise AttributeError("Cannot provide property %s in properties dict."%key)

        self.properties['x'] = self.image_pos.x
        self.properties['y'] = self.image_pos.y
        self.properties['u'] = self.field_pos.x
        self.properties['v'] = self.field_pos.y

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
        coordinates and u,v are teh position in field coordinates.

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
