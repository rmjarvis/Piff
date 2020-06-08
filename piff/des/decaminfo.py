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
.. module:: decaminfo
"""

from ..star import Star, StarFit, StarData
import numpy as np

class DECamInfo(object):
    """ decaminfo is a class used to contain DECam geometry information and various utility routines
    """
    _infoDict = None

    @property
    def infoDict(self):
        if DECamInfo._infoDict is None:
            # info returns a dictionary chock full of info on the DECam geometry
            # keyed by the CCD name
            DECamInfo._infoDict = {}

            # store a dictionary for each CCD, keyed by the CCD name
            # AJR 9/14/2012 fixed these to agree with the DS9 coordinate system
            DECamInfo._infoDict["S1"] =  {"xCenter":  -16.908,"yCenter":-191.670, "FAflag":False, "CCDNUM":25}
            DECamInfo._infoDict["S2"]  = {"xCenter":  -16.908,"yCenter":-127.780, "FAflag":False, "CCDNUM":26}
            DECamInfo._infoDict["S3"]  = {"xCenter":  -16.908,"yCenter": -63.890, "FAflag":False, "CCDNUM":27}
            DECamInfo._infoDict["S4"]  = {"xCenter":  -16.908,"yCenter":   0.000, "FAflag":False, "CCDNUM":28}
            DECamInfo._infoDict["S5"]  = {"xCenter":  -16.908,"yCenter":  63.890, "FAflag":False, "CCDNUM":29}
            DECamInfo._infoDict["S6"]  = {"xCenter":  -16.908,"yCenter": 127.780, "FAflag":False, "CCDNUM":30}
            DECamInfo._infoDict["S7"]  = {"xCenter":  -16.908,"yCenter": 191.670, "FAflag":False, "CCDNUM":31}
            DECamInfo._infoDict["S8"]  = {"xCenter":  -50.724,"yCenter":-159.725, "FAflag":False, "CCDNUM":19}
            DECamInfo._infoDict["S9"]  = {"xCenter":  -50.724,"yCenter": -95.835, "FAflag":False, "CCDNUM":20}
            DECamInfo._infoDict["S10"] = {"xCenter":  -50.724,"yCenter": -31.945, "FAflag":False, "CCDNUM":21}
            DECamInfo._infoDict["S11"] = {"xCenter":  -50.724,"yCenter":  31.945, "FAflag":False, "CCDNUM":22}
            DECamInfo._infoDict["S12"] = {"xCenter":  -50.724,"yCenter":  95.835, "FAflag":False, "CCDNUM":23}
            DECamInfo._infoDict["S13"] = {"xCenter":  -50.724,"yCenter": 159.725, "FAflag":False, "CCDNUM":24}
            DECamInfo._infoDict["S14"] = {"xCenter":  -84.540,"yCenter":-159.725, "FAflag":False, "CCDNUM":13}
            DECamInfo._infoDict["S15"] = {"xCenter":  -84.540,"yCenter": -95.835, "FAflag":False, "CCDNUM":14}
            DECamInfo._infoDict["S16"] = {"xCenter":  -84.540,"yCenter": -31.945, "FAflag":False, "CCDNUM":15}
            DECamInfo._infoDict["S17"] = {"xCenter":  -84.540,"yCenter":  31.945, "FAflag":False, "CCDNUM":16}
            DECamInfo._infoDict["S18"] = {"xCenter":  -84.540,"yCenter":  95.835, "FAflag":False, "CCDNUM":17}
            DECamInfo._infoDict["S19"] = {"xCenter":  -84.540,"yCenter": 159.725, "FAflag":False, "CCDNUM":18}
            DECamInfo._infoDict["S20"] = {"xCenter": -118.356,"yCenter":-127.780, "FAflag":False, "CCDNUM":8 }
            DECamInfo._infoDict["S21"] = {"xCenter": -118.356,"yCenter": -63.890, "FAflag":False, "CCDNUM":9 }
            DECamInfo._infoDict["S22"] = {"xCenter": -118.356,"yCenter":   0.000, "FAflag":False, "CCDNUM":10}
            DECamInfo._infoDict["S23"] = {"xCenter": -118.356,"yCenter":  63.890, "FAflag":False, "CCDNUM":11}
            DECamInfo._infoDict["S24"] = {"xCenter": -118.356,"yCenter": 127.780, "FAflag":False, "CCDNUM":12}
            DECamInfo._infoDict["S25"] = {"xCenter": -152.172,"yCenter": -95.835, "FAflag":False, "CCDNUM":4 }
            DECamInfo._infoDict["S26"] = {"xCenter": -152.172,"yCenter": -31.945, "FAflag":False, "CCDNUM":5 }
            DECamInfo._infoDict["S27"] = {"xCenter": -152.172,"yCenter":  31.945, "FAflag":False, "CCDNUM":6 }
            DECamInfo._infoDict["S28"] = {"xCenter": -152.172,"yCenter":  95.835, "FAflag":False, "CCDNUM":7 }
            DECamInfo._infoDict["S29"] = {"xCenter": -185.988,"yCenter": -63.890, "FAflag":False, "CCDNUM":1 }
            DECamInfo._infoDict["S30"] = {"xCenter": -185.988,"yCenter":   0.000, "FAflag":False, "CCDNUM":2 }
            DECamInfo._infoDict["S31"] = {"xCenter": -185.988,"yCenter":  63.890, "FAflag":False, "CCDNUM":3 }
            DECamInfo._infoDict["N1"]  = {"xCenter": 16.908,  "yCenter":-191.670, "FAflag":False, "CCDNUM":32}
            DECamInfo._infoDict["N2"]  = {"xCenter": 16.908,  "yCenter":-127.780, "FAflag":False, "CCDNUM":33}
            DECamInfo._infoDict["N3"]  = {"xCenter": 16.908,  "yCenter": -63.890, "FAflag":False, "CCDNUM":34}
            DECamInfo._infoDict["N4"]  = {"xCenter": 16.908,  "yCenter":   0.000, "FAflag":False, "CCDNUM":35}
            DECamInfo._infoDict["N5"]  = {"xCenter": 16.908,  "yCenter":  63.890, "FAflag":False, "CCDNUM":36}
            DECamInfo._infoDict["N6"]  = {"xCenter": 16.908,  "yCenter": 127.780, "FAflag":False, "CCDNUM":37}
            DECamInfo._infoDict["N7"]  = {"xCenter": 16.908,  "yCenter": 191.670, "FAflag":False, "CCDNUM":38}
            DECamInfo._infoDict["N8"]  = {"xCenter": 50.724,  "yCenter":-159.725, "FAflag":False, "CCDNUM":39}
            DECamInfo._infoDict["N9"]  = {"xCenter": 50.724,  "yCenter": -95.835, "FAflag":False, "CCDNUM":40}
            DECamInfo._infoDict["N10"] = {"xCenter": 50.724,  "yCenter": -31.945, "FAflag":False, "CCDNUM":41}
            DECamInfo._infoDict["N11"] = {"xCenter": 50.724,  "yCenter":  31.945, "FAflag":False, "CCDNUM":42}
            DECamInfo._infoDict["N12"] = {"xCenter": 50.724,  "yCenter":  95.835, "FAflag":False, "CCDNUM":43}
            DECamInfo._infoDict["N13"] = {"xCenter": 50.724,  "yCenter": 159.725, "FAflag":False, "CCDNUM":44}
            DECamInfo._infoDict["N14"] = {"xCenter": 84.540,  "yCenter":-159.725, "FAflag":False, "CCDNUM":45}
            DECamInfo._infoDict["N15"] = {"xCenter": 84.540,  "yCenter": -95.835, "FAflag":False, "CCDNUM":46}
            DECamInfo._infoDict["N16"] = {"xCenter": 84.540,  "yCenter": -31.945, "FAflag":False, "CCDNUM":47}
            DECamInfo._infoDict["N17"] = {"xCenter": 84.540,  "yCenter":  31.945, "FAflag":False, "CCDNUM":48}
            DECamInfo._infoDict["N18"] = {"xCenter": 84.540,  "yCenter":  95.835, "FAflag":False, "CCDNUM":49}
            DECamInfo._infoDict["N19"] = {"xCenter": 84.540,  "yCenter": 159.725, "FAflag":False, "CCDNUM":50}
            DECamInfo._infoDict["N20"] = {"xCenter": 118.356, "yCenter":-127.780, "FAflag":False, "CCDNUM":51}
            DECamInfo._infoDict["N21"] = {"xCenter": 118.356, "yCenter": -63.890, "FAflag":False, "CCDNUM":52}
            DECamInfo._infoDict["N22"] = {"xCenter": 118.356, "yCenter":   0.000, "FAflag":False, "CCDNUM":53}
            DECamInfo._infoDict["N23"] = {"xCenter": 118.356, "yCenter":  63.890, "FAflag":False, "CCDNUM":54}
            DECamInfo._infoDict["N24"] = {"xCenter": 118.356, "yCenter": 127.780, "FAflag":False, "CCDNUM":55}
            DECamInfo._infoDict["N25"] = {"xCenter": 152.172, "yCenter": -95.835, "FAflag":False, "CCDNUM":56}
            DECamInfo._infoDict["N26"] = {"xCenter": 152.172, "yCenter": -31.945, "FAflag":False, "CCDNUM":57}
            DECamInfo._infoDict["N27"] = {"xCenter": 152.172, "yCenter":  31.945, "FAflag":False, "CCDNUM":58}
            DECamInfo._infoDict["N28"] = {"xCenter": 152.172, "yCenter":  95.835, "FAflag":False, "CCDNUM":59}
            DECamInfo._infoDict["N29"] = {"xCenter": 185.988, "yCenter": -63.890, "FAflag":False, "CCDNUM":60}
            DECamInfo._infoDict["N30"] = {"xCenter": 185.988, "yCenter":   0.000, "FAflag":False, "CCDNUM":61}
            DECamInfo._infoDict["N31"] = {"xCenter": 185.988, "yCenter":  63.890, "FAflag":False, "CCDNUM":62}
            DECamInfo._infoDict["FS1"] = {"xCenter": -152.172,"yCenter": 143.7525,"FAflag":True , "CCDNUM":66}
            DECamInfo._infoDict["FS2"] = {"xCenter": -185.988,"yCenter": 111.8075,"FAflag":True , "CCDNUM":65}
            DECamInfo._infoDict["FS3"] = {"xCenter": -219.804,"yCenter":  15.9725,"FAflag":True , "CCDNUM":63}
            DECamInfo._infoDict["FS4"] = {"xCenter": -219.804,"yCenter": -15.9725,"FAflag":True , "CCDNUM":64}
            DECamInfo._infoDict["FN1"] = {"xCenter": 152.172, "yCenter": 143.7525,"FAflag":True , "CCDNUM":67}
            DECamInfo._infoDict["FN2"] = {"xCenter": 185.988, "yCenter": 111.8075,"FAflag":True , "CCDNUM":68}
            DECamInfo._infoDict["FN3"] = {"xCenter": 219.804, "yCenter":  15.9725,"FAflag":True , "CCDNUM":69}
            DECamInfo._infoDict["FN4"] = {"xCenter": 219.804, "yCenter": -15.9725,"FAflag":True , "CCDNUM":70}

        return DECamInfo._infoDict

    def _getinfoArray(self):
        vals = np.zeros((71, 2))
        for key in self.infoDict:
            infoDict = self.infoDict[key]
            vals[infoDict['CCDNUM']][0] = infoDict['xCenter']
            vals[infoDict['CCDNUM']][1] = infoDict['yCenter']
        return vals


    def __init__(self,**inputDict):

        self.infoDict
        self.mmperpixel = 0.015

        # ccddict returns the chip name when given a chip number
        # so ccddict[70] = 'FN4'
        self.ccddict = {}
        for keyi in self.infoDict.keys():
            self.ccddict.update(
                {self.infoDict[keyi]['CCDNUM']: keyi}
                )
        self.infoArr = self._getinfoArray()
        # get edges.
        pixHalfSize = 1024 * np.ones(self.infoArr.shape) * self.mmperpixel
        # for < 63, y is 2x bigger than x
        pixHalfSize[:63, 1] *= 2
        self.infoLowerLeftCorner = self.infoArr - pixHalfSize
        self.infoUpperRightCorner = self.infoArr + pixHalfSize

    def getPosition(self, chipnums, ix, iy):
        """Given chipnum and pixel coordinates return focal_plane coordinates [mm]

        :param chipnums:        Array of ccd numbers.
        :param ix, iy:          Arrays of x and y coordinates, in pixels on a ccd.

        :returns xPos, yPos:    Arrays of x and y coordinates in mm on the focal plane.
        """
        # do getPosition but with chipnum instead
        chipnums = np.array(chipnums, dtype=int, copy=False)

        xpixHalfSize = 1024. * np.ones(chipnums.shape)
        ypixHalfSize = 1024. * np.ones(chipnums.shape)
        ypixHalfSize = np.where(chipnums > 62, 1024., 2048.)
        xCenter = self.infoArr[chipnums][:, 0]
        yCenter = self.infoArr[chipnums][:, 1]

        xPos = xCenter + (ix - xpixHalfSize + 0.5) * self.mmperpixel
        yPos = yCenter + (iy - ypixHalfSize + 0.5) * self.mmperpixel

        return xPos, yPos

    def getPixel_chipnum(self, chipnums, xPos, yPos):
        """Given chipnum and focal_plane coordinates [mm] return pixel coordinates

        :param chipnums:    Array of ccd numbers.
        :param xPos, yPos:  Arrays of x and y coordinates, in mm on the focal plane

        :returns ix, iy:    Arrays of x and y coordinates in pixels
        """
        # do getPixel but with chipnum instead
        chipnums = np.array(chipnums, dtype=int, copy=False)
        
        xpixHalfSize = 1024. * np.ones(chipnums.shape)
        ypixHalfSize = 1024. * np.ones(chipnums.shape)
        ypixHalfSize = np.where(chipnums > 62, 1024., 2048.)
        xCenter = self.infoArr[chipnums][:, 0]
        yCenter = self.infoArr[chipnums][:, 1]

        ix = (xPos - xCenter) / self.mmperpixel + xpixHalfSize - 0.5
        iy = (yPos - yCenter) / self.mmperpixel + ypixHalfSize - 0.5

        return ix, iy

    def getPixel(self, xPos, yPos):
        """Given focal_plane coordinates [mm] return pixel coordinates

        :param xPos, yPos:          Arrays of x and y coordinates, in mm on the focal plane

        :returns chipnums ix, iy:   Arrays of chipnumbers, x and y coordinates in pixels
        """
        # get chipnums
        pos = np.vstack((xPos, yPos)).T
        # (Nsamp, Nchip, Ndim) = (len(xPos), 71, 2)
        # first entry is a fake entry. Skip it!
        conds = ((pos[:, None, 0] >= self.infoLowerLeftCorner[None, 1:, 0]) *
                 (pos[:, None, 0] <= self.infoUpperRightCorner[None, 1:, 0]) *
                 (pos[:, None, 1] >= self.infoLowerLeftCorner[None, 1:, 1]) *
                 (pos[:, None, 1] <= self.infoUpperRightCorner[None, 1:, 1]))
        chipnums = np.argwhere(conds)[:, 1] + 1

        ix, iy = self.getPixel_chipnum(chipnums, xPos, yPos)
        return chipnums, ix, iy

    def pixel_to_focal_stardata(self, stardata):
        """Take stardata and add focal plane position to properties

        :param stardata:    The stardata with property 'chipnum'

        :returns stardata:  New stardata with updated properties
        """
        # stardata needs to have chipnum as a property!
        focal_x, focal_y = self.getPosition(
            np.array([stardata['chipnum']]), np.array([stardata['x']]), np.array([stardata['y']]))
        properties = stardata.properties.copy()
        properties['focal_x'] = focal_x[0]
        properties['focal_y'] = focal_y[0]
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key,None)
        return StarData(image=stardata.image,
                        image_pos=stardata.image_pos,
                        weight=stardata.weight,
                        pointing=stardata.pointing,
                        properties=properties)

    def pixel_to_focal(self, star):
        """Take star and add focal plane position to properties

        :param star:    The star with property 'chipnum'

        :returns star:  New star with updated properties
        """
        return Star(self.pixel_to_focal_stardata(star.data), star.fit)

    def pixel_to_focalList(self, stars):
        """Take stars and add focal plane position to properties

        :param stars:     Starlist with property 'chipnum'

        :returns starsl:  New stars with updated properties
        """
        return [self.pixel_to_focal(star) for star in stars]

    def get_nominal_wcs(self, chipnum):
        """Given a chip, return a galsim wcs object with reasonable values.

        Useful for generating fake stars.

        :param chipnum:     Chip we are looking at

        :returns wcs:       Galsim wcs object

        .. note::
            This is a EuclideanWCS, NOT a CelestialWCS!
            All I have done here is create a galsim wcs object whose center (u,v) = (0,0)
            corresponds to the focal plane (focal_x, focal_y) = (0, 0) and who has a reasonable
            jacobian transformation about the center. For simplicity, set dudx, dvdy = 0, and dudy,
            dvdx = -0.26. This is pretty close to what the y1 test images look like...
        """
        import galsim

        # get center and convert to pixels.
        xpix, ypix = self.getPixel_chipnum([chipnum], [0], [0])
        # now we know that dudx etc, so convert xpix and ypix to uarcsec varcsec
        arcsec_over_pixel = 0.26
        # also a minus sign because the axes also flip
        uarcsec = ypix * -arcsec_over_pixel
        varcsec = xpix * -arcsec_over_pixel
        world_origin = galsim.PositionD(uarcsec, varcsec)
        wcs = galsim.AffineTransform(0, -arcsec_over_pixel, -arcsec_over_pixel, 0,
                                     world_origin=-world_origin)
        return wcs
