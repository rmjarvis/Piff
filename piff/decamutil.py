#
# $Rev::                                                              $:
# $Author::                                                           $:
# $LastChangedDate::                                                  $:
#
# Utility methods for DECam
#

import numpy as np

class decaminfo(object):
    """ decaminfo is a class used to contain DECam geometry information and various utility routines
    """

    def info(self):
        # info returns a dictionary chock full of info on the DECam geometry
        # keyed by the CCD name

        infoDict = {}

        # store a dictionary for each CCD, keyed by the CCD name
        # AJR 9/14/2012 fixed these to agree with the DS9 coordinate system
        infoDict["S1"] =  {"xCenter":  -16.908,"yCenter":-191.670, "FAflag":False, "CCDNUM":25}
        infoDict["S2"]  = {"xCenter":  -16.908,"yCenter":-127.780, "FAflag":False, "CCDNUM":26}
        infoDict["S3"]  = {"xCenter":  -16.908,"yCenter": -63.890, "FAflag":False, "CCDNUM":27}
        infoDict["S4"]  = {"xCenter":  -16.908,"yCenter":   0.000, "FAflag":False, "CCDNUM":28}
        infoDict["S5"]  = {"xCenter":  -16.908,"yCenter":  63.890, "FAflag":False, "CCDNUM":29}
        infoDict["S6"]  = {"xCenter":  -16.908,"yCenter": 127.780, "FAflag":False, "CCDNUM":30}
        infoDict["S7"]  = {"xCenter":  -16.908,"yCenter": 191.670, "FAflag":False, "CCDNUM":31}
        infoDict["S8"]  = {"xCenter":  -50.724,"yCenter":-159.725, "FAflag":False, "CCDNUM":19}
        infoDict["S9"]  = {"xCenter":  -50.724,"yCenter": -95.835, "FAflag":False, "CCDNUM":20}
        infoDict["S10"] = {"xCenter":  -50.724,"yCenter": -31.945, "FAflag":False, "CCDNUM":21}
        infoDict["S11"] = {"xCenter":  -50.724,"yCenter":  31.945, "FAflag":False, "CCDNUM":22}
        infoDict["S12"] = {"xCenter":  -50.724,"yCenter":  95.835, "FAflag":False, "CCDNUM":23}
        infoDict["S13"] = {"xCenter":  -50.724,"yCenter": 159.725, "FAflag":False, "CCDNUM":24}
        infoDict["S14"] = {"xCenter":  -84.540,"yCenter":-159.725, "FAflag":False, "CCDNUM":13}
        infoDict["S15"] = {"xCenter":  -84.540,"yCenter": -95.835, "FAflag":False, "CCDNUM":14}
        infoDict["S16"] = {"xCenter":  -84.540,"yCenter": -31.945, "FAflag":False, "CCDNUM":15}
        infoDict["S17"] = {"xCenter":  -84.540,"yCenter":  31.945, "FAflag":False, "CCDNUM":16}
        infoDict["S18"] = {"xCenter":  -84.540,"yCenter":  95.835, "FAflag":False, "CCDNUM":17}
        infoDict["S19"] = {"xCenter":  -84.540,"yCenter": 159.725, "FAflag":False, "CCDNUM":18}
        infoDict["S20"] = {"xCenter": -118.356,"yCenter":-127.780, "FAflag":False, "CCDNUM":8 }
        infoDict["S21"] = {"xCenter": -118.356,"yCenter": -63.890, "FAflag":False, "CCDNUM":9 }
        infoDict["S22"] = {"xCenter": -118.356,"yCenter":   0.000, "FAflag":False, "CCDNUM":10}
        infoDict["S23"] = {"xCenter": -118.356,"yCenter":  63.890, "FAflag":False, "CCDNUM":11}
        infoDict["S24"] = {"xCenter": -118.356,"yCenter": 127.780, "FAflag":False, "CCDNUM":12}
        infoDict["S25"] = {"xCenter": -152.172,"yCenter": -95.835, "FAflag":False, "CCDNUM":4 }
        infoDict["S26"] = {"xCenter": -152.172,"yCenter": -31.945, "FAflag":False, "CCDNUM":5 }
        infoDict["S27"] = {"xCenter": -152.172,"yCenter":  31.945, "FAflag":False, "CCDNUM":6 }
        infoDict["S28"] = {"xCenter": -152.172,"yCenter":  95.835, "FAflag":False, "CCDNUM":7 }
        infoDict["S29"] = {"xCenter": -185.988,"yCenter": -63.890, "FAflag":False, "CCDNUM":1 }
        infoDict["S30"] = {"xCenter": -185.988,"yCenter":   0.000, "FAflag":False, "CCDNUM":2 }
        infoDict["S31"] = {"xCenter": -185.988,"yCenter":  63.890, "FAflag":False, "CCDNUM":3 }
        infoDict["N1"]  = {"xCenter": 16.908,  "yCenter":-191.670, "FAflag":False, "CCDNUM":32}
        infoDict["N2"]  = {"xCenter": 16.908,  "yCenter":-127.780, "FAflag":False, "CCDNUM":33}
        infoDict["N3"]  = {"xCenter": 16.908,  "yCenter": -63.890, "FAflag":False, "CCDNUM":34}
        infoDict["N4"]  = {"xCenter": 16.908,  "yCenter":   0.000, "FAflag":False, "CCDNUM":35}
        infoDict["N5"]  = {"xCenter": 16.908,  "yCenter":  63.890, "FAflag":False, "CCDNUM":36}
        infoDict["N6"]  = {"xCenter": 16.908,  "yCenter": 127.780, "FAflag":False, "CCDNUM":37}
        infoDict["N7"]  = {"xCenter": 16.908,  "yCenter": 191.670, "FAflag":False, "CCDNUM":38}
        infoDict["N8"]  = {"xCenter": 50.724,  "yCenter":-159.725, "FAflag":False, "CCDNUM":39}
        infoDict["N9"]  = {"xCenter": 50.724,  "yCenter": -95.835, "FAflag":False, "CCDNUM":40}
        infoDict["N10"] = {"xCenter": 50.724,  "yCenter": -31.945, "FAflag":False, "CCDNUM":41}
        infoDict["N11"] = {"xCenter": 50.724,  "yCenter":  31.945, "FAflag":False, "CCDNUM":42}
        infoDict["N12"] = {"xCenter": 50.724,  "yCenter":  95.835, "FAflag":False, "CCDNUM":43}
        infoDict["N13"] = {"xCenter": 50.724,  "yCenter": 159.725, "FAflag":False, "CCDNUM":44}
        infoDict["N14"] = {"xCenter": 84.540,  "yCenter":-159.725, "FAflag":False, "CCDNUM":45}
        infoDict["N15"] = {"xCenter": 84.540,  "yCenter": -95.835, "FAflag":False, "CCDNUM":46}
        infoDict["N16"] = {"xCenter": 84.540,  "yCenter": -31.945, "FAflag":False, "CCDNUM":47}
        infoDict["N17"] = {"xCenter": 84.540,  "yCenter":  31.945, "FAflag":False, "CCDNUM":48}
        infoDict["N18"] = {"xCenter": 84.540,  "yCenter":  95.835, "FAflag":False, "CCDNUM":49}
        infoDict["N19"] = {"xCenter": 84.540,  "yCenter": 159.725, "FAflag":False, "CCDNUM":50}
        infoDict["N20"] = {"xCenter": 118.356, "yCenter":-127.780, "FAflag":False, "CCDNUM":51}
        infoDict["N21"] = {"xCenter": 118.356, "yCenter": -63.890, "FAflag":False, "CCDNUM":52}
        infoDict["N22"] = {"xCenter": 118.356, "yCenter":   0.000, "FAflag":False, "CCDNUM":53}
        infoDict["N23"] = {"xCenter": 118.356, "yCenter":  63.890, "FAflag":False, "CCDNUM":54}
        infoDict["N24"] = {"xCenter": 118.356, "yCenter": 127.780, "FAflag":False, "CCDNUM":55}
        infoDict["N25"] = {"xCenter": 152.172, "yCenter": -95.835, "FAflag":False, "CCDNUM":56}
        infoDict["N26"] = {"xCenter": 152.172, "yCenter": -31.945, "FAflag":False, "CCDNUM":57}
        infoDict["N27"] = {"xCenter": 152.172, "yCenter":  31.945, "FAflag":False, "CCDNUM":58}
        infoDict["N28"] = {"xCenter": 152.172, "yCenter":  95.835, "FAflag":False, "CCDNUM":59}
        infoDict["N29"] = {"xCenter": 185.988, "yCenter": -63.890, "FAflag":False, "CCDNUM":60}
        infoDict["N30"] = {"xCenter": 185.988, "yCenter":   0.000, "FAflag":False, "CCDNUM":61}
        infoDict["N31"] = {"xCenter": 185.988, "yCenter":  63.890, "FAflag":False, "CCDNUM":62}
        infoDict["FS1"] = {"xCenter": -152.172,"yCenter": 143.7525,"FAflag":True , "CCDNUM":66}
        infoDict["FS2"] = {"xCenter": -185.988,"yCenter": 111.8075,"FAflag":True , "CCDNUM":65}
        infoDict["FS3"] = {"xCenter": -219.804,"yCenter":  15.9725,"FAflag":True , "CCDNUM":63}
        infoDict["FS4"] = {"xCenter": -219.804,"yCenter": -15.9725,"FAflag":True , "CCDNUM":64}
        infoDict["FN1"] = {"xCenter": 152.172, "yCenter": 143.7525,"FAflag":True , "CCDNUM":67}
        infoDict["FN2"] = {"xCenter": 185.988, "yCenter": 111.8075,"FAflag":True , "CCDNUM":68}
        infoDict["FN3"] = {"xCenter": 219.804, "yCenter":  15.9725,"FAflag":True , "CCDNUM":69}
        infoDict["FN4"] = {"xCenter": 219.804, "yCenter": -15.9725,"FAflag":True , "CCDNUM":70}

        return infoDict

    def getinfoArray(self):
        vals = np.zeros((71, 2))
        for key in self.infoDict:
            infoDict = self.infoDict[key]
            vals[infoDict['CCDNUM']][0] = infoDict['xCenter']
            vals[infoDict['CCDNUM']][1] = infoDict['yCenter']
        return vals


    def __init__(self,**inputDict):

        self.infoDict = self.info()
        self.mmperpixel = 0.015

        # ccddict returns the chip name when given a chip number
        self.ccddict = {}
        for keyi in self.infoDict.keys():
            self.ccddict.update(
                {self.infoDict[keyi]['CCDNUM']: keyi}
                )
        self.infoArr = self.getinfoArray()

    def getPosition_extnum(self, extnums, ix, iy):
        # do getPosition but with extnum instead
        xpixHalfSize = 1024. * np.ones(len(extnums))
        ypixHalfSize = 1024. * np.ones(len(extnums))
        ypixHalfSize = np.where(extnums > 62, 1024., 2048.)
        xCenter = self.infoArr[extnums][:, 0]
        yCenter = self.infoArr[extnums][:, 1]

        xPos = xCenter + (ix - xpixHalfSize + 0.5) * self.mmperpixel
        yPos = yCenter + (iy - ypixHalfSize + 0.5) * self.mmperpixel

        return xPos, yPos

    def getPixel_extnum(self, extnums, xPos, yPos):
        # do getPixel but with extnum instead
        xpixHalfSize = 1024. * np.ones(len(extnums))
        ypixHalfSize = 1024. * np.ones(len(extnums))
        ypixHalfSize = np.where(extnums > 62, 1024., 2048.)
        xCenter = self.infoArr[extnums][:, 0]
        yCenter = self.infoArr[extnums][:, 1]

        ix = (xPos - xCenter) / self.mmperpixel + xpixHalfSize - 0.5
        iy = (yPos - yCenter) / self.mmperpixel + ypixHalfSize - 0.5

        return ix, iy

    def getPosition(self,extname,ix,iy):
        # return the x,y position in [mm] for a given CCD and pixel number
        # note that the ix,iy are Image pixels - overscans removed - and start at zero

        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        # calculate positions
        xPos = ccdinfo["xCenter"] + (ix-xpixHalfSize+0.5)*self.mmperpixel
        yPos = ccdinfo["yCenter"] + (iy-ypixHalfSize+0.5)*self.mmperpixel

        return xPos,yPos

    def getPixel(self,extname,xPos,yPos):
        # given a coordinate in [mm], return pixel number

        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        # calculate positions
        ix = (xPos - ccdinfo["xCenter"]) / self.mmperpixel + xpixHalfSize - 0.5
        iy = (yPos - ccdinfo["yCenter"]) / self.mmperpixel + ypixHalfSize - 0.5

        return ix,iy

    def getPixel_no_extname(self, xPos, yPos):
        # get pixel coordinate without specifying extname

        coord_mm = [xPos, yPos]

        # determine extname
        for extname in self.infoDict:
            ccdinfo = self.infoDict[extname]

            # CCD size in pixels
            if ccdinfo["FAflag"]:
                xpixHalfSize = 1024.
                ypixHalfSize = 1024.
            else:
                xpixHalfSize = 1024.
                ypixHalfSize = 2048.

            xmin = ccdinfo["xCenter"] - xpixHalfSize * self.mmperpixel
            xmax = ccdinfo["xCenter"] + xpixHalfSize * self.mmperpixel
            ymin = ccdinfo["yCenter"] - ypixHalfSize * self.mmperpixel
            ymax = ccdinfo["yCenter"] + ypixHalfSize * self.mmperpixel
            bounds = [[xmin, xmax], [ymin, ymax]]
            # bounds are [[xmin, xmax], [ymin, ymax]]
            inside = np.multiply(*[(coord_mm[i] > bounds[i][0]) *
                                   (coord_mm[i] < bounds[i][1])
                                   for i in range(2)])
            if inside:
                # we have found our extname!
                break
        ix, iy = self.getPixel(extname, xPos, yPos)
        return ix, iy

