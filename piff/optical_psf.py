
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
.. module:: optical_psf
"""

from __future__ import print_function

from .psf import PSF
from .interp import Interp
from .model import Model

class Optical(PSF):
    """A class that covers the optical portion of the PSF.

    The usual way to create a PSF is through one of the two factory functions::

        >>> optical = piff.Optical.build(images=images, pos=pos, model=model, interp=interp, ...)
        >>> optical = piff.Optical.read(file_name=file_name, ...)

    The first is used to build an Optical PSF model from the data.
    The second is used to read in an Optical PSF model from disk.

    NOTE: not sure about where the optics drawing goes. probably model? so hsm goes
    in a separate place to fit stuff
    """

    def __init__(self, model, interp):
        self.model = model
        self.interp = interp

    @classmethod
    def build(cls, stars, model, interp, logger=None):
        """The main driver function to build an Optical PSF model from data.

        :param stars:       A list of StarData instances.
        :param model:       A Model instance that defines how to model the individual PSFs
                            at the location of each star.
        :param interp:      An Interp instance that defines how to do the interpolation of the
                            data vectors (produced by model for each star).
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: an Optical PSF instance
        """
        if logger:
            logger.info("Start building Optical PSF using %s stars", len(stars))
            logger.debug("Model is %s", model)
            logger.debug("Interp is %s", interp)

        # reduce stars to positions and model outputs

        # load up base zernikes for each position

        # solve the interpolation model

        # Return this as a PSF instance
        if logger:
            logger.debug("Done building PSF")
        return cls(model, interp)

class Zernike(Interp):
    # a class that takes in positions and returns zernikes from reference to file
    # and interp-specific corrections

"""
# add donutlib to path
import sys
sys.path.append('/nfs/slac/g/ki/ki18/cpd/Projects/DES/Donut')
from donutlib.makedonut import makedonut
import galsim
import galsim.optics
hdu = fits.open('/nfs/slac/g/ki/ki18/cpd/Projects/DES/Donut/unittest.0001-debug.fits')
pupil_plane_im = hdu[1].data
pupil_plane_img = galsim.Image(pupil_plan_im)



# AJR params
outerRadius = 0.7174
innerRadius = 0.301
zLength = 4#.274419
fLength = 11.719
pixelSize = 15.0e-6
F = zLength / (2 * outerRadius)
# aaron plays between 19 mm thick and 50 mm thick
strut_thickness = 0.050 * (1462.526 / 4010.) / 2.0 # conversion factor is nebulous?!


# aberrations here have FOUR 0s up front, instead of 3!!
# diam in meters
# lam in nanometers
def md_gs(ZernikeArray, rzero=0.125, xDECam=0, yDECam=0, paramDict={'lam': wavelength * 1e9,
                                                                    'diam': zLength,
#                                                                     'obscuration': innerRadius / outerRadius,
#                                                                     'nstruts': 4, 
#                                                                     'strut_thick': strut_thickness,
#                                                                     'strut_angle': 45 * galsim.degrees},
                                                                    'pupil_plane_im': pupil_plane_img,
                                                                    'oversampling': int(8 / factor),
                                                                    'pad_factor': 1.5},
          imageDict=imageDict):
    opt = galsim.optics.OpticalPSF(aberrations=[0,0,0,0] + list(ZernikeArray), **paramDict)
    # to get array: opt._optimage.array
    # convolve with kolmogorov
    # lam in nm again
    # this is from the kolmogorov class instructions
    lam = wavelength * 1e9  # nm
    r0 = rzero #* (lam/500)**-1.2  # meters
    lam_over_r0 = (lam * 1.e-9) / r0  # radians
    lam_over_r0 *= 206265  # Convert to arcsec
    atm = galsim.Kolmogorov(lam_over_r0=lam_over_r0)

    # convolve
    psf = galsim.Convolve([opt, atm])
    donut = psf.drawImage(**imageDict)
    # get moments for galsim
    moments = donut.FindAdaptiveMom()
    print('galsim donut sigma', moments.moments_sigma)
    print('galsim donut shape', moments.observed_shape)
    print('galsim donut centroid', moments.moments_centroid)
    donut = donut.array
    donut /= donut.sum()
    return donut
"""
