
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
.. module:: optical_model
"""

from __future__ import print_function

import galsim
import fitsio
import numpy

from .psf import PSF
from .interp import Interp
from .model import Model
from .stardata import StarData
from .starfit import Star, StarFit
from .gaussian_model import Gaussian

class Optical(Model):
    def __init__(self, rzero=0.1, sigma=0., g1=0., g2=0., pupil_path='', lam=500., **kwargs):
        """Initialize the Optical Model

        :param rzero:       Atmospheric seeing. Usually in the 0.1 - 0.2 range.
        :param g1, g2:      Shear to apply to final image. Simulates vibrational modes.
        :param sigma:       Convolve with gaussian of size sigma.
        :param pupil_path:  If a path is given, load up a pupil image, else
                            make image from galsim parameters.
        """

        # catch any kwargs passed along...
        self.kwargs = kwargs

        self.lam = lam
        self.pupil_path = pupil_path
        OpticalPSF = {'diam': 4.274419,
                      'lam': lam}
        if pupil_path:
            # load the pupil
            pupil_plane = fitsio.read(pupil_path)
            pupil_plane_im = galsim.Image(pupil_plane)
            OpticalPSF['pupil_plane_im'] = pupil_plane_im
        else:
            # make fake pupil
            OpticalPSF['obscuration'] = 0.301 / 0.7174
            OpticalPSF['nstruts'] = 4
            # aaron plays between 19 mm thick and 50 mm thick
            OpticalPSF['strut_thick'] = 0.050 * (1462.526 / 4010.) / 2.0 # conversion factor is nebulous?!
            OpticalPSF['strut_angle'] = 45 * galsim.degrees
        self.OpticalPSF = OpticalPSF
        # Update any OpticalPSF elements in kwargs
        if 'OpticalPSF' in kwargs:
            OpticalPSF = kwargs.pop('OpticalPSF')
            self.OpticalPSF.update(OpticalPSF)
        # deal with Kolmogorov from OpticalPSF

        # gaussian sigma
        self.sigma = sigma
        # shear
        self.g1 = g1
        self.g2 = g2

        # atmosphere
        self.rzero = rzero
        if rzero != 0:
            r0 = rzero * (lam / 500) ** -1.2  # meters
            lam_over_r0 = (lam * 1.e-9) / r0  # radians
            lam_over_r0 *= 206265  # Convert to arcsec
        else:
            lam_over_r0 = 0
        self.lam_over_r0 = lam_over_r0

    def fit(self, star):
        """Warning: This method just updates the fit with the chisq and dof!

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        image, weight, image_pos = star.data.getImage()
        # make image from self.draw
        model_image = self.draw(star).data.getImage()[0]

        # compute chisq
        chisq = numpy.std(image.array - model_image.array)
        dof = numpy.count_nonzero(weight.array) - 6

        fit = StarFit(star.fit.params, flux=star.fit.flux, center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A numpy array with [z4, z5, z6...z11]

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = []
        # gaussian
        if self.sigma != 0:
            gaussian = galsim.Gaussian(sigma=self.sigma)
            prof.append(gaussian)
        # atmosphere
        if self.lam_over_r0 != 0:
            atm = galsim.Kolmogorov(lam_over_r0=self.lam_over_r0)
            prof.append(atm)
        # optics
        aberrations = [0,0,0,0] + list(params)
        if len(aberrations) <= 4:
            # no optics here
            pass
        else:
            optics = galsim.OpticalPSF(aberrations=aberrations, **self.OpticalPSF)
            prof.append(optics)
            # convolve together
        if len(prof) == 0:
            raise Exception('No profile returned by model!')
        elif len(prof) == 1:
            prof = prof[0]
        else:
            prof = galsim.Convolve(prof)

        if self.g1 != 0 or self.g2 != 0:
            # no shearing
            # shear constant mode
            prof = prof.shear(g1=self.g1, g2=self.g2)

        return prof

    def draw(self, star):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        import galsim
        prof = self.getProfile(star.fit.params)
        center = galsim.PositionD(*star.fit.center)
        offset = star.data.image_pos + center - star.data.image.trueCenter()
        image = prof.drawImage(star.data.image.copy(), method='no_pixel', offset=offset)
        data = StarData(image, star.data.image_pos, star.data.weight)
        return Star(data, star.fit)

    def writeParameters(self, fits, extname):
        """Write parameters of Model to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        dtypes = [('RZERO', numpy.float64),
                  ('SIGMA', numpy.float64),
                  ('G1', numpy.float64),
                  ('G2', numpy.float64),
                  ('LAM', numpy.float64),
                  ('LAM_OVER_R0', numpy.float64),
                  # need to account for pupil_path == '' having length 0!
                  ('PUPIL_PATH', 'S{0}'.format(len(self.pupil_path) + 1)),
                  ]
        data = numpy.empty(1, dtype=dtypes)

        # assign
        data['PUPIL_PATH'] = self.pupil_path
        data['RZERO'] = self.rzero
        data['SIGMA'] = self.sigma
        data['G1'] = self.g1
        data['G2'] = self.g2
        data['LAM'] = self.lam
        data['LAM_OVER_R0'] = self.lam_over_r0

        # write to fits
        fits.write_table(data, extname=extname)

    def readParameters(self, fits, extname):
        """Read parameters of Model from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        # header = fits[extname].read_header()
        data = fits[extname].read()

        # using the dtypes above, assign variables probably via init process
        pupil_path = data['PUPIL_PATH'][0].strip()
        rzero = data['RZERO'][0]
        sigma = data['SIGMA'][0]
        g1 = data['G1'][0]
        g2 = data['G2'][0]
        lam = data['LAM'][0]

        self.__init__(rzero=rzero, sigma=sigma, g1=g1, g2=g2, pupil_path=pupil_path, lam=lam)
