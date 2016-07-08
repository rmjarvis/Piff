
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
import numpy as np

from .model import Model
from .star import Star, StarFit, StarData

des_pupil_template = {'obscuration': 0.301 / 0.7174,
                      'nstruts': 4,
                      'diam': 4.274419,  # meters
                      # aaron plays between 19 mm thick and 50 mm thick
                      'strut_thick': 0.050 * (1462.526 / 4010.) / 2.0, # conversion factor is nebulous?!
                      'strut_angle': 45 * galsim.degrees}

class Optical(Model):
    def __init__(self, rzero=0.1, sigma=0., g1=0., g2=0., pupil_path='None', lam=500., optical_template='des', **kwargs):
        """Initialize the Optical Model

        :param rzero:               Atmospheric seeing. Usually in the 0.1 - 0.2 range. [default: 0.1]
        :param g1, g2:              Shear to apply to final image. Simulates vibrational modes. [default: 0]
        :param sigma:               Convolve with gaussian of size sigma. [default: 0]
        :param pupil_path:          If a path is given, load up a pupil image, else
                                    make image from galsim parameters referencing optical_template [default: None]
        :param lam:                 Wavelength of observations in nanometers [default: 500]
        :param optical_template:    If no pupil plane image is given, create one from a set of templates. [default: 'des']
        """

        # catch any kwargs passed along...
        self.kwargs = {
            'rzero': rzero,
            'g1': g1,
            'g2': g2,
            'sigma': sigma,
            'pupil_path': pupil_path,
            'lam': lam,
            'optical_template': optical_template,
            }

        self.lam = lam
        self.pupil_path = pupil_path
        optical_psf_kwargs = {'lam': lam}
        if pupil_path != 'None':
            # load the pupil
            pupil_plane = fitsio.read(pupil_path)
            pupil_plane_im = galsim.Image(pupil_plane)
            optical_psf_kwargs['pupil_plane_im'] = pupil_plane_im
        else:
            # make fake pupil from template
            if optical_template == 'des':
                optical_psf_kwargs.update(des_pupil_template)
            elif type(optical_template) == dict:
                optical_psf_kwargs.update(optical_template)
            else:
                raise Exception('Unrecognized optical template {0}'.format(optical_template))
        self.optical_psf_kwargs = optical_psf_kwargs

        # atmosphere
        if rzero != 0:
            r0 = rzero * (lam / 500) ** -1.2  # meters
            lam_over_r0 = (lam * 1.e-9) / r0  # radians
            lam_over_r0 *= 206265  # Convert to arcsec
        else:
            lam_over_r0 = 0
        self.lam_over_r0 = lam_over_r0
        # self.optical_psf_kwargs['lam_over_r0'] = self.lam_over_r0

    def fit(self, star):
        """Warning: This method just updates the fit with the chisq and dof!

        :param star:    A Star instance

        :returns: a new Star with the fitted parameters in star.fit
        """
        image = star.image
        weight = star.weight
        # make image from self.draw
        model_image = self.draw(star).image

        # compute chisq
        chisq = np.std(image.array - model_image.array)
        dof = np.count_nonzero(weight.array) - 6

        fit = StarFit(star.fit.params, flux=star.fit.flux, center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A np array with [z4, z5, z6...z11]

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = []
        # gaussian
        if self.kwargs['sigma'] != 0:
            gaussian = galsim.Gaussian(sigma=self.kwargs['sigma'])
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
            optics = galsim.OpticalPSF(aberrations=aberrations, **self.optical_psf_kwargs)
            prof.append(optics)
            # convolve together
        if len(prof) == 0:
            raise Exception('No profile returned by model!')
        elif len(prof) == 1:
            prof = prof[0]
        else:
            prof = galsim.Convolve(prof)

        if self.kwargs['g1'] != 0 or self.kwargs['g2'] != 0:
            # no shearing
            # shear constant mode
            prof = prof.shear(g1=self.kwargs['g1'], g2=self.kwargs['g2'])

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
