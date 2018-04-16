
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
import coord
import fitsio
import numpy as np

from .model import Model
from .star import Star, StarFit, StarData

# The only one here by default is 'des', but this allows people to easily add another template
optical_templates = {
    'des': { 'obscuration': 0.301 / 0.7174,
             'nstruts': 4,
             'diam': 4.274419,  # meters
             'lam': 700, # nm
             # aaron plays between 19 mm thick and 50 mm thick
             'strut_thick': 0.050 * (1462.526 / 4010.) / 2.0, # conversion factor is nebulous?!
             'strut_angle': 45 * galsim.degrees,
             'r0': 0.1,
           },
}

class Optical(Model):
    def __init__(self, template=None, logger=None, **kwargs):
        """Initialize the Optical Model

        There are potentially three components to this model that are convolved together.

        First, there is an optical component, which uses a galsim.OpticalPSF to model the
        profile.  The aberrations are considered fitted parameters, but the other attributes
        are fixed and are given at initialization.  These parameters are passed to GalSim, so
        they have the same definitions as used there.

        :param diam:            Diameter of telescope aperture in meters. [required (but cf.
                                template option)]
        :param lam:             Wavelength of observations in nanometers. [required (but cf.
                                template option)]
        :param obscuration:     Linear dimension of central obscuration as fraction of pupil
                                linear dimension, [0., 1.). [default: 0]
        :param nstruts:         Number of radial support struts to add to the central obscuration.
                                [default: 0]
        :param strut_thick:     Thickness of support struts as a fraction of pupil diameter.
                                [default: 0.05]
        :param strut_angle:     Angle made between the vertical and the strut starting closest to
                                it, defined to be positive in the counter-clockwise direction.
                                [default: 0. * galsim.degrees]
        :param pupil_plane_im:  The name of a file containing the pupil plane image to use instead
                                of creating one from obscuration, struts, etc. [default: None]

        Second, there may be an atmospheric component, which uses a galsim.Kolmogorov to
        model the profile.

        :param fwhm:            The full-width half-max of the atmospheric part of the PSF.
                                [default: None]
        :param r0:              The Fried parameter in units of meters to use to calculate fwhm
                                as fwhm = 0.976 lam / r0. [default: None]

        Finall, there is allowed to be a final Gaussian component and an applied shear.

        :param sigma:           Convolve with gaussian of size sigma. [default: 0]
        :param g1, g2:          Shear to apply to final image. Simulates vibrational modes.
                                [default: 0]

        Since there are a lot of parameters here, we provide the option of setting many of them
        from a template value.  e.g. template = 'des' will use the values stored in the dict
        piff.optical_model.optical_templates['des'].

        :param template:        A key word in the dict piff.optical_model.optical_template to use
                                for setting values of these parameters.  [default: None]

        If you use a template as well as other specific parameters, the specific parameters will
        override the values from the template.  e.g.  to simulate what DES would be like at
        lambda=1000 nm (the default is 700), you could do:

                >>> model = piff.OpticalModel(template='des', lam=1000)
        """
        logger = galsim.config.LoggerWrapper(logger)
        # If pupil_angle and strut angle are provided as strings, eval them.
        for key in ['pupil_angle', 'strut_angle']:
            if key in kwargs and isinstance(kwargs[key],str):
                kwargs[key] = eval(kwargs[key])

        # Copy over anything from the template dict, but let the direct kwargs override anything
        # in the template.
        self.kwargs = {}
        if template is not None:
            if template not in optical_templates:
                raise ValueError("Unknown template specified: %s"%template)
            self.kwargs.update(optical_templates[template])
        # Do this second, so specified kwargs override anything from the template
        self.kwargs.update(kwargs)

        # Some of these aren't documented above, but allow them anyway.
        optical_psf_keys = ('lam', 'diam', 'lam_over_diam', 'scale_unit',
                            'circular_pupil', 'obscuration', 'interpolant',
                            'oversampling', 'pad_factor', 'suppress_warning',
                            'nstruts', 'strut_thick', 'strut_angle',
                            'pupil_angle', 'pupil_plane_scale', 'pupil_plane_size')
        self.optical_psf_kwargs = { key : self.kwargs[key] for key in self.kwargs
                                                           if key in optical_psf_keys }

        # Deal with the pupil plane image now so it only needs to be loaded from disk once.
        if 'pupil_plane_im' in kwargs:
            pupil_plane_im = kwargs.pop('pupil_plane_im')
            if isinstance(pupil_plane_im, str):
                logger.debug('Loading pupil_plane_im from {0}'.format(pupil_plane_im))
                pupil_plane_im = galsim.fits.read(pupil_plane_im)
            self.optical_psf_kwargs['pupil_plane_im'] = pupil_plane_im

        kolmogorov_keys = ('lam', 'r0', 'lam_over_r0', 'scale_unit',
                           'fwhm', 'half_light_radius', 'r0_500')
        self.kolmogorov_kwargs = { key : self.kwargs[key] for key in self.kwargs
                                                          if key in kolmogorov_keys }
        # If lam is the only one, then remove it -- we don't have a Kolmogorov component then.
        if self.kolmogorov_kwargs.keys() == ['lam']:
            self.kolmogorov_kwargs = {}
        # Also, let r0=0 or None indicate that there is no kolmogorov component
        if 'r0' in self.kolmogorov_kwargs and not self.kolmogorov_kwargs['r0']:
            self.kolmogorov_kwargs = {}

        # Store the Gaussian and shear parts
        self.sigma = kwargs.pop('sigma',None)
        self.g1 = kwargs.pop('g1',None)
        self.g2 = kwargs.pop('g2',None)

        # Check that no unexpected parameters were passed in:
        extra_kwargs = [k for k in kwargs if k not in optical_psf_keys and k not in kolmogorov_keys]
        if len(extra_kwargs) > 0:
            raise TypeError('__init__() got an unexpected keyword argument %r'%extra_kwargs[0])

        # Check for some required parameters.
        if 'diam' not in self.optical_psf_kwargs:
            raise TypeError("Required keyword argument 'diam' not found")
        if 'lam' not in self.optical_psf_kwargs:
            raise TypeError("Required keyword argument 'lam' not found")

        # pupil_angle and strut_angle won't serialize properly, so repr them now in self.kwargs.
        for key in ['pupil_angle', 'strut_angle']:
            if key in self.kwargs:
                self.kwargs[key] = repr(self.kwargs[key])

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

        var = np.zeros(len(star.fit.params)) 
        fit = StarFit(star.fit.params, params_var=var, flux=star.fit.flux,
                      center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A np array with [z4, z5, z6...z11]

        :returns: a galsim.GSObject instance
        """
        import galsim
        prof = []
        # gaussian
        if self.sigma is not None:
            gaussian = galsim.Gaussian(sigma=self.sigma)
            prof.append(gaussian)
        # atmosphere
        if len(self.kolmogorov_kwargs) > 0:
            atm = galsim.Kolmogorov(**self.kolmogorov_kwargs)
            prof.append(atm)
        # optics
        if params is None or len(params) == 0:
            # no optics here
            pass
        else:
            aberrations = [0,0,0,0] + list(params)
            optics = galsim.OpticalPSF(aberrations=aberrations, **self.optical_psf_kwargs)
            prof.append(optics)
            # convolve together

        if len(prof) == 0:
            raise RuntimeError('No profile returned by model!')

        prof = galsim.Convolve(prof)

        if self.g1 is not None or self.g2 is not None:
            prof = prof.shear(g1=self.g1, g2=self.g2)

        return prof

    def draw(self, star, copy_image=True):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        import galsim
        prof = self.getProfile(star.fit.params)
        center = galsim.PositionD(*star.fit.center)
        offset = star.data.image_pos + center - star.data.image.true_center
        if copy_image:
            image = star.image.copy()
        else:
            image = star.image
        prof.drawImage(image, method='no_pixel', offset=offset)
        data = StarData(image, star.data.image_pos, star.data.weight)
        return Star(data, star.fit)
