
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

    Second, there may be an atmospheric component, which uses either a galsim.Kolmogorov or
    galsim.VonKarman to model the profile.

    :param fwhm:            The full-width half-max of the atmospheric part of the PSF.
                            [default: None]
    :param r0:              The Fried parameter in units of meters to use for the Kolmogorov
                            profile. [default: None]
    :param L0:              The outer scale in units of meters if desired, in which case
                            the atmospheric part will be a VonKarman. [default: None]

    Finally, there is allowed to be a final Gaussian component and an applied shear.

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
    _method = 'no_pixel'
    _centered = True
    _model_can_be_offset = False

    def __init__(self, template=None, logger=None, **kwargs):
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
        opt_keys = ('lam', 'diam', 'lam_over_diam', 'scale_unit',
                    'circular_pupil', 'obscuration', 'interpolant',
                    'oversampling', 'pad_factor', 'suppress_warning',
                    'nstruts', 'strut_thick', 'strut_angle',
                    'pupil_angle', 'pupil_plane_scale', 'pupil_plane_size')
        self.opt_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in opt_keys }

        # Deal with the pupil plane image now so it only needs to be loaded from disk once.
        if 'pupil_plane_im' in kwargs:
            pupil_plane_im = kwargs.pop('pupil_plane_im')
            if isinstance(pupil_plane_im, str):
                logger.debug('Loading pupil_plane_im from {0}'.format(pupil_plane_im))
                pupil_plane_im = galsim.fits.read(pupil_plane_im)
            self.opt_kwargs['pupil_plane_im'] = pupil_plane_im

        atm_keys = ('lam', 'r0', 'lam_over_r0', 'scale_unit',
                    'fwhm', 'half_light_radius', 'r0_500', 'L0')
        self.atm_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in atm_keys }
        # If lam is the only one, then remove it -- we don't have a Kolmogorov component then.
        if list(self.atm_kwargs.keys()) == ['lam']:
            self.atm_kwargs = {}
        # Also, let r0=0 or None indicate that there is no atm component
        if 'r0' in self.atm_kwargs and not self.atm_kwargs['r0']:
            self.atm_kwargs = {}

        # Store the Gaussian and shear parts
        self.sigma = kwargs.pop('sigma',None)
        self.g1 = kwargs.pop('g1',None)
        self.g2 = kwargs.pop('g2',None)

        # Check that no unexpected parameters were passed in:
        extra_kwargs = [k for k in kwargs if k not in opt_keys and k not in atm_keys]
        if len(extra_kwargs) > 0:
            raise TypeError('__init__() got an unexpected keyword argument %r'%extra_kwargs[0])

        # Check for some required parameters.
        if 'diam' not in self.opt_kwargs:
            raise TypeError("Required keyword argument 'diam' not found")
        if 'lam' not in self.opt_kwargs:
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
        dof = np.count_nonzero(weight.array)

        var = np.zeros(len(star.fit.params)) 
        fit = StarFit(star.fit.params, params_var=var, flux=star.fit.flux,
                      center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject

        :param params:      A np array with [z4, z5, z6...z11]

        :returns: a galsim.GSObject instance
        """
        prof = []
        # gaussian
        if self.sigma is not None:
            gaussian = galsim.Gaussian(sigma=self.sigma)
            prof.append(gaussian)

        # atmosphere
        if len(self.atm_kwargs) > 0:
            if 'L0' in self.atm_kwargs and self.atm_kwargs['L0'] is not None:
                atm = galsim.VonKarman(**self.atm_kwargs)
            else:
                atm = galsim.Kolmogorov(**self.atm_kwargs)
            prof.append(atm)

        # optics
        if params is None or len(params) == 0:
            # no aberrations.  Just the basic opt_kwargs
            optics = galsim.OpticalPSF(**self.opt_kwargs)
        else:
            aberrations = [0,0,0,0] + list(params)
            optics = galsim.OpticalPSF(aberrations=aberrations, **self.opt_kwargs)

        # convolve together
        prof.append(optics)

        if len(prof) == 1:
            prof = prof[0]
        else:
            prof = galsim.Convolve(prof)

        if self.g1 is not None or self.g2 is not None:
            prof = prof.shear(g1=self.g1, g2=self.g2)

        return prof

