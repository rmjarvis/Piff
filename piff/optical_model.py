# Note: to use this file with a kolmogorov atmosphere, small adjustments may be needed. This works with vonkarman atmosphere but has not been tested with kolmogorov atmosphere.


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
import copy
import numpy as np

from .model import Model
from .star import Star, StarFit, StarData
from .config import LoggerWrapper

# The only one here by default is 'des', but this allows people to easily add another template
optical_templates = {
    'des': { 'obscuration': 0.301 / 0.7174, #Note: VonKarman is standard. If you want a kolmogorov atmosphere you need to both specify that you are using a kolmogorov atmosphere model and also say you are using the des_kolmogorov template.
             'nstruts': 4,
             'diam': 4.274419,  # meters
             'lam': 700, # nm
             # aaron plays between 19 mm thick and 50 mm thick
             'strut_thick': 0.050 * (1462.526 / 4010.) / 2.0, # conversion factor is nebulous?!
             'strut_angle': 45 * galsim.degrees,
             'r0': 0.15,
             'L0': 25.0, #Note: in an actual fit this will appear in the kolmogorov_kwargs but likely not change; however, it will also appear in the optatmo_psf_kwargs and it necessarily will change there as opt_L0 is being fit. This will make L0 unique in that it is the only member of optatmo_psf_kwargs that changes for the fit. In a sense, this happens with r0 also since r0 from kolmogorov_kwargs doesn't change, but in that case you vary the "size" parameter in the fit and then use r0 = 0.15/size during the fit, rather than have an r0 that varies in optatmo_psf_kwargs.
           },
    'des_kolmogorov': { 'obscuration': 0.301 / 0.7174,
             'nstruts': 4,
             'diam': 4.274419,  # meters
             'lam': 700, # nm
             # aaron plays between 19 mm thick and 50 mm thick
             'strut_thick': 0.050 * (1462.526 / 4010.) / 2.0, # conversion factor is nebulous?!
             'strut_angle': 45 * galsim.degrees,
             'r0': 0.15,
           },
}

class Optical(Model):
    def __init__(self, template=None, vary_atmosphere=True, vary_optics=True, logger=None, **kwargs):
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
        :param L0:              The VonKarman outer scale [default: None]

        Finally, there is allowed to be a final Gaussian component and an applied shear.

        :param sigma:           Convolve with gaussian of size sigma. [default: None]

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
                           'fwhm', 'half_light_radius', 'r0_500', 'L0')
        #kolmogorov_keys = ('lam', 'r0', 'lam_over_r0', 'scale_unit',
        #                   'fwhm', 'half_light_radius', 'r0_500')
        self.kolmogorov_kwargs = { key : self.kwargs[key] for key in self.kwargs
                                                          if key in kolmogorov_keys }
        # If lam is the only one, then remove it -- we don't have a Kolmogorov component then.
        if self.kolmogorov_kwargs.keys() == ['lam']:
            self.kolmogorov_kwargs = {}
        # Also, let r0=0 or None indicate that there is no kolmogorov component
        if 'r0' in self.kolmogorov_kwargs and not self.kolmogorov_kwargs['r0']:
            self.kolmogorov_kwargs = {}


        #It turns out, not using the below is too slow.
        self.gsparams = galsim.GSParams(
            minimum_fft_size=32,  # 128
            # maximum_fft_size=4096,  # 4096
            # stepk_minimum_hlr=5,  # 5
            # folding_threshold=5e-3,  # 5e-3
            # maxk_threshold=1e-3,  # 1e-3
            # kvalue_accuracy=1e-5,  # 1e-5
            # xvalue_accuracy=1e-5,  # 1e-5
            # table_spacing=1.,  # 1
            )
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 0.5
            # self.optical_psf_kwargs['pad_factor'] = 1.1  # does work
            # self.optical_psf_kwargs['pad_factor'] = 1.0
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 0.5
            # self.optical_psf_kwargs['oversampling'] = 1.1  # does work
            # self.optical_psf_kwargs['oversampling'] = 1.0

        if len(self.kolmogorov_kwargs) > 0 and 'L0' in self.kolmogorov_kwargs:
            logger.debug('Creating VonKarman Atmosphere')
            self.atmo = galsim.VonKarman(**self.kolmogorov_kwargs)
            sigma = kwargs.pop('sigma',None)
            if sigma is not None:
                logger.debug('Found extra sigma = {0}. It will be unused'.format(sigma))
        elif len(self.kolmogorov_kwargs) > 0:
            logger.debug('Creating Kolmogorov Atmosphere')
            self.atmo = galsim.Kolmogorov(**self.kolmogorov_kwargs)
            sigma = kwargs.pop('sigma',None)
            if sigma is not None:
                logger.debug('Found extra sigma = {0}. It will be unused'.format(sigma))
        else:
            logger.debug('No kolmogorov atmosphere found.')
            sigma = kwargs.pop('sigma',None)
            logger.debug('Filling with gaussian sigma = {0}'.format(sigma))
            if sigma is not None:
                self.atmo = galsim.Gaussian(sigma=sigma)
            else:
                self.atmo = None

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

        self.vary_atmosphere = vary_atmosphere
        self.vary_optics = vary_optics
        self.kwargs['vary_atmosphere'] = vary_atmosphere
        self.kwargs['vary_optics'] = vary_optics

