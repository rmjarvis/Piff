
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
    'des': { 'obscuration': 0.301 / 0.7174,
             'nstruts': 4,
             'diam': 4.274419,  # meters
             'lam': 700, # nm
             # aaron plays between 19 mm thick and 50 mm thick
             'strut_thick': 0.050 * (1462.526 / 4010.) / 2.0, # conversion factor is nebulous?!
             'strut_angle': 45 * galsim.degrees,
             'r0': 0.15,
             'L0': 25.0,
           },
}

class Optical(Model):
    def __init__(self, template=None, vary_atmosphere=True, vary_optics=True, logger=None, fastfit=True, **kwargs):
        """Initialize the Optical Model

        There are potentially three components to this model that are convolved together.

        First, there is an optical component, which uses a galsim.OpticalPSF to model the
        profile.  The aberrations are considered fitted parameters, but the other attributes
        are fixed and are given at initialization.  These parameters are passed to GalSim, so
        they have the same definitions as used there.

        :param fastfit:        If True, will lower requirements for galsim fourier transforms yielding considerable speedup during fitting
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


        self.gsparams = galsim.GSParams()

        if len(self.kolmogorov_kwargs) > 0:
            logger.debug('Creating Kolmogorov Atmosphere')
            self.atmo = galsim.VonKarman(**self.kolmogorov_kwargs)
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
        self._fastfit = fastfit
        self.kwargs['vary_atmosphere'] = vary_atmosphere
        self.kwargs['vary_optics'] = vary_optics
        self.kwargs['fastfit'] = fastfit

    def _fit_residual(self, lmparams, star, logger=None):
        logger = LoggerWrapper(logger)

        image, weight, image_pos = star.data.getImage()
        all_params = lmparams.valuesdict().values()

        flux, du, dv = all_params[:3]
        params = all_params[3:]


        prof = self.getProfile(params, logger=logger).shift(du, dv) * flux

        # draw
        # Equivalent to galsim.Image(image, dtype=float), but without the sanity checks.
        model_image = galsim._Image(np.empty_like(image.array, dtype=float),
                                    image.bounds, image.wcs)
        prof.drawImage(model_image,
                       offset=(image_pos - model_image.true_center))

        # Caculate sqrt(weight) * (model_image - image) in place for efficiency.
        model_image.array[:,:] -= image.array
        model_image.array[:,:] *= np.sqrt(weight.array)
        chi = model_image.array.ravel()

        # logger.debug('current params, chi2 / dof of {0}:'.format(np.sum(chi ** 2) / len(chi)))
        # logger.debug(str(all_params))
        return chi

    def fit(self, star, params0=None, fastfit=None, logger=None, **kwargs):
        """Fit star parameters. Depending on the model settings, may fit only flux and centering, or may also fit other parameters

        :param star:    A Star instance
        :param params0: Initial set of parameters for fit. If None, will choose some (reasonable) defaults
        :param **kwargs: A set of parameters to pass in for changing the
                            way lmfit does the minimization.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star with the fitted parameters in star.fit
        """
        lmfit_kwargs = {'method': 'leastsq', 'epsfcn': 1e-5, 'maxfev': 1000}
        lmfit_kwargs.update(**kwargs)
        logger = LoggerWrapper(logger)
        import lmfit

        if fastfit is None:
            fastfit = self._fastfit

        # update convolution if fastfit
        if fastfit:
            logger.debug('fastfit mode activated')
            self._save_gsparams = self.gsparams
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
            # speedup in optical modeling
            self._save_optical_psf_kwargs = copy.deepcopy(self.optical_psf_kwargs)
            if 'pad_factor' not in self.optical_psf_kwargs:
                self.optical_psf_kwargs['pad_factor'] = 0.5
                # self.optical_psf_kwargs['pad_factor'] = 1.1  # does work
                # self.optical_psf_kwargs['pad_factor'] = 1.0
            if 'oversampling' not in self.optical_psf_kwargs:
                self.optical_psf_kwargs['oversampling'] = 0.5
                # self.optical_psf_kwargs['oversampling'] = 1.1  # does work
                # self.optical_psf_kwargs['oversampling'] = 1.0

        # make initial lmparams
        lmparams = lmfit.Parameters()

        flux = star.fit.flux
        if flux == 1.:
            # a pretty reasonable first guess is to just take the sum of the pixels
            flux = star.image.array.sum()
        lmparams.add('flux', value=flux, vary=True, min=0.0)
        lmparams.add('du', value=star.fit.center[0], vary=True, min=-0.3, max=0.3)
        lmparams.add('dv', value=star.fit.center[1], vary=True, min=-0.3, max=0.3)

        # atmo params
        if params0 is None:
            size0 = 1
            g10 = 0
            g20 = 0
        else:
            size0, g10, g20 = params0[:3]

        min_size = 0.45
        max_size = 2.0
        max_g = 0.4
        lmparams.add('size', value=size0, vary=self.vary_atmosphere, min=min_size, max=max_size)
        lmparams.add('g1', value=g10, vary=self.vary_atmosphere, min=-max_g, max=max_g)
        lmparams.add('g2', value=g20, vary=self.vary_atmosphere, min=-max_g, max=max_g)

        # sanity checks
        if size0 < min_size:
            logger.warning('Initial size is less than recommended minimum: {0} < {1}'.format(size0, min_size))
        if size0 > max_size:
            logger.warning('Initial size is greater than recommended maximum: {0} > {1}'.format(size0, max_size))
        if g10 > max_g:
            logger.warning('Initial g1 is greater than recommended maximum: {0} > {1}'.format(g10, max_g))
        if g10 < -max_g:
            logger.warning('Initial g1 is less than recommended minimum: {0} < {1}'.format(g10, -max_g))
        if g20 > max_g:
            logger.warning('Initial g2 is greater than recommended maximum: {0} > {1}'.format(g20, max_g))
        if g20 < -max_g:
            logger.warning('Initial g2 is less than recommended minimum: {0} < {1}'.format(g20, -max_g))

        # optics params
        # if params0 passed, use it to guess size, otherwise default to 4-11
        if params0 is None:
            n_optics_params = 8
            optics_params = np.zeros(n_optics_params)
        else:
            n_optics_params = len(params0) - 3
            optics_params = params0[3:]
        for i in range(n_optics_params):
            lmparams.add('zernike_{0}'.format(i + 4), value=optics_params[i], vary=self.vary_optics, min=-2, max=2)

        # run fit
        results = lmfit.minimize(self._fit_residual, lmparams, args=(star, logger,), **lmfit_kwargs)
        logger.debug(lmfit.fit_report(results, min_correl=0.5))

        if fastfit:
            logger.debug('fastfit mode deactivated')
            self.gsparams = self._save_gsparams
            self.optical_psf_kwargs = self._save_optical_psf_kwargs

        # extract values
        flux = results.params['flux'].value
        du = results.params['du'].value
        dv = results.params['dv'].value
        center = (du, dv)
        chisq = results.chisqr
        dof = results.nfree
        fit_params = np.zeros(len(results.params) - 3)
        params_var = np.zeros(len(results.params) - 3)
        for i, key in enumerate(results.params):
            indx = i - 3  # first three are flux and center
            if key in ['flux', 'du', 'dv']:
                continue
            param = results.params[key]
            fit_params[indx] = param.value
            if hasattr(param, 'stderr'):
                params_var[indx] = param.stderr ** 2

        fit = StarFit(fit_params, params_var=params_var, flux=flux, center=center, chisq=chisq, dof=dof)
        return Star(star.data, fit)


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

    def getProfile(self, params, logger=None):
        """Get a version of the model as a GalSim GSObject

        :param params:      A np array with [size, g1, g2, z4, z5, z6...]

        :returns: a galsim.GSObject instance
        """
        logger = LoggerWrapper(logger)
        import galsim

        if params is None:
            size = 1
            g1 = 0
            g2 = 0
            optics_params = []
            logger.warning('entered getProfile of optical model with star lacking fit parameters. Entering default values and skipping optical aberrations')
        else:
            size, g1, g2 = params[:3]
            optics_params = params[3:]
        # atmo
        prof = []
        if self.atmo is not None:
            # * 1. to prevent error in galsim dilate
            prof.append(self.atmo.dilate(size * 1.).shear(g1=g1, g2=g2))
        else:
            logger.warning('No atmosphere model found')

        # optics
        if len(optics_params) == 0:
            # no optics here; this should behave like a gsobject
            pass
        else:
            aberrations = [0,0,0,0] + list(optics_params)
            optics = galsim.OpticalPSF(aberrations=aberrations, gsparams=self.gsparams, **self.optical_psf_kwargs)
            prof.append(optics)

        if len(prof) == 0:
            raise RuntimeError('No profile returned by model!')
        #if len(prof) == 1:
        #    prof = prof[0]
        ## convolve together
        #elif len(prof) > 1:
        #    prof = galsim.Convolve(prof, gsparams=self.gsparams)
        if len(prof) == 1:
            prof = prof[0]
        else:
            prof = galsim.Convolve(prof)

        if self.g1 is not None or self.g2 is not None:
            prof = prof.shear(g1=self.g1, g2=self.g2)

        return prof

    def draw(self, star, copy_image=True, logger=None):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.
        :param copy_image:          If False, will use the same image object.
                                    If True, will copy the image and then overwrite it.
                                    [default: True]

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        logger = LoggerWrapper(logger)
        prof = self.getProfile(star.fit.params, logger=logger).shift(star.fit.center) * star.fit.flux
        if copy_image:
            image = star.image.copy()
        else:
            image = star.image
        prof.drawImage(image, method='auto', offset=(star.image_pos-image.true_center))
        properties = star.data.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key, None)
        data = StarData(image=image,
                        image_pos=star.data.image_pos,
                        weight=star.data.weight,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        values_are_sb=star.data.values_are_sb,
                        orig_weight=star.data.orig_weight,
                        properties=properties)
        return Star(data, star.fit)
