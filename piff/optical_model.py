
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
    'des': { 'obscuration': 0.301 / 0.7174, # from Zemax DECam model
             'nstruts': 4,
             'diam': 4.010,  # meters
             'lam': 700, # nm
             'strut_thick': 3.5 * 0.019 * (1462.526 / 4010.) / 2.0,  #updated to match big DES donuts, ignores one larger spider leg
             'strut_angle': 45 * galsim.degrees,
             'pad_factor': 1 ,
             'oversampling': 1,
             'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
           },
    'desdonut': { 'obscuration': 0.301 / 0.7174, # from Zemax DECam model
             'nstruts': 4,
             'diam': 4.010,  # meters
             'lam': 700, # nm
             'strut_thick': 3.5 * 0.019 * (1462.526 / 4010.) / 2.0,  #updated to match big DES donuts, ignores one larger spider leg
             'strut_angle': 45 * galsim.degrees,
             'pad_factor': 8,
             'oversampling': 1,
             'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion

           },
}

gsparams_templates = {
    'star': {  'minimum_fft_size': 32, 'folding_threshold': 0.02},
    'starby2': {  'minimum_fft_size': 64, 'folding_threshold': 0.01 },
    'starby4': {  'minimum_fft_size': 128, 'folding_threshold': 0.005 },
    'donut': { 'minimum_fft_size': 128, 'folding_threshold': 0.005 },
}

class Optical(Model):

    _method = 'optatmomodel'
    _centered = True

    def __init__(self, optical='des', gsparams='star', atmo_type='VonKarman', logger=None, **kwargs):
        """Initialize the Optical+Atmosphere Model

        There are four components to this model that are convolved together.

        First, there is an optical component, which uses a galsim.OpticalPSF to model the
        profile. These parameters are passed to GalSim, so
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
        :param pad_factor       Numerical factor by which to pad the pupil plane FFT [default: 1]
        :param oversampling     Numerical factor by which to oversample the FFT [default: 1]
        :param pupil_plane_im:  The name of a file containing the pupil plane image to use instead
                                of creating one from obscuration, struts, etc. [default: None]
        :param mirror_figure_im: The name of a file containing an additional phase contribution, for example
                                 from the figure of the primary mirror. [default: None]
        :param sigma            Gaussian CCD diffusion in arcsec [default:0.]

        Second, there is an atmospheric component, which uses either a galsim.Kolmogorov or
        galsim.VonKarman to model the profile.

        :param atmo_type        The name of the Atmospheric kernel. [default 'VonKarman']

        Finally, there is both an additional Gaussian component to describe CCD diffusion and separately an applied shear.

        Since there are a lot of parameters here, we provide the option of setting many of them
        from a template value.  e.g. template = 'des' will use the values stored in the dict
        piff.optatmo_model.aperture_templates['des'].

        :param optical :        A key word in the dict piff.optatmo_model.aperture_template to use
                                for setting values of these aperture parameters.  [default: None]
        :param gsparams:        A key word in the dict piff.optatmo_model.gsparams_template to use
                                for setting values of these Galsim parameters.  [default: None]

        If you use a template as well as other specific parameters, the specific parameters will
        override the values from the template.  e.g.  to simulate what DES would be like at
        lambda=1000 nm (the default is 700), you could do:

                >>> model = piff.OptatmoModel(optical='des', lam=1000)
        """
        self.logger = galsim.config.LoggerWrapper(logger)

        # If pupil_angle and strut angle are provided as strings, eval them.
        for key in ['pupil_angle', 'strut_angle']:
            if key in kwargs and isinstance(kwargs[key],str):
                kwargs[key] = eval(kwargs[key])

        # Copy over anything from the aperture and gsparams template dict, but let the direct kwargs override anything
        self.kwargs = {}
        if optical is not None:
            if optical not in optical_templates:
                raise ValueError("Unknown aperture template specified: %s"%optical)
            self.kwargs.update(optical_templates[optical])

        if gsparams is not None:
            if gsparams not in gsparams_templates:
                raise ValueError("Unknown gsparams template specified: %s"%gsparams)
            self.kwargs.update(gsparams_templates[gsparams])

        # Do this last, so specified kwargs override anything from the templates
        self.kwargs.update(kwargs)

        # Sift out optical and gsparams keys. Some of these aren't documented above, but allow them anyway.
        opt_keys = ('lam', 'diam', 'lam_over_diam', 'scale_unit',
                    'circular_pupil', 'obscuration', 'interpolant',
                    'oversampling', 'pad_factor', 'suppress_warning',
                    'nstruts', 'strut_thick', 'strut_angle',
                    'pupil_angle', 'pupil_plane_scale', 'pupil_plane_size','sigma')
        self.opt_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in opt_keys }

        gsparams_keys = ('minimum_fft_size','folding_threshold')
        self.gsparams_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in gsparams_keys }

        # Deal with the pupil plane image now so it only needs to be loaded from disk once.
        # TODO actually use this image in the aperture
        if 'pupil_plane_im' in kwargs:
            pupil_plane_im = kwargs.pop('pupil_plane_im')
            if isinstance(pupil_plane_im, str):
                logger.debug('Loading pupil_plane_im from {0}'.format(pupil_plane_im))
                pupil_plane_im = galsim.fits.read(pupil_plane_im)
            self.opt_kwargs['pupil_plane_im'] = pupil_plane_im

        # Store the Atmospheric Kernel type
        self.atmo_type = kwargs.pop('atmo_type','VonKarman')

        # Check that no unexpected parameters were passed in:
        extra_kwargs = [k for k in kwargs if k not in opt_keys and k not in gsparams_keys]
        if len(extra_kwargs) > 0:
            raise TypeError('__init__() got an unexpected keyword argument %r'%extra_kwargs[0])

        # Check for some required parameters.
        if 'diam' not in self.opt_kwargs:
            raise TypeError("Required keyword argument 'diam' not found")
        if 'lam' not in self.opt_kwargs:
            raise TypeError("Required keyword argument 'lam' not found")
        self.diam = self.opt_kwargs['diam']
        self.lam = self.opt_kwargs['lam']

        # pupil_angle and strut_angle won't serialize properly, so repr them now in self.kwargs.
        for key in ['pupil_angle', 'strut_angle']:
            if key in self.kwargs:
                self.kwargs[key] = repr(self.kwargs[key])

        # build the Galsim optical aperture here to cache it
        # TODO: need to check all these parameters are filled, or fill in defaults....
        self.aperture = galsim.Aperture(diam=self.opt_kwargs['diam'], obscuration=self.opt_kwargs['obscuration'],
                                    nstruts=self.opt_kwargs['nstruts'], strut_thick=self.opt_kwargs['strut_thick'],
                                    strut_angle=self.opt_kwargs['strut_angle'],
                                    pupil_plane_scale=None, pupil_plane_size=None,
                                    oversampling=self.opt_kwargs['oversampling'], pad_factor=self.opt_kwargs['pad_factor'],
                                    gsparams=galsim.GSParams(**self.gsparams_kwargs))

        # dictionary for cache of Galsim interp_objects
        self.cache = {}


    # fit is currently a DUMMY routine
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

    def getProfile(self, zernike_coeff, r0, L0=None, g1=0., g2=0.):
        """Get a version of the model as a GalSim GSObject

        :param zernike_coeff:      A ndarray or list with [0, 0, 0, 0, z4, z5, z6...z11...]
        :param r0:                 Value of r0 for Atmospheric Kernel
        :param L0:                 Value of L0 for Atmospheric Kernel. [default = None]
        :param g1:                 Value of g1 to Shear PSF [default=0.]
        :param g2:                 Value of g2 to Shear PSF [default=0.]

        :returns: a galsim.GSObject instance of the combined OptAtmo PSF
        """
        prof = []

        # TODO: build caches of each profile, in case it isn't changing...

        # gaussian for CCD Diffusion TODO: make it effectively Gaussian on the Focal Plane, undoing shear,size of the WCS
        if self.opt_kwargs['sigma']!=0.0:
            gaussian = galsim.Gaussian(sigma=self.opt_kwargs['sigma'])
            prof.append(gaussian)

        # atmospheric kernel
        if self.atmo_type == 'VonKarman':
            #if 'atm' in self.cache:
            #    atm = self.cache['atm']
            #    #TODO check that the cache'd version has the same parameters as what we want...
            #else:
            atm = galsim.VonKarman(lam=self.lam, r0=r0, L0=L0, flux=1.0, gsparams=galsim.GSParams(**self.gsparams_kwargs))
            self.cache['atm'] = atm
        else:
            atm = galsim.Kolmogorov(lam=self.lam, r0=r0, flux=1.0, gsparams=galsim.GSParams(**self.gsparams_kwargs))
        prof.append(atm)

        # optics
        optics = galsim.OpticalPSF(lam=self.lam,diam=self.diam,aper=self.aperture,
                                       aberrations=zernike_coeff,gsparams=galsim.GSParams(**self.gsparams_kwargs))
        prof.append(optics)

        # convolve
        prof = galsim.Convolve(prof)

        # shear
        if g1!=0.0 or g2!=0.0:
            prof = prof.shear(g1=g1, g2=g2)

        return prof

    # TODO: this could go into util.py, doesn't use any class properties currently
    def drawProfile(self, star, prof, params, use_fit=True, copy_image=True):
        """Generate PSF image for a given star and profile

        :param star:        Star instance holding information needed for
                            interpolation including the weight array,  image or field position,  and WCS into which
                            PSF will be rendered.
        :param prof:        A galsim profile
        :param params:      Params associated with profile to put in the Star's StarFit object.
        :param use_fit:     Bool [default: True] shift the profile by a star's
                            fitted center and multiply by its fitted flux

        :returns:   Star instance with its image filled with rendered PSF
        """

        # use flux and center properties
        if use_fit:
            prof = prof.shift(star.fit.center) * star.fit.flux

        # get image,weight and image_pos from the input star
        image, weight, image_pos = star.data.getImage()
        if copy_image:
            image_model = image.copy()
        else:
            image_model = image

        # draw the profile into this image, using its wcs
        prof.drawImage(image_model, method='auto', center=star.image_pos)

        # get properties from the input star
        properties = star.data.properties.copy()

        # Get rid of keys that constructor doesn't want to see:
        for key in ['x', 'y', 'u', 'v']:
            properties.pop(key, None)

        # build the output date, with the new image but all other quantities from the input star
        # make sure that hsm is reset here... add star.local_wcs here?
        data = StarData(image=image_model,
                        image_pos=star.data.image_pos,
                        weight=star.data.weight,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        orig_weight=star.data.orig_weight,
                        properties=properties)
        fit = StarFit(params,
                      flux=star.fit.flux,
                      center=star.fit.center)

        # build new star
        newstar = Star(data,fit)

        return newstar
