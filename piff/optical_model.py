
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
import os
import galsim
import coord
import fitsio
import numpy as np
from functools import lru_cache

from .model import Model
from .star import Star

optical_templates = {
    'des': {
        # This is normally what to use when working with DES (or other DECam) images.
        'diam': 4.010,  # meters
        'lam': 700, # nm
        'pad_factor': 1 ,
        'oversampling': 1,
        'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
        'mirror_figure_im': 'DECam_236392_finegrid512_nm_uv.fits',
        'pupil_plane_im': 'DECam_pupil_512uv.fits'
    },
    'des_simple': {
        # An approximation of the DECam pupil plane using simple struts and obscuration.
        'diam': 4.010,  # meters
        'lam': 700, # nm
        'obscuration': 0.301 / 0.7174, # from Zemax DECam model
        'nstruts': 4,
        'strut_thick': 0.0166,  # 66.5mm thick / 4010mm pupil
                                # (tuned to match DECam big donut images)
        'strut_angle': 45 * galsim.degrees,
        'pad_factor': 1 ,
        'oversampling': 1,
        'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
    },
    'des_128': {
        # Same as 'des', but using a smaller (128x128) pupil plane image.
        'diam': 4.010,  # meters
        'lam': 700, # nm
        'pad_factor': 1 ,
        'oversampling': 1,
        'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
        'mirror_figure_im': 'DECam_236392_finegrid512_nm_uv.fits',
        'pupil_plane_im': 'DECam_pupil_128uv.fits'
    },
    'des_donut': {
        # Same as 'des', but with a larger pad_factor, appropriate when rendering very
        # out-of-focus images (aka "donuts").
        'diam': 4.010,  # meters
        'lam': 700, # nm
        'pad_factor': 8,
        'oversampling': 1,
        'sigma': 8.0 * (0.263/15.0), # 8micron sigma CCD diffusion
        'mirror_figure_im': 'DECam_236392_finegrid512_nm_uv.fits',
        'pupil_plane_im': 'DECam_pupil_512uv.fits'
    },
}

gsparams_templates = {
    # Two approximations to speed things up somewhat by
    # 1. lowering the minimum size of the FFT image GalSim wants to make
    # 2. increasing the "folding_threshold", which sets the scale size in k-space.
    'fast': {  'minimum_fft_size': 64, 'folding_threshold': 0.01 },
    'faster': {  'minimum_fft_size': 32, 'folding_threshold': 0.02},
}

def update_file_name(kwargs, key):
    from .meta_data import data_dir
    if key not in kwargs:
        return
    file_name = kwargs[key]
    if not isinstance(file_name, str) or os.path.isfile(file_name):
        return
    new_file_name = os.path.join(data_dir, file_name)

    if not os.path.isfile(new_file_name):
        raise ValueError("File %s not found for %s" % (file_name, key))
    kwargs[key] = new_file_name

class Optical(Model):
    """
    Initialize the Optical+Atmosphere Model

    Use type name "Optical" in a config field to use this model.

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
    :param base_aberrations: A list of aberrations to which any fitted aberrations are added.
                            [default: None]

    The next two parameters are specific to our treatment of the Optical PSF. If given,
    they are used to create an additional galsim.PhaseScreen describing the mirror's own
    contribution to the wavefront.

    :param mirror_figure_im: The name of a file containing an additional phase contribution, for
                             example from the figure of the primary mirror. [default: None]
    :param mirror_figure_scale: The pixel scale to use in mirror_figure_im.  If omittited,
                                Piff will use the pixel scale in the image itself. [default: None]

    Second, there is an atmospheric component, which uses either a galsim.Kolmogorov or
    galsim.VonKarman to model the profile.

    :param atmo_type        The name of the Atmospheric kernel. [default 'VonKarman']

    Finally, there is both an additional Gaussian component to describe CCD diffusion and
    separately an applied shear.

    :param sigma            Gaussian CCD diffusion in arcsec [default:0.]

    Since there are a lot of parameters here, we provide the option of setting many of them
    from a template value.  e.g. template = 'des' will use the values stored in the dict
    piff.optical_model.optical_templates['des'].

    :param template:        A key word in the dict piff.optical_model.optical_templates to use
                            for setting values of these aperture parameters.  [default: None]
    :param gsparams:        A key word in the dict piff.optical_model.gsparams_templates to use
                            for setting values of these Galsim parameters.  [default: None]

    If you use a template as well as other specific parameters, the specific parameters will
    override the values from the template.  e.g.  to simulate what DES would be like at
    lambda=1000 nm (the default is 700), you could do:

            >>> model = piff.Optical(template='des', lam=1000)

    Note that the Zernike coeffients (zernike_coeff), Atmospheric parameters (ie. r0,L0) and Shear (g1,g2) are
    fitted parameters, passed via the arguments to getProfile.
    """
    _type_name = 'Optical'
    _method = 'auto'
    _centered = False
    _model_can_be_offset = False

    def __init__(self, template=None, gsparams=None, atmo_type='VonKarman', logger=None, **kwargs):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.set_num(None)

        # If pupil_angle and strut angle are provided as strings, eval them.
        for key in kwargs:
            if key.endswith('angle'):
                kwargs[key] = galsim.config.ParseValue(kwargs, key, {}, galsim.Angle)[0]

        # Copy over anything from the aperture and gsparams template dict, but let the direct
        # kwargs override anything
        self.kwargs = {}
        if template is not None:
            if template not in optical_templates:
                raise ValueError("Unknown optical template specified: %s"%template)
            self.kwargs.update(optical_templates[template])

        if gsparams is None:
            self.gsparams = None
        else:
            if isinstance(gsparams, galsim.GSParams):
                pass
            elif gsparams in gsparams_templates:
                gsparams = galsim.GSParams(**gsparams_templates[gsparams])
            else:
                raise ValueError("Unknown gsparams template specified: %s"%gsparams)
            self.gsparams = gsparams

        # Do this last, so specified kwargs override anything from the templates
        self.kwargs.update(kwargs)

        # Update file names if they are in the data_dir.
        for key in ('pupil_plane_im', 'mirror_figure_im'):
            update_file_name(self.kwargs, key)

        # Sift out optical and gsparams and other keys. Some of these aren't documented above, but
        # allow them anyway.
        opt_keys = ('lam', 'diam', 'lam_over_diam', 'scale_unit',
                    'circular_pupil', 'obscuration', 'interpolant',
                    'oversampling', 'pad_factor', 'suppress_warning',
                    'nstruts', 'strut_thick', 'strut_angle',
                    'pupil_plane_im', 'pupil_angle', 'pupil_plane_scale', 'pupil_plane_size')
        self.opt_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in opt_keys }

        gsparams_keys = ('minimum_fft_size','folding_threshold')
        gsparams_kwargs = { key : self.kwargs[key] for key in self.kwargs if key in gsparams_keys }
        if gsparams_kwargs:
            if self.gsparams is not None:
                raise ValueError("Cannot provide both gsparams and %s",list(gsparams_kwargs))
            else:
                self.gsparams = galsim.GSParams(**gsparams_kwargs)

        # Check that no unexpected parameters were passed in:
        other_keys = ('sigma', 'mirror_figure_im', 'mirror_figure_scale', 'base_aberrations')
        extra_kwargs = [k for k in kwargs if k not in opt_keys + gsparams_keys + other_keys]
        if len(extra_kwargs) > 0:
            raise TypeError('__init__() got an unexpected keyword argument %r'%extra_kwargs[0])

        # Save sigma if present.
        self.sigma = self.kwargs.get('sigma', 0.)

        # Check for some required parameters.
        if 'diam' not in self.opt_kwargs:
            raise TypeError("Required keyword argument 'diam' not found")
        if 'lam' not in self.opt_kwargs:
            raise TypeError("Required keyword argument 'lam' not found")
        self.diam = self.opt_kwargs['diam']
        self.lam = self.opt_kwargs['lam']

        # Load the mirror figure screen and save it as a PhaseScreen
        self.mirror_figure_screen = None
        if 'mirror_figure_im' in self.kwargs:
            mirror_figure_im = self.kwargs['mirror_figure_im']
            if isinstance(mirror_figure_im, str):
                self.logger.debug('Loading mirror_figure_im from {0}'.format(mirror_figure_im))
                mirror_figure_im = galsim.fits.read(mirror_figure_im)
            scale = self.kwargs.get('mirror_figure_scale', mirror_figure_im.scale)
            npix_v, npix_u = mirror_figure_im.array.shape

            # build u,v grid points
            mirror_figure_u = np.linspace(-(npix_u-1)/2 * scale, (npix_u-1)/2 * scale, num=npix_u)
            mirror_figure_v = np.linspace(-(npix_v-1)/2 * scale, (npix_v-1)/2 * scale, num=npix_v)

            # build the LUT for the mirror figure, and save it
            mirror_figure_table = galsim.LookupTable2D(mirror_figure_u, mirror_figure_v,
                                                       mirror_figure_im.array)
            self.mirror_figure_screen = galsim.UserScreen(mirror_figure_table, diam=self.diam)

        # Store the Atmospheric Kernel type
        self.atmo_type = atmo_type

        # pupil_angle and strut_angle won't serialize properly, so repr them now in self.kwargs.
        for key in ['pupil_angle', 'strut_angle']:
            if key in self.kwargs:
                self.kwargs[key] = repr(self.kwargs[key])

        # build the Galsim optical aperture here to cache it
        self.aperture = galsim.Aperture(**self.opt_kwargs)

        # define the param array elements for this model
        self.nZ = 37
        self.idx_z0 = 0
        self.idx_r0 = self.nZ + 1  # add 1 since nZ+1 Zernikes are stored, ie. 0:nZ is stored
        self.idx_L0 = self.idx_r0 + 1
        self.idx_g1 = self.idx_L0 + 1
        self.idx_g2 = self.idx_g1 + 1
        self.param_len = self.idx_g2 + 1
        self.base_aberrations = np.zeros(self.nZ+1)
        base_aberrations = kwargs.get('base_aberrations', None)
        if base_aberrations is not None:
            self.base_aberrations[0:len(base_aberrations)] += base_aberrations

    # fit is currently a DUMMY routine
    def fit(self, star, logger=None, convert_func=None, draw_method=None):
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
        dof = np.count_nonzero(weight.array) - 3   #3 DOF in flux,centroid

        params = star.fit.get_params(self._num)
        var = np.zeros_like(params)
        return Star(star.data,
                    star.fit.newParams(params, params_var=var, num=self._num, chisq=chisq, dof=dof))

    def initialize(self, star, logger=None, default_init=None):
        """Initialize the given star's fit parameters.

        :param star:            The Star to initialize.
        :param logger:          A logger object for logging debug info. [default: None]
        :param default_init:    Ignored. [default: None]

        :returns: a new initialized Star.
        """
        params = self.kwargs_to_params()
        params_var = np.zeros_like(params)
        fit = star.fit.newParams(params, params_var=params_var, num=self._num)
        star = Star(star.data, fit)
        return star

    @lru_cache(maxsize=8)
    def getOptics(self, zernike_coeff):
        """Get the strictly Optics PSF component.  Use two phase screens, one from
        mirror_figure_im, and the other from zernike aberrations

        :param zernike_coeff:   An array or list with Zernike Coefficients, in units of waves.
                                Array indexed as [0, 0, z2, z3, z4,...z11...]

        :returns: a galsim.GSObject instance
        """

        if self.mirror_figure_screen:

            optical_screen = galsim.OpticalScreen(diam=self.diam, aberrations=zernike_coeff,
                                                  lam_0=self.lam)
            screens = galsim.PhaseScreenList([optical_screen, self.mirror_figure_screen])
            optics_psf = galsim.PhaseScreenPSF(screen_list=screens, aper=self.aperture,
                                               lam=self.lam, gsparams=self.gsparams)

        else:
            optics_psf = galsim.OpticalPSF(lam=self.lam, diam=self.diam, aper=self.aperture,
                                           aberrations=zernike_coeff, gsparams=self.gsparams)

        return optics_psf

    @lru_cache(maxsize=8)
    def getAtmosphere(self, r0, L0=None, g1=0.0, g2=0.0):
        """ Get the Atmospheric PSF component.

        :param r0:          Fried parameter
        :param L0:          Outer scale
        :param g1:          Shear g1 component
        :param g2:          Shear g2 component

        :returns: a galsim.GSObject instance
        """
        if self.atmo_type == 'VonKarman':
            if L0 is None or L0==0.:
                raise ValueError("No value specified for VonKarman L0 ")
            else:
                atm = galsim.VonKarman(lam=self.lam, r0=r0, L0=L0, flux=1.0, gsparams=self.gsparams)
        elif self.atmo_type == 'Kolmogorov':
            atm = galsim.Kolmogorov(lam=self.lam, r0=r0, flux=1.0, gsparams=self.gsparams)
        elif self.atmo_type == 'None' or self.atmo_type == None:
            atm = galsim.DeltaFunction()
        else:
            raise ValueError("Invalid atmo_type ",self.atmo_type)

        # shear
        if g1!=0.0 or g2!=0.0:
            atm = atm.shear(g1=g1, g2=g2)
        return atm

    def params_to_kwargs(self,params):
        """Fill kwarg from params ndarray or list

        :param params               An ndarray or list with parameters in order

        :returns: a dictionary of named parameters
        """
        kwargs = {}
        kwargs['zernike_coeff'] = params[self.idx_z0:self.idx_z0+self.nZ+1]
        kwargs['r0'] = params[self.idx_r0]
        kwargs['L0'] = params[self.idx_L0]
        kwargs['g1'] = params[self.idx_g1]
        kwargs['g2'] = params[self.idx_g2]
        return kwargs

    def kwargs_to_params(self,zernike_coeff=[0,0,0,0,0],r0=0.15,L0=10.,g1=0.,g2=0.):
        """Fill params ndarray from kwargs

        :param zernike_coeff:      A ndarray or list with [0, 0, z2, z3, z4, z5, z6...z11..z37]
                                   [default=[0,0,0,0,0]]
        :param r0:                 Value of r0 for Atmospheric Kernel. [default=0.15]
        :param L0:                 Value of L0 for Atmospheric Kernel. [default=10.0]
        :param g1:                 Value of g1 to Shear PSF [default=0.]
        :param g2:                 Value of g2 to Shear PSF [default=0.]

        :returns: an ndarray of parameters
        """
        params = np.zeros(self.param_len)

        # adapt to a zernike_coeff array shorter or longer than 0:37+1
        zlen = len(zernike_coeff)
        if zlen<=self.nZ+1:
            params[self.idx_z0:self.idx_z0+zlen] = zernike_coeff
        else:
            params[self.idx_z0:self.idx_z0+self.nZ+1] = zernike_coeff[0:self.nZ+1]

        params[self.idx_r0] = r0
        params[self.idx_L0] = L0
        params[self.idx_g1] = g1
        params[self.idx_g2] = g2
        return params


    def getProfile(self,params=None,zernike_coeff=None,r0=None,L0=None,g1=0.,g2=0.):
        """Get a version of the model as a GalSim GSObject

        :param params          An ndarray with the parameters ordered via idx_PARAM data members
        :param zernike_coeff   A list of the Zernike coefficients in units of [waves]
        :param r0              Atmospheric Fried parameter [meters]
        :param L0              Atmospheric Outer scale [meters]
        :param g1              Atmospheric g1 Shear
        :param g2              Atmospheric g2 Shear

        :returns: a galsim.GSObject instance of the combined optics, atmosphere and diffusion PSF
        """

        # decode input, if params array or list is present use it alone,
        # otherwise parameters are in getProfile argument list
        if params is not None :
            *zernike_coeff,r0,L0,g1,g2 = params
        aberrations = self.base_aberrations.copy()
        if zernike_coeff is not None:
            aberrations[:len(zernike_coeff)] += zernike_coeff

        # list of PSF components
        prof = []

        # gaussian for CCD Diffusion
        # note that the WCS may make it effectively Sheared on the Focal Plane
        # consider removing that shear by giving this Gaussian opposite shear to the WCS
        if self.sigma > 0.:
            gaussian = galsim.Gaussian(sigma=self.sigma, gsparams=self.gsparams)
            prof.append(gaussian)

        # atmospheric kernel
        if self.atmo_type not in ['None', None]:
            atmopsf = self.getAtmosphere(r0, L0, g1, g2)
            prof.append(atmopsf)

        # optics
        optics = self.getOptics(tuple(aberrations))

        if len(prof) == 0:
            return optics
        else:
            prof.append(optics)
            return galsim.Convolve(prof, gsparams=self.gsparams)
