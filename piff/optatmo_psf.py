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
.. module:: optatmo_psf
"""

from __future__ import print_function

import galsim
import numpy as np

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .gsobject_model import GSObjectModel, Kolmogorov, Gaussian
from .star import Star, StarFit, StarData
from .util import hsm, hsm_error

class OptAtmoPSF(PSF):
    """Combine Optical and Atmospheric PSFs together
    """

    def __init__(self, atmo_interp=None, analytic_coefs=None, optatmo_psf_kwargs={}, optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None, n_optfit_stars=0, shape_weights=[0.5, 1, 1],  shape_unnormalized=True, shape_method='hsm', fov_radius=1., jmax_pupil=11, jmax_focal=10,  min_optfit_snr=0, optfit_optimize='analytic', logger=None):
        """
        Instead of arbitrarily putting different models in and dealing with
        convolutions of models, put into single piece here.

        Fitting is done with lmfit, which also lets us specificy which params
        are fixed as we desire. This is akin to the gsobject, but we can
        explicitly fit parameters.

        Further, the interpolation and model are integrated together.

        Things I am thinking about right now that we will need and I am likely
        to forget:
            - A function to strip star parameters for the atmosphere
              interpolant to fit
            - A function to go from u,v position to zernike polynomials. Could
              one day be replaced by galsim implementation
            - make sure when we fit the constant piece (which includes size, g1,
              g2) that the atmosphere piece does NOT go beyond constant
              component

        In the end, this will completely replace the Optical Model

        Fit Combined Atmosphere and Optical PSF in two stage process

        :param atmo_interp:             Piff Interpolant object that represents
                                        the atmospheric interpolation
        :param analytic_coefs:          Terms in analytic breakdown of zernike
                                        to shape transformation. Only used in
                                        when optfit_optimize = analytic
        :param optatmo_psf_kwargs:        Terms that set the state of the PSF,
                                        excepting the atmospheric interpolant
        :param optical_psf_kwargs:      Arguments to pass into galsim
                                        opticalpsf object
        :param kolmogorov_kwargs:       Arguments to pass into galsim
                                        kolmogorov object
        :param reference_wavefront:     Reference interpolator for the optical
                                        wavefront. Takes in stars, resturns
                                        aberrations. Default is to not include.
        :param n_optfit_stars:          [default: 0] If > 0, randomly sample
                                        only n_optfit_stars for the fit
        :param shape_weights:           Array or list of shape_weights for
                                        comparing gaussian shapes in fit
                                        [default: [0.5, 1, 1], so downweight
                                        size]
        :param shape_unnormalized:      [Default: True] Sets whether the shapes
                                        we evaluate are in the normalized or
                                        unnormalized bases. If unnormalized,
                                        the shapes have units arcsec^2
        :param shape_method:            [Default: 'hsm'] Decide whether to use
                                        'hsm' or 'lmfit' to measure the shapes.
        :param fov_radius:              [Default: 1.] Radius of telescope in
                                        u,v coordinates
        :param jmax_pupil:              Number of pupil-basis zernikes in
                                        Optical model [default: 11]
        :param jmax_focal:              Number of field-basis zernikes in
                                        Optical model [default: 11]
        :param min_optfit_snr:          minimum snr from star property required
                                        for optical portion of fit. If 0,
                                        ignored. [default: 0]
        :param optfit_optimize:         When fitting the optical psf, optimize
                                        to pixels, shapes, or analytic formulae
                                        for shapes [pixel, moments,
                                        analytic=default]
        :param logger:                  A logger object for logging debug info.
                                        [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # atmo_interp is a parsed class
        self.atmo_interp = atmo_interp
        self.analytic_coefs = analytic_coefs
        self.optical_psf_kwargs = optical_psf_kwargs
        self.kolmogorov_kwargs = kolmogorov_kwargs
        self.reference_wavefront = reference_wavefront

        self.min_optfit_snr = min_optfit_snr
        self.n_optfit_stars = n_optfit_stars
        self.optfit_optimize = optfit_optimize

        #####
        # setup double zernike piece
        #####
        if jmax_pupil < 4:
            # why do an optatmo if you have no optical?
            raise ValueError('OptAtmo PSF requires at least 4 aberrations; found {0}'.format(jmax_pupil))
        self.jmax_pupil = jmax_pupil
        if jmax_focal < 1:
            # need at least some constant piece of focal
            raise ValueError('OptAtmo PSF requires at least a constant field zernike; found {0}'.format(jmax_focal))
        self.jmax_focal = jmax_focal

        self.fov_radius = fov_radius

        # Field-of-view does not have obscuration, so obscuration=0 and annular=False here.
        self._noll_coef_field = galsim.phase_screens._noll_coef_array(self.jmax_focal, 0.0, False)

        init_error = 10000
        self.optatmo_psf_kwargs = {
            'size': 1.0,  'fix_size': False,  'error_size': init_error,
            'g1':   0,    'fix_g1':   False,  'error_g1':   init_error,
            'g2':   0,    'fix_g2':   False,  'error_g2':   init_error,
            }
        self.keys = [ 'size', 'g1', 'g2', ]
        # throw in default zernike parameters
        # only fit zernikes starting at 4 / defocus
        for zi in range(4, self.jmax_pupil + 1):
            for dxy in range(1, self.jmax_focal + 1):
                zkey = 'zUV{0:03d}_zXY{1:03d}'.format(zi, dxy)
                self.keys.append(zkey)
                # initial value
                self.optatmo_psf_kwargs[zkey] = 0
                # fix if greater than max focal zernike
                if dxy > self.jmax_focal:
                    self.optatmo_psf_kwargs['fix_' + zkey] = True
                else:
                    self.optatmo_psf_kwargs['fix_' + zkey] = False
                # # we shall not specify in advance a zernike limit
                # self.optatmo_psf_kwargs['limit_' + zkey] = (-2, 2)
                # initial placeholder for error in parameter
                self.optatmo_psf_kwargs['error_' + zkey] = init_error
        try:
            self.optatmo_psf_kwargs.update(optatmo_psf_kwargs)
        except TypeError:
            # should just be from loading, and is fixed by _finish_read
            pass

        # create initial aberrations_field from optatmo_psf_kwargs
        logger.info("Initializing optatmopsf state")
        self.aberrations_field = np.zeros((self.jmax_pupil, self.jmax_focal),
                                          dtype=float)
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger)

        # since we haven't fit the interpolator, yet, disable atmosphere
        logger.info('Disabling Varying Atmosphere')
        self._enable_atmosphere = False

        # set up hardcoded gsparams for _considerable_ speedup
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
        # if not specified in advance, decrease pad_factor and oversampling for speedup in optical modeling
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 0.5
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 0.5
        # set up gsobject model to measure shapes
        logger.info("Loading Kolmogorov GSObject for modeling shapes")
        # if we wanted, this could become an argument for some other model
        self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)

        # deal with shape piece
        self.shape_method = shape_method
        self.shape_unnormalized = shape_unnormalized
        self.shape_weights = np.array(shape_weights)
        # normalize shape_weights
        self.shape_weights /= self.shape_weights.sum()
        self._opt_shapes = []
        self._opt_shape_errors = []

        # kwargs
        self.kwargs = {'fov_radius': self.fov_radius,
                       'jmax_pupil': self.jmax_pupil,
                       'jmax_focal': self.jmax_focal,
                       'analytic_coefs': self.analytic_coefs,
                       'optical_psf_kwargs': self.optical_psf_kwargs,
                       'kolmogorov_kwargs': self.kolmogorov_kwargs,
                       'min_optfit_snr': self.min_optfit_snr,
                       'n_optfit_stars': self.n_optfit_stars,
                       'optfit_optimize': self.optfit_optimize,
                       'shape_unnormalized': self.shape_unnormalized,
                       'shape_method': self.shape_method,
                       # junk entries to be overwritten in _finish_read function
                       'optatmo_psf_kwargs': 0,
                       'atmo_interp': 0,
                       'reference_wavefront': 0,
                       'shape_weights': 0,
                       }

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to
        use for initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict,
                                config['psf']
        :param logger:          A logger object for logging debug info.
                                [default: None]

        :returns:               a kwargs dict to pass to the initializer
        """
        config_psf = config_psf.copy()  # Don't alter the original dict.

        kwargs = config_psf.copy()

        # do processing as appropriate
        # set up optical and atmosphere psf kwargs using the optical model
        if 'optical_psf_kwargs' in config_psf:
            optical_psf_kwargs = config_psf['optical_psf_kwargs']
        else:
            optical_psf_kwargs = {}

        optical = Optical(logger=logger, **optical_psf_kwargs)
        kwargs['optical_psf_kwargs'] = optical.optical_psf_kwargs
        kolmogorov_kwargs = optical.kolmogorov_kwargs
        if 'kolmogorov_kwargs' in config_psf:
            kolmogorov_kwargs.update(config_psf['kolmogorov_kwargs'])
        if kolmogorov_kwargs.keys() == ['lam']:
            kolmogorov_kwargs = {'half_light_radius': 1.0}
        # Also, let r0=0 or None indicate that there is no kolmogorov component
        if 'r0' in kolmogorov_kwargs and not kolmogorov_kwargs['r0']:
            kolmogorov_kwargs = {'half_light_radius': 1.0}
        kwargs['kolmogorov_kwargs'] = kolmogorov_kwargs

        # optical field
        if 'optatmo_psf_kwargs' in config_psf:
            kwargs['optatmo_psf_kwargs'] = config_psf['optatmo_psf_kwargs']

        # atmo interp
        kwargs['atmo_interp'] = Interp.process(config_psf['atmo_interp'], logger=logger)

        # process reference_wavefront kwargs
        reference_wavefront_kwargs = {}
        if 'reference_wavefront' in config_psf:
            reference_wavefront_kwargs.update(config_psf['reference_wavefront'])
            logger.info("Making reference wavefront")
            reference_wavefront = Interp.process(reference_wavefront_kwargs, logger=logger)
        else:
            logger.info("Skipping reference wavefront")
            reference_wavefront = None
        kwargs['reference_wavefront'] = reference_wavefront

        # process analytic formula
        if 'analytic_coefs' in config_psf:
            analytic_coefs = config_psf['analytic_coefs']
            if isinstance(analytic_coefs, str):
                analytic_coefs = np.load(analytic_coefs)
        else:
            analytic_coefs = None
        kwargs['analytic_coefs'] = analytic_coefs

        return kwargs

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """

        # write the atmo interp if it exists
        if self.atmo_interp:
            self.atmo_interp.write(fits, extname + '_atmo_interp', logger)

        # write reference wavefront if it exists
        if self.reference_wavefront:
            self.reference_wavefront.write(fits, extname + '_reference_wavefront', logger)

        # write the final fitted state
        dtypes = [('shape_weights', '3f4')]
        for key in self.optatmo_psf_kwargs:
            if 'fix_' in key:
                dtypes.append((key, bool))
            else:
                dtypes.append((key, float))
        data = np.zeros(1, dtype=dtypes)
        data['shape_weights'] = self.shape_weights
        for key in self.optatmo_psf_kwargs:
            data[key][0] = self.optatmo_psf_kwargs[key]

        fits.write_table(data, extname=extname + '_solution')
        logger.info('Wrote optatmopsf state')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # read the atmo interp
        if extname + '_atmo_interp' in fits:
            self.atmo_interp = Interp.read(fits, extname + '_atmo_interp', logger)
            self._enable_atmosphere = True
        else:
            self.atmo_interp = None
            self._enable_atmosphere = False

        # read reference wavefront
        if extname + '_reference_wavefront' in fits:
            self.reference_wavefront = Interp.read(fits, extname + '_reference_wavefront', logger)
        else:
            self.reference_wavefront = None

        # read the final state, update the psf
        data = fits[extname + '_solution'].read()
        for key in data.dtype.names:
            if key == 'shape_weights':
                self.shape_weights = data['shape_weights'][0]
            else:
                self.optatmo_psf_kwargs[key] = data[key][0]
        logger.info('Reloading optatmopsf state')
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger)

    def fit(self, stars, wcs, pointing, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the
                                telescope pointing.
                                [Note: pointing should be None if the WCS is
                                not a CelestialWCS]
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """

        # do first pass of flux and centers for the stars
        self.stars = []
        for star in stars:
            shape, error = self.measure_shape(star, return_error=True, logger=logger)
            star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))
            star.data.properties['shape'] = shape
            star.data.properties['shape_error'] = error
            self.stars.append(star)

        self.wcs = wcs
        self.pointing = pointing

        # fit optical interpolant piece including constant atmosphere to
        # either: analytic moments, actual moments, pixels. Mode is decided in
        # __init__ of PSF
        logger.info('Fitting optics')
        self.fit_optics(self.stars, logger=logger, **kwargs)

        # fit atmosphere
        logger.info('Fitting atmosphere')
        self.fit_atmosphere(self.stars, logger=logger, **kwargs)

        # enable atmosphere interpolation now that we have solved the interp
        logger.info('Enabling Atmosphere')
        self._enable_atmosphere = True

    def _getParamsList_aberrations_field(self, stars):
        # get zernike parameters and mean of kolmogorov from optical model
        # get field dependent aberrations_pupil (including atmosphere)
        # collect u and v from stars
        u = np.array([star.data['u'] for star in stars])
        v = np.array([star.data['v'] for star in stars])
        r = (u + 1j * v) / self.fov_radius
        rsqr = np.abs(r) ** 2
        aberrations_pupil = np.array([galsim.utilities.horner2d(rsqr, r, ca, dtype=complex).real
                               for ca in self._coef_arrays_field])

        # aberrations_pupil.shape = (ncoefs, nstars), but we want (nstars, ncoefs)
        aberrations_pupil = aberrations_pupil.T

        return aberrations_pupil

    def getParamsList(self, stars):
        """Get params for a list of stars.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           Params  [size, g1, g2, z4, z5...] for each star
        """
        params = np.zeros((len(stars), self.jmax_pupil), dtype=float)

        aberrations_pupil = self._getParamsList_aberrations_field(stars)
        params += aberrations_pupil

        if self.reference_wavefront:
            # TODO: this could be remade to just be for u, v coordinates centered about the center instead of pixel_to_focal stuff
            stars = [Star(star.data, None) for star in stars]
            stars = self.reference_wavefront.interpolateList(stars)
            aberrations_reference_wavefront = np.array([star_interpolated.fit.params for star_interpolated in stars])
            # put aberrations_reference_wavefront
            # reference wavefront starts at z4 but may not span full range of aberrations used
            n_reference_aberrations = aberrations_reference_wavefront.shape[1]
            if n_reference_aberrations + 3 < self.jmax_pupil:
                params[:, 3: n_reference_aberrations + 3] += aberrations_reference_wavefront
            else:
                # we have more jmax_pupil than reference wavefront
                params[:, 3:] += aberrations_reference_wavefront[:, :self.jmax_pupil - 3]

        # get kolmogorov parameters from atmosphere model, but only if we said so
        if self._enable_atmosphere:
            # strip star fit
            stars = [Star(star.data, None) for star in stars]
            stars = self.atmo_interp.interpolateList(stars)
            aberrations_atmo_star = np.array([star.fit.params for star in stars])
            params[:, 0:3] += aberrations_atmo_star

        return params

    def getParams(self, star):
        """Get params for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Params  [size, g1, g2, z4, z5...]
        """
        return self.getParamsList([star])[0]

    def getProfile(self, star):
        """Get galsim profile for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Galsim profile
        """
        # get parameters
        params = self.getParams(star)
        prof = self._profile(params)
        return prof

    def _profile(self, params):
        """Get galsim profile for a given params
        :param params:      [size, g1, g2, z4, z5...]
        :returns:           Galsim profile
        """

        # optics
        aberrations = np.zeros(4 + len(params[3:]))
        aberrations[4:] = params[3:]
        profs = []
        if np.any(aberrations != 0):
            opt = galsim.OpticalPSF(aberrations=aberrations, gsparams=self.gsparams, **self.optical_psf_kwargs)
            profs.append(opt)

        # atmosphere
        size = params[0]
        g1 = params[1]
        g2 = params[2]
        atmo = self.atmo_model.dilate(size).shear(g1=g1, g2=g2)
        profs.append(atmo)

        # convolve together!
        if len(profs) > 1:
            prof = galsim.Convolve(profs, gsparams=self.gsparams)
        else:
            prof = profs[0]

        return prof

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Star instance with its image filled with rendered
                            PSF
        """

        # get profile and params
        params = self.getParams(star)
        prof = self._profile(params)
        star = self.drawProfileStar(star, prof, params)
        return star

    def drawProfileStar(self, star, prof, params):
        """Generate PSF image for a given star and profile

        :param profile:     A galsim profile
        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Star instance with its image filled with rendered
                            PSF
        """
        # since I do this in a couple different places, let's do it in one place
        image = self.drawProfile(star, prof)
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
        fit = StarFit(params,
                      flux=star.fit.flux,
                      center=star.fit.center)
        return Star(data, fit)

    @classmethod
    def drawProfile(cls, star, prof):
        """Given a profile and a star with an image, draw thes star

        :param profile:     Profile to draw with
        :param star:        Star to draw onto

        :returns:           An image
        """
        image, weight, image_pos = star.data.getImage()
        image_model = image.copy()
        image_model = prof.drawImage(image_model, method='auto', offset=(star.image_pos-image_model.true_center))
        return image_model

    @staticmethod
    def shape_convert_to_unnormalized(scale, g1, g2):
        shear = galsim.Shear(g1=g1, g2=g2)
        e1norm = shear.e1
        e2norm = shear.e2
        # absgsq = g1**2 + g2**2
        # g2e = 2. / (1.+absgsq)
        # e1norm = g1 * g2e
        # e2norm = g2 * g2e
        e0 = np.sqrt(4 * scale ** 4 / (1 - e1norm ** 2 - e2norm ** 2))
        e1 = e1norm * e0
        e2 = e2norm * e0
        return e0, e1, e2

    @staticmethod
    def shape_convert_to_normalized(e0, e1, e2):
        e1norm = e1 / e0
        e2norm = e2 / e0
        # e_max = 0.5
        # if np.abs(e1norm) > e_max:
        #     print('Warning! e1norm = {0:.2e} / {1:.2e} = {2:.2e} > {3:.2e}! Setting to max'.format(e1, e0, e1norm, e_max))
        #     e1norm = e_max * np.sign(e1norm)
        # if np.abs(e2norm) > e_max:
        #     print('Warning! e2norm = {0:.2e} / {1:.2e} = {2:.2e} > {3:.2e}! Setting to max'.format(e2, e0, e2norm, e_max))
        #     e2norm = e_max * np.sign(e2norm)
        shear = galsim.Shear(e1=e1norm, e2=e2norm)
        g1 = shear.g1
        g2 = shear.g2
        # absesq = e1norm ** 2 + e2norm ** 2
        # e2g = 1. / (1. + np.sqrt(1. - absesq))
        # g1 = e1norm * e2g
        # g2 = e2norm * e2g
        scale = np.sqrt(np.sqrt((e0 ** 2 - e1 ** 2 - e2 ** 2)) * 0.5)
        return scale, g1, g2

    @staticmethod
    def shape_convert_errors_to_unnormalized(sigma_scale, sigma_g1, sigma_g2, scale, g1, g2):

        # power of sympy
        beta = np.sqrt(scale ** 4 * (g1 ** 2 + g2 ** 2 + 1) ** 2 / (g1 ** 4 + 2 * g1 ** 2 * g2 ** 2 - 2 * g1 ** 2 + g2 ** 4 - 2 * g2 ** 2 + 1))
        gamma = (g1 ** 2 + g2 ** 2 - 1) * (g1 ** 2 + g2 ** 2 + 1)

        de0dscale = 4 / scale * beta
        de0dg1 = -8 * g1 * beta / gamma
        de0dg2 = -8 * g2 * beta / gamma

        de1dscale = 8 * g1 * beta / (scale * (g1 ** 2 + g2 ** 2 + 1))
        de1dg1 = -4 * beta * (g1 ** 2 - g2 ** 2 + 1) / gamma
        de1dg2 = -8 * g1 * g2 * beta / gamma

        de2dscale = 8 * g2 * beta / (scale * (g1 ** 2 + g2 ** 2 + 1))
        de2dg1 = -8 * g1 * g2 * beta / gamma
        de2dg2 = -4 * beta * (-g1 ** 2 + g2 ** 2 + 1) / gamma

        sigma_e0 = np.sqrt(de0dscale ** 2 * sigma_scale ** 2 + de0dg1 ** 2 * sigma_g1 ** 2 + de0dg2 ** 2 * sigma_g2 ** 2)
        sigma_e1 = np.sqrt(de1dscale ** 2 * sigma_scale ** 2 + de1dg1 ** 2 * sigma_g1 ** 2 + de1dg2 ** 2 * sigma_g2 ** 2)
        sigma_e2 = np.sqrt(de2dscale ** 2 * sigma_scale ** 2 + de2dg1 ** 2 * sigma_g1 ** 2 + de2dg2 ** 2 * sigma_g2 ** 2)

        return sigma_e0, sigma_e1, sigma_e2

    @staticmethod
    def shape_convert_errors_to_normalized(sigma_e0, sigma_e1, sigma_e2, e0, e1, e2):
        alpha = np.sqrt(1 - (e1 / e0) ** 2 - (e2 / e0) ** 2)

        dg1de0 = -e1 * e0 ** -2 * 1 / (1 + alpha) * (1 + (1 - alpha ** 2) / alpha)
        dg1de1 = 1. / (e0 * (1 + alpha)) * (1 + e1 ** 2 / (e0 ** 2 * alpha))
        dg1de2 = e1 * e2 * e0 ** -3. / (alpha * (1 + alpha) ** 2.)

        dg2de0 = -e2 * e0 ** -2 * 1 / (1 + alpha) * (1 + (1 - alpha ** 2) / alpha)
        dg2de1 = e1 * e2 * e0 ** -3. / (alpha * (1 + alpha) ** 2.)
        dg2de2 = 1. / (e0 * (1 + alpha)) * (1 + e2 ** 2 / (e0 ** 2 * alpha))

        dsigmade0 = np.sqrt(2) / (4 * np.sqrt(e0) * alpha ** (3. / 2.))
        dsigmade1 = -np.sqrt(2) / (4 * np.sqrt(e0) * alpha ** (3. / 2.))
        dsigmade2 = -np.sqrt(2) / (4 * np.sqrt(e0) * alpha ** (3. / 2.))

        sigma_sigma = np.sqrt(dsigmade0 ** 2 * sigma_e0 ** 2 + dsigmade1 ** 2 * sigma_e1 ** 2 + dsigmade2 ** 2 * sigma_e2 ** 2)
        sigma_g1 = np.sqrt(dg1de0 ** 2 * sigma_e0 ** 2 + dg1de1 ** 2 * sigma_e1 ** 2 + dg1de2 ** 2 * sigma_e2 ** 2)
        sigma_g2 = np.sqrt(dg2de0 ** 2 * sigma_e0 ** 2 + dg2de1 ** 2 * sigma_e1 ** 2 + dg2de2 ** 2 * sigma_e2 ** 2)
        return sigma_sigma, sigma_g1, sigma_g2

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           List of Star instances with its image filled with
                            rendered PSF

        Slightly different from drawStar because we get all params at once
        """
        # get all params at once
        params = self.getParamsList(stars)
        # now step through to make the stars
        stars_drawn = [self.drawProfileStar(star, self._profile(param), param) for param, star in zip(params, stars)]
        return stars_drawn

    def _update_optatmopsf(self, optatmo_psf_kwargs={}, logger=None):
        """Update aberrations_field attribute, recompute coef arrays for the
        field

        :param _coef_arrays_field:      The arrays that take an input complex
                                        number r = u + iv and its square abs(u
                                        + iv) ** 2 to compute the Zernike
                                        aberrations
        :param aberrations_field:       A two-dimensional array of
                                        coefficients.  First index is over nm,
                                        i.e. the pupil aberrations.  The second
                                        index is over rs, i.e., the field
                                        dependence. Both are defined according
                                        to Noll convention
        :param logger:                  A logger object for logging debug info
        """
        if len(optatmo_psf_kwargs) == 0:
            optatmo_psf_kwargs = self.optatmo_psf_kwargs
            keys = self.keys
        else:
            keys = optatmo_psf_kwargs.keys()

        aberrations_changed = False
        for key in keys:
            # some checks
            if 'limit_' in key:
                continue
            elif 'error_' in key:
                continue
            elif 'fix_' in key:
                continue

            # size, g1, g2 mean constant terms
            if key == 'size':
                uv = 1
                xy = 1
            elif key == 'g1':
                uv = 2
                xy = 1
            elif key == 'g2':
                uv = 3
                xy = 1
            else:
                # zUV012_zXY034; kludgey as hell
                uv = int(key[3:6])
                xy = int(key[10:13])
                if uv < 4:
                    raise ValueError('Not allowed to fit pupil zernike {0} less than {2}, key {1}!'.format(uv, key, 4))
                elif xy < 1:
                    raise ValueError('Not allowed to fit focal zernike {0} less than {2} !, key {1}!'.format(xy, key, 1))
                elif uv > self.jmax_pupil:
                    raise ValueError('Not allowed to fit pupil zernike {0}, greater than {2}, key {1}!'.format(uv, key, self.jmax_pupil))
                elif xy > self.jmax_focal:
                    raise ValueError('Not allowed to fit focal zernike {0} greater than {2} !, key {1}!'.format(xy, key, self.jmax_focal))

            old_value = self.aberrations_field[uv - 1, xy - 1]
            new_value = optatmo_psf_kwargs[key]
            if old_value != new_value:
                if logger: logger.debug('Updating Zernike parameter {0} from {1:+.2e} + {3:+.2e} = {2:+.2e}'.format(key, old_value, new_value, new_value - old_value))
                self.aberrations_field[uv - 1, xy - 1] = new_value
                aberrations_changed = True
        if logger: logger.debug('---------- Recomputing field zernike coefs')

        if aberrations_changed:
            # One coef_array for each wavefront aberration
            # shape (jmax_pupil, maxn_focal, maxm_focal)
            self._coef_arrays_field = np.array([np.dot(self._noll_coef_field, a)
                                                for a in self.aberrations_field])

    def measure_shape(self, star, return_error=True, logger=None):
        if self.shape_method == 'lmfit':
            return self.measure_shape_lmfit(star, shape_unnormalized=self.shape_unnormalized, return_error=return_error, logger=logger)
        elif self.shape_method == 'hsm':
            return self.measure_shape_hsm(star, shape_unnormalized=self.shape_unnormalized, return_error=return_error, logger=logger)

    def measure_shape_hsm(self, star, shape_unnormalized=True, return_error=True, logger=None):
        if return_error:
            # do in unnormalized basis by default
            flux, u0, v0, e0, e1, e2, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, return_debug=False)
            if shape_unnormalized:
                shape = np.array([flux, u0, v0, e0, e1, e2])
                error = np.array([sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2])
            else:
                # convert back
                sigma, g1, g2 = self.shape_convert_to_normalized(e0, e1, e2)
                shape = np.array([flux, u0, v0, sigma, g1, g2])
                sigma_scale, sigma_g1, sigma_g2 = self.shape_convert_errors_to_normalized(sigma_e0, sigma_e1, sigma_e2, e0, e1, e2)

                error = np.array([sigma_flux, sigma_u0, sigma_v0, sigma_scale, sigma_g1, sigma_g2])

            return shape, error
        else:
            # do in normalized basis by default
            flux, u0, v0, sigma, g1, g2, flag = hsm(star)
            if shape_unnormalized:
                e0, e1, e2 = self.shape_convert_to_unnormalized(sigma, g1, g2)
                shape = np.array([flux, u0, v0, e0, e1, e2])
            else:
                shape = np.array([flux, u0, v0, sigma, g1, g2])
            return shape

    def measure_shape_lmfit(self, star, shape_unnormalized=True, return_error=True, logger=None):
        import lmfit
        # we also fit flux, du, dv but those are less important

        # put in initial guesses for flux, du, dv if they exist
        # all stars have fits with fluxes and centers defined, even if they may just be default values
        flux = star.fit.flux
        du, dv = star.fit.center

        # create lmparameters
        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=flux, vary=True, min=0.0)
        params.add('du', value=du, vary=True, min=-1, max=1)
        params.add('dv', value=dv, vary=True, min=-1, max=1)
        # limit size of perturbations to size and ellipticity
        min_size = 0.2
        max_g = 0.4
        params.add('size', value=1., vary=True, min=min_size)
        params.add('g1', value=0, vary=True, min=-max_g, max=max_g)
        params.add('g2', value=0, vary=True, min=-max_g, max=max_g)

        # do fit
        results = lmfit.minimize(self._measure_shape_residual, params,
                                 args=(star, logger,))
        if logger: logger.debug(lmfit.fit_report(results))
        flux, du, dv, size, g1, g2 = results.params.valuesdict().values()
        if results.errorbars:
            error = np.sqrt(np.diag(results.covar))
        else:
            if logger: logger.debug('Cannot estimate errors!')
            min_err = 0.01  # must change for unnormalized basis stuff
            error = np.array([min_err, min_err, min_err, min_err, min_err, min_err])

        if shape_unnormalized:
            # convert fit to unnormalized basis, assuming no covariance between fitted parameters
            e0, e1, e2 = self.shape_convert_to_unnormalized(size, g1, g2)
            shape = np.array([flux, du, dv, e0, e1, e2])
            sigma_size, sigma_g1, sigma_g2 = error[3:6]
            sigma_e0, sigma_e1, sigma_e2 = self.shape_convert_errors_to_unnormalized(sigma_size, sigma_g1, sigma_g2, size, g1, g2)

        else:
            shape = np.array([flux, du, dv, size, g1, g2])

        if not results.success:
            if logger: logger.debug('Failure measuring star shape!')
            shape *= np.nan
            error *= np.nan
        if return_error:
            return shape, error
        else:
            return shape

    def _measure_shape_residual(self, lmparams, star, logger=None):
        # modify params with lmparams
        flux, du, dv, size, g1, g2 = lmparams.valuesdict().values()

        prof = self.atmo_model.dilate(size).shear(g1=g1, g2=g2).shift(du, dv) * flux
        # calculate chi2
        image, weight, image_pos = star.data.getImage()
        image_model = self.drawProfile(star, prof)
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

    def fit_optics(self, stars, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of
        operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        import lmfit
        if logger:
            logger.info("Start fitting Optical fit using lmfit")

        # decide optimization function
        if self.optfit_optimize == 'pixel':
            resid = self._fit_optics_residual_pixel
        elif self.optfit_optimize == 'moments':
            resid = self._fit_optics_residual_moments
        elif self.optfit_optimize == 'analytic':
            resid = self._fit_optics_residual_analytic
        else:
            raise KeyError('Optical Fitter Algorithm {0} not allowed'.format(self.optfit_optimize))

        # measure shapes until we have enough stars
        self._opt_stars = []
        self._opt_snrs = []
        self._opt_shapes = []
        self._opt_shape_errors = []
        self._opt_indices = []  # which stars made it

        indices = np.arange(len(stars))
        if self.n_optfit_stars and self.n_optfit_stars < len(stars):
            np.random.shuffle(indices)
            max_stars = self.n_optfit_stars
        else:
            max_stars = len(stars)

        for indx in indices:
            if len(self._opt_stars) > max_stars:
                if logger:
                    logger.info("Finishing opt_star measurements at {0} stars".format(len(self._opt_stars)))
                break

            if logger:
                logger.debug("Measuring shape of star {0}".format(indx))
            star = stars[indx]
            snr = self.measure_snr(star)
            if self.min_optfit_snr > 0:
                if snr < self.min_optfit_snr:
                    if logger:
                        logger.debug("Skipping star {0} because SNR {1} < {2}".format(indx, snr, self.min_optfit_snr))
                    continue

            # idea here is that even for pixels, if we can't fit a shape, the
            # star is probably borked and should be skipped
            if 'shape' in star.data.properties:
                shape = star.data.properties['shape']
                error = star.data.properties['shape_error']
            else:
                shape, error = self.measure_shape(star, return_error=True, logger=logger)

            # put the properties into the starfit
            star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))

            self._opt_stars.append(star)
            self._opt_snrs.append(snr)
            self._opt_shapes.append(shape)
            self._opt_shape_errors.append(error)
            self._opt_indices.append(indx)

        # arrayify
        self._opt_snrs = np.array(self._opt_snrs)
        self._opt_shapes = np.array(self._opt_shapes)
        self._opt_shape_errors = np.array(self._opt_shape_errors)
        self._opt_indices = np.array(self._opt_indices)

        if len(self._opt_stars) < max_stars:
            if logger:
                logger.info("Using {0} stars instead of desired {1} out of {2}".format(len(self._opt_stars), max_stars, len(stars)))

        # create lmparameters
        params = lmfit.Parameters()
        # step through keys
        for key in self.keys:
            value = self.optatmo_psf_kwargs[key]
            vary = not self.optatmo_psf_kwargs['fix_' + key]
            if 'limit_' + key in self.optatmo_psf_kwargs:
                min, max = self.optatmo_psf_kwargs['limit_' + key]
            else:
                min, max = (None, None)
            params.add(key, value=value, vary=vary, min=min, max=max)

        # do fit
        results = lmfit.minimize(resid, params, args=(logger,))#, maxfev=self.kwargs['max_iterations'])

        key_i = 0
        for key in self.keys:
            if not self.optatmo_psf_kwargs['fix_' + key]:
                val = results.params.valuesdict()[key]
                self.optatmo_psf_kwargs[key] = val

                err = np.sqrt(results.covar[key_i, key_i])
                self.optatmo_psf_kwargs['error_' + key] = err
                key_i += 1
        self._update_optatmopsf(logger=logger)

        # set final fit
        if logger:
            logger.info('Optical fit from lmfit parameters:')
            logger.info(lmfit.fit_report(results))

        # save this for debugging purposes
        self._opt_results = results

    @staticmethod
    def measure_snr(star):
        """Calculate the signal-to-noise of a given star. Copied from input.py

        :param star:        Input star, with stamp, weight

        :returns: the SNR value.
        """
        # The S/N value that we use will be the weighted total flux where the
        # weight function is the star's profile itself.  This is the maximum
        # S/N value that any flux measurement can possibly produce, which will
        # be closer to an in-practice S/N than using all the pixels equally.
        #
        # F = Sum_i w_i I_i^2
        # var(F) = Sum_i w_i^2 I_i^2 var(I_i)
        #        = Sum_i w_i I_i^2             <--- Assumes var(I_i) = 1/w_i
        #
        # S/N = F / sqrt(var(F))
        image, weight, image_pos = star.data.getImage()
        I = image.array
        w = weight.array
        flux = (w*I*I).sum(dtype=float)
        snr = flux**0.5
        return snr

    def _fit_optics_residual_pixel(self, lmparams, logger=None):
        stars = self._opt_stars
        shapes = self._opt_shapes

        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(params, logger)

        # get optical params
        stars_params = self.getParamsList(stars)

        # calculate chi
        chi = np.array([])
        for i, star in enumerate(stars):
            opt_params = stars_params[i]
            # use fits from measure_shapes (which didn't include optical
            # component) for flux, du, dv. This is not technically correct,
            # because shifting coma and trefoil shifts the mean positions
            # about, but modifying that entails a factor of ~60 slowdown from
            # having to take about that many steps to fit the flux, du, dv for
            # given set of optical parameters
            flux, du, dv = shapes[i][:3]

            # get profile; modify based on flux and shifts
            profile = self._profile(opt_params).shift(du, dv) * flux

            # draw star
            image_model = self.drawProfile(star, profile)

            # compute chi2
            image, weight, image_pos = star.data.getImage()
            chi_i = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
            # if logger: logger.debug('Star {0}: flux {1:.2e}, du {2:+.2e}, dv {3:+.2e}, params {4}, chi^2 sum {5} / {6}'.format(i, flux, du, dv, str(opt_params), (chi_i ** 2).sum(), len(chi_i)))
            chi = np.hstack((chi, chi_i))

        return chi

    # TODO: this is definitely broken!
    def _fit_optics_residual_moments(self, lmparams, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(params, logger)

        # generate stars
        stars_model = self.drawStarList(self._opt_stars)

        # measure their shapes and calculate chi
        chi = np.array([])
        for i, star, shape, error in zip(np.arange(len(stars_model)), stars_model, self._opt_shapes, self._opt_shape_errors):
            shape_model = self.measure_shape(star, return_error=False, logger=logger)
            if np.any(shape_model != shape_model):
                if logger:
                    logger.debug('Warning! Shape measurement failed for model of star {0}'.format(i))
                continue

            # don't care about flux, du, dv here
            chi_i = (self.shape_weights * (shape_model - shape) / error)[3:]
            chi = np.hstack((chi, chi_i))

        return chi

    def _fit_optics_residual_analytic(self, lmparams, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(params, logger)

        # get star params
        zernikes = self.getParamsList(self._opt_stars)

        # generate analytic star moments
        shapes_model = self.analytic_zernike_shapes(zernikes, self.analytic_coefs)

        # calculate chi
        chi = (self.shape_weights[None] * (shapes_model - self._opt_shapes[:, 3:]) / self._opt_shape_errors[:, 3:]).flatten()

        return chi

    def fit_atmosphere(self, stars, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of
        operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        if self._enable_atmosphere:
            logger.info("Setting _enable_atmosphere == False. Was {0}".format(self._enable_atmosphere))
            self._enable_atmosphere = False

        # TODO: chisquare cutting?

        # fit models
        logger.info("Fitting atmo model")
        opt_params_stars = self.getParamsList(stars)
        model_fitted_stars = [self._fit_model(star, opt_params=opt_params, vary_shape=True, logger=logger)
                              for star, opt_params in zip(stars, opt_params_stars)]

        # fit interpolant
        logger.info("Initializing atmo interpolator")
        initialized_stars = self.atmo_interp.initialize(model_fitted_stars, logger=logger)

        logger.info("Fitting atmo interpolant")
        self.atmo_interp.solve(initialized_stars, logger=logger)

    def _fit_model(self, star, opt_params, vary_shape=True, logger=None):
        import lmfit
        # create lmparameters
        # put in initial guesses for flux, du, dv if they exist
        flux = star.fit.flux
        du, dv = star.fit.center
        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=flux, vary=True, min=0.0)
        params.add('du', value=du, vary=True, min=-1, max=1)
        params.add('dv', value=dv, vary=True, min=-1, max=1)

        if vary_shape:
            # we must also cut the min and max based on opt_params to avoid things
            # like large ellipticities or small sizes
            min_size = 0.2
            max_g = 0.4
            opt_size = opt_params[0]
            opt_g1 = opt_params[1]
            opt_g2 = opt_params[2]
            params.add('size', value=0, vary=True, min=min_size - opt_size)
            params.add('g1', value=0, vary=True, min=-max_g - opt_g1, max=max_g - opt_g1)
            params.add('g2', value=0, vary=True, min=-max_g - opt_g2, max=max_g - opt_g2)

        if logger: logger.debug('Initial parameters: {0}'.format(str(params)))

        # do fit
        results = lmfit.minimize(self._fit_model_residual, params,
                                 args=(star, opt_params, vary_shape, logger,))
        if logger: logger.debug(lmfit.fit_report(results))
        if vary_shape:
            flux, du, dv, size, g1, g2 = results.params.valuesdict().values()
            params = np.array([ size, g1, g2 ])
        else:
            flux, du, dv = results.params.valuesdict().values()
            params = None
        center = (du, dv)
        fit = StarFit(params, flux=flux, center=center)
        return Star(star.data, fit)

    def _fit_model_residual(self, lmparams, star, opt_params, vary_shape=True, logger=None):

        # modify params with lmparams
        params = opt_params.copy()
        if vary_shape:
            flux, du, dv, size, g1, g2 = lmparams.valuesdict().values()
            params[0] = opt_params[0] + size
            params[1] = opt_params[1] + g1
            params[2] = opt_params[2] + g2
        else:
            flux, du, dv = lmparams.valuesdict().values()

        if logger: logger.debug('Making Profile with shape terms {0} {1} {2} {3} {4} {5}'.format(flux, du, dv, params[0], params[1], params[2]))
        prof = self._profile(params).shift(du, dv) * flux

        # calculate chi2
        image, weight, image_pos = star.data.getImage()
        image_model = self.drawProfile(star, prof)
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

    @staticmethod
    def analytic_zernike_shapes(zernikes, coefs):
        # transform into full index
        params_onehot = np.vstack((np.ones(len(zernikes)).T, zernikes.T)).T
        nsample, ndim = params_onehot.shape
        # all params can be up to square * size
        ppoly_full = (params_onehot[:, :, None, None] * params_onehot[:, None, :, None] * params_onehot[:, :2][:, None, None, :])
        ppoly = ppoly_full.reshape(nsample, 2 * ndim * ndim)

        # apply model
        shapes = np.array([np.dot(ppoly, coef) for coef in coefs]).T
        return shapes