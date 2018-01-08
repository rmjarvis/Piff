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

# TODO: lmfit has issues

from __future__ import print_function

import galsim
import numpy as np
import copy

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .model import ModelFitError
# from .gsobject_model import GSObjectModel, Kolmogorov, Gaussian
from .star import Star, StarFit, StarData
from .util import hsm, hsm_error, measure_snr, write_kwargs, read_kwargs

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=False)
from .cypoly import cypoly

class OptAtmoPSF(PSF):
    """Combine Optical and Atmospheric PSFs together
    """

    def __init__(self, atmo_interp=None, analytic_coefs=None, optatmo_psf_kwargs={}, optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None, n_optfit_stars=0, shape_weights=[0.5, 1, 1],  shape_unnormalized=True, shape_method='hsm', fov_radius=4500., jmax_pupil=11, jmax_focal=10,  min_optfit_snr=0, optfit_optimize='analytic', logger=None):
        """
        Fit Combined Atmosphere and Optical PSF in two stage process

        :param atmo_interp:             Piff Interpolant object that represents
                                        the atmospheric interpolation
        :param analytic_coefs:          Terms in analytic breakdown of zernike
                                        to shape transformation. Only used in
                                        when optfit_optimize = analytic. It is
                                        formatted as [coefs, indices], with
                                        each of those being 3 deep (one for
                                        each of the three second moment shapes)
        :param optatmo_psf_kwargs:      Terms that set the state of the PSF,
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

        # If pupil_angle and strut angle are provided as strings, eval them.
        try:
            for key in ['pupil_angle', 'strut_angle']:
                if key in optical_psf_kwargs and isinstance(optical_psf_kwargs[key],str):
                    optical_psf_kwargs[key] = eval(optical_psf_kwargs[key])
        except TypeError:
            # we can end up saving optical_psf_kwargs as 0, so fix that
            optical_psf_kwargs = {}
            if logger: logger.warning('Warning! Invalid optical psf kwargs. Putting in empty dictionary')

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

        self.optatmo_psf_kwargs = {
                'size': 1.0,  'fix_size': False, 'min_size': 0.4,
                'g1':   0,    'fix_g1':   False, 'min_g1': -0.4, 'max_g1': 0.4,
                'g2':   0,    'fix_g2':   False, 'min_g2': -0.4, 'max_g2': 0.4,
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

                # default to unfixing all possible combinations
                self.optatmo_psf_kwargs['fix_' + zkey] = False
                # can optionally fix an entire UV or XY aberrations if we want
                fix_keyUV = 'fix_zUV{0:03d}'.format(zi)
                if fix_keyUV in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyUV]
                fix_keyXY = 'fix_zXY{0:03d}'.format(zi)
                if fix_keyXY in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyXY]

                zmax = 2
                self.optatmo_psf_kwargs['min_' + zkey] = -zmax
                self.optatmo_psf_kwargs['max_' + zkey] = zmax
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
        # if not specified in advance, decrease pad_factor and oversampling for
        # speedup in optical modeling
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 0.5
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 0.5


        # set up gsobject model to measure shapes
        logger.info("Loading Kolmogorov GSObject for modeling shapes")
        # if we wanted, this could become an argument for some other model, ala
        # gsobject. For now, we fix to Kolmogorov
        try:
            self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
        except TypeError:
            logger.warning('Warning! Invalid kolmogorov kwargs. Putting in dictionary with fwhm=1')
            self.kolmogorov_kwargs = {'fwhm': 1}
            self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)

        # deal with shape piece
        self.shape_method = shape_method
        self.shape_unnormalized = shape_unnormalized
        self.shape_weights = np.array(shape_weights)
        # normalize shape_weights
        self.shape_weights /= self.shape_weights.sum()
        # note: these params are not saved
        self._opt_shapes = []
        self._opt_shape_errors = []

        # kwargs
        self.kwargs = {'fov_radius': self.fov_radius,
                       'jmax_pupil': self.jmax_pupil,
                       'jmax_focal': self.jmax_focal,
                       'min_optfit_snr': self.min_optfit_snr,
                       'n_optfit_stars': self.n_optfit_stars,
                       'optfit_optimize': self.optfit_optimize,
                       'shape_unnormalized': self.shape_unnormalized,
                       'shape_method': self.shape_method,
                       # junk entries to be overwritten in _finish_read function
                       'analytic_coefs': 0,
                       'optatmo_psf_kwargs': 0,
                       'atmo_interp': 0,
                       'reference_wavefront': 0,
                       'shape_weights': 0,
                       'optical_psf_kwargs': 0,
                       'kolmogorov_kwargs': 0,
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
        if 'atmo_interp' in config_psf:
            if config_psf['atmo_interp'] in [None, 'none', 'None']:
                kwargs['atmo_interp'] = None
            else:
                kwargs['atmo_interp'] = Interp.process(config_psf['atmo_interp'], logger=logger)
        else:
            kwargs['atmo_interp'] = None

        # process reference_wavefront kwargs
        reference_wavefront_kwargs = {}
        if 'reference_wavefront' in config_psf:
            if config_psf['reference_wavefront'] in [None, 'none', 'None']:
                logger.info("Skipping reference wavefront")
                reference_wavefront = None
            else:
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
            if analytic_coefs in [None, 'none', 'None']:
                analytic_coefs = None
            else:
                if isinstance(analytic_coefs, str):
                    analytic_coefs = np.load(analytic_coefs).item()
                # make sure the analytic_coefs are in a reasonable format
                after_burner_array = np.array([coef for coef in analytic_coefs['after_burners']])
                indices = [np.array(index).astype(np.int64) for index in analytic_coefs['indices']]
                coefs = [np.array(coef).astype(np.float64) for coef in analytic_coefs['coefs']]
                analytic_coefs = [coefs, indices, after_burner_array]
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
            self.atmo_interp.write(fits, extname + '_atmo_interp')

        if not self.analytic_coefs == None:
            shape_index = len(self.analytic_coefs[1][0][0])
            dtype = [('coefs', 'f4'), ('indices', '{0}i4'.format(shape_index)), ('shape', 'i4')]
            coefs = []
            indices = []
            shape = []
            for i, coef, index in zip(range(len(self.analytic_coefs[0])), self.analytic_coefs[0], self.analytic_coefs[1]):
                coefs += coef.tolist()
                indices += index.tolist()
                shape += [i] * len(index)
            data = np.zeros(len(shape), dtype=dtype)
            data['coefs'] = coefs
            data['indices'] = indices
            data['shape'] = shape
            fits.write_table(data, extname=extname + '_analytic')
            # and afterburner
            after_burner = np.zeros(len(self.analytic_coefs[2]), dtype=[('offset', 'f4'), ('slope', 'f4')])
            after_burner_array = np.array([coef for coef in self.analytic_coefs[2]])
            after_burner['offset'] = after_burner_array[:, 0]
            after_burner['slope'] = after_burner_array[:, 1]
            fits.write_table(after_burner, extname=extname + '_analytic_afterburner')

        # write reference wavefront if it exists
        if self.reference_wavefront:
            self.reference_wavefront.write(fits, extname + '_reference_wavefront')

        # write optical_psf_kwargs
        # pupil_angle and strut_angle won't serialize properly, so repr them now in self.kwargs['optical_psf_kwargs'].
        optical_psf_kwargs = {}
        for key in self.optical_psf_kwargs:
            if key in ['pupil_angle', 'strut_angle']:
                optical_psf_kwargs[key] = repr(self.optical_psf_kwargs[key])
            else:
                optical_psf_kwargs[key] = self.optical_psf_kwargs[key]
        write_kwargs(fits, extname + '_optical_psf_kwargs', optical_psf_kwargs)

        # write kolmogorov_kwargs
        write_kwargs(fits, extname + '_kolmogorov_kwargs', self.kolmogorov_kwargs)

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
            self.atmo_interp = Interp.read(fits, extname + '_atmo_interp')
            self._enable_atmosphere = True
        else:
            self.atmo_interp = None
            self._enable_atmosphere = False

        if extname + '_analytic' in fits:
            data = fits[extname + '_analytic'].read()
            coefs_flat = data['coefs']
            indices_flat = data['indices']
            shape = data['shape']
            after_burner = fits[extname + '_analytic_afterburner'].read()
            after_burner_array = np.array([after_burner['offset'], after_burner['slope']]).T.astype(np.float64)
            possible_shapes = np.sort(np.unique(shape))
            analytic_coefs = [[], [], after_burner_array]
            for i in possible_shapes:
                analytic_coefs[0].append(np.array(coefs_flat[shape == i]).astype(np.float64))
                analytic_coefs[1].append(np.array(indices_flat[shape == i]).astype(np.int64))
            self.analytic_coefs = analytic_coefs
        else:
            self.analytic_coefs = None

        # read optical_psf_kwargs
        self.optical_psf_kwargs = read_kwargs(fits, extname=extname + '_optical_psf_kwargs')
        # If pupil_angle and strut angle are provided as strings, eval them.
        for key in ['pupil_angle', 'strut_angle']:
            if key in self.optical_psf_kwargs and isinstance(self.optical_psf_kwargs[key],str):
                self.optical_psf_kwargs[key] = eval(self.optical_psf_kwargs[key])
        logger.info('Reloading optatmopsf optical psf kwargs')

        # read kolmogorov_kwargs
        self.kolmogorov_kwargs = read_kwargs(fits, extname=extname + '_kolmogorov_kwargs')
        # set the atmo_model
        self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
        logger.info('Reloading optatmopsf atmo model')

        # read reference wavefront
        if extname + '_reference_wavefront' in fits:
            self.reference_wavefront = Interp.read(fits, extname + '_reference_wavefront')
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
        # temporary params for ensuring correct_profile works
        params = self.getParamsList(stars)
        for star_i, star in enumerate(stars):
            try:
                shape, error = self.measure_shape(star, return_error=True, logger=logger)
                star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))
                star.data.properties['shape'] = shape
                star.data.properties['shape_error'] = error
                # stars sometimes fail at correct_profile even though shape
                # measurement of the usual star is fine. I think it has to do
                # with the mask
                param = params[star_i]
                profile = self._profile(param)
                star_test = self._correct_profile(profile, star, shape)
                self.stars.append(star)
            except (ModelFitError, RuntimeError) as e:
                # something went wrong with this star
                if logger: logger.warn('Star {0} failed shape estimation. Skipping'.format(star_i))

        self.wcs = wcs
        self.pointing = pointing

        # fit optical interpolant piece including constant atmosphere to
        # either: analytic moments, actual moments, pixels. Mode is decided in
        # __init__ of PSF
        if logger: logger.info('Fitting optics')
        self.fit_optics(self.stars, logger=logger, **kwargs)

        if self.atmo_interp:
            # fit atmosphere
            if logger: logger.info('Fitting atmosphere')
            self.fit_atmosphere(self.stars, logger=logger, **kwargs)

            # enable atmosphere interpolation now that we have solved the interp
            if logger: logger.info('Enabling Atmosphere')
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
        # do_opt = np.any(aberrations != 0)
        do_opt = True  # for now, force it; not doing the opt actually does change the shape slightly!
        if do_opt:
            opt = galsim.OpticalPSF(aberrations=aberrations, gsparams=self.gsparams, **self.optical_psf_kwargs)

        # atmosphere
        size = params[0]
        g1 = params[1]
        g2 = params[2]
        atmo = self.atmo_model.dilate(size).shear(g1=g1, g2=g2)

        # convolve together!
        if do_opt:
            prof = galsim.Convolve([opt, atmo], gsparams=self.gsparams)
        else:
            prof = atmo

        return prof

    def drawStar(self, star, correct_flux_center=False):
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
        star = self.drawProfileStar(star, prof, params, correct_flux_center)
        return star

    def drawProfileStar(self, star, prof, params, correct_flux_center=False):
        """Generate PSF image for a given star and profile

        :param profile:     A galsim profile
        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Star instance with its image filled with rendered
                            PSF
        """
        # since I do this in a couple different places, let's do it in one place
        if correct_flux_center:
            # use flux and center properties
            prof = prof.shift(star.fit.center[0], star.fit.center[1]) * star.fit.flux
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
        """Given reduced shear and size, convert to unnormalized second moment
        shape parameterization

        :param scale:   size of object
        :param g1,2:    reduced shear of object
        :returns:       e0, e1, e2 = (Mxx + Myy, Mxx - Myy, 2 Mxy) unnormalized
                        second moments

        Note: This is what the galsim shear code is doing under the hood
        # absgsq = g1**2 + g2**2
        # g2e = 2. / (1.+absgsq)
        # e1norm = g1 * g2e
        # e2norm = g2 * g2e
        """
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
        """Given second moment shapes, return reduced shear.

        :param e0:      Mxx + Myy; unnormalized second moment size
        :param e1:      Mxx - Myy; corresponds to ellipticity 1
        :param e2:      2 Mxy; corresponds to ellipticity 1

        :returns:       Size and reduced shears

        Note: This is what the galsim shear code is doing under the hood
        # absesq = e1norm ** 2 + e2norm ** 2
        # e2g = 1. / (1. + np.sqrt(1. - absesq))
        # g1 = e1norm * e2g
        # g2 = e2norm * e2g
        """
        e1norm = e1 / e0
        e2norm = e2 / e0
        shear = galsim.Shear(e1=e1norm, e2=e2norm)
        g1 = shear.g1
        g2 = shear.g2
        scale = np.sqrt(np.sqrt((e0 ** 2 - e1 ** 2 - e2 ** 2)) * 0.5)
        return scale, g1, g2

    @staticmethod
    def shape_convert_errors_to_unnormalized(sigma_scale, sigma_g1, sigma_g2, scale, g1, g2):
        """Turn errors on size and reduced shear into errors on second moments,
        assuming no covariance between shape parameters, via propogation of
        errors.

        :params sigma...:       Errors on size and reduced shear
        :params scale, g1,2:    Size and reduced shear

        :returns sigma e0,1,2:  Errors on second moment shapes
        """

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
        """Turn errors on second moment shapes into errors on size and reduced
        shear, assuming no covariance between shape parameters, via propogation
        of errors.

        :params sigma...:           Errors on second moment shapes
        :params e0,1,2:             Second moment shapes

        :returns sigma sigma,g1,2:  Errors on size and reduced shears
        """
        alpha = np.sqrt(1 - (e1 / e0) ** 2 - (e2 / e0) ** 2)
        beta = e0 ** 2 - e1 ** 2 - e2 ** 2

        dsigmade0 = np.sqrt(2) * e0 / (4 * beta ** (3. / 4.))
        dsigmade1 = -np.sqrt(2) * e1 / (4 * beta ** (3. / 4.))
        dsigmade2 = -np.sqrt(2) * e2 / (4 * beta ** (3. / 4.))

        dg1de0 = -e1 * e0 ** -2 / (alpha * (alpha + 1))
        dg1de1 = (e0 * e0 * alpha + e0 * e0 - e2 * e2) / (e0 ** 3 * alpha * (alpha + 1) ** 2)
        dg1de2 = e1 * e2 / (e0 ** 3 * (alpha + 1) ** 2 * alpha)

        dg2de0 = -e2 * e0 ** -2 / (alpha * (alpha + 1))
        dg2de1 = e1 * e2 / (e0 ** 3 * (alpha + 1) ** 2 * alpha)
        dg2de2 = (e0 * e0 * alpha + e0 * e0 - e1 * e1) / (e0 ** 3 * alpha * (alpha + 1) ** 2)

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
            if 'error_' in key:
                continue
            elif 'fix_' in key:
                continue
            elif 'min_' in key:
                continue
            elif 'max_' in key:
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
                if logger: logger.debug('Updating Zernike parameter {0} from {1:+.4e} + {3:+.4e} = {2:+.4e}'.format(key, old_value, new_value, new_value - old_value))
                self.aberrations_field[uv - 1, xy - 1] = new_value
                aberrations_changed = True

        if aberrations_changed:
            if logger: logger.debug('---------- Recomputing field zernike coefficients')
            # One coef_array for each wavefront aberration
            # shape (jmax_pupil, maxn_focal, maxm_focal)
            self._coef_arrays_field = np.array([np.dot(self._noll_coef_field, a)
                                                for a in self.aberrations_field])

    def measure_shape(self, star, return_error=True, logger=None):
        """Measure the shape of a star

        :param star:            Star we want to measure
        :param return_error:    Bool. Also measure the error? [default: True]
        :param logger:          A logger object for logging debug info

        :returns:               An array of [flux, u0, v0, size, ellipticity 1,
                                ellipticity2], where the size and ellipticity
                                bases depend on whether the OptAtmoPSF object
                                has self.shape_unnormalized = 'hsm' or 'lmfit'.
                                Will also return the 1d errors if
                                return_error=True
        """
        if self.shape_method == 'lmfit':
            return self.measure_shape_lmfit(star, shape_unnormalized=self.shape_unnormalized, return_error=return_error, logger=logger)
        elif self.shape_method == 'hsm':
            return self.measure_shape_hsm(star, shape_unnormalized=self.shape_unnormalized, return_error=return_error, logger=logger)

    def measure_shape_hsm(self, star, shape_unnormalized=True, return_error=True, logger=None):
        """Measure the shape of a star using the HSM algorithm

        :param star:                Star we want to measure
        :param shape_unnormalized:  Bool. If True, return shape measurement
                                    (and error if return_error) in unnormalized
                                    basis. Note: HSM by default measures in the
                                    unnormalized basis, and then converts to
                                    normalized.  The piff hsm algorithm does
                                    that conversion for us, while the hsm_error
                                    does not.
        :param return_error:        Bool. If True, also measure the error
                                    [default: True]
        :param logger:              A logger object for logging debug info

        :returns:                   Shape (and error if return_error) in
                                    specified basis
        """
        if return_error:
            # do in unnormalized basis by default
            flux, u0, v0, e0, e1, e2, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, return_debug=False, logger=logger, return_error=return_error)
        else:
            flux, u0, v0, e0, e1, e2 = hsm_error(star, return_debug=False, logger=logger, return_error=False)
        # flux is underestimated empirically
        flux = flux / 0.92

        if shape_unnormalized:
            shape = np.array([flux, u0, v0, e0, e1, e2])
        else:
            # convert back
            sigma, g1, g2 = self.shape_convert_to_normalized(e0, e1, e2)
            shape = np.array([flux, u0, v0, sigma, g1, g2])
        if np.any(shape != shape):
            raise ModelFitError

        if logger: logger.debug('Measured Shape is {0}'.format(str(shape)))
        if return_error:
            if shape_unnormalized:
                error = np.array([sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2])
            else:
                sigma_scale, sigma_g1, sigma_g2 = self.shape_convert_errors_to_normalized(sigma_e0, sigma_e1, sigma_e2, e0, e1, e2)
                error = np.array([sigma_flux, sigma_u0, sigma_v0, sigma_scale, sigma_g1, sigma_g2])
            if logger: logger.debug('Measured Error is {0}'.format(str(error)))
            return shape, error
        else:
            return shape

    def measure_shape_lmfit(self, star, shape_unnormalized=False, return_error=True, logger=None):
        """Measure the shape of a star using lmfit.

        :param star:                Star we want to measure
        :param shape_unnormalized:  Bool. If True, return shape measurement
                                    (and error if return_error) in unnormalized
                                    basis. lmfit by default fits in normalized
                                    basis
        :param return_error:        Bool. If True, also return the error
                                    [default: True]
        :param logger:              A logger object for logging debug info

        :returns:                   Shape (and error if return_error) in
                                    specified basis
        """
        import lmfit
        # we also fit flux, du, dv but those are less important

        # put in initial guesses for flux, du, dv if they exist
        # all stars have fits with fluxes and centers defined, even if they may just be default values
        flux = star.fit.flux
        if flux == 1.0:
            # use as a starting guess the sum of the array
            flux = star.image.array.sum()
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
        results = lmfit.minimize(self._fit_model_residual, params,
                                 args=(star, np.array([0., 0., 0.]), True, logger,), method='leastsq', epsfcn=1e-8)
        if logger: logger.debug(lmfit.fit_report(results))
        flux, du, dv, size, g1, g2 = results.params.valuesdict().values()
        if shape_unnormalized:
            # convert fit to unnormalized basis, assuming no covariance between fitted parameters
            e0, e1, e2 = self.shape_convert_to_unnormalized(size, g1, g2)
            shape = np.array([flux, du, dv, e0, e1, e2])
        else:
            shape = np.array([flux, du, dv, size, g1, g2])

        if results.errorbars:
            error = np.sqrt(np.diag(results.covar))
            if shape_unnormalized:
                # convert fit to unnormalized basis, assuming no covariance between fitted parameters
                sigma_size, sigma_g1, sigma_g2 = error[3:6]
                sigma_e0, sigma_e1, sigma_e2 = self.shape_convert_errors_to_unnormalized(sigma_size, sigma_g1, sigma_g2, size, g1, g2)
                error[3] = sigma_e0
                error[4] = sigma_e1
                error[5] = sigma_e2
        else:
            if logger: logger.debug('Cannot estimate errors!')
            min_err = 0.01  # must change for unnormalized basis stuff
            error = np.array([min_err, min_err, min_err, min_err, min_err, min_err])

        if not results.success:
            if logger: logger.debug('Failure measuring star shape!')
            shape *= np.nan
            error *= np.nan
        if np.any(shape != shape):
            raise ModelFitError
        if return_error:
            return shape, error
        else:
            return shape

    def _fit_optics_make_Dfun(self, func, lmparams, eps):
        """Takes in a _fit_optics_residual_* function and returns the function that gives you gradients. We basically need to do this because the numerical differentiation done in lmfit can attempt finite differences with too small of values, resulting in inaccurate derivatives"""
        var = lmparams.valuesdict()
        keys = []
        for key in lmparams:
            param = lmparams[key]
            if param.vary:
                keys.append(param.name)

        # construct function based on keys that are NOT varying
        def Dfun(params, stars, shapes, shape_errors, return_indices, logger=None):
            return_indices_dfun = True
            # use centered stencil, with eps based on key and fitting parameters
            gradients = []

            # ONLY do this if we are using hsm and with moments. lmfit doesn't seem to cooperate well enough...
            if self.shape_method == 'hsm' and self.optfit_optimize == 'moments':
                # need the values of the zernke polynomials as a function of XY
                # position. Shape is (focal zernikes, stars)
                u = np.array([star.data['u'] for star in stars])
                v = np.array([star.data['v'] for star in stars])
                r = (u + 1j * v) / self.fov_radius
                rsqr = np.abs(r) ** 2
                # have to do some funky transposing to make the ordering work out
                zernikes = np.array([galsim.utilities.horner2d(rsqr, r, ca.T,
                    dtype=complex).real for ca in self._noll_coef_field.T])
            for i in range(len(keys)):
                key = keys[i]

                if self.shape_method == 'hsm' and self.optfit_optimize == 'moments':
                    # IF we are doing zernikes. then zUVi_zXYj is proportional to zUVi_zXY1
                    if 'zXY' in key:
                        key_split = key.split('_')
                        key_check = key_split[0] + '_zXY001'
                        # note we assume that the fitter always does zXY001 first. I think this is reasonable
                        if key_check in keys and key_check != key:
                            # find index
                            grad_old = gradients[keys.index(key_check)]
                            j = int(key.split('zXY')[-1])
                            zernike_i = zernikes[j - 1]
                            # PROBLEM: grad_old is, e.g. total number of pixels, or moments, times the number of stars
                            reps = len(grad_old) // len(zernike_i)
                            gradient = grad_old * np.tile(zernike_i, reps)
                            gradients.append(gradient)
                            continue

                # do positive and negative steps
                if logger:
                    logger.debug('Calculating gradient for {0}, centered at {1:+.4e} +- {2:.4e}'.format(key, params[key].value, eps))
                params[key].set(value=params[key].value + eps)
                chi_pos, indices_pos = func(params, stars, shapes, shape_errors, return_indices_dfun, logger)
                if logger:
                    logger.debug('chi positive chi2 / dof for {2} is {0:.4e} / {1}'.format(np.sum(np.square(chi_pos[indices_pos])), len(chi_pos[indices_pos]), key))
                params[key].set(value=params[key].value - 2 * eps)
                chi_neg, indices_neg = func(params, stars, shapes, shape_errors, return_indices_dfun, logger)
                if logger:
                    logger.debug('chi negative chi2 / dof for {2} is {0:.4e} / {1}'.format(np.sum(np.square(chi_neg[indices_neg])), len(chi_neg[indices_neg]), key))
                # and set it back
                params[key].set(value=params[key].value + eps)

                conds = (indices_pos * indices_neg).astype(bool)

                # measure gradient
                gradient = ((chi_pos - chi_neg) / (2 * eps))
                gradient = np.where(conds, gradient, 0)

                # dfun wants -dchi/dtheta
                gradient *= -1

                if logger:
                    logger.debug('average gradient for {0} across data is {1:+.4e} for {2} entries'.format(key, np.mean(gradient), len(gradient)))
                # record gradient
                gradients.append(gradient)
                if gradient.sum() == 0:
                    import ipdb; ipdb.set_trace()

            gradients = np.array(gradients)
            if logger: logger.debug('Gradient shape is {0}'.format(str(gradients.shape)))
            return gradients

        return Dfun

    def fit_optics(self, stars, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of
        operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        import lmfit
        if logger:
            logger.info("Start fitting Optical fit using {0} and {1}".format(self.shape_method, self.optfit_optimize))

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
            if len(self._opt_stars) + 1 > max_stars:
                if logger: logger.info("Finishing opt_star measurements at {0} stars".format(len(self._opt_stars)))
                break

            star = stars[indx]
            snr = self.measure_snr(star)
            if self.min_optfit_snr > 0:
                if snr < self.min_optfit_snr:
                    if logger: logger.info("Skipping star {0} because SNR {1} < {2}".format(indx, snr, self.min_optfit_snr))
                    continue

            # idea here is that even for pixels, if we can't fit a shape, the
            # star is probably borked and should be skipped
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
            if logger: logger.info("Using {0} stars instead of desired {1} out of {2}".format(len(self._opt_stars), max_stars, len(stars)))

        params = self._fit_optics_params(self.optatmo_psf_kwargs)

        # decide optimization function
        if self.optfit_optimize == 'pixel':
            resid = self._fit_optics_residual_pixel
            Dfun = self._fit_optics_make_Dfun(resid, params, eps=1e-4)
            col_deriv = 1
        elif self.optfit_optimize == 'moments':
            resid = self._fit_optics_residual_moments
            Dfun = self._fit_optics_make_Dfun(resid, params, eps=1e-3)
            col_deriv = 1
        elif self.optfit_optimize == 'analytic':
            resid = self._fit_optics_residual_analytic
            Dfun = None
            col_deriv = 0
        else:
            raise KeyError('Optical Fitter Algorithm {0} not allowed'.format(self.optfit_optimize))
        return_indices = False  # in the normal residual function if a star fails, get rid of it
        # do fit
        # TODO: temporary check for when things fail...
        results = lmfit.minimize(resid, params, args=(self._opt_stars, self._opt_shapes, self._opt_shape_errors, return_indices, logger,), method='leastsq', Dfun=Dfun, col_deriv=col_deriv)
        # try:
        #     results = lmfit.minimize(resid, params, args=(self._opt_stars, self._opt_shapes, self._opt_shape_errors, return_indices, logger,), method='leastsq', Dfun=Dfun)
        # except ValueError as e:
        #     print(e)
        #     # import ipdb; ipdb.set_trace()
        #     raise e

        key_i = 0
        for key in self.keys:
            if not self.optatmo_psf_kwargs['fix_' + key]:
                val = results.params.valuesdict()[key]
                self.optatmo_psf_kwargs[key] = val

                try:
                    err = np.sqrt(results.covar[key_i, key_i])
                    self.optatmo_psf_kwargs['error_' + key] = err
                except TypeError:
                    # covar is None for Reasons.
                    placeholder_error = 10000
                    if logger: logger.warning('No Error calculated for parameter {0}! Replacing with large number {1}!'.format(key, placeholder_error))
                    self.optatmo_psf_kwargs['error_' + key] = placeholder_error
                key_i += 1
        self._update_optatmopsf(logger=logger)

        # set final fit
        if logger:
            logger.info('Optical fit from lmfit parameters:')
            logger.info(lmfit.fit_report(results))

        # save this for debugging purposes
        self._opt_results = results

    def _fit_optics_params(self, optatmo_psf_kwargs):
        import lmfit
        # create lmparameters
        params = lmfit.Parameters()
        # step through keys
        for key in self.keys:
            value = optatmo_psf_kwargs[key]
            if 'fix_' + key in optatmo_psf_kwargs:
                vary = not optatmo_psf_kwargs['fix_' + key]
            else:
                vary = True
            if 'min_' + key in optatmo_psf_kwargs:
                min = optatmo_psf_kwargs['min_' + key]
            else:
                min = None
            if 'max_' + key in optatmo_psf_kwargs:
                max = optatmo_psf_kwargs['max_' + key]
            else:
                max = None
            params.add(key, value=value, vary=vary, min=min, max=max)
        return params

    @staticmethod
    def measure_snr(star):
        """Calculate the signal-to-noise of a given star. Calls util
        measure_snr function

        :param star:    Input star, with stamp, weight

        :returns:       the SNR value.
        """
        return measure_snr(star)

    def _correct_profile(self, profile, star, shape, logger=None):
        # draw uncorrected star
        star_model = self.drawProfileStar(star, profile, None)
        # model star's weight array should only be 0's and 1's
        weight = star_model.data.weight.array
        star_model.data.weight.array[:] = np.where(weight, 1., 0)
        shape_model = self.measure_shape(star_model, return_error=False, logger=logger)
        flux_model, u_model, v_model = shape_model[:3]
        flux_star, u_star, v_star = shape[:3]
        du = u_star - star.fit.center[0] - u_model
        dv = v_star - star.fit.center[1] - v_model
        if self.shape_method == 'hsm':
            # not sure why this seems to work better...
            flux_star = (star.image.array * star.image.array * star.weight.array).sum()
            flux_star_model = (star_model.image.array * star_model.image.array * star_model.weight.array).sum()
        flux = flux_star

        profile = profile.shift(du, dv) * flux
        if logger: logger.debug('Corrected star by .shift({0}, {1}) * {2}'.format(du, dv, flux))

        return profile

    def _fit_optics_residual_pixel(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(params, logger)

        # get optical params
        stars_params = self.getParamsList(stars)

        # calculate chi
        chi = np.array([])
        returned_indices = np.array([])
        for i, star in enumerate(stars):
            opt_params = stars_params[i]
            shape = shapes[i]

            try:
                # get profile; modify based on flux and shifts
                profile_uncorrected = self._profile(opt_params)

                # correct profile with hsm
                profile = self._correct_profile(profile_uncorrected, star, shape, logger=logger)

                # draw star
                image_model = self.drawProfile(star, profile)
                indices = np.ones_like(image_model)
            except (ModelFitError, RuntimeError) as e:
                if logger:
                    logger.warn(str(e))
                    logger.warn('Star {0}\'s model failed to be drawn.'.format(i))
                    logger.warn('Parameters are {0}'.format(str(opt_params)))
                    logger.warn('Input parameters are {0}'.format(str(params)))
                    logger.warn('Pretending we didn\'t measure this star.')
                if return_indices:
                    if logger: logger.warn('Filling with infinite chi')
                    image_model = star.data.getImage()[0] * np.inf
                    indices = np.zeros_like(image_model)
                else:
                    if logger: logger.warn('Pretending we didn\'t measure this star.')
                    continue

            # compute chi2
            image, weight, image_pos = star.data.getImage()
            chi_i = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
            chi = np.hstack((chi, chi_i))
            returned_indices = np.hstack((returned_indices, indices.flatten()))
        if logger: logger.debug('Current Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        if return_indices:
            return chi, returned_indices
        else:
            return chi

    def _fit_optics_residual_moments(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(params, logger)

        # get optical params
        stars_params = self.getParamsList(stars)

        # measure their shapes and calculate chi
        chi = np.array([])
        returned_indices = np.array([])
        # shapes_model = np.array([])
        for i, star in enumerate(stars):
            opt_params = stars_params[i]
            shape = shapes[i]
            error = shape_errors[i]

            try:
                # get profile; modify based on flux and shifts
                profile = self._profile(opt_params)

                # # correct profile with hsm
                # profile = self._correct_profile(profile, star, shape, logger=logger)

                # measure final shape
                star_model = self.drawProfileStar(star, profile, opt_params)
                # model star's weight array should only be 0's and 1's
                weight = star_model.data.weight.array
                star_model.data.weight.array[:] = np.where(weight, 1., 0)
                shape_model = self.measure_shape(star_model, return_error=False)#, logger=logger)
                if np.any(shape_model != shape_model):
                    if logger: logger.warn('Star {0} returned nan shape'.format(i))
                    indices = np.zeros_like(shape_model, dtype=bool)
                else:
                    indices = np.ones_like(shape_model, dtype=bool)
            except (ModelFitError, RuntimeError) as e:
                if logger:
                    logger.warn(str(e))
                    logger.warn('Star {0}\'s model failed to be drawn.'.format(i))
                    logger.warn('Parameters are {0}'.format(str(opt_params)))
                    logger.warn('Input parameters are {0}'.format(str(params)))
                import ipdb; ipdb.set_trace()
                if return_indices:
                    if logger: logger.warn('Filling with infinite chi')
                    shape_model = shape * np.inf
                    indices = np.zeros_like(shape_model, dtype=bool)
                else:
                    if logger: logger.warn('Pretending we didn\'t measure this star.')
                    continue

            # don't care about flux, du, dv here
            chi_i = self.shape_weights * ((shape_model - shape) / error)[3:]
            # if logger:
            #     logger.debug('Current params for star_i {0} is {1}'.format(i, opt_params))
            #     logger.debug('Current Shape for star_i {0} is {1}'.format(i, shape))
            #     logger.debug('Current Model Shape for star_i {0} is {1}'.format(i, shape_model))
            #     logger.debug('Current Chi for star_i {0} is {1}'.format(i, chi_i))
            chi = np.hstack((chi, chi_i))
            returned_indices = np.hstack((returned_indices, indices[3:]))
            # shapes_model = np.hstack((shapes_model, shape_model[3:]))
        if logger: logger.debug('Current Total Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        if return_indices:
            return chi, returned_indices
        else:
            return chi

    # TODO: OK wtf SOMETIMES the VERY LAST ROW fails.
    def _fit_optics_residual_analytic(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        # convert lmparams instance
        lmparam = lmparams.valuesdict()

        # update psf
        self._update_optatmopsf(lmparam, logger)

        # get star params
        params = self.getParamsList(stars)

        # generate analytic star moments
        shapes_model = self.analytic_shapes(params, self.analytic_coefs)

        # calculate chi
        shapes = shapes[:, 3:]
        errors = shape_errors[:, 3:]
        chi = (self.shape_weights[None] * (shapes_model - shapes) / errors).flatten()
        # print(chi[::3])
        # print(chi[1::3])
        # print(chi[2::3])
        # print('e0')
        # print(shapes[:, 0])
        # print(shapes_model[:, 0])
        # print('e1')
        # print(shapes[:, 1])
        # print(shapes_model[:, 1])
        # print('e2')
        # print(shapes[:, 2])
        # print(shapes_model[:, 2])
        if logger: logger.debug('Current Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        returned_indices = np.ones_like(chi)
        if return_indices:
            return chi, returned_indices
        else:
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
        model_fitted_stars = []
        for star_i, star, opt_params in zip(range(len(stars)), stars, opt_params_stars):
            try:
                model_fitted_star = self._fit_model(star, opt_params=opt_params, vary_shape=True, logger=logger)
                model_fitted_stars.append(model_fitted_star)
            except Exception as e:
                logger.warn('{0}'.format(str(e)))
                logger.warn('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))

        # fit interpolant
        logger.info("Initializing atmo interpolator")
        initialized_stars = self.atmo_interp.initialize(model_fitted_stars, logger=logger)

        logger.info("Fitting atmo interpolant")
        self.atmo_interp.solve(initialized_stars, logger=logger)

        self._model_fitted_stars = model_fitted_stars

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
            min_size = 0.43
            max_g = 0.4
            opt_size = opt_params[0]
            opt_g1 = opt_params[1]
            opt_g2 = opt_params[2]
            params.add('size', value=max([min_size - opt_size, 0]), vary=True, min=min_size - opt_size)
            params.add('g1', value=0, vary=True, min=-max_g - opt_g1, max=max_g - opt_g1)
            params.add('g2', value=0, vary=True, min=-max_g - opt_g2, max=max_g - opt_g2)

        # do fit
        results = lmfit.minimize(self._fit_model_residual, params,
                                 args=(star, opt_params, vary_shape, logger,),
                                 method='leastsq', epsfcn=1e-8,
                                 maxfev=200)
        if logger: logger.debug(lmfit.fit_report(results))
        if vary_shape:
            flux, du, dv, size, g1, g2 = results.params.valuesdict().values()
            params = np.array([ size, g1, g2 ])
            if results.errorbars:
                params_var = np.diag(results.covar)
            else:
                params_var = np.zeros(len(params))
        else:
            flux, du, dv = results.params.valuesdict().values()
            params = None
            params_var = None
        center = (du, dv)
        fit = StarFit(params, params_var=params_var, flux=flux, center=center)
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

        prof = self._profile(params).shift(du, dv) * flux

        # calculate chi2
        image, weight, image_pos = star.data.getImage()
        image_model = self.drawProfile(star, prof)
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

    @staticmethod
    def analytic_shapes(params, analytic_coefs):
        # TODO: docs
        coefs = analytic_coefs[0]
        indices = analytic_coefs[1]
        after_burners = analytic_coefs[2]

        # transform into full index
        params_onehot = np.vstack((np.ones(len(params)).T, np.ones(len(params)).T, params.T)).T.astype(np.float64) # extra set of ones which we turn into 1/size
        params_onehot[:, 1] = 1. / params_onehot[:, 2]
        # apply model
        shapes = np.array([cypoly(params_onehot, coef.astype(np.float64), index.astype(np.int64)) * afb[1] + afb[0]
                           for coef, index, afb in zip(coefs, indices, after_burners)]).T
        if np.any(shapes != shapes) or np.any(~np.isfinite(shapes)):
            # TODO: figure out why this happens. delete
            print(params)
            print(params_onehot)
            print(shapes)
            # import ipdb; ipdb.set_trace()
            raise ValueError('Bad shape values')

        return shapes
