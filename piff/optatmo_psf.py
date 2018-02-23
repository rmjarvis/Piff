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
import numba

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .outliers import Outliers
from .model import ModelFitError
# from .gsobject_model import GSObjectModel, Kolmogorov, Gaussian
from .star import Star, StarFit, StarData
from .util import hsm_error, measure_snr, write_kwargs, read_kwargs
from .config import LoggerWrapper

class OptAtmoPSF(PSF):

    """Combine Optical and Atmospheric PSFs together
    """

    def __init__(self, atmo_interp=None, outliers=None, analytic_coefs=None, do_fit_size=True, do_fit_optics=True, optatmo_psf_kwargs={}, optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None, n_optfit_stars=0, fov_radius=4500., jmax_pupil=11, jmax_focal=10, min_optfit_snr=0, logger=None):
        """
        Fit Combined Atmosphere and Optical PSF in two stage process.

        :param atmo_interp:             Piff Interpolant object that represents
                                        the atmospheric interpolation
        :param outliers:                Optionally, an Outliers instance used
                                        to remove outliers during atmosphere
                                        fit.  [default: None]
        :param analytic_coefs:          Terms in analytic breakdown of zernike
                                        to shape transformation. Only used in
                                        It is formatted as [coefs, indices],
                                        with each of those being 3 deep (one
                                        for each of the three second moment
                                        shapes)
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
        :param fov_radius:              [Default: 1.] Radius of telescope in
                                        u,v coordinates
        :param jmax_pupil:              Number of pupil-basis zernikes in
                                        Optical model. Inclusive and in Noll
                                        convention. [default: 11]
        :param jmax_focal:              Number of focal-basis zernikes in
                                        Optical model. Inclusive and in Noll
                                        convention. [default: 11]
        :param min_optfit_snr:          minimum snr from star property required
                                        for optical portion of fit. If 0,
                                        ignored. [default: 0]
        :param logger:                  A logger object for logging debug info.
                                        [default: None]

        Notes
        -----
        Our model of the PSF is the convolution of an elliptical Kolmogorov
        with an optics model:

            PSF = convolve(Kolmogorov(size, g1, g2), Optics(defocus, etc))

        Call [size, g1, g2, defocus, astigmatism-y, astigmatism-x, ...] a_k,
        with k starting at 1 so that the Zernike terms like defocus can keep
        the noll convention. Thus, we call the size a_1, g1 (confusingly) a_2,
        and so on. The goal of this PSF model is to return a_k given focal
        plane coordinates u, v. So, for the i-th star:

        a_{ik} (u_i, v_i) = \sum^{jmax_focal}_{\ell=1} b_{k \ell} Z_{\ell} (u_i, v_i)
                            + a^{reference}_{k} (u_i, v_i) [if k >= 4]
                            + atmo_interp(u_i, v_i) [if k < 4]

        We note that b_{k \ell} = 0 if k in [1, 2, 3] and \ell > 1, which is to
        say that we fit a constant atmosphere and let the atmo_interp deal with
        differences from constant. b_{k \ell} is called a Double Zernike
        Decomposition. The fitting process can be broken down into two major
        steps:

        1. Fit b_{k \ell} by looking at the field pattern of the shapes e_{ij}
            -   First, we use an analytic relation for e_{ij}:

                    e_{ij} = f(a_{ik}; analytic_coefs)

                This relation is very fast. We pass in b_{k \ell} to a least
                squares minimization and generally fit these terms on the order
                of a few minutes.

                These analytic_coefs are specific to the instrument and should
                be recalculated for different telescopes.  I fitted the
                analytic coefs to up to fourth in combinations of up to three
                terms, e.g. z_i z_j z_k z_\ell with \ell = at least one of i,
                j, k.

            -   The analytic relation is not perfect, and will overestimate the
                size. I believe this is comes from noise in the pixels and from
                the effects of masking, neither of which are taken into account
                in the analytic relation. It is a simple fix, however: simply
                take a few stars, grid search b_{1 1} (ie constant size), and
                adjust accordingly.

            -   Finally we fit the full optical model to the shapes by actually
                drawing and measuring the shapes of the stars, such that:

                e_{ij} = HSM(drawStar(a_{ik})) and a_{ik} come from above

                This step is slow, however, the analytic relations usually get
                us very close to the right answer, so you don't usually have to
                run this step to get good results.

        2. Fit atmo_interp.
            -   a_{ik} = a^{optics}_{ik} + a^{atmosphere}_{ik} for k < 4, where
                a^{optics}_{ik} = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i).
                We directly find a^{atmosphere}_{ik} for each star by
                minimizing the chi2 of the pixels of the observed star and the
                model as drawn here.

            -   After finding a^{atmosphere}_{ik}, we fit the atmo_interp to
                interpolate those parameters as a function of focal plane
                position (u_i, v_i).

        """
        logger = LoggerWrapper(logger)

        # If pupil_angle and strut angle are provided as strings, eval them.
        # TODO: these two bits are not tested!
        try:
            for key in ['pupil_angle', 'strut_angle']:
                if key in optical_psf_kwargs and isinstance(optical_psf_kwargs[key],str):
                    optical_psf_kwargs[key] = eval(optical_psf_kwargs[key])
        except TypeError:
            # we can end up saving optical_psf_kwargs as 0, so fix that
            optical_psf_kwargs = {}
            logger.warning('Warning! Invalid optical psf kwargs. Putting in empty dictionary')
        # we can end up saving optatmo_psf_kwargs as 0, so for now we pass it
        # as empty. This will be overwritten later in _finish_read
        if optatmo_psf_kwargs == 0:
            optatmo_psf_kwargs = {}
        # same with kolmogorov kwargs
        if kolmogorov_kwargs == 0:
            kolmogorov_kwargs = {}

        self.do_fit_size = do_fit_size
        self.do_fit_optics = do_fit_optics

        self.outliers = outliers
        # atmo_interp is a parsed class
        self.atmo_interp = atmo_interp
        self.analytic_coefs = analytic_coefs
        self.optical_psf_kwargs = optical_psf_kwargs
        self.kolmogorov_kwargs = kolmogorov_kwargs
        self.reference_wavefront = reference_wavefront

        self.min_optfit_snr = min_optfit_snr
        self.n_optfit_stars = n_optfit_stars

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
                'size': 1.0,  'fix_size': False, 'min_size': 0.45, 'max_size': 2.0,
                'g1':   0,    'fix_g1':   False, 'min_g1': -0.4, 'max_g1': 0.4,
                'g2':   0,    'fix_g2':   False, 'min_g2': -0.4, 'max_g2': 0.4,
            }
        self.keys = [ 'size', 'g1', 'g2', ]
        # throw in default zernike parameters
        # only fit zernikes starting at 4 / defocus
        for zi in range(4, self.jmax_pupil + 1):
            for dxy in range(1, self.jmax_focal + 1):
                zkey = 'zPupil{0:03d}_zFocal{1:03d}'.format(zi, dxy)
                self.keys.append(zkey)

                # default to unfixing all possible combinations
                self.optatmo_psf_kwargs['fix_' + zkey] = False
                # can optionally fix an entire Pupil or Focal aberrations if we want
                # TODO: not tested
                fix_keyPupil = 'fix_zPupil{0:03d}'.format(zi)
                if fix_keyPupil in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyPupil]
                # TODO: not tested
                fix_keyFocal = 'fix_zFocal{0:03d}'.format(dxy)
                if fix_keyFocal in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyFocal]

                zmax = 1.
                self.optatmo_psf_kwargs['min_' + zkey] = -zmax
                self.optatmo_psf_kwargs['max_' + zkey] = zmax

                # initial value. If there is no reference wavefront it helps
                # the fitter to pass along nonzero values to non-fixed
                # parameters
                if self.reference_wavefront or self.optatmo_psf_kwargs['fix_' + zkey]:
                    self.optatmo_psf_kwargs[zkey] = 0
                else:
                    self.optatmo_psf_kwargs[zkey] = np.random.random() * (0.1 - -0.1) + -0.1
        try:
            self.optatmo_psf_kwargs.update(optatmo_psf_kwargs)
        # TODO: not tested
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

        # TODO: make these optional changes
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
        # TODO: not tested
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 0.5
        # TODO: not tested
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 0.5


        # set up gsobject model to measure shapes
        logger.info("Loading Kolmogorov GSObject for modeling shapes")
        # if we wanted, this could become an argument for some other model, ala
        # gsobject. For now, we fix to Kolmogorov
        try:
            self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
        # TODO: not tested
        except TypeError:
            logger.warning('Warning! Invalid kolmogorov kwargs. Putting in dictionary with fwhm=1')
            self.kolmogorov_kwargs = {'fwhm': 1}
            self.atmo_model = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)

        # max size of shapes allowed in fit_analytic, fit_size, fit_optics
        self._max_shapes = np.array([1.5, 0.15, 0.15])
        # weighting of shapes in fit_analytic, fit_size, fit_optics
        self._shape_weights = np.array([0.2, 0.4, 0.4])

        # kwargs
        self.kwargs = {'fov_radius': self.fov_radius,
                       'jmax_pupil': self.jmax_pupil,
                       'jmax_focal': self.jmax_focal,
                       'min_optfit_snr': self.min_optfit_snr,
                       'n_optfit_stars': self.n_optfit_stars,
                       'do_fit_size': self.do_fit_size,
                       'do_fit_optics': self.do_fit_optics,
                       # junk entries to be overwritten in _finish_read function
                       'analytic_coefs': 0,
                       'optatmo_psf_kwargs': 0,
                       'atmo_interp': 0,
                       'reference_wavefront': 0,
                       'optical_psf_kwargs': 0,
                       'kolmogorov_kwargs': 0,
                       'outliers': 0,
                       }

        # cache parameters to cut down on lookup
        self._cache = False
        self._aberrations_reference_wavefront = None

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
        logger = LoggerWrapper(logger)
        config_psf = config_psf.copy()  # Don't alter the original dict.

        kwargs = config_psf.copy()

        # do processing as appropriate
        # set up optical and atmosphere psf kwargs using the optical model
        # TODO: not tested
        if 'optical_psf_kwargs' in config_psf:
            optical_psf_kwargs = config_psf['optical_psf_kwargs']
        else:
            optical_psf_kwargs = {}

        optical = Optical(logger=logger, **optical_psf_kwargs)
        kwargs['optical_psf_kwargs'] = optical.optical_psf_kwargs
        kolmogorov_kwargs = optical.kolmogorov_kwargs
        # TODO: not tested
        if 'kolmogorov_kwargs' in config_psf:
            kolmogorov_kwargs.update(config_psf['kolmogorov_kwargs'])
        # TODO: not tested
        if kolmogorov_kwargs.keys() == ['lam']:
            kolmogorov_kwargs = {'half_light_radius': 1.0}
        # Also, let r0=0 or None indicate that there is no kolmogorov component
        # TODO: not tested
        if 'r0' in kolmogorov_kwargs and not kolmogorov_kwargs['r0']:
            kolmogorov_kwargs = {'half_light_radius': 1.0}
        kwargs['kolmogorov_kwargs'] = kolmogorov_kwargs

        # optical field
        # TODO: not tested
        if 'optatmo_psf_kwargs' in config_psf:
            kwargs['optatmo_psf_kwargs'] = config_psf['optatmo_psf_kwargs']

        # atmo interp
        # TODO: not tested
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
            # TODO: not tested
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
        # TODO: not tested
        if 'analytic_coefs' in config_psf:
            analytic_coefs = config_psf['analytic_coefs']
            if analytic_coefs in [None, 'none', 'None']:
                analytic_coefs = None
            else:
                if isinstance(analytic_coefs, str):
                    analytic_coefs = np.load(analytic_coefs).item()
                # make sure the analytic_coefs are in a reasonable format
                indices = []
                coefs = []
                # purge coefs and indices with j higher than jmax
                if 'jmax_pupil' in kwargs:
                    jmax_pupil = kwargs['jmax_pupil']
                else:
                    # TODO: not tested
                    jmax_pupil = 11
                for index, coef in zip(analytic_coefs['indices'], analytic_coefs['coefs']):
                    index = np.array(index).astype(np.int64)
                    coef = np.array(coef).astype(np.float64)
                    # purge based on jmax_pupil
                    conds_full = index <= jmax_pupil  # +1 because of one-hot encoding
                    conds = np.all(conds_full, axis=1)
                    if conds.sum() != conds.size:
                        logger.warning('Analytic Coefs allow indices up to {0}, but jmax_pupil is only {1}. Cutting {2} out of {3} entries'.format(np.max(index), jmax_pupil, conds.size - conds.sum(), conds.size))
                        index = index[conds]
                        coef = coef[conds]

                    indices.append(index)
                    coefs.append(coef)
                analytic_coefs = [coefs, indices]
        else:
            # TODO: not tested
            analytic_coefs = None
        kwargs['analytic_coefs'] = analytic_coefs

        if 'outliers' in kwargs:
            outliers = Outliers.process(kwargs.pop('outliers'), logger=logger)
            kwargs['outliers'] = outliers

        return kwargs

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        logger = LoggerWrapper(logger)

        # write the atmo interp if it exists
        if self.atmo_interp:
            self.atmo_interp.write(fits, extname + '_atmo_interp')
        if self.outliers:
            self.outliers.write(fits, extname + '_outliers')
            logger.debug("Wrote the PSF outliers to extension %s",extname + '_outliers')

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
        dtypes = []
        for key in self.optatmo_psf_kwargs:
            if 'fix_' in key:
                dtypes.append((key, bool))
            else:
                dtypes.append((key, float))
        data = np.zeros(1, dtype=dtypes)
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
        logger = LoggerWrapper(logger)
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
            possible_shapes = np.sort(np.unique(shape))
            analytic_coefs = [[], []]
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
            self.optatmo_psf_kwargs[key] = data[key][0]
        logger.info('Reloading optatmopsf state')
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger)
        if extname + '_outliers' in fits:
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            self.outliers = None

    def fit(self, stars, wcs, pointing,
            chisq_threshold=0.1, max_iterations=30, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the
                                telescope pointing.
                                [Note: pointing should be None if the WCS is
                                not a CelestialWCS]
        :param chisq_threshold: Change in reduced chisq at which iteration will
                                terminate during atmosphere fit.  [default: 0.1]
        :param max_iterations:  Maximum number of iterations to try during
                                atmosphere fit. [default: 30]
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)

        # do first pass of flux and centers for the stars
        self.stars = []
        self.star_shapes = []
        self.star_errors = []
        self.star_snrs = []
        for star_i, star in enumerate(stars):
            logger.debug('Measuring shape of star {0}'.format(star_i))
            try:
                shape, error = self.measure_shape(star, return_error=True, logger=logger)
                star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))
                star.data.properties['shape'] = shape
                star.data.properties['shape_error'] = error
                snr = self.measure_snr(star)

                self.stars.append(star)
                self.star_shapes.append(shape)
                self.star_errors.append(error)
                self.star_snrs.append(snr)
            except (ModelFitError, RuntimeError) as e:
                # something went wrong with this star
                logger.warning(str(e))
                logger.warning('Star {0} failed shape estimation. Skipping'.format(star_i))
        self.star_shapes = np.array(self.star_shapes)
        self.star_errors = np.array(self.star_errors)
        self.star_snrs = np.array(self.star_snrs)

        # determine stars used in optical fit
        self.fit_optics_indices = np.arange(len(self.stars))
        # cut self.fit_optics_indices based on snr and max_shapes
        conds_snr = (self.star_snrs >= self.min_optfit_snr)
        conds_shape = (np.all(np.abs(self.star_shapes[:, 3:]) <= self._max_shapes, axis=1))
        self.fit_optics_indices = self.fit_optics_indices[conds_snr * conds_shape]
        logger.info('Cutting {0} stars for fitting the optics based on SNR > {1} ({2} stars) and on maximum shapes ({3} stars)'.format(len(stars) - len(self.fit_optics_indices), self.min_optfit_snr, len(self.stars) - np.sum(conds_snr), len(stars) - np.sum(conds_snr)))
        if self.n_optfit_stars and self.n_optfit_stars < len(self.fit_optics_indices):
            logger.info('Cutting from {0} to {1} stars'.format(len(self.fit_optics_indices), self.n_optfit_stars))
            max_stars = self.n_optfit_stars
            np.random.shuffle(self.fit_optics_indices)
            self.fit_optics_indices = self.fit_optics_indices[:max_stars]
        else:
            max_stars = len(self.stars)
            if len(self.fit_optics_indices) < max_stars:
                logger.info("Using {0} stars instead of desired {1}".format(max_stars, self.n_optfit_stars))

        self.fit_optics_stars = [self.stars[indx] for indx in self.fit_optics_indices]
        self.fit_optics_star_shapes = self.star_shapes[self.fit_optics_indices]
        self.fit_optics_star_errors = self.star_errors[self.fit_optics_indices]

        self.wcs = wcs
        self.pointing = pointing

        if self.analytic_coefs in ['skip', 'Skip', None, 'none', 'None', 0]:
            pass
        else:
            conds = conds_snr * conds_shape
            logger.info('Fitting analytic for {0} stars'.format(conds.sum()))
            self.fit_analytic_stars = [s for s, star_snr in zip(self.stars, self.star_snrs) if star_snr >= self.min_optfit_snr]
            self.fit_analytic_stars = [s for s, c in zip(self.stars, conds) if c]
            self.fit_analytic_star_shapes = self.star_shapes[conds]
            self.fit_analytic_star_errors = self.star_errors[conds]
            self.fit_analytic(self.fit_analytic_stars, self.fit_analytic_star_shapes, self.fit_analytic_star_errors, logger=logger, **kwargs)

        if self.do_fit_size:
            # first just fit the size. This gets us a lot of the way there
            logger.info('Fitting size')
            n_fit_size = 200
            self.fit_size_indices = np.arange(len(self.fit_optics_stars))
            if n_fit_size < len(self.fit_optics_stars):

                logger.debug('Cutting from {0} to {1} stars for fit_size because we do not need that many!'.format(len(self.fit_optics_stars), n_fit_size))
                np.random.shuffle(self.fit_size_indices)
                self.fit_size_indices = self.fit_size_indices[:n_fit_size]
            self.fit_size_stars = [self.fit_optics_stars[indx] for indx in self.fit_size_indices]
            self.fit_size_star_shapes = self.fit_optics_star_shapes[self.fit_size_indices]
            self.fit_size_star_errors = self.fit_optics_star_errors[self.fit_size_indices]
            self.fit_size(self.fit_size_stars, self.fit_size_star_shapes, self.fit_size_star_errors, logger=logger, **kwargs)

        if self.do_fit_optics:
            # fit optical interpolant piece with constant atmosphere based on
            # moments. Can be skipped.
            logger.info('Fitting optics')
            self.fit_optics(self.fit_optics_stars, self.fit_optics_star_shapes, self.fit_optics_star_errors, logger=logger, **kwargs)

        # fit atmosphere. Can also be skipped
        if self.atmo_interp in ['skip', 'Skip', None, 'none', 'None', 0]:
            pass
        else:
            logger.info('Fitting atmosphere')
            stars_fit_atmosphere = self.fit_atmosphere(self.stars, chisq_threshold=chisq_threshold, max_iterations=max_iterations, logger=logger, **kwargs)
            self.stars = stars_fit_atmosphere

            # enable atmosphere interpolation now that we have solved the interp
            logger.info('Enabling Atmosphere')
            self._enable_atmosphere = True

    def _getParamsList_aberrations_field(self, stars):
        """Get params for a list of stars from the aberrations

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           Params  [size, g1, g2, z4, z5...] for each star

        Notes
        -----
        We have a set of coefficients b_{k \ell} that describe the Zernike
        decomposition. Then, for the i-th star at position (u_i, v_i), we get
        param a_{ik} as:

            a_{ik} = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i)
        """
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

    def getParamsList(self, stars, logger=None):
        """Get params for a list of stars.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           Params  [size, g1, g2, z4, z5...] for each star

        Notes
        -----
        For the i-th star, we have param a_{ik} = a_{ik}^{optics} +
        a_{ik}^{reference} + a_{ik}^{atmo_interp}.  We note that for k < 4,
        a_{ik}^{reference} = 0, k >= 4 a_{ik}^{atmo_interp} = 0. If no
        reference is provided to the PSF, then that piece is zero, and
        similarly with the atmo_interp.  When we initially produce the PSF, the
        atmo_interp is not fitted, and so calls to this function will skip the
        atmosphere. _enable_atmosphere is set to True when atmo_interp is
        finally fitted.
        """
        logger = LoggerWrapper(logger)
        params = np.zeros((len(stars), self.jmax_pupil), dtype=np.float64)

        logger.debug('Getting aberrations from misalignment')
        aberrations_pupil = self._getParamsList_aberrations_field(stars)
        params += aberrations_pupil

        if self.reference_wavefront:
            if self._cache:
                logger.debug('Getting cached reference wavefront aberrations')
                # use precomputed cache. WARNING! assumes stars are the same as in cache!
                aberrations_reference_wavefront = self._aberrations_reference_wavefront
            else:
                logger.debug('Getting reference wavefront aberrations')
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
            if self.atmo_interp is None:
                logger.warning('Attempting to retrieve atmospheric interpolations, but we have no atmospheric interpolant! Ignoring')
            else:
                logger.debug('Getting atmospheric aberrations')
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

    def getProfile(self, params, logger=None):
        """Get galsim profile for a given params
        :param params:      [size, g1, g2, z4, z5...]
        :returns:           Galsim profile
        """
        logger = LoggerWrapper(logger)

        # optics
        aberrations = np.zeros(4 + len(params[3:]))
        aberrations[4:] = params[3:]
        opt = galsim.OpticalPSF(aberrations=aberrations, gsparams=self.gsparams, **self.optical_psf_kwargs)

        # atmosphere
        size = params[0]
        g1 = params[1]
        g2 = params[2]
        atmo = self.atmo_model.dilate(size).shear(g1=g1, g2=g2)

        # convolve together
        prof = galsim.Convolve([opt, atmo], gsparams=self.gsparams)

        return prof

    def drawProfile(self, star, prof, params, use_fit=True):
        """Generate PSF image for a given star and profile

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :param profile:     A galsim profile
        :param params:      Params associated with profile to put in the star.
        :param use_fit:     Bool [default: True] shift the profile by a star's
                            fitted center and multiply by its fitted flux

        :returns:           Star instance with its image filled with rendered
                            PSF
        """
        # use flux and center properties
        if use_fit:
            prof = prof.shift(star.fit.center) * star.fit.flux
        image, weight, image_pos = star.data.getImage()
        image_model = image.copy()
        image_model = prof.drawImage(image_model, method='auto', offset=(star.image_pos-image_model.true_center))
        properties = star.data.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key, None)
        data = StarData(image=image_model,
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
        prof = self.getProfile(params)
        star = self.drawProfile(star, prof, params)
        return star

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
        stars_drawn = [self.drawProfile(star, self.getProfile(param), param) for param, star in zip(params, stars)]
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
        logger = LoggerWrapper(logger)
        if len(optatmo_psf_kwargs) == 0:
            # TODO: not tested
            optatmo_psf_kwargs = self.optatmo_psf_kwargs
            keys = self.keys
        else:
            keys = optatmo_psf_kwargs.keys()

        aberrations_changed = False
        for key in keys:
            # some checks
            if 'error_' in key:
                # TODO: not tested
                continue
            elif 'fix_' in key:
                continue
            elif 'min_' in key:
                continue
            elif 'max_' in key:
                continue

            # size, g1, g2 mean constant terms
            if key == 'size':
                pupil_index = 1
                focal_index = 1
            elif key == 'g1':
                pupil_index = 2
                focal_index = 1
            elif key == 'g2':
                pupil_index = 3
                focal_index = 1
            else:
                # zPupil012_zFocal034; kludgey as hell
                pupil_index = int(key.split('zPupil')[-1].split('_')[0])
                focal_index = int(key.split('zFocal')[-1])
                if pupil_index < 4:
                    raise ValueError('Not allowed to fit pupil zernike {0} less than {2}, key {1}!'.format(pupil_index, key, 4))
                elif focal_index < 1:
                    raise ValueError('Not allowed to fit focal zernike {0} less than {2} !, key {1}!'.format(focal_index, key, 1))
                elif pupil_index > self.jmax_pupil:
                    raise ValueError('Not allowed to fit pupil zernike {0}, greater than {2}, key {1}!'.format(pupil_index, key, self.jmax_pupil))
                elif focal_index > self.jmax_focal:
                    raise ValueError('Not allowed to fit focal zernike {0} greater than {2} !, key {1}!'.format(focal_index, key, self.jmax_focal))

            old_value = self.aberrations_field[pupil_index - 1, focal_index - 1]
            new_value = optatmo_psf_kwargs[key]

            if old_value != new_value:
                if 'fix_' + key in optatmo_psf_kwargs:
                    if optatmo_psf_kwargs['fix_' + key]:
                        logger.warning('Warning! Changing key {0} which is designated as fixed from {1} to {2}!'.format(key, old_value, new_value))
                logger.debug('Updating Zernike parameter {0} from {1:+.4e} + {3:+.4e} = {2:+.4e}'.format(key, old_value, new_value, new_value - old_value))
                self.aberrations_field[pupil_index - 1, focal_index - 1] = new_value
                aberrations_changed = True

        if aberrations_changed:
            logger.debug('---------- Recomputing field zernike coefficients')
            # One coef_array for each wavefront aberration
            # shape (jmax_pupil, maxn_focal, maxm_focal)
            self._coef_arrays_field = np.array([np.dot(self._noll_coef_field, a)
                                                for a in self.aberrations_field])

    def measure_shape(self, star, return_error=True, logger=None):
        """Measure the shape of a star using the HSM algorithm

        :param star:                Star we want to measure
        :param return_error:        Bool. If True, also measure the error
                                    [default: True]
        :param logger:              A logger object for logging debug info

        :returns:                   Shape (and error if return_error) in
                                    specified basis
        """
        logger = LoggerWrapper(logger)
        if return_error:
            # do in unnormalized basis by default
            flux, u0, v0, e0, e1, e2, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, return_debug=False, logger=logger, return_error=return_error)
            error = np.array([sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2])
        else:
            flux, u0, v0, e0, e1, e2 = hsm_error(star, return_debug=False, logger=logger, return_error=False)

        # flux is underestimated empirically
        flux = flux / 0.92

        shape = np.array([flux, u0, v0, e0, e1, e2])

        if np.any(shape != shape):
            # TODO: not tested. Add test for terrible image ethat should fail
            raise ModelFitError

        logger.debug('Measured Shape is {0}'.format(str(shape)))
        if return_error:
            logger.debug('Measured Error is {0}'.format(str(error)))
            return shape, error
        else:
            return shape

    @staticmethod
    def measure_snr(star):
        """Calculate the signal-to-noise of a given star. Calls util
        measure_snr function

        :param star:    Input star, with stamp, weight

        :returns:       the SNR value.
        """
        return measure_snr(star)

    def fit_analytic(self, stars, shapes, errors, logger=None, **kwargs):
        """Fit interpolated PSF model to star shapes.

        :param shapes:      A list of Star shapes
        :param errors:      A list of Star shape errors
        :param logger:      A logger object for logging debug info.
                            [default: None]

        Notes
        -----
        This model leverages the fact that the j-th measured HSM shape of the
        i-th star, e_{ij}, is pretty well approximated by a polynomial in the
        input params [size, g1, g2, defocus, ...], which we call a_{ik}. So,
        I have an analytic function I defined in advance that takes f(a_{ik})
        and returns e_{ij}. The optical model is specified at given focal
        plane coordinates [u, v] by a sum over Zernike polynomials:

        a_{ik} (u_i, v_i) = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i) 
                            + a^{reference}_{k}(u_i, v_i)

        Having measured the shapes of stars, e_{ij} with errors \sigma_{ij}, we
        then find the optimal b_{k \ell}
        """
        logger = LoggerWrapper(logger)
        import lmfit
        logger.info("Start fitting analytic Optical")

        # save reference wavefront values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_cache(stars, logger=logger)

        lmparams = self._fit_optics_lmparams(self.optatmo_psf_kwargs, self.keys)
        return_indices = False  # in the normal residual function if a star fails, get rid of it
        results = lmfit.minimize(self._fit_analytic_residual, lmparams, args=(stars, shapes, errors, return_indices, logger,))
        key_i = 0
        for key in self.keys:
            if not self.optatmo_psf_kwargs['fix_' + key]:
                val = results.params.valuesdict()[key]
                self.optatmo_psf_kwargs[key] = val

                try:
                    err = np.sqrt(results.covar[key_i, key_i])
                    self.optatmo_psf_kwargs['error_' + key] = err
                except (TypeError, AttributeError):
                    # covar is None for Reasons.
                    placeholder_error = 10000
                    logger.warning('No Error calculated for parameter {0}! Replacing with large number {1}!'.format(key, placeholder_error))
                    self.optatmo_psf_kwargs['error_' + key] = placeholder_error
                key_i += 1
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)

        # set final fit
        logger.info('Analytic Optical fit from lmfit parameters:')
        logger.info(lmfit.fit_report(results))
        try:
            logger.info('Success: {0}'.format(str(results.success)))
        except AttributeError:
            pass
        try:
            logger.info('Status: {0}'.format(str(results.status)))
        except AttributeError:
            pass
        try:
            logger.info('Message: {0}'.format(str(results.message)))
        except AttributeError:
            pass
        try:
            logger.info('lmdif_message: {0}'.format(str(results.lmdif_message)))
        except AttributeError:
            pass

        # save this for debugging purposes
        self._fit_analytic_results = results

        if self.reference_wavefront: self._delete_cache(logger=logger)

    # TODO: update docs
    def fit_size(self, stars, shapes, shape_errors, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of
        operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)
        import lmfit
        logger.info("Start fitting Optical fit of size alone")

        # save reference wavefront values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_cache(stars, logger=logger)

        # make kwargs with only size
        dparam = 0.1
        param = self.optatmo_psf_kwargs['size']
        fit_size_kwargs = {'size': param, 'min_size': param - dparam, 'max_size': param + dparam}

        lmparams = self._fit_optics_lmparams(fit_size_kwargs, ['size'])

        # do fit
        return_indices = False  # in the normal residual function if a star fails, get rid of it
        results = lmfit.minimize(self._fit_size_residual, lmparams, args=(stars, shapes, shape_errors, return_indices, logger,), method='brute', Ns=201)  # 1e-3 steps

        # set final fit
        logger.info('Optical fit from lmfit parameters:')
        logger.info(lmfit.fit_report(results))
        try:
            logger.info('Success: {0}'.format(str(results.success)))
            logger.info('Message: {0}'.format(str(results.message)))
            logger.info('lmdif_message: {0}'.format(str(results.lmdif_message)))
        except AttributeError:
            pass

        self.optatmo_psf_kwargs['size'] = results.params.valuesdict()['size']
        if results.errorbars:
            err = np.sqrt(results.covar[0, 0])
            self.optatmo_psf_kwargs['error_size'] = err
        else:
            logger.warning('No error calculated for size in fit_size')
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)

        # save this for debugging purposes
        self._fit_size_results = results

        if self.reference_wavefront: self._delete_cache(logger=logger)

    # TODO: update kwargs
    def fit_optics(self, stars, shapes, shape_errors, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of
        operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)
        import lmfit
        logger.info("Start fitting Optical fit for {0} stars and {1} shapes".format(len(stars), len(shapes)))

        # save reference wavefront values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_cache(stars, logger=logger)

        lmparams = self._fit_optics_lmparams(self.optatmo_psf_kwargs, self.keys)

        # do fit
        Dfun = self._fit_optics_make_Dfun(self._fit_optics_residual, lmparams, eps=1e-3)
        return_indices = False  # in the normal residual function if a star fails, get rid of it
        results = lmfit.minimize(self._fit_optics_residual, lmparams, args=(stars, shapes, shape_errors, return_indices, logger,), method='leastsq', Dfun=Dfun, col_deriv=1)

        # set final fit
        logger.info('Optical fit from lmfit parameters:')
        logger.info(lmfit.fit_report(results))
        try:
            logger.info('Success: {0}'.format(str(results.success)))
            logger.info('Message: {0}'.format(str(results.message)))
            logger.info('lmdif_message: {0}'.format(str(results.lmdif_message)))
        except AttributeError:
            pass

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
                    logger.warning('No Error calculated for parameter {0}! Replacing with large number {1}!'.format(key, placeholder_error))
                    self.optatmo_psf_kwargs['error_' + key] = placeholder_error
                key_i += 1
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)

        # save this for debugging purposes
        self._fit_optics_results = results

        if self.reference_wavefront: self._delete_cache(logger=logger)

    def fit_atmosphere(self, stars,
                       chisq_threshold=0.1, max_iterations=30, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of
        operations. Will also reject with outliers

        :param stars:           A list of Star instances.
        :param chisq_threshold: Change in reduced chisq at which iteration will
                                terminate. If no outliers is provided, this is
                                ignored. [default: 0.1]
        :param max_iterations:  Maximum number of iterations to try. If no
                                outliers is provided, this is ignored.
                                [default: 30]
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)

        if self._enable_atmosphere:
            # TODO: not tested
            logger.info("Setting _enable_atmosphere == False. Was {0}".format(self._enable_atmosphere))
            self._enable_atmosphere = False

        # fit models
        logger.info("Initial Fitting atmo model")
        new_stars = []
        params = self.getParamsList(stars)
        for star_i, star in zip(range(len(stars)), stars):
            try:
                model_fitted_star = self.fit_model(star, params=params[star_i], vary_shape=True, logger=logger)
                new_stars.append(model_fitted_star)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                # TODO: not tested
                logger.warning('{0}'.format(str(e)))
                logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
        stars = new_stars

        # fit interpolant
        logger.info("Initializing atmo interpolator")
        stars = self.atmo_interp.initialize(stars, logger=logger)

        logger.info("Fitting atmo interpolant")
        # Begin iterations.  Very simple convergence criterion right now.
        if self.outliers is None:
            # with no outliers, no need to do the below cycle
            self.atmo_interp.solve(stars, logger=logger)
        else:
            oldchisq = 0.
            for iteration in range(max_iterations):
                nremoved = 0
                logger.warning("Iteration %d: Fitting %d stars", iteration+1, len(stars))

                new_stars = []
                params = self.getParamsList(stars)
                for star_i, star in zip(range(len(stars)), stars):
                    try:
                        model_fitted_star = self.fit_model(star, params=params[star_i], vary_shape=True, logger=logger)
                        new_stars.append(model_fitted_star)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:
                        # TODO: not tested
                        logger.warning('{0}'.format(str(e)))
                        logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))
                stars = new_stars


                logger.debug("             Calculating the interpolation")
                self.atmo_interp.solve(stars, logger=logger)

                # Refit and recenter all stars, collect stats
                logger.debug("             Re-fluxing stars")
                new_stars = []
                interpolated_stars = self.atmo_interp.interpolateList(stars)
                for s_interp in interpolated_stars:
                    try:
                        new_star = self.reflux(s_interp,logger=logger)  # fit params come from model fit of just flux, du, dv while other params come from interpolation
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:  # pragma: no cover
                        logger.warning("Caught exception:")
                        logger.warning("Failed trying to reflux star at %s.  Excluding it.",
                                           s_interp.image_pos)
                        nremoved += 1
                    else:
                        new_stars.append(new_star)
                stars = new_stars

                if self.outliers:
                    # Perform outlier rejection
                    logger.debug("             Looking for outliers")
                    stars, nremoved1 = self.outliers.removeOutliers(stars, logger=logger)
                    if nremoved1 == 0:
                        logger.debug("             No outliers found")
                    else:
                        logger.info("             Removed %d outliers", nremoved1)
                    nremoved += nremoved1

                chisq = np.sum([s.fit.chisq for s in stars])
                dof   = np.sum([s.fit.dof for s in stars])
                logger.warning("             Total chisq = %.2f / %d dof", chisq, dof)

                # Very simple convergence test here:
                # Note, the lack of abs here means if chisq increases, we also stop.
                # Also, don't quit if we removed any outliers.
                if (nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < chisq_threshold*dof):
                    break
                oldchisq = chisq

            else:
                logger.warning("PSF fit did not converge.  Max iterations = %d reached.", max_iterations)

        return stars

    def reflux(self, star, logger=None):
        """Fit the Model to the star's data, varying only the flux and center. This puts one of the options for fit_model into the regular Piff syntax.

        :param star:        A Star instance
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance, with updated flux, center, chisq, dof
        """
        logger = LoggerWrapper(logger)
        refluxed_star = self.fit_model(star, params=self.getParams(star), vary_shape=False, logger=logger)
        return refluxed_star

    # TODO: vary_shape = False not tested
    # TODO: not putting in params not tested
    # TODO: variation with _enable_atmosphere not tested
    # TODO: how come the initial fluxes and the fluxes used in fits are so wildly different?!?!
    def fit_model(self, star, params, vary_shape=True, return_results=False, logger=None):
        logger = LoggerWrapper(logger)
        import lmfit
        # create lmparameters
        # put in initial guesses for flux, du, dv if they exist
        flux = star.fit.flux
        du, dv = star.fit.center
        lmparams = lmfit.Parameters()
        # Order of params is important!
        lmparams.add('flux', value=flux, vary=True, min=0.0)
        lmparams.add('du', value=du, vary=True, min=-1, max=1)
        lmparams.add('dv', value=dv, vary=True, min=-1, max=1)

        # we must also cut the min and max based on opt_params to avoid things
        # like large ellipticities or small sizes
        min_size = self.optatmo_psf_kwargs['min_size']
        max_size = self.optatmo_psf_kwargs['max_size']
        max_g = self.optatmo_psf_kwargs['max_g1']
        # getParams puts in atmosphere terms

        fit_size = 0
        fit_g1 = 0
        fit_g2 = 0
        opt_size = params[0]
        opt_g1 = params[1]
        opt_g2 = params[2]
        lmparams.add('size', value=fit_size, vary=vary_shape, min=min_size - opt_size, max=max_size - opt_size)
        lmparams.add('g1', value=fit_g1,   vary=vary_shape, min=-max_g - opt_g1, max=max_g - opt_g1)
        lmparams.add('g2', value=fit_g2,   vary=vary_shape, min=-max_g - opt_g2, max=max_g - opt_g2)

        # do fit
        results = lmfit.minimize(self._fit_model_residual, lmparams,
                                 args=(star, params, logger,),
                                 method='leastsq', epsfcn=1e-8,
                                 maxfev=200)
        logger.debug(lmfit.fit_report(results))
        flux = results.params['flux'].value
        du = results.params['du'].value
        dv = results.params['dv'].value
        if vary_shape:
            fit_params = np.array([results.params['size'].value, results.params['g1'].value, results.params['g2'].value])
            if results.errorbars:
                params_var = np.diag(results.covar)[3:]
            else:
                params_var = np.zeros(len(params))
        else:
            try:
                fit_params = star.fit.params
            except AttributeError:
                fit_params = None
            try:
                params_var = star.fit.params_var
            except AttributeError:
                params_var = None
        center = (du, dv)
        chisq = results.chisqr
        dof = results.nfree
        fit = StarFit(fit_params, params_var=params_var, flux=flux, center=center,
                      chisq=chisq, dof=dof)
        star_fit = Star(star.data, fit)
        if return_results:
            return star_fit, results
        else:
            return star_fit

    # TODO: docs, not tested
    def adjustStar(self, star, logger=None):
        # idea: given a star, find the optimal flux, offset PSF for comparison
        param = self.getParams(star)
        star_adjusted = self.fit_model(star, param, vary_shape=False, logger=logger)
        return star_adjusted

    # TODO: docs, not tested
    def adjustStarList(self, stars, logger=None):
        params = self.getParamsList(stars)
        stars_adjusted = []
        for star, param in zip(stars, params):
            star_adjusted = self.fit_model(star, param, vary_shape=False, logger=logger)
            stars_adjusted.append(star_adjusted)
        return stars_adjusted

    @staticmethod
    def analytic_shapes(params, analytic_coefs, logger=None):
        logger = LoggerWrapper(logger)
        # TODO: docs
        coefs = analytic_coefs[0]
        indices = analytic_coefs[1]

        # transform into full index
        params_onehot = np.vstack((np.ones(len(params)).T, params.T)).T.astype(np.float64)
        # apply model
        shapes = np.array([poly(params_onehot, coef.astype(np.float64), index.astype(np.int64))
                           for coef, index in zip(coefs, indices)]).T
        if np.any(shapes != shapes) or np.any(~np.isfinite(shapes)):
            # shouldn't happen unless something wacko happens
            raise ValueError('Bad shape values')

        return shapes

    def _fit_analytic_residual(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        logger = LoggerWrapper(logger)
        # update psf
        self._update_optatmopsf(lmparams.valuesdict(), logger)

        # get star params
        params = self.getParamsList(stars)

        # generate analytic star moments
        shapes_model = self.analytic_shapes(params, self.analytic_coefs)
        shape_weights = self._shape_weights

        # calculate chi
        shapes = shapes[:, 3:]
        errors = shape_errors[:, 3:]
        chi = (shape_weights[None] * (shapes_model - shapes) / errors).flatten()
        logger.debug('Current Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        returned_indices = np.ones_like(chi)  # all stars should return reasonable chi
        if return_indices:
            return chi, returned_indices.astype(bool)
        else:
            return chi

    def _fit_optics_lmparams(self, optatmo_psf_kwargs, keys):
        import lmfit
        # create lmparameters
        lmparams = lmfit.Parameters()
        # step through keys
        for key in keys:
            if key in optatmo_psf_kwargs:
                value = optatmo_psf_kwargs[key]
            else:
                value = 0
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
            lmparams.add(key, value=value, vary=vary, min=min, max=max)
        return lmparams

    def _fit_size_residual(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        logger = LoggerWrapper(logger)
        # extract chi on size only and remove the shape_weight
        if return_indices:
            chi, returned_indices = self._fit_optics_residual(lmparams, stars, shapes, shape_errors, return_indices, logger)
            chi = chi[0::3] / self._shape_weights[0]
            returned_indices = returned_indices[0::3]
            return chi, returned_indices
        else:
            chi = self._fit_optics_residual(lmparams, stars, shapes, shape_errors, return_indices, logger)
            chi = chi[0::3] / self._shape_weights[0]
            return chi

    def _fit_optics_residual(self, lmparams, stars, shapes, shape_errors, return_indices=False, logger=None):
        logger = LoggerWrapper(logger)
        # update psf
        self._update_optatmopsf(lmparams.valuesdict(), logger)

        # get optical params
        opt_params = self.getParamsList(stars)

        # measure their shapes and calculate chi
        chi = np.array([])
        returned_indices = np.array([])
        for i, star in enumerate(stars):
            params = opt_params[i]
            shape = shapes[i]
            error = shape_errors[i]

            try:
                # get profile; modify based on flux and shifts
                profile = self.getProfile(params)

                # measure final shape
                star_model = self.drawProfile(star, profile, params)
                # # TODO: I don't think we need to do this??
                # # model star's weight array should only be 0's and 1's
                # weight = star_model.data.weight.array
                # star_model.data.weight.array[:] = np.where(weight, 1., 0)
                shape_model = self.measure_shape(star_model, return_error=False)
                if np.any(shape_model != shape_model):
                    logger.warning('Star {0} returned nan shape'.format(i))
                    indices = np.zeros_like(shape_model, dtype=bool)
                else:
                    indices = np.ones_like(shape_model, dtype=bool)
            except (ModelFitError, RuntimeError) as e:
                logger.warning(str(e))
                logger.warning('Star {0}\'s model failed to be drawn and measured.'.format(i))
                logger.warning('Parameters are {0}'.format(str(params)))
                logger.warning('Input parameters are {0}'.format(str(lmparams.valuesdict())))
                logger.warning('Filling with zero chi')
                shape_model = shape
                indices = np.zeros_like(shape_model, dtype=bool)

            # don't care about flux, du, dv here
            chi_i = self._shape_weights * ((shape_model - shape) / error)[3:]
            chi = np.hstack((chi, chi_i))
            returned_indices = np.hstack((returned_indices, indices[3:]))
        logger.debug('Current Total Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        returned_indices = returned_indices.astype(bool)
        if return_indices:
            return chi, returned_indices
        else:
            return chi

    # TODO: whole func not tested
    def _fit_optics_make_Dfun(self, func, lmparams, eps):
        """Takes in a _fit_optics_residual_* function and returns the function that gives you gradients. We basically need to do this because the numerical differentiation done in lmfit can attempt finite differences with too small of values, resulting in inaccurate derivatives. We use a centered stencil

        :param func:        The function whose derivative we want to calculate
        :param lmparams:    The lmparameters used in fit
        :param eps:         The step size of the numerical derivative

        :returns Dfun:      Function that returns the gradient of chi in appropriate form for lmfit
        """
        keys = []
        for key in lmparams:
            param = lmparams[key]
            if param.vary:
                keys.append(param.name)

        # construct function based on keys that are NOT varying
        def Dfun(params, stars, shapes, shape_errors, return_indices, logger=None):
            logger = LoggerWrapper(logger)
            return_indices_dfun = True
            # use centered stencil, with eps based on key and fitting parameters
            gradients = []

            # make stencil calculations
            # need the values of the zernke polynomials as a function of Focal
            # position. Shape is (focal zernikes, stars)
            u = np.array([star.data['u'] for star in stars])
            v = np.array([star.data['v'] for star in stars])
            r = (u + 1j * v) / self.fov_radius
            rsqr = np.abs(r) ** 2
            # have to do some funky transposing to make the ordering work out
            zernikes = np.array([galsim.utilities.horner2d(rsqr, r, ca.T,
                dtype=complex).real for ca in self._noll_coef_field.T])
            logger.debug('Calculating gradient for {0} stars, {1} shapes'.format(len(stars), len(shapes)))
            for i in range(len(keys)):
                key = keys[i]

                # IF we are doing zernikes. then zPupili_zFocalj is proportional to zPupili_zFocal1
                if 'zFocal' in key:
                    key_split = key.split('_')
                    key_check = key_split[0] + '_zFocal001'
                    # note we assume that the fitter always does zFocal001 first. I think this is reasonable
                    if key_check in keys and key_check != key:
                        # find index
                        grad_old = gradients[keys.index(key_check)]
                        j = int(key.split('zFocal')[-1])
                        zernike_i = zernikes[j - 1]
                        # need to repeat zernikes 
                        reps = len(grad_old) // len(zernike_i)
                        gradient = grad_old * np.tile(zernike_i, reps)
                        gradients.append(gradient)
                        continue

                # do positive and negative steps
                logger.debug('Calculating gradient for {0}, centered at {1:+.4e} +- {2:.4e}'.format(key, params[key].value, eps))
                params[key].set(value=params[key].value + eps)
                chi_pos, indices_pos = func(params, stars, shapes, shape_errors, return_indices_dfun, logger)
                params[key].set(value=params[key].value - 2 * eps)
                chi_neg, indices_neg = func(params, stars, shapes, shape_errors, return_indices_dfun, logger)
                # and set it back
                params[key].set(value=params[key].value + eps)

                # catch if any failed on either step
                conds = (indices_pos * indices_neg).astype(bool)

                # measure gradient
                gradient = ((chi_pos - chi_neg) / (2 * eps))
                gradient = np.where(conds, gradient, 0)

                # dfun wants -dchi/dtheta
                gradient *= -1

                logger.debug('average gradient for {0} across data is {1:+.4e} for {2} entries'.format(key, np.mean(gradient), len(gradient)))
                # record gradient
                gradients.append(gradient)
                # if gradient.sum() == 0:
                #     import ipdb; ipdb.set_trace()

            gradients = np.array(gradients)
            logger.debug('Gradient shape is {0}'.format(str(gradients.shape)))
            return gradients

        return Dfun

    # TODO: untested
    def _create_cache(self, stars, logger=None):
        """Save aberrations from reference wavefront. This is useful if we want
        to keep calling getParams but we aren't changing the positions of the
        stars. The reference_wavefront.interpolateList call is relatively
        expensive, so we save the results from that step and skip it if we can.

        :param stars:   A list of stars
        :param logger:  A logger object for logging debug info [default: None]
        """
        if self.reference_wavefront:
            logger.debug('Caching reference aberrations')
            self._cache = True
            clean_stars = [Star(star.data, None) for star in stars]
            interp_stars = self.reference_wavefront.interpolateList(clean_stars)
            aberrations_reference_wavefront = np.array([star_interpolated.fit.params for star_interpolated in interp_stars])
            self._aberrations_reference_wavefront = aberrations_reference_wavefront
        else:
            logger.debug('Cache called, but no reference wavefront. Skipping')
            self._cache = False
            self._aberrations_reference_wavefront = None

    # TODO: untested
    def _delete_cache(self, logger=None):
        """Delete reference wavefront cache.
        :param logger:  A logger object for logging debug info [default: None]
        """
        if self.reference_wavefront:
            logger.debug('Clearing cache of reference aberrations')
        else:
            logger.debug('Delete cache called, but no reference wavefront. Skipping')
        self._cache = False
        self._aberrations_reference_wavefront = None

    # TODO: vary_shape = False not tested
    def _fit_model_residual(self, lmparams, star, params, logger=None):
        logger = LoggerWrapper(logger)

        # modify params with lmparams
        params = params.copy()
        flux, du, dv, size, g1, g2 = lmparams.valuesdict().values()
        params[0] = params[0] + size
        params[1] = params[1] + g1
        params[2] = params[2] + g2

        prof = self.getProfile(params).shift(du, dv) * flux

        # calculate chi2
        image, weight, image_pos = star.data.getImage()
        image_model = self.drawProfile(star, prof, params, use_fit=False).image
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

# some functions to deal with the analytic poly
@numba.jit
def poly(X, coef, indices):
    """Deals with complicated sparse polynomials

    :param X:           array of values [nstar, nvar] we make polynomial out of
                        that is one-hot (ie the first term of each entry is 1)
    :param coef:        sets of coefficients [ncoef]
    :param indices:     sets which indices we are multiplying together [ncoef, norder]

    :returns y:         [nstar] values of the polynomial
    """

    nstar = X.shape[0]
    ncoef = coef.shape[0]
    norder = indices.shape[1]

    y = np.zeros(nstar, dtype=np.float64)

    for i in range(0, nstar):
        for j in range(0, ncoef):
            term = 1
            for k in range(0, norder):
                indx = indices[j, k]
                term *= X[i, indx]
            term *= coef[j]
            y[i] += term

    return y

@numba.jit
def poly_full(X, indices):
    """Deals with complicated sparse polynomials

    :param X:           array of values [nstar, nvar] we make polynomial out of
                        that is one-hot (ie the first term of each entry is 1)
    :param indices:     sets which indices we are multiplying together [ncoef, norder]

    :returns Xpoly:     [nstar, ncoef] values of the polynomial
    """

    nstar = X.shape[0]
    ncoef = indices.shape[0]
    norder = indices.shape[1]

    Xpoly = np.zeros((nstar, ncoef), dtype=np.float64)

    for i in range(0, nstar):
        for j in range(0, ncoef):
            term = 1
            for k in range(0, norder):
                indx = indices[j, k]
                term *= X[i, indx]
            Xpoly[i, j] = term

    return Xpoly
