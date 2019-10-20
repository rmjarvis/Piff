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
import coord
import numpy as np
import scipy
import copy
import os
import sklearn
import pickle

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .outliers import Outliers
from .model import ModelFitError
from .star import Star, StarFit, StarData
from .util import measure_snr, write_kwargs, read_kwargs
from galsim.config import LoggerWrapper

class wavefrontmap(object):
    """a class used to build and access a Wavefront map - zernike coefficients vs. X,Y

    Aaron Roodman (C) SLAC National Accelerator Laboratory, Stanford University 2018.
    """
    _cache = {}

    def __init__(self, fname):
        self.fname = fname
        if fname in self._cache:
            self.x, self.y, self.zcoeff, self.nZernikeLast, self.interpDict = self._cache[fname]
        else:
            from scipy.interpolate import Rbf
            # init contains all initializations which are done only once for all fits

            mapdict = pickle.load(open(fname,'rb'))
            self.x = mapdict['x']
            self.y = mapdict['y']
            self.zcoeff = mapdict['zcoeff']
            self.nZernikeLast = self.zcoeff.shape[1]

            self.interpDict = {}
            for iZ in range(3,self.nZernikeLast):    # numbering is such that iZ=3 is zern4
                self.interpDict[iZ] = Rbf(self.x, self.y, self.zcoeff[:,iZ])
            self._cache[fname] = (self.x, self.y, self.zcoeff, self.nZernikeLast, self.interpDict)

    def get(self,x,y,nZernikeFirst=4):
        # start with defocus (zern4) and go up to the highest zernike found in the higher order
        # reference wavefront here
        # fill an array with Zernike coefficients for this x,y in the Map

        zout = np.zeros((self.nZernikeLast-nZernikeFirst+1))
        for iZactual in range(nZernikeFirst,self.nZernikeLast+1):
            iZ = iZactual-1
            zout[iZactual-nZernikeFirst] = self.interpDict[iZ](x,y)

        return zout

class OptAtmoPSF(PSF):
    """Combine Optical and Atmospheric PSFs together

    Fit Combined Atmosphere and Optical PSF in two stage process.

    :param atmo_interp:             Piff Interpolant object that represents the atmospheric
                                    interpolation
    :param outliers:                Optionally, an Outliers instance used to remove outliers
                                    during atmosphere fit.  [default: None]
    :param optatmo_psf_kwargs:      Terms that set the state of the PSF, excepting the atmospheric
                                    interpolant
    :param optical_psf_kwargs:      Arguments to pass into galsim opticalpsf object
    :param kolmogorov_kwargs:       Arguments to pass into galsim kolmogorov or vonkarman object
    :param reference_wavefront:     Reference interpolator for the optical wavefront. Takes in
                                    stars, returns aberrations. Default is to not include.
    :param n_optfit_stars:          If > 0, randomly sample only n_optfit_stars for the optical fit.
                                    Only use n_optfit_stars if doing a test run, not a serious fit.
                                    [default: 0]
    :param fov_radius:              Radius of telescope in u,v coordinates [default: 1]
    :param jmax_pupil:              Number of pupil-basis zernikes in Optical model. Inclusive and
                                    in Noll convention. [default: 11]
    :param jmax_focal:              Number of focal-basis zernikes in Optical model. Inclusive and
                                    in Noll convention. [default: 11]
    :param min_optfit_snr:          Minimum snr from star property required for optical portion of
                                    fit. If 0, ignored. [default: 0]
    :param fit_optics_mode:         Choose ['random_forest', 'shape', 'pixel'] for optics fitting
                                    mode. [default: 'pixel'; random_forest is invalid in py2.7]
    :param higher_order_reference_wavefront_file: A string with the path and filename of the
                                    pickle containing the higher order reference wavefront.
                                    [default: None]
    :param init_with_rf:            Initialize the fit with the random_forest? [default: False;
                                    invalid for py2.7]
    :param random_forest_shapes_model_pickles_location: A string with the path to the folder
                                    containing the random forest model pickles for the random_forest
                                    fit. [default: None]
    :param atmosphere_model:        Choose ['kolmogorov', 'vonkarman']. Selects the galsim object
                                    used for the atmospheric piece.  Note that the default is
                                    vonkarman and to use kolmogorov would require some changes to
                                    input.py.
    :param atmo_mad_outlier:        If True, when computing atmosphere interps remove 5 sigma
                                    outliers from a MAD cut
    :param shape_weights:           A list of weights for the different moments to be used in the
                                    chisq fit
    :param test_fraction:           The fraction of stars to reserve for testing. [default: 0.2]
    :param logger:                  A logger object for logging debug info.
                                    [default: None]
    Notes
    -----
    Our model of the PSF is the convolution of a sheared Kolmogorov/VonKarman
    with an optics model. If Vonkarman, also have L0, the VonKarman outer scale:
        PSF = convolve(Vonkarman(size, g1, g2, L0), Optics(defocus, etc))
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
    Decomposition. Note that L0 is considered separately and only has a constant
    piece. The fitting process can be broken down into three major steps:
    1. Fit b_{k \ell} by looking at the field pattern of the shapes.
        -   First, we use a random forest model to find this approximately:
            This is very fast. This random forest model may need to be
            recalculated for different telescopes.
        -   The random forest model may misestimate the
            size. To account for this, we do a second fit: simply
            take a few stars, grid search b_{1 1} (ie constant size), and
            adjust accordingly.
        -   Finally we do the full optical fit, including using L0.
    2. Do individual star fit to atmospheric parameters.
        -   a_{ik} = a^{optics}_{ik} + a^{atmosphere}_{ik} for k < 4, where
            a^{optics}_{ik} = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i).
            We directly find a^{atmosphere}_{ik} for each star by
            minimizing the chi2 of the pixels of the observed star and the
            model as drawn here.
    2. Fit atmo_interp.
        -   After finding a^{atmosphere}_{ik}, we fit the atmo_interp to
            interpolate those parameters as a function of focal plane
            position (u_i, v_i).
    """
    def __init__(self, atmo_interp=None, outliers=None, optatmo_psf_kwargs={},
                 optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None,
                 n_optfit_stars=0, fov_radius=4500., jmax_pupil=11,
                 jmax_focal=10, min_optfit_snr=0, fit_optics_mode='pixel',
                 higher_order_reference_wavefront_file=None, init_with_rf=False,
                 random_forest_shapes_model_pickles_location=None,
                 atmosphere_model='vonkarman', atmo_mad_outlier=False, max_shapes = [],
                 shape_weights=[], reference_wavefront_zernikes_list=[],
                 higher_order_reference_wavefront_zernikes_list=[], test_fraction=0.2,
                 logger=None, **kwargs):
        logger = LoggerWrapper(logger)

        # If pupil_angle and strut angle are provided as strings, eval them.
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

        self.outliers = outliers
        # atmo_interp is a parsed class
        self.atmo_interp = atmo_interp
        self.optical_psf_kwargs = optical_psf_kwargs
        self.kolmogorov_kwargs = kolmogorov_kwargs
        self.reference_wavefront = reference_wavefront
        self.higher_order_reference_wavefront_file = higher_order_reference_wavefront_file
        if self.higher_order_reference_wavefront_file in [None, 'none', 'None', 'NONE']:
            # here we save the specified higher order reference wavefront as an instance of the
            # wavefrontmap class.
            self.higher_order_reference_wavefront = None
        else:
            self.higher_order_reference_wavefront = wavefrontmap(
                self.higher_order_reference_wavefront_file)
        self.min_optfit_snr = min_optfit_snr
        self.n_optfit_stars = n_optfit_stars

        #####
        # setup double zernike piece
        #####
        if jmax_pupil < 4:
            # why do an optatmo if you have no optical?
            raise ValueError('OptAtmo PSF requires at least 4 aberrations; found {0}'.format(
                             jmax_pupil))
        self.jmax_pupil = jmax_pupil
        if jmax_focal < 1:
            # need at least some constant piece of focal
            raise ValueError('OptAtmo PSF requires at least a constant field zernike ' +
                             'found {0}'.format(jmax_focal))
        self.jmax_focal = jmax_focal

        self.fov_radius = fov_radius

        self._noll_coef_field = galsim.zernike._noll_coef_array(self.jmax_focal, 0.0)

        min_sizes = {'kolmogorov': 0.45, 'vonkarman': 0.7}
        self.optatmo_psf_kwargs = {
            'L0' : 25.0 if atmosphere_model == 'vonkarman' else -1.,
            'fix_L0':   atmosphere_model == 'kolmogorov',
            'min_L0': 5.0,
            'max_L0': 100.0,
            'size': 1.0,
            'fix_size': False,
            'min_size': min_sizes[atmosphere_model],
            'max_size': 3.0,
            'g1':   0,
            'fix_g1':   False,
            'min_g1': -0.4,
            'max_g1': 0.4,
            'g2':   0,
            'fix_g2':   False,
            'min_g2': -0.4,
            'max_g2': 0.4,
        }
        self.keys = ['size', 'g1', 'g2', 'L0']

        # throw in default zernike parameters
        # only fit zernikes starting at 4 / defocus
        for zi in range(4, self.jmax_pupil + 1):
            for dxy in range(1, self.jmax_focal + 1):
                zkey = 'zPupil{0:03d}_zFocal{1:03d}'.format(zi, dxy)
                self.keys.append(zkey)

                # default to unfixing all possible combinations
                self.optatmo_psf_kwargs['fix_' + zkey] = False
                # can optionally fix an entire Pupil or Focal aberrations if we want
                fix_keyPupil = 'fix_zPupil{0:03d}'.format(zi)
                if fix_keyPupil in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyPupil]
                fix_keyFocal = 'fix_zFocal{0:03d}'.format(dxy)
                if fix_keyFocal in optatmo_psf_kwargs:
                    self.optatmo_psf_kwargs['fix_' + zkey] += optatmo_psf_kwargs[fix_keyFocal]

                zmax = 1.  # don't allow the solutions to go crazy
                self.optatmo_psf_kwargs['min_' + zkey] = -zmax
                self.optatmo_psf_kwargs['max_' + zkey] =  zmax

                # initial value. If there is no reference wavefront it helps
                # the fitter to pass along nonzero values to non-fixed
                # parameters
                if self.reference_wavefront or self.optatmo_psf_kwargs['fix_' + zkey]:
                    #TODO: make this work with either of the reference wavefronts
                    self.optatmo_psf_kwargs[zkey] = 0
                else:
                    initial_value = np.random.random() * (0.1 - -0.1) + -0.1
                    logger.debug('Setting initial {0} to randomly generated value {1}'.format(
                                 zkey, initial_value))
                    self.optatmo_psf_kwargs[zkey] = initial_value
        # update aberrations from our kwargs
        try:
            self.optatmo_psf_kwargs.update(optatmo_psf_kwargs)
        except TypeError:
            # this means the dictionary got saved as 0 in the kwargs.
            # this is fixed in _finish_read
            pass

        # create initial aberrations_field from optatmo_psf_kwargs
        logger.debug("Initializing optatmopsf state")
        self.aberrations_field = np.zeros((self.jmax_pupil, self.jmax_focal),
                                          dtype=float)
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger)

        # since we haven't fit the interpolator, yet, disable atmosphere
        self._enable_atmosphere = False

        # We don't actually need the OpticalPSF rendering to be super accurate.
        # So dial down the default GalSim accuracy settings somewhat to get improved speed.
        self.gsparams = galsim.GSParams(
            minimum_fft_size=32,            # default 128
            folding_threshold=0.02,         # default 0.005
            maxk_threshold=0.01,            # default 0.001
        )
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 1.0    # defautl 1.5
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 1.0  # defautl 1.5

        # max size of shapes allowed
        self._max_shapes = np.array([1.5, 0.12, 0.12, 0.15, 0.15, 0.15, 0.15, 1.5, 5.0, 50.0])
        if len(max_shapes) > 0:
            if len(max_shapes) != len(self._max_shapes):
                raise ValueError('Specified {0} max shapes, but need to specify {1}!'.format(
                                 len(max_shapes), len(self._max_shapes)))
            for i, si in enumerate(max_shapes):
                self._max_shapes[i] = si
        # weighting of shapes
        self._shape_weights = np.array([0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        if len(shape_weights) > 0:
            if len(shape_weights) != len(self._shape_weights):
                raise ValueError('Specified {0} shape weights, but need to specify {1}!'.format(
                                 len(shape_weights), len(self._shape_weights)))
            for i, si in enumerate(shape_weights):
                self._shape_weights[i] = si

        for reference_wavefront_zernike in reference_wavefront_zernikes_list:
            if reference_wavefront_zernike in higher_order_reference_wavefront_zernikes_list:
                raise ValueError('Zernike {0} from reference_wavefront_zernike_list also found in '
                                 'higher_order_reference_wavefront_zernikes_list, but these lists '
                                 'cannot overlap!'.format(reference_wavefront_zernike))
        for hiz in higher_order_reference_wavefront_zernikes_list:
            if hiz > 37:
                raise ValueError('Zernike {0} from higher_reference_wavefront_zernike_list found; '
                                 'cannot use higher order reference wavefront with zernikes > z37 '
                                 'because conversion from zout_camera (AOS system) coordinates to '
                                 'zout_sky (Galsim) coordinates (inspired by thesis of Chris '
                                 'Davis) for higher order reference wavefront values are not '
                                 'recorded here above z37'.format(
                                    higher_order_reference_wavefront_zernike))
        # reference wavefront zernikes list
        self._reference_wavefront_zernikes_list = list(range(4, 12)) + [14, 15]
        if len(reference_wavefront_zernikes_list) > 0:
            self._reference_wavefront_zernikes_list = reference_wavefront_zernikes_list
            self._reference_wavefront_zernikes_list = [
                    int(reference_wavefront_zernike)
                    for reference_wavefront_zernike in self._reference_wavefront_zernikes_list]
        # higher order reference wavefront zernikes list
        self._higher_order_reference_wavefront_zernikes_list = [12, 13] + list(range(16,38))
        if len(higher_order_reference_wavefront_zernikes_list) > 0:
            self._higher_order_reference_wavefront_zernikes_list = \
                higher_order_reference_wavefront_zernikes_list
            self._higher_order_reference_wavefront_zernikes_list = [
                int(higher_order_reference_wavefront_zernike)
                for higher_order_reference_wavefront_zernike in \
                    self._higher_order_reference_wavefront_zernikes_list]

        self.fit_optics_mode = fit_optics_mode
        self.random_forest_shapes_model_pickles_location = \
            random_forest_shapes_model_pickles_location
        if atmosphere_model not in ['kolmogorov', 'vonkarman']:
            raise KeyError('Atmosphere model {0} not allowed! '
                           'choose either kolmogorov or vonkarman'.format(atmosphere_model))
        self.atmosphere_model = atmosphere_model

        self.atmo_mad_outlier = atmo_mad_outlier
        self.test_fraction = test_fraction
        self.init_with_rf = init_with_rf

        # kwargs
        self.kwargs = {
            'fov_radius': self.fov_radius,
            'max_shapes': self._max_shapes,
            'shape_weights': self._shape_weights,
            'reference_wavefront_zernikes_list': self._reference_wavefront_zernikes_list,
            'higher_order_reference_wavefront_zernikes_list':
                self._higher_order_reference_wavefront_zernikes_list,
            'jmax_pupil': self.jmax_pupil,
            'jmax_focal': self.jmax_focal,
            'min_optfit_snr': self.min_optfit_snr,
            'n_optfit_stars': self.n_optfit_stars,
            'fit_optics_mode': self.fit_optics_mode,
            'higher_order_reference_wavefront_file': self.higher_order_reference_wavefront_file,
            'init_with_rf': self.init_with_rf,
            'random_forest_shapes_model_pickles_location':
                self.random_forest_shapes_model_pickles_location,
            'atmosphere_model': self.atmosphere_model,
            'atmo_mad_outlier': self.atmo_mad_outlier,
            'test_fraction': self.test_fraction,
            # junk entries to be overwritten in _finish_read function
            'optatmo_psf_kwargs': 0,
            'atmo_interp': 0,
            'reference_wavefront': 0,
            'optical_psf_kwargs': 0,
            'kolmogorov_kwargs': 0,
            'outliers': 0,
        }

        # cache parameters to cut down on lookup
        self._caches = False
        self._aberrations_reference_wavefronts = None

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to
        use for initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info.  [default: None]

        :returns:               a kwargs dict to pass to the initializer
        """
        logger = LoggerWrapper(logger)
        config_psf = config_psf.copy()  # Don't alter the original dict.

        kwargs = config_psf.copy()
        kwargs.pop('type',None)

        # do processing as appropriate
        # set up optical and atmosphere psf kwargs using the optical model
        optical_psf_kwargs = config_psf.pop('optical_psf_kwargs', {})

        optical = Optical(logger=logger, **optical_psf_kwargs)
        kwargs['optical_psf_kwargs'] = optical.optical_psf_kwargs
        kolmogorov_kwargs = optical.kolmogorov_kwargs
        if 'kolmogorov_kwargs' in config_psf:
            kolmogorov_kwargs.update(config_psf['kolmogorov_kwargs'])
        # if we only have lam (which we expect from Optical models), then put in a placeholder fwhm
        # Also, let r0=0 or None indicate that there is no kolmogorov (or vonkarman) component
        if (kolmogorov_kwargs.keys() == ['lam'] or
            ('r0' in kolmogorov_kwargs and not kolmogorov_kwargs['r0'])):
            kolmogorov_kwargs = {'fwhm': 1.0}
        kwargs['kolmogorov_kwargs'] = kolmogorov_kwargs

        #custom shape weights for the moments used in fitting
        if 'optatmo_psf_kwargs' in config_psf:
            kwargs['optatmo_psf_kwargs'] = config_psf['optatmo_psf_kwargs']

        # atmo interp may be skipped; this is usually done so the atmospheric fitting is done in
        # the PIFF fitting pipeline
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
            if config_psf['reference_wavefront'] in [None, 'none', 'None', 'NONE']:
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

        # write reference wavefront if it exists
        if self.reference_wavefront:
            self.reference_wavefront.write(fits, extname + '_reference_wavefront')

        # write optical_psf_kwargs
        # pupil_angle and strut_angle won't serialize properly, so repr them now in
        # self.kwargs['optical_psf_kwargs'].
        optical_psf_kwargs = {}
        for key in self.optical_psf_kwargs:
            if key in ['pupil_angle', 'strut_angle']:
                optical_psf_kwargs[key] = repr(self.optical_psf_kwargs[key])
            else:
                optical_psf_kwargs[key] = self.optical_psf_kwargs[key]
        write_kwargs(fits, extname + '_optical_psf_kwargs', optical_psf_kwargs)

        # write kolmogorov_kwargs
        write_kwargs(fits, extname + '_kolmogorov_kwargs', self.kolmogorov_kwargs)

        # write the final fitted state of model
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

        # read optical_psf_kwargs
        self.optical_psf_kwargs = read_kwargs(fits, extname=extname + '_optical_psf_kwargs')
        # If pupil_angle and strut angle are provided as strings, eval them.
        for key in ['pupil_angle', 'strut_angle']:
            if key in self.optical_psf_kwargs and isinstance(self.optical_psf_kwargs[key],str):
                self.optical_psf_kwargs[key] = eval(self.optical_psf_kwargs[key])
        logger.info('Reloading optatmopsf optical psf kwargs')

        # read kolmogorov_kwargs
        self.kolmogorov_kwargs = read_kwargs(fits, extname=extname + '_kolmogorov_kwargs')
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
        logger.info('checking for outliers')
        if extname + '_outliers' in fits:
            logger.info('Reloading outliers')
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            logger.info('Skipping outliers')
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
                                If no outliers is provided, then this does nothing.
        :param max_iterations:  Maximum number of iterations to try during
                                atmosphere fit. [default: 30]
                                If no outliers is provided, then this does nothing.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)
        ntest = int(self.test_fraction * len(stars))
        test_indx = np.random.choice(len(stars), ntest, replace=False)
        test_stars = []
        train_stars = []
        for star_i, star in enumerate(stars):
            if star_i in test_indx:
                test_stars.append(star)
            else:
                train_stars.append(star)

        if self.higher_order_reference_wavefront_file in [None, 'none', 'None', 'NONE']:
            # here we save the specified higher order reference wavefront as an instance of the
            # wavefrontmap class. Note that this step is necessary here in case fit() is called by
            # itself and the creation of the higher order reference wavefront was not done
            # elsewhere.
            self.higher_order_reference_wavefront = None
        else:
            self.higher_order_reference_wavefront = wavefrontmap(
                self.higher_order_reference_wavefront_file)
        self.wcs = wcs
        self.pointing = pointing

        do_shapes = True
        do_errors = self.init_with_rf or self.fit_optics_mode in ['shape', 'random_forest']

        # do first pass of flux, centers, and shapes for the train stars
        # train stars that fail this step are going to constantly fail the fit, so
        # we get rid of them
        self.stars = []
        self.star_shapes = []
        self.star_errors = []
        self.star_snrs = []
        for star_i, star in enumerate(train_stars):
            logger.debug('Measuring shape of train star {0}'.format(star_i))
            try:
                star = Star(star.data, StarFit(None))
                if do_shapes:
                    shape = self.measure_shape_orthogonal(star, logger=logger)
                    star.fit.flux = shape[0]
                    star.fit.center = shape[1], shape[2]
                else:
                    shape = None
                # shapes measured here include flux, center, 2nd, 3rd, and orthogonal radial
                # moments up to eighth moments
                if do_errors:
                    error = self.measure_error_orthogonal(star, logger=logger)
                else:
                    error = None
                # errors measured here include flux, center, 2nd, 3rd, and orthogonal radial
                # moments up to eighth moments
                star.data.properties['shape'] = shape
                star.data.properties['shape_error'] = error
                snr = measure_snr(star)

                self.stars.append(star)
                self.star_shapes.append(shape)
                self.star_errors.append(error)
                self.star_snrs.append(snr)
            except (ModelFitError, RuntimeError) as e:
                # something went wrong with this star
                logger.warning(str(e))
                logger.warning('Train Star {0} failed shape estimation. Skipping'.format(star_i))
        self.star_shapes = np.array(self.star_shapes)
        self.star_errors = np.array(self.star_errors)
        self.star_snrs = np.array(self.star_snrs)

        # do first pass of flux, centers, and shapes for the test stars
        # test stars that fail this step are eliminated just like the train stars were before
        self.test_stars = []
        self.test_star_shapes = []
        self.test_star_errors = []
        self.test_star_snrs = []
        for star_i, star in enumerate(test_stars):
            logger.debug('Measuring shape of test star {0}'.format(star_i))
            try:
                shape = self.measure_shape_orthogonal(star, logger=logger)
                # shapes measured here include flux, center, 2nd, 3rd, and orthogonal radial
                # moments up to eighth moments
                error = self.measure_error_orthogonal(star, logger=logger)
                # errors measured here include flux, center, 2nd, 3rd, and orthogonal radial
                # moments up to eighth moments
                star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))
                star.data.properties['shape'] = shape
                star.data.properties['shape_error'] = error
                snr = measure_snr(star)

                self.test_stars.append(star)
                self.test_star_shapes.append(shape)
                self.test_star_errors.append(error)
                self.test_star_snrs.append(snr)
            except (ModelFitError, RuntimeError) as e:
                # something went wrong with this star
                logger.warning(str(e))
                logger.warning('Test Star {0} failed shape estimation. Skipping'.format(star_i))
        self.test_star_shapes = np.array(self.test_star_shapes)
        self.test_star_errors = np.array(self.test_star_errors)
        self.test_star_snrs = np.array(self.test_star_snrs)

        # do a max shapes cut for the train stars
        conds_shape = (np.all(np.abs(self.star_shapes[:, 3:]) <= self._max_shapes, axis=1))

        # do a max shapes cut for the test stars
        test_conds_shape = (np.all(
                np.abs(self.test_star_shapes[:, 3:]) <= self._max_shapes, axis=1))

        # also a MAD cut
        med = np.nanmedian(
            np.concatenate([self.star_shapes[:, 3:], self.test_star_shapes[:, 3:]], axis=0),
            axis=0)
        mad = np.nanmedian(
            np.abs(np.concatenate([self.star_shapes[:, 3:], self.test_star_shapes[:, 3:]], axis=0)
                   - med[None]), axis=0)
        logger.debug('MAD values: {0}'.format(str(mad)))

        # do MAD cut for the train stars
        madx = np.abs(self.star_shapes[:, 3:] - med[None])
        conds_mad = (np.all(madx <= 1.48 * 5 * mad, axis=1))

        # do MAD cut for the test stars
        test_madx = np.abs(self.test_star_shapes[:, 3:] - med[None])
        test_conds_mad = (np.all(test_madx <= 1.48 * 5 * mad, axis=1))

        # apply the aforementioned max shapes and MAD cuts for the train stars
        self.stars_indices = np.arange(len(self.stars))
        self.stars_indices = self.stars_indices[conds_shape * conds_mad]
        self.stars = [self.stars[indx] for indx in self.stars_indices]
        self.star_shapes = self.star_shapes[self.stars_indices]
        self.star_errors = self.star_errors[self.stars_indices]
        self.star_snrs = self.star_snrs[self.stars_indices]

        # apply the aforementioned max shapes and MAD cuts for the test stars
        self.test_stars_indices = np.arange(len(self.test_stars))
        self.test_stars_indices = self.test_stars_indices[test_conds_shape * test_conds_mad]
        self.test_stars = [self.test_stars[indx] for indx in self.test_stars_indices]
        self.test_star_shapes = self.test_star_shapes[self.test_stars_indices]
        self.test_star_errors = self.test_star_errors[self.test_stars_indices]
        self.test_star_snrs = self.test_star_snrs[self.test_stars_indices]

        # do an snr cut for the stars for the fit and record how many have been cut and why so far
        # in the logger
        self.fit_optics_indices = np.arange(len(self.stars))
        conds_snr = (self.star_snrs >= self.min_optfit_snr)
        self.fit_optics_indices = self.fit_optics_indices[conds_snr]
        logger.info('Cutting to {0} stars for fitting the optics based on SNR > {1} ({2} stars) '
                    'on maximum shapes ({3} stars) and on a 5 sigma outlier cut ({4} stars)'.format(
                        len(self.fit_optics_indices), self.min_optfit_snr,
                        len(conds_snr) - np.sum(conds_snr), len(conds_shape) - np.sum(conds_shape),
                        len(conds_mad) - np.sum(conds_mad)))

        # cut further if we have more stars for fit than n_optfit_stars.
        # Warning: only use n_optfit_stars if doing a test run, not a serious fit. This limits the
        # ability to get the 500 highest SNR stars for the fit.
        if (self.n_optfit_stars and self.n_optfit_stars <
                len(self.fit_optics_indices) and self.n_optfit_stars <= 500):
            logger.info('Cutting from {0} to {1} stars for the fit, as requested in '
                        'n_optfit_stars. Warning: at least 500 (highest SNR) stars recommended '
                        'for the optical fit. Only use n_optfit_stars if doing a test run, not a '
                        'serious fit.'.format(len(self.fit_optics_indices), self.n_optfit_stars))
            max_stars = self.n_optfit_stars
            np.random.shuffle(self.fit_optics_indices)
            self.fit_optics_indices = self.fit_optics_indices[:max_stars]
        else:
            if len(self.fit_optics_indices) < self.n_optfit_stars and self.n_optfit_stars > 0:
                logger.info('{0} stars remaining after cuts instead of the {1} requested using '
                            'n_optfit_stars. Of these, the (at most) 500 highest SNR stars will be '
                            'passed on to the optical fit. Note: only use n_optfit_stars if doing '
                            'a test run, not a serious fit.'.format(max_stars, self.n_optfit_stars))
            if (self.n_optfit_stars and self.n_optfit_stars <
                    len(self.fit_optics_indices) and self.n_optfit_stars > 500):
                logger.info('{0} stars remaining after cuts. Of these, the (at most) 500 highest '
                            'SNR stars will be passed on to the optical fit. Cutting down to {1} '
                            'stars has been requested using n_optfit_stars; however, since this '
                            'number is more than 500 this will be ignored. Only use '
                            'n_optfit_stars if doing a test run, not a serious fit.'.format(
                                len(self.fit_optics_indices), self.n_optfit_stars))

        self.fit_optics_stars = [self.stars[indx] for indx in self.fit_optics_indices]
        self.fit_optics_star_shapes = self.star_shapes[self.fit_optics_indices]
        self.fit_optics_star_errors = self.star_errors[self.fit_optics_indices]

        # cut down to 500 highest SNR stars for fit if have more than that remaining.
        if len(self.fit_optics_stars) > 500:
            bright_train_stars = []
            bright_train_star_shapes = []
            bright_train_star_errors = []
            snrs = []
            for fit_optics_star in self.fit_optics_stars:
                snrs.append(-measure_snr(fit_optics_star))
            snrs = np.array(snrs)
            order = np.argsort(snrs)

            for o, order_entry in enumerate(order):
                if o < 500:
                    bright_train_stars.append(self.fit_optics_stars[order_entry])
                    bright_train_star_shapes.append(self.fit_optics_star_shapes[order_entry])
                    bright_train_star_errors.append(self.fit_optics_star_errors[order_entry])

            self.fit_optics_stars = bright_train_stars
            self.fit_optics_star_shapes = np.array(bright_train_star_shapes)
            self.fit_optics_star_errors = np.array(bright_train_star_errors)

        # perform initial fit in "random_forest" mode, which uses a random forest model (this model
        # is trained to return shapes based on what fit parameters you give it)
        # the fit parameters here are the optical fit parameters and the average of the atmospheric
        # fit parameters
        if self.init_with_rf:
            self.fit_optics(self.fit_optics_stars, self.fit_optics_star_shapes,
                            self.fit_optics_star_errors, mode='random_forest',
                            logger=logger, **kwargs)

        # first just fit the optical size parameter to correct size offset
        # the size parameter is proportional to 1/r0, where r0 is the Fried parameter
        # the "optics" size is the average of this across the focal plane, whereas
        # the "atmospheric" size is the deviation from this average at different points in the
        # focal plane.
        # only use the e0 moment to fit
        # fit_size() is used before the full optical fit because it makes that fit faster
        moments_list = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2",
                        "orth4", "orth6", "orth8"]
        self.length_of_moments_list = len(moments_list)
        self.fit_size(self.fit_optics_stars, logger=logger, **kwargs)

        # do a fit to moments ("shape" mode) or pixels ("pixel" mode), whichever is specified in
        # the yaml file. Nothing happens here if "random_forest" mode is chosen
        # this is the "optical" fit; despite being called that the fit parameters here are the
        # optical fit parameters and the across-the-focal-plane average of the atmospheric fit
        # parameters
        self.total_redchi_across_iterations = []
        if self.fit_optics_mode == 'random_forest' and self.init_with_rf:
            # already did it, so can pass
            pass
        elif self.fit_optics_mode in ['shape', 'pixel', 'random_forest']:
            self.fit_optics(self.fit_optics_stars, self.fit_optics_star_shapes,
                            self.fit_optics_star_errors, mode=self.fit_optics_mode, logger=logger,
                            ftol=1.e-3, **kwargs)
                            #looser convergence criteria used than default of ftol=1.e-7
        else:
            # an unrecognized mode is simply ignored
            logger.warning('Found unrecognized fit_optics_mode {0}. Ignoring'.format(
                           self.fit_optics_mode))

        logger.info("len(self.stars): {0}".format(len(self.stars)))
        # one extra round of outlier rejection using the pull from the moments (only up to third
        # moments)
        if self.fit_optics_mode == 'shape':
            for s, stars in enumerate([self.stars, self.test_stars, self.fit_optics_stars]):
                if s == 0:
                    logger.info("now prepping pull cuts for self.stars; in other words the train stars")
                if s == 1:
                    logger.info("now prepping pull cuts for self.test_stars")
                if s == 2:
                    logger.info("now prepping pull cuts for self.fit_optics_stars; in other words the "
                                "subset train stars specifically used in the optical fit")
                data_shapes_all_stars = []
                data_errors_all_stars = []
                model_shapes_all_stars = []
                for star in stars:
                    data_shapes_all_stars.append(self.measure_shape_third_moments(star))
                    data_errors_all_stars.append(self.measure_error_third_moments(star))
                    model_shapes_all_stars.append(self.measure_shape_third_moments(self.drawStar(star)))
                data_shapes_all_stars = np.array(data_shapes_all_stars)[:,3:]
                data_errors_all_stars = np.array(data_errors_all_stars)[:,3:]
                model_shapes_all_stars = np.array(model_shapes_all_stars)[:,3:]
                pull_all_stars = ((data_shapes_all_stars - model_shapes_all_stars) /
                                    data_errors_all_stars)
                # pull is (data-model)/error
                logger.debug("data_shapes_all_stars: {0}".format(data_shapes_all_stars))
                logger.debug("model_shapes_all_stars: {0}".format(model_shapes_all_stars))
                logger.debug("data_errors_all_stars: {0}".format(data_errors_all_stars))
                logger.debug("pull_all_stars: {0}".format(pull_all_stars))
                conds_pull = (np.all(np.abs(pull_all_stars) <= 4.0, axis=1))
                # all stars with more than 4.0 pull are thrown out
                conds_pull_e0 = (np.abs(pull_all_stars[:,0]) <= 4.0)
                conds_pull_e1 = (np.abs(pull_all_stars[:,1]) <= 4.0)
                conds_pull_e2 = (np.abs(pull_all_stars[:,2]) <= 4.0)
                if s == 0:
                    self.stars = np.array(self.stars)[conds_pull].tolist()
                if s == 1:
                    self.test_stars = np.array(self.test_stars)[conds_pull].tolist()
                if s == 2:
                    self.number_of_outliers_optical = np.array(
                        [len(self.fit_optics_stars) - np.sum(conds_pull_e0),
                         len(self.fit_optics_stars) - np.sum(conds_pull_e1),
                         len(self.fit_optics_stars) - np.sum(conds_pull_e2)])
                    self.number_of_stars_pre_cut_optical = len(self.fit_optics_stars)
                    self.fit_optics_stars = np.array(self.fit_optics_stars)[conds_pull].tolist()
                    self.number_of_stars_post_cut_optical = len(self.fit_optics_stars)
                    self.pull_mean_optical = np.nanmean(pull_all_stars[:,:3], axis=0)
                    # the mean pull (only second moments) for stars used in the fit is later used to
                    # find outliers among exposures
                    self.pull_rms_optical = np.sqrt(np.nanmean(np.square(pull_all_stars[:,:3]),axis=0))
                    self.pull_all_stars_optical = pull_all_stars


            number_of_stars_used_in_optical_chi = \
                len(self.final_optical_chi)//self.length_of_moments_list
            logger.info("total chisq for optical chi: {0}".format(
                np.sum(np.square(self.final_optical_chi))))
            for tm, test_moment in enumerate(moments_list):
                logger.debug("total chisq for optical chi for {0}: {1}".format(
                    test_moment,
                    np.sum(np.square(self.final_optical_chi)[tm::self.length_of_moments_list])))
            logger.info("total dof for optical chi: {0}".format(len(self.final_optical_chi)))
            logger.info("number_of_stars_used_in_optical_chi: {0}".format(
                        number_of_stars_used_in_optical_chi))

            # record the chi
            self.chisq_all_stars_optical = np.empty(number_of_stars_used_in_optical_chi)
            for s in range(0,number_of_stars_used_in_optical_chi):
                self.chisq_all_stars_optical[s] = np.sum(np.square(
                    self.final_optical_chi[s*self.length_of_moments_list :
                                           s*self.length_of_moments_list+self.length_of_moments_list]))
        # this is the "atmospheric" fit.
        # we start here with the optical fit parameters and the average values of the atmospheric
        # parameters found in the optical fit and hold those fixed.
        # we float only the deviation of these atmospheric parameters from the average here.
        # this fit can be skipped and usually is in order to do the atmospheric fit with the PIFF
        # fitting pipeline
        if self.atmo_interp in ['skip', 'Skip', None, 'none', 'None', 0]:
            pass
        else:
            stars_fit_atmosphere, stars_fit_atmosphere_stripped = self.fit_atmosphere(
                self.stars, chisq_threshold=chisq_threshold, max_iterations=max_iterations,
                logger=logger, **kwargs)
            self.stars = stars_fit_atmosphere
            # keeps all fit params and vars, but does NOT include any removed with outliers

            # enable atmosphere interpolation now that we have solved the interp
            logger.info('Enabling Interpolated Atmosphere')
            self._enable_atmosphere = True

    def _getParamsList_aberrations_field(self, stars):
        """Get params for a list of stars from the aberrations
        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :returns:           Params  [size, g1, g2, z4, z5...] for each star
                            where all params that are not "z_number" are
                            atmospheric params (average across the focal plane).
        Notes
        -----
        We have a set of coefficients b_{k \ell} that describe the Zernike
        decomposition. Then, for the i-th star at position (u_i, v_i), we get
        param a_{ik} as:
            a_{ik} = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i)
        """
        # collect u and v from stars
        u = np.array([star.data['u'] for star in stars])
        v = np.array([star.data['v'] for star in stars])
        r = (u + 1j * v) / self.fov_radius
        rsqr = np.abs(r) ** 2
        # get [size, g1, g2, z4, z5...]

        # There is a bug in GalSim 2.2 with dtype=complex, so the below doesn't work.
        #return np.array([galsim.utilities.horner2d(rsqr, r, ca, dtype=complex).real
        #                 for ca in self._coef_arrays_field]).T  # (nstars, ncoefs)
        # However, using the _horner2d function does work.  (Plus, it's probably slightly faster.)
        aberrations_pupil = np.empty((len(self._coef_arrays_field), len(stars)), dtype=float)
        res = np.empty_like(rsqr, dtype=complex)
        temp = np.empty_like(rsqr, dtype=complex)
        for i, ca in enumerate(self._coef_arrays_field):
            galsim.utilities._horner2d(rsqr, r, ca, res, temp)
            aberrations_pupil[i,:] = res.real
        return aberrations_pupil.T

    def getParamsList(self, stars, logger=None):
        """Get params for a list of stars.
        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :returns:   Params  [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...]
                    for each star where all params that are not "z_number" are atmospheric params.
                    Those labelled "opt_something" are the averages of these atmospheric params
                    across the focal plane and those labelled "atm_something" are the deviations
                    from these averages for stars at different points in the focal plane.
        Notes
        -----
        For the i-th star, we have param a_{ik} = a_{ik}^{optics} +
        a_{ik}^{reference} + a_{ik}^{atmo_interp}.  We note that for k < 4,
        a_{ik}^{reference} = 0, k >= 4 a_{ik}^{atmo_interp} = 0. In other
        words, the reference wavefronts have nothing to say about the size, g1,
        g2, and the atmo interp has nothing to say about z4, z5. If no
        reference is provided to the PSF, then that piece is zero, and
        similarly with the atmo_interp.  When we initially produce the PSF, the
        atmo_interp is not fitted, and so calls to this function will skip the
        atmosphere. _enable_atmosphere is set to True when atmo_interp is
        finally fitted.
        """
        logger = LoggerWrapper(logger)
        params = np.zeros((len(stars), self.jmax_pupil + 4), dtype=np.float64)

        logger.debug('Getting aberrations from optical / mean system')
        aberrations_pupil = self._getParamsList_aberrations_field(stars)
        params[:, 4:] += aberrations_pupil

        if self.reference_wavefront or self.higher_order_reference_wavefront:
            if self._caches and self._aberrations_reference_wavefronts.shape[0] == len(stars):
                logger.debug('Getting cached reference wavefront aberrations from all reference '
                             'wavefronts')
                # Use precomputed cache for reference wavefronts.
                # Assumes stars are the same as in cache!
                aberrations_reference_wavefronts = self._aberrations_reference_wavefronts
            else:
                # obtain reference wavefront zernike values for all stars
                if self.reference_wavefront:
                    logger.debug('Getting reference wavefront aberrations')
                    clean_stars = [Star(star.data, None) for star in stars]
                    interp_stars = self.reference_wavefront.interpolateList(clean_stars)
                    aberrations_reference_wavefront = np.array(
                        [star_interpolated.fit.params for star_interpolated in interp_stars])
                else:
                    aberrations_reference_wavefront = None
                if self.higher_order_reference_wavefront:
                    logger.debug('Getting higher order reference wavefront aberrations')
                    aberrations_higher_order_reference_wavefront = np.zeros([len(stars), 34])
                    for s, star in enumerate(stars):
                        aberrations_higher_order_reference_wavefront[s] = \
                            self.get_aberrations_higher_order_reference_wavefront_for_one_star(star)
                else:
                    aberrations_higher_order_reference_wavefront = None
                # combine obtained zernike values from both reference wavefront and higher order
                # reference wavefront
                if (aberrations_reference_wavefront is not None and
                    aberrations_higher_order_reference_wavefront is not None):
                    highest_zernike = int(np.max(
                        self._reference_wavefront_zernikes_list +
                        self._higher_order_reference_wavefront_zernikes_list))
                elif aberrations_reference_wavefront is not None:
                    highest_zernike = np.max(self._reference_wavefront_zernikes_list)
                elif aberrations_higher_order_reference_wavefront is not None:
                    highest_zernike = np.max(self._higher_order_reference_wavefront_zernikes_list)
                aberrations_reference_wavefronts = np.zeros([len(stars), highest_zernike - 3])
                # the shape of this is [number of stars, number of zernikes up to highest zernike
                # requested from one of the reference wavefronts]; note that the 3 is here because
                # we start with defocus (z4) first, fill in zernikes from the reference wavefront;
                # which zernikes you want from the reference wavefront are can be specified in the
                # yaml file
                if aberrations_reference_wavefront is not None:
                    for reference_wavefront_zernike in self._reference_wavefront_zernikes_list:
                        aberrations_reference_wavefronts[:,reference_wavefront_zernike - 4] = \
                            aberrations_reference_wavefront[:,reference_wavefront_zernike - 4]
                        # the 4 is here because we start with defocus (z4)
                # second, fill in zernikes from the higher order reference wavefront;
                # which zernikes you want from the higher order reference wavefront can also be
                # specified in the yaml file
                if aberrations_higher_order_reference_wavefront is not None:
                    for higher_order_reference_wavefront_zernike in \
                            self._higher_order_reference_wavefront_zernikes_list:
                        max_zern = higher_order_reference_wavefront_zernike - 4
                        # the 4 is here because we start with defocus (z4)
                        aberrations_reference_wavefronts[:,max_zern] = \
                            aberrations_higher_order_reference_wavefront[:,max_zern]
            # put aberrations_reference_wavefronts
            # reference wavefronts start at z4 but may not span full range of aberrations used
            n_reference_aberrations = aberrations_reference_wavefronts.shape[1]
            if n_reference_aberrations + 3 < self.jmax_pupil:
                # the 3 is here because we start with defocus (z4)
                params[:, 7:7+n_reference_aberrations] += aberrations_reference_wavefronts
            else:
                # we have more jmax_pupil than reference wavefront
                params[:, 7:] += aberrations_reference_wavefronts[:, :self.jmax_pupil - 3]
                # the 3 is here because we start with defocus (z4)
        # get kolmogorov parameters from atmosphere model, but only if we said so
        if self._enable_atmosphere:
            if self.atmo_interp is None:
                logger.warning('Attempting to retrieve atmospheric interpolations, but we have no '
                               'atmospheric interpolant! Ignoring')
            else:
                logger.debug('Getting atmospheric aberrations')
                # strip star fit
                stars = [Star(star.data, None) for star in stars]
                stars = self.atmo_interp.interpolateList(stars)
                aberrations_atmo_star = np.array([star.fit.params for star in stars])
                params[:, 0:3] += aberrations_atmo_star
        if self.atmosphere_model == 'vonkarman':
            # set the vonkarman outer scale, L0
            params[:, 3] = self.optatmo_psf_kwargs['L0']
        else:
            params[:, 3] = -1.

        return params

    def getParams(self, star):
        """Get params for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:   Params  [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...]
                    where all params that are not "z_number" are atmospheric params. Those labelled
                    "opt_something" are the averages of these atmospheric params across the focal
                    plane and those labelled "atm_something" are the deviations from these averages
                    for stars at different points in the focal plane.
        """
        return self.getParamsList([star])[0]

    def getOpticalProfile(self, star, params):
        """Get the optical part of the profile.

        :param star:    The Star for which to get the optical profile
        :param params:  The parameters array.  cf. `getProfile`.

        :returns: a galsim.OpticalPSF instance
        """
        aberrations = np.zeros(4 + len(params[7:]))
        # fill piston etc with 0
        aberrations[4:] = params[7:] * (700.0/self.optical_psf_kwargs['lam'])
        # aberrations are scaled according to the wavelength

        if hasattr(star, '_last_opt') and np.array_equal(star._last_opt_aber, aberrations):
            # It is not uncommon to repeat this with the same aberrations array.
            # E.g. when fitting something other than the optical part of the profile.
            # So we save the last values used and the resulting OpticalPSF.  Then various
            # back-end calculations don't need to be redone by GalSim.
            opt = star._last_opt
        elif hasattr(self, '_optical_psf_aper_kwargs'):
            # It is more efficient to only make the OpticalPSF aperture once and save it,
            # so if we have already done so, use the modified kwargs with aper.
            opt = galsim.OpticalPSF(aberrations=aberrations,
                                    gsparams=self.gsparams,
                                    **self._optical_psf_aper_kwargs)
            star._last_opt = opt
            star._last_opt_aber = aberrations
        else:
            # This is the first time into this function.  So make OpticalPSF with the regular
            # kwargs, but then make the faster aper version.
            opt = galsim.OpticalPSF(aberrations=aberrations,
                                    gsparams=self.gsparams,
                                    **self.optical_psf_kwargs)
            star._last_opt = opt
            star._last_opt_aber = aberrations
            self._optical_psf_aper_kwargs = self.optical_psf_kwargs.copy()
            for key in ['obcuration', 'circular_pupil', 'nstruts', 'strut_thick', 'strut_angle',
                        'oversampling', 'pad_factor', 'pupil_plane_im', 'pupil_angle',
                        'pupil_plane_scale', 'pupil_plane_size']:
                self._optical_psf_aper_kwargs.pop(key, None)
            self._optical_psf_aper_kwargs['aper'] = opt._aper

        return opt

    def getProfile(self, star, params=None, logger=None):
        """Get galsim profile for a given params

        :param star:    The Star for which to get a profile.
        :param params:  [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...].
                        where all params that are not "z_number" are atmospheric params. Those
                        labelled "opt_something" are the averages of these atmospheric params
                        across the focal plane and those labelled "atm_something" are the deviations
                        from these averages for stars at different points in the focal plane. Note
                        how this means that, for example, atm_size and opt_size are added together
                        for the Kolmogorov model

        :returns: a galsim.GSObject instance
        """
        logger = LoggerWrapper(logger)

        if params is None:
            params = self.getParams(star)

        # optics
        opt = self.getOpticalProfile(star, params)

        # atmosphere
        # add stochastic (labelled "atm") and constant (labelled "opt") pieces together
        size = params[0] + params[4]
        g1 = params[1] + params[5]
        g2 = params[2] + params[6]
        L0 = params[3]
        if L0 < 0:
            atmo = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
            atmo = atmo.dilate(size)
        else:
            kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                      'r0': self.kolmogorov_kwargs['r0'] / size,
                      'L0': L0,}
            atmo = galsim.VonKarman(gsparams=self.gsparams, **kwargs)
        atmo = atmo.shear(g1=g1, g2=g2)

        # convolve together
        prof = galsim.Convolve([opt, atmo], gsparams=self.gsparams)

        return prof

    def drawProfile(self, star, prof, params, use_fit=True, copy_image=True):
        """Generate PSF image for a given star and profile

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :param profile:     A galsim profile
        :param params:      Params associated with profile to put in the star.
        :param use_fit:     Bool [default: True] shift the profile by a star's
                            fitted center and multiply by its fitted flux

        :returns:   Star instance with its image filled with rendered PSF
        """
        # use flux and center properties
        if use_fit:
            prof = prof.shift(star.fit.center) * star.fit.flux
        image, weight, image_pos = star.data.getImage()
        if copy_image:
            image_model = image.copy()
        else:
            image_model = image
        prof.drawImage(image_model, method='auto', center=star.image_pos)
        properties = star.data.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key, None)
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
        return Star(data, fit)

    def draw_fitted_star_given_fitted_image_and_flux(self, x, y, fitted_image, pointing, flux):
        """Creates the appropriate Star instance for a given image (usually a fitted image),
        position, pointing, and flux.

        :param x:               x coordinate of the star's position
        :param y:               y coordinate of the star's position
        :param fitted_image:    Image of star
        :param pointing:        Pointing of star
        :param flux:            Flux of star

        :returns:               Star instance with its image filled
        """
        star = Star.makeTarget(x=x, y=y, image=fitted_image, pointing=pointing, flux=flux)
        return star

    def drawStar(self, star, params=None, copy_image=True):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :param params:      If already known, the parameters for this star. [default: None,
                            in which case getParams(star) will be called.]

        :returns:   Star instance with its image filled with rendered PSF
        """
        if params is None:
            params = self.getParams(star)
        prof = self.getProfile(star, params)
        star = self.drawProfile(star, prof, params, copy_image=copy_image)
        return star

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars.

        Slightly different from drawStar because we get all params at once

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:       List of Star instances with its image filled with rendered PSF
        """
        # get all params at once
        params = self.getParamsList(stars)
        # now step through to make the stars
        stars_drawn = []
        for param, star in zip(params, stars):
            try:
                stars_drawn.append(self.drawProfile(star, self.getProfile(star, param), param,
                                                    copy_image=copy_image))
            except:
                stars_drawn.append(None)

        return stars_drawn

    def _update_optatmopsf(self, optatmo_psf_kwargs={}, logger=None):
        """Update the state of the PSF's field components

        :param optatmo_psf_kwargs:      A dictionary containing the keys we are
                                        updating, like "zPupil004_zFocal001" or
                                        "size" (in this example "size" is
                                        proportional to the average of 1/r0
                                        across the focal plane, r0 being the
                                        Fried parameter)
        :param logger:                  A logger object for logging debug info
        """
        logger = LoggerWrapper(logger)
        if len(optatmo_psf_kwargs) == 0:
            optatmo_psf_kwargs = self.optatmo_psf_kwargs
            keys = self.keys
        else:
            keys = optatmo_psf_kwargs.keys()

        aberrations_changed = False
        for key in keys:
            # skip some keys that often show up in the argument
            if 'error_' in key:
                continue
            elif 'fix_' in key:
                continue
            elif 'min_' in key:
                continue
            elif 'max_' in key:
                continue
            elif 'starcenter_' in key:
                continue

            # size, g1, g2, L0 mean constant atmospheric terms. These are called "opt_size", etc.
            # elsewhere as opposed to "atm_size," etc. which are the deviations from these means.
            if key == 'size':
                pupil_index = 1
                focal_index = 1
            elif key == 'g1':
                pupil_index = 2
                focal_index = 1
            elif key == 'g2':
                pupil_index = 3
                focal_index = 1
            elif key == 'L0':
                pass
            else:
                # zPupil012_zFocal034 is an example of a key
                pupil_index = int(key.split('zPupil')[-1].split('_')[0])
                focal_index = int(key.split('zFocal')[-1])
                if pupil_index < 4:
                    raise ValueError('Not allowed to fit pupil zernike {0} less than {2}, '
                                     'key {1}!'.format(pupil_index, key, 4))
                elif focal_index < 1:
                    raise ValueError('Not allowed to fit focal zernike {0} less than {2} !, '
                                     'key {1}!'.format(focal_index, key, 1))
                elif pupil_index > self.jmax_pupil:
                    raise ValueError('Not allowed to fit pupil zernike {0}, greater than {2}, '
                                     'key {1}!'.format(pupil_index, key, self.jmax_pupil))
                elif focal_index > self.jmax_focal:
                    raise ValueError('Not allowed to fit focal zernike {0} greater than {2} !, '
                                     'key {1}!'.format(focal_index, key, self.jmax_focal))

            if key != 'L0':
                old_value = self.aberrations_field[pupil_index - 1, focal_index - 1]
            else:
                old_value = self.optatmo_psf_kwargs['L0']
            new_value = optatmo_psf_kwargs[key]

            # figure out if we really need to recompute the coef arrays
            if old_value != new_value:
                if 'fix_' + key in optatmo_psf_kwargs:
                    if optatmo_psf_kwargs['fix_' + key]:
                        logger.warning('Warning! Changing key {0} which is designated as fixed '
                                       'from {1} to {2}!'.format(key, old_value, new_value))
                logger.debug('Updating Zernike parameter {0} from {1:+.4e} + {3:+.4e} = '
                             '{2:+.4e}'.format(key, old_value, new_value, new_value - old_value))
                if key != 'L0':
                    self.aberrations_field[pupil_index - 1, focal_index - 1] = new_value
                    aberrations_changed = True
                else:
                    self.optatmo_psf_kwargs['L0'] = new_value

        if aberrations_changed:
            logger.debug('---------- Recomputing field zernike coefficients')
            # One coef_array for each wavefront aberration
            # shape (jmax_pupil, maxn_focal, maxm_focal)
            self._coef_arrays_field = np.array([np.dot(self._noll_coef_field, a)
                                                for a in self.aberrations_field])

    def measure_shape(self, star, return_error=True, logger=None):
        """Measure the shape of a star using the HSM algorithm. Does not go beyond second moments.
        :param star:                Star we want to measure
        :param return_error:        Bool. If True, also measure the error
                                    [default: True]
        :param logger:              A logger object for logging debug info
        :returns:                   Shape (and error if return_error) in
                                    unnormalized basis. Does not go beyond
                                    second moments.
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2,
        #          sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2
        values = star.calculate_moments(logger=logger, errors=return_error)
        errors = np.array(values[6:])
        values = np.array(values[:6])
        logger.debug('Measured Shape is {0}'.format(str(values)))

        if True:
            from .util import hsm
            hsm = hsm(star)
            pix_area = star.data.pixel_area
            values[0] *= hsm[0] * pix_area * star.data.weight.array.mean()
            values[1] += hsm[1]
            values[2] += hsm[2]
            values[3:] *= 2

        if return_error:
            if True:
                errors[0] *= (hsm[0] * pix_area * star.data.weight.array.mean())**2
                errors[3:] *= 4

            logger.debug('Measured Error is {0}'.format(str(errors)))
            return values, np.sqrt(errors)
        else:
            return values

    def measure_shape_third_moments(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm. Goes up to third moments.

        Does not return error.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info

        :returns:                   Shape in unnormalized basis. Goes up
                                    to third moments.
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2
        values = star.calculate_moments(logger=logger, third_order=True)
        values = np.array(values)

        if True:
            # This converts from natural moments to the version Ares had
            # The tests pass without this, but I think that just means they weren't really
            # sufficiently robust.  Probably should just disable this and redo the RF with the
            # new moment definitions.
            from .util import hsm
            hsm = hsm(star)
            values[0] *= hsm[0] * star.data.pixel_area * star.data.weight.array.mean()
            values[1] += hsm[1]
            values[2] += hsm[2]
            values[3:] *= 2

        return values

    def measure_error_third_moments(self, star, logger=None):
        """Measure the shape error of a star using the HSM algorithm. Goes up to third moments.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info

        :returns:                   Shape Error in unnormalized basis. Goes up
                                    to third moments.
        """
        logger = LoggerWrapper(logger)

        # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
        #          sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2
        values = star.calculate_moments(logger=logger, third_order=True, errors=True)
        errors = np.array(values[10:])

        if True:
            from .util import hsm
            hsm = hsm(star)
            errors[0] *= (hsm[0] * star.data.pixel_area * star.data.weight.array.mean())**2
            errors[3:] *= 4

        return np.sqrt(errors)

    def get_aberrations_higher_order_reference_wavefront_for_one_star(self, star):
        """Gets higher order reference wavefront zernike values for star. Then, converts
        them from zout_camera (AOS system) coordinates to zout_sky (Galsim) coordinates,
        inspired by thesis of Chris Davis.

        :param star:            Star instance to find higher order reference wavefront zernike
                                values

        :returns:               Zernike values from higher order reference wavefront
        """
        x_value = star.data.local_wcs._x(star.data['u'], star.data['v'])
        y_value = star.data.local_wcs._y(star.data['u'], star.data['v'])
        x_value = x_value * (15.0/1000.0)
        y_value = y_value * (15.0/1000.0)
        zout_camera = self.higher_order_reference_wavefront.get(x=x_value, y=y_value)
        if len(zout_camera) < 34:
            zout_camera_zeros = np.zeros(34)
            zout_camera_zeros += zout_camera
            zout_camera = zout_camera_zeros
        zout_sky = np.array([
            zout_camera[0], zout_camera[1], -zout_camera[2], zout_camera[4],
            zout_camera[3], -zout_camera[6], -zout_camera[5], zout_camera[7],
            -zout_camera[8], zout_camera[9], zout_camera[10], -zout_camera[11],
            zout_camera[13], zout_camera[12], -zout_camera[15], -zout_camera[14],
            zout_camera[17], zout_camera[16], zout_camera[18], zout_camera[19],
            -zout_camera[20], -zout_camera[21], zout_camera[22], zout_camera[23],
            -zout_camera[24], zout_camera[26], zout_camera[25], -zout_camera[28],
            -zout_camera[27], zout_camera[30], zout_camera[29], -zout_camera[32],
            -zout_camera[31], zout_camera[33]])
        return zout_sky

    def measure_shape_orthogonal(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.

        Goes up to third moments plus orthogonal radial moments up to eighth moments.
        Does not return error.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info

        :returns:   Shape in unnormalized basis. Goes up to third moments plus orthogonal radial
                    moments up to eighth moments
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, xi4, xi6, xi8
        values = star.calculate_moments(logger=logger, third_order=True, radial=True)
        values = np.array(values)

        if True:
            # This converts from natural moments to the version Ares had
            # The tests pass without this, but I think that just means they weren't really
            # sufficiently robust.  Probably should just disable this and redo the RF with the
            # new moment definitions.
            from .util import hsm
            hsm = hsm(star)
            values[0] *= hsm[0] * star.data.pixel_area * star.data.weight.array.mean()
            values[1] += hsm[1]
            values[2] += hsm[2]
            values[3:] *= 2

        # flux is underestimated empirically
        # MJ: I don't think this ^ is true.  But then, it isn't expected to return the real flux.
        #     For a Gaussian, M00 is flux / (4 pi sigma^2).
        #     Or for your version, it is flux^2 pixel_scale^2 mean(w) / (4 pi sigma^2).
        #     So probably that just happened to come out as 0.92 for whatever test you did.
        #values[0] = values[0] / 0.92

        return values

    def measure_error_orthogonal(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.

        Goes up to third moments plus orthogonal radial moments up to eighth moments.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info

        :returns:   Shape Error in unnormalized basis. Goes up to third moments plus orthogonal
                    radial moments up to eighth moments.  to fourth moments.
        """
        logger = LoggerWrapper(logger)

        # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
        #          sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2,
        #          sigma_orth4, sigma_orth6, sigma_orth8
        values = star.calculate_moments(logger=logger, third_order=True, radial=True, errors=True)
        errors = np.array(values[13:])

        if True:
            from .util import hsm
            hsm = hsm(star)
            errors[0] *= (hsm[0] * star.data.pixel_area * star.data.weight.array.mean())**2
            errors[3:] *= 4

        return np.sqrt(errors)

    @property
    def regr_dict(self):
        if not hasattr(self, '_regr_dict'):
            # load up random forest model (used only in "random_forest" mode)
            self._regr_dict = {}
            for m, moment in enumerate(np.array(
                    ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"])):
                with open("{0}/random_forest_shapes_model_{1}.pickle".format(
                        self.random_forest_shapes_model_pickles_location, moment), 'rb') as f:
                    try:
                        regr = pickle.load(f)
                    except:
                        raise OSError('Random forest model pickle failed to load.')
                    version = regr.__getstate__()['_sklearn_version']
                    if version != sklearn.__version__:
                        logger.error('sklearn version changed from the one used to make the '
                                     'random forest file.  This might not work.')
                    self._regr_dict[moment] = regr
        return self._regr_dict

    def fit_optics(self, stars, shapes, errors, mode, logger=None, ftol=1.e-7, **kwargs):
        """Fit interpolated PSF model to star shapes.

        It is important to note that although this fit is referred to as the "optical" fit we
        still fit the average of the atmospheric parameters across the focal plane here. Finding
        the deviation of these atmospheric parameters from the average is then done later in the
        fit_atmosphere() function. For example, there is an atmospheric parameter known as the
        "size" parameter (which is proportional to 1/r0 with r0 being the Fried parameter) whose
        average we fit in this function. Finding the deviation of this size parameter from the
        average is then done later in the fit_atmosphere() function.

        :param stars:       A list of Stars
        :param shapes:      A list of premeasured Star shapes
        :param errors:      A list of premeasured Star shape errors
        :param mode:        Parameter mode ['random_forest', 'shape', 'pixel']. Dictates which
                            residual function we use.
        :param logger:      A logger object for logging debug info.  [default: None]
        :param ftol:        One of the convergence criteria for the optical fit. Based on relative
                            change in the chi after an iteration. Smaller ftol is stricter and
                            takes longer to converge. Not used in "random_forest" mode.
                            [default: 1.e-7]
        Notes
        -----
        This model leverages an initial random forest model fit and
        fit on the average of the atmospheric "size" parameter across
        the focal plane. The optical model is specified at given focal
        plane coordinates [u, v] by a sum over Zernike polynomials:
        a_{ik} (u_i, v_i) = \sum_{\ell} b_{k \ell} Z_{\ell} (u_i, v_i)
                            + a^{reference}_{k}(u_i, v_i)
        Having measured the shapes of stars, with errors \sigma_{ij}, we
        then find the optimal b_{k \ell}
        """
        import scipy
        from .util import estimate_cov_from_jac

        logger = LoggerWrapper(logger)
        logger.info("Start fitting Optical in {0} mode for {1} stars".format(mode, len(stars)))

        # save reference wavefronts' values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_caches(stars, logger=logger)

        fit_keys = [key for key in self.keys
                    if not self.optatmo_psf_kwargs.get('fix_'+key,True)]
        params = [self.optatmo_psf_kwargs[key] for key in fit_keys]

        if mode == 'random_forest':
            results = scipy.optimize.least_squares(
                    self._fit_random_forest_residual, params,
                    args=(stars, fit_keys, shapes, errors, self.regr_dict, logger,),
                    diff_step=1e-5, ftol=1.e-3, xtol=1.e-4)
        elif mode == 'shape':
            results = scipy.optimize.least_squares(
                    self._fit_optics_residual, params,
                    args=(stars, fit_keys, shapes, errors, logger,),
                    diff_step=1e-5, ftol=1.e-3, xtol=1.e-4)
        elif mode == 'pixel':
            for i in range(len(stars)):
                params.append(stars[i].center[0])
                params.append(stars[i].center[1])
            results = scipy.optimize.least_squares(
                    self._fit_optics_pixel_residual, params,
                    jac=self._fit_optics_pixel_jac,
                    args=(stars, fit_keys, logger,),
                    diff_step=1e-5, ftol=1.e-3, xtol=1.e-4)
        else:
            raise KeyError('Unrecognized fit mode: {0}'.format(mode))

        logger.info('Results from {0} optical fit:'.format(mode))
        logger.info(results.message)
        if not results.success:
            raise RuntimeError("fit failed")
        # Estimate covariance matrix from jacobian
        cov = estimate_cov_from_jac(results.jac)
        for i, key in enumerate(fit_keys):
            self.optatmo_psf_kwargs[key] = results.x[i]
            self.optatmo_psf_kwargs['error_' + key] = cov[i,i]

        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)

        # remove saved values from the reference wavefronts' caches when we are done with the fit
        if self.reference_wavefront: self._delete_caches(logger=logger)

    def fit_size(self, stars, logger=None, **kwargs):
        """Adjusts the optics size parameter found in the random forest fit.

        The "optics" size is the average of this across the focal plane, whereas "atmospheric"
        size is the deviation from this average at different points in the focal plane.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)
        import scipy
        logger.info("Start fitting Optical fit of size alone")

        # Get the current parameters.  Everything but opt_size is constant here.
        opt_params = self.getParamsList(stars)

        # Make sure the stars have a decent flux, centroid estimate
        stars = [self.reflux(star, param, logger=logger)
                 for param, star in zip(opt_params,stars)]

        # get the optical parts of the profiles
        optical_profiles = []
        for i, star in enumerate(stars):
            params = opt_params[i]
            opt = self.getOpticalProfile(star, params)
            optical_profiles.append(opt)

        # do size fit
        results = scipy.optimize.least_squares(
                self._fit_size_residual,
                [np.log(self.optatmo_psf_kwargs['size'])],
                jac=self._fit_size_jac,
                args=(stars, opt_params, optical_profiles, logger,),
                diff_step=1.e-4, ftol=1.e-3, xtol=1.e-4)

        size = np.exp(results.x[0])
        logger.info("finished optics size fit: size = %s",size)
        logger.info(results.message)
        self.optatmo_psf_kwargs['size'] = size
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)


    # Note: This is not currently being used. Instead, the atmospheric fitting is currently being
    # done in the PIFF fitting pipeline itself. As a result, it has not been updated in a while
    # and it is not known if it is compatible with the current version of the PIFF fitting pipeline.
    def fit_atmosphere(self, stars, chisq_threshold=0.1, max_iterations=30, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of
        operations (will also reject with outliers). We start here with the
        optical fit parameters and the average values of the atmospheric
        parameters found in the optical fit and hold those fixed. We float only
        the deviation of these atmospheric parameters from the average here.

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
            logger.info("Setting _enable_atmosphere == False. Was {0}".format(
                self._enable_atmosphere))
            self._enable_atmosphere = False

        # fit models
        logger.info("Initial Fitting atmo model")
        params = self.getParamsList(stars)
        model_fitted_stars = []
        for star_i, star in zip(range(len(stars)), stars):
            try:
                model_fitted_star = self.fit_model(star, params=params[star_i], logger=logger)
                model_fitted_stars.append(model_fitted_star)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.warning('{0}'.format(str(e)))
                logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring '
                               'star in atmosphere fit'.format(star_i))

        logger.debug('Stripping star fit params down to just atmosphere params for fitting with '
                     'the atmo_interp')
        stripped_stars = self.stripStarList(model_fitted_stars, logger=logger)
        stars = stripped_stars

        if self.atmo_mad_outlier:
            logger.info('Stripping MAD outliers from star fit params of atmosphere')
            params = np.array([s.fit.params for s in stars])
            madp = np.abs(params - np.median(params, axis=0)[np.newaxis])
            madcut = np.all(madp <= 5 * 1.48 * np.median(madp)[np.newaxis] + 1e-8, axis=1)
            mad_stars = []
            for si, s, keep in zip(range(len(stars)), stars, madcut):
                if keep:
                    mad_stars.append(s)
                else:
                    logger.debug('Removing star {0} based on MAD. params are {1}'.format(
                                 si, str(params[si])))
            if len(mad_stars) != len(stars):
                logger.info('Stripped stars from {0} to {1} based on 5sig MAD cut'.format(
                            len(stars), len(mad_stars)))
            stars = mad_stars
        fitted_model_params = [s.fit.params for s in stars]
        fitted_model_params_var = [s.fit.params_var for s in stars]

        # fit interpolant
        logger.info("Initializing atmo interpolator")
        stars = self.atmo_interp.initialize(stars, logger=logger)

        logger.info("Fitting atmo interpolant")
        # Begin iterations.  Very simple convergence criterion right now.
        if self.outliers is None:
            # with no outliers, no need to do the below cycle
            self.atmo_interp.solve(stars, logger=logger)
        else:
            # get the params again after all the stars
            oldchisq = 0.
            for iteration in range(max_iterations):
                nremoved = 0
                logger.info("Iteration %d: Fitting %d stars", iteration+1, len(stars))

                #####
                # outliers
                #####
                # solve atmo_interp with the reduced set of stars
                stars = self.atmo_interp.initialize(stars, logger=logger)
                self.atmo_interp.solve(stars, logger=logger)

                # create new stars including atmo interp
                params = self.getParamsList(stars)
                stars_interp = self.atmo_interp.interpolateList(stars)
                aberrations_atmo_star = np.array([star.fit.params for star in stars_interp])
                params[:, 0:3] += aberrations_atmo_star

                # refluxing star and get chisq
                refluxed_stars = [self.reflux(star, param, logger=logger)
                                  for param, star in zip(params,stars_interp)]

                # put back into the refluxed stars the fitted model params. This way when outliers
                # returns the new list, we won't have to refit those parameters (which will be the
                # same as earlier)
                reparam_stars = []
                for params, params_var, star in zip(fitted_model_params, fitted_model_params_var,
                                                    refluxed_stars):
                    fit = StarFit(params, params_var=params_var, flux=star.fit.flux,
                                  center=star.fit.center,
                                  chisq=star.fit.chisq, dof=star.fit.dof,
                                  alpha=star.fit.alpha, beta=star.fit.beta)
                    new_star = Star(star.data, fit)
                    reparam_stars.append(new_star)

                # Perform outlier rejection
                logger.debug("             Looking for outliers")
                nonoutlier_stars, nremoved1 = self.outliers.removeOutliers(
                        reparam_stars, logger=logger)
                if nremoved1 == 0:
                    logger.debug("             No outliers found")
                else:
                    logger.info("             Removed %d outliers", nremoved1)
                nremoved += nremoved1

                stars = nonoutlier_stars

                chisq = np.sum([s.fit.chisq for s in stars])
                dof   = np.sum([s.fit.dof for s in stars])
                logger.info("             Total chisq = %.2f / %d dof", chisq, dof)

                # Very simple convergence test here:
                # Note, the lack of abs here means if chisq increases, we also stop.
                # Also, don't quit if we removed any outliers.
                if (nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < chisq_threshold*dof):
                    break
                oldchisq = chisq

            else:
                logger.warning("PSF fit did not converge.  Max iterations = %d reached.",
                               max_iterations)

        return model_fitted_stars, stars

    def fit_model(self, star, params, logger=None):
        """Fit model to star's pixel data.

        :param star:        A Star instance
        :param params:      An array of initial star parameters like one would
                            get from getParams
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance and results, with updated flux,
                            center, chisq, dof, and fit params and params_var
        """
        from .util import estimate_cov_from_jac

        logger = LoggerWrapper(logger)

        # Make the optical profile, constant for this part of the fit.
        optical_profile = self.getOpticalProfile(star, params)

        # Start with the current parameters
        flux = star.fit.flux
        if flux == 1.:
            # a pretty reasonable first guess is to just take the sum of the pixels
            flux = star.image.array.sum()
        du, dv = star.fit.center
        fit_size, fit_g1, fit_g2 = params[0:3]

        # acquire the values of opt_size, opt_g1, opt_g2, and opt_L0
        opt_L0, opt_size, opt_g1, opt_g2 = params[3:7]
        fit_size += opt_size  # Do fits with full size, g1, g2
        fit_g1 += opt_g1
        fit_g2 += opt_g2

        # parameters to fit:
        # Use log(flux) and log(size), so we don't have to worry about going negative
        fit_params = [np.log(flux), du, dv, np.log(fit_size), fit_g1, fit_g2]

        # Find the solution
        results = scipy.optimize.least_squares(
                self._fit_model_residual, fit_params,
                args=(star, optical_profile, opt_L0, logger,),
                ftol=1.e-3, xtol=1.e-4)

        logger.debug(results.message)
        if not results.success:
            raise RuntimeError('Not successful fit')

        g1, g2 = results.x[4:6]
        if np.abs(g1) > 0.4 or np.abs(g2) > 0.4:
            raise RuntimeError('Bad fit.  g1,g2 = %f,%f is probably unphysical'%(g1,g2))

        fit_params = np.zeros_like(params)
        fit_params[0] = np.exp(results.x[3])  # size = exp(logsize)
        fit_params[1:3] = results.x[4:6]      # g1, g2
        fit_params[0:3] -= params[4:7]        # subtract off opt_* part
        fit_params[3:] = params[3:]           # fill in the other params that were constant here

        # Estimate covariance matrix from jacobian
        cov = estimate_cov_from_jac(results.jac)
        params_var = np.zeros_like(fit_params)
        params_var[0:3] = cov.diagonal()[3:6]
        params_var[0] *= fit_params[0]**2     # var(size) = size**2 * var(logsize)

        # Return results as a new Star instance
        logflux, du, dv = results.x[0:3]
        flux = np.exp(logflux)
        center = (du, dv)
        chisq = results.cost * 2
        dof = len(results.fun) - len(results.x)
        fit = StarFit(fit_params, params_var=params_var, flux=flux, center=center,
                      chisq=chisq, dof=dof)
        star_fit = Star(star.data, fit)
        return star_fit

    def stripStarList(self, stars, logger=None):
        """take star fits and strip fit params to just the first three
        parameters, which correspond to the atmospheric terms. Keep flux and
        center but get rid of everything else
        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        :returns:               A list of stars with only num_keep fit params
        """
        num_keep = 3
        new_stars = []
        for star_i, star in enumerate(stars):
            try:
                fit_params = star.fit.params
                new_fit_params = fit_params[:num_keep]
            except RuntimeError:
                logger.debug("Star {0} has no fit params".format(star_i))
                new_fit_params = None
            try:
                fit_params_var = star.fit.params_var
                new_fit_params_var = fit_params_var[:num_keep]
            except RuntimeError:
                logger.debug("Star {0} has no fit params_var".format(star_i))
                new_fit_params_var = None
            new_fit = StarFit(new_fit_params, params_var=new_fit_params_var,
                              flux=star.fit.flux, center=star.fit.center)
            new_star = Star(star.data, new_fit)
            new_stars.append(new_star)
        return new_stars

    def reflux(self, star, params=None, logger=None):
        """Fit the Model to the star's data, varying only the flux and center.

        :param star:        A Star instance
        :param params:      If already known, the parameters for this star. [default: None,
                            in which case getParams(star) will be called.]
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance, with updated flux, center, chisq, dof
        """

        def _resid(x, psf, prof, image, weight, image_pos, model):
            # residual as a function of x = (flux, du, dv)
            flux = np.exp(x[0])
            center = x[1:]
            prof = prof.shift(center) * flux
            prof.drawImage(model, method='auto', center=image_pos)
            return (np.sqrt(weight.array) * (model.array - image.array)).flatten()

        logger = LoggerWrapper(logger)
        if params is None:
            params = self.getParams(star)

        # Make a new Star to use as a temp value in _resid.
        # We'll also use this as the return value, but it's ok to modify in the resid function.
        star = Star(star.data, star.fit.copy())

        # Use current flux, center as initial guess for x0.
        logflux = np.log(star.fit.flux)
        if logflux == 0.:
            # Then initial flux is exactly 1.0.  Probably not a good guess.
            # Use the image sum as a better initial guess.
            logflux = np.log(np.sum(star.data.image.array))
        du, dv = star.fit.center
        prof = self.getProfile(star, params, logger=logger)
        image, weight, image_pos = star.data.getImage()
        model = image.copy()  # Temporary image for drawing the model image.
        results = scipy.optimize.least_squares(_resid, x0=[logflux, du, dv],
                                               args=(self, prof, image, weight, image_pos, model),
                                               diff_step=1.e-4, ftol=1.e-3, xtol=1.e-4)

        # Update return value with fit results
        star.fit.flux = np.exp(results.x[0])
        star.fit.center = results.x[1:]
        star.fit.chisq = results.cost*2
        return star

    def _fit_random_forest_residual(self, params, stars, fit_keys, shapes, shape_errors,
                                    regr_dictionary, logger=None):
        """Residual function for fitting optics via random forest model.

        This is what is done in "random_forest" mode.

        :param params:          Numpy array with parameters to fit.  First parameters for each
                                key in fit_keys, then (u,v) for each star.
        :param stars:           A list of Stars
        :param fit_keys:        Key names for the initial values in params array.
        :param shapes:          A list of premeasured Star shapes
        :param errors:          A list of premeasured Star shape errors
        :param regr_dictionary: A dictionary containing the random forest
                                models used to get the stars' moments based on their fit parameters.
        :param logger:          A logger object for logging debug info.
                                [default: None]

        :returns chi:           Chi of observed shapes to model shapes
        """
        logger = LoggerWrapper(logger)
        # update psf
        n_opt = len(fit_keys)
        self._update_optatmopsf(dict(zip(fit_keys, params[:n_opt])), logger=logger)

        # get star params
        params_all = self.getParamsList(stars)
        param_values_all_stars = params_all[:,4:4+11]
        number_of_rows, number_of_columns = param_values_all_stars.shape
        if number_of_columns < 11:
            param_values_all_stars_copy = copy.deepcopy(param_values_all_stars)
            param_values_all_stars = np.zeros((number_of_rows,11))
            param_values_all_stars[:,:number_of_columns] = param_values_all_stars_copy

        # generate the stars' moments using random forest model and the fit parameters of the stars
        #note: only up to third moments used for the random forest fit
        shapes_model_list = []
        for m, moment in enumerate(np.array(
                ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"])):
            regr = regr_dictionary[moment]
            shapes_model_list.append(regr.predict(param_values_all_stars))
        shapes_model = np.column_stack(tuple(shapes_model_list))
        shape_weights = self._shape_weights[:7]

        # calculate chi. Exclude measurements of flux and centroids
        shapes = shapes[:, 3:10]
        errors = shape_errors[:, 3:10]
        chi = (shape_weights[None] * (shapes_model - shapes) / errors).flatten() #chi is
        logger.debug('Current Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        # chi is a one-dimensional numpy array, containing
        # moment_weight*((moment_model-moment)/moment_error)
        # for up to all moments up to third moments, for all stars
        # the model moments in this case are based on what the random forest model
        # returns for a model star with a given set of fit parameters
        return chi

    def _fit_size_residual(self, x, stars, opt_params, optical_profiles, logger=None):
        """Residual function for fitting the optics size parameter to the
        observed e0 moment. The size parameter is proportional to 1/r0, r0
        being the Fried parameter. The "optics" size is the average of this
        across the focal plane, whereas "atmospheric" size is the deviation
        from this average at different points in the focal plane.

        :param x:               numpy array with [logsize]
        :param stars:           A list of Stars
        :param opt_params:      The full parameter list
        :param optical_profiles:    A list of optical profiels, constant during this fit
        :param logger:          A logger object for logging debug info.
                                [default: None]

        :returns chi:           Chi of observed e0 to model e0
        """
        logger = LoggerWrapper(logger)
        opt_size = np.exp(x[0])

        chis = []
        atmo_profiles = []
        for i, star in enumerate(stars):

            # Finish making the profile using optical_profile and the given opt_params
            # for this star
            params = opt_params[i]
            size = params[0] + opt_size
            g1 = params[1] + params[5]
            g2 = params[2] + params[6]
            L0 = params[3]
            if L0 < 0:
                atmo = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
                atmo = atmo.dilate(size)
            else:
                kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                          'r0': self.kolmogorov_kwargs['r0'] / size,
                          'L0': L0,}
                atmo = galsim.VonKarman(gsparams=self.gsparams, **kwargs)
            atmo = atmo.shear(g1=g1, g2=g2)
            atmo_profiles.append(atmo)

            prof = galsim.Convolve([optical_profiles[i], atmo], gsparams=self.gsparams)
            prof = prof.shift(star.fit.center)

            # Draw model
            image, weight, image_pos = star.data.getImage()
            model = prof.drawImage(image.copy(), method='auto', center=image_pos)

            # Calculate chi for this star
            image_flux = np.sum(image.array * weight.array)
            model_flux = np.sum(model.array * weight.array)
            model *= image_flux/model_flux  # Don't worry about flux differences
            chi = (np.sqrt(weight.array) * (model.array - image.array)).flatten()
            chis.append(chi)

        chi = np.concatenate(chis)

        # Save some things for possible use by jacobian
        self._fit_size_cache_params = x
        self._fit_size_cache_chis = chis
        self._fit_size_cache_chi0 = chi
        self._fit_size_cache_atmo_profiles = atmo_profiles

        chisq = np.sum(chi**2)
        logger.info("size = %s: chisq = %s",opt_size, chisq)
        return chi

    def _fit_size_jac(self, x, stars, opt_params, optical_profiles, logger=None):
        """Jacobian calculation for _fit_size_residual.

        :param x:               numpy array with [logsize]
        :param stars:           A list of Stars
        :param opt_params:      The full parameter list
        :param optical_profiles:    A list of optical profiels, constant during this fit
        :param logger:          A logger object for logging debug info.
                                [default: None]

        :returns jac:   Jacobian array, d(chi)/d(logsize)
        """
        logger = LoggerWrapper(logger)
        opt_size = np.exp(x[0])

        if not np.array_equal(x, self._fit_size_cache_params):
            # This sets the cache items if they aren't already correct
            self._fit_size_residual(x, stars, opt_params, optical_profiles)

        chis = self._fit_size_cache_chis
        chi0 = self._fit_size_cache_chi0
        atmo_profiles = self._fit_size_cache_atmo_profiles

        jac = np.zeros((len(chi0), 1), dtype=float)

        # Array of the start/stop indices in chi0 for each star:
        indx = np.zeros(len(chis)+1, dtype=int)
        indx[1:] = np.cumsum([len(c) for c in chis])

        dlogsize = 1.e-4

        for i, star in enumerate(stars):

            atmo = atmo_profiles[i].dilate(1.+dlogsize)

            prof = galsim.Convolve([optical_profiles[i], atmo], gsparams=self.gsparams)
            prof = prof.shift(star.fit.center)

            # Draw model
            image, weight, image_pos = star.data.getImage()
            model = prof.drawImage(image.copy(), method='auto', center=image_pos)

            # Calculate chi for this star
            image_flux = np.sum(image.array * weight.array)
            model_flux = np.sum(model.array * weight.array)
            model *= image_flux/model_flux  # Don't worry about flux differences
            chi = (np.sqrt(weight.array) * (model.array - image.array)).flatten()

            jac[indx[i]:indx[i+1],0] = (chi-chi0[indx[i]:indx[i+1]]) / dlogsize

        return jac

    def _fit_optics_residual(self, params, stars, fit_keys, shapes, shape_errors, logger=None):
        """Residual function for fitting the optical fit parameters and the average values of the
        atmospheric fit parameters to the observed shapes.

        :param params:          Numpy array with parameters to fit.  First parameters for each
                                key in fit_keys, then (u,v) for each star.
        :param stars:           A list of Stars
        :param fit_keys:        Key names for the initial values in params array.
        :param shapes:          A list of premeasured Star shapes
        :param shape_errors:    A list of premeasured Star shape errors
        :param logger:          A logger object for logging debug info.  [default: None]

        :returns chi:             Chi of observed shapes to model shapes

        Notes
        -----
        This is done by forward modeling the PSF and measuring its shape via HSM
        """
        logger = LoggerWrapper(logger)
        logger.debug('start residual: current params = %s',params)
        # update psf
        n_opt = len(fit_keys)
        self._update_optatmopsf(dict(zip(fit_keys, params[:n_opt])), logger=logger)

        # get optical params
        opt_params = self.getParamsList(stars)

        # measure their shapes and calculate chi
        chi = np.array([])
        for i, star in enumerate(stars):
            params = opt_params[i]
            shape = shapes[i]
            error = shape_errors[i]

            try:
                # get profile; modify based on flux and shifts
                profile = self.getProfile(star, params)

                # measure final shape
                star_model = self.drawProfile(star, profile, params)
                shape_model = self.measure_shape_orthogonal(star_model)
                if np.any(shape_model != shape_model):
                    logger.warning('Star {0} returned nan shape'.format(i))
                    logger.warning('Parameters are {0}'.format(str(params)))
                    logger.warning('Input parameters are {0}'.format(str(params)))
                    logger.warning('Filling with zero chi')
                    shape_model = shape
            except (ModelFitError, RuntimeError) as e:
                logger.warning(str(e))
                logger.warning('Star {0}\'s model failed to be drawn and measured.'.format(i))
                logger.warning('Parameters are {0}'.format(str(params)))
                logger.warning('Input parameters are {0}'.format(str(params)))
                logger.warning('Filling with zero chi')
                shape_model = shape

            # don't care about flux, du, dv here
            chi_i = self._shape_weights * (((shape_model - shape) / error)[3:])
            chi = np.hstack((chi, chi_i))

        self.final_optical_chi = chi
        # chi is a one-dimensional numpy array, containing
        # moment_weight*((moment_model-moment)/moment_error)
        # for all moments, for all stars
        chisq = np.sum(chi**2)
        logger.info("chisq = %s",chisq)
        return chi

    # not necessarily set up to work with vonkarman atmosphere; also not currently used
    # because too slow
    def _fit_optics_pixel_residual(self, params, stars, fit_keys, logger=None):
        """Residual function for fitting all stars using pixel-based residuals.

        :param params:      Numpy array with parameters to fit.  First parameters for each
                            key in fit_keys, then (u,v) for each star.
        :param stars:       A list of Stars
        :param fit_keys:    Key names for the initial values in params array.
        :param logger:      A logger object for logging debug info.
                            [default: None]

        :returns chi:   Chi of observed pixels of all stars to model pixels after fitting for flux,
                        centering, and atmospheric size / ellipticity
        """
        logger = LoggerWrapper(logger)
        logger.debug('start residual: current params = %s',params)
        # update psf
        n_opt = len(fit_keys)
        self._update_optatmopsf(dict(zip(fit_keys, params[:n_opt])), logger=logger)

        # get optical params
        opt_params = self.getParamsList(stars)

        size = self.optatmo_psf_kwargs['size']
        if hasattr(self, '_fit_optical_cache_size') and self._fit_optical_cache_size == size:
            make_mask = False
            masks = self._fit_optical_cache_masks
        else:
            make_mask = True
            masks = [None] * len(stars)

        chis = []
        for i, star in enumerate(stars):
            opt_param_i = opt_params[i]

            # get profile; modify based on flux and shifts
            prof = self.getProfile(star, opt_param_i)

            prof = prof.shift(params[n_opt + 2*i], params[n_opt + 2*i + 1])

            # Draw model
            image, weight, image_pos = star.data.getImage()
            model = prof.drawImage(image.copy(), method='auto', center=image_pos)

            # Most of the information about aberrations is on the edge of the profile.
            # The center is dominated by the Kolmogorov seeing.
            # So limit our chisq to 1 < r < 2.
            # TODO: This range is kind of empirical based on test_optics_and_fit_model()
            #       Should probably do a more systematic test to figure out the optical range.
            #       Also maybe better to use a weight as a function of r?
            if make_mask:
                _, _, u, v = star.data.getDataVector(include_zero_weight=True)
                rsq = (u**2 + v**2) / size**2
                mask = (1 < rsq) & (rsq < 4)
                masks[i] = mask
            else:
                mask = masks[i]

            data = image.array.ravel()[mask]
            weight = weight.array.ravel()[mask]
            model = model.array.ravel()[mask]

            # Calculate chi for this star
            image_flux = np.sum(data * weight)
            model_flux = np.sum(model * weight)
            model *= image_flux/model_flux  # Don't worry about flux differences
            chi = np.sqrt(weight) * (model - data)
            chis.append(chi)

        if make_mask:
            self._fit_optical_cache_size = size
            self._fit_optical_cache_masks = masks

        chi = np.concatenate(chis)

        # Save some things for possible use by jacobian
        self._fit_optical_cache_params = params
        self._fit_optical_cache_optparams = opt_params
        self._fit_optical_cache_chis = chis
        self._fit_optical_cache_chi0 = chi

        chisq = np.sum(chi**2)
        logger.info("chisq = %s",chisq)
        return chi

    def _fit_optics_pixel_jac(self, params, stars, fit_keys, logger=None):
        """Find jacobian corresponding to _fit_optics_pixel_residual.

        :param params:      Numpy array with parameters to fit.  First parameters for each
                            key in fit_keys, then (u,v) for each star.
        :param stars:       A list of Stars
        :param fit_keys:    Key names for the initial values in params array.
        :param logger:      A logger object for logging debug info.
                            [default: None]

        :returns jac:   Jacobian array, d(chi)/d(params)
        """
        logger = LoggerWrapper(logger)
        logger.debug('start jacobian: current params = %s',params)
        # update psf
        n_opt = len(fit_keys)

        if not np.array_equal(params, self._fit_optical_cache_params):
            # This sets the cache items if they aren't already correct
            self._fit_optics_pixel_residual(params, stars, fit_keys)

        opt_params = self._fit_optical_cache_optparams
        chis = self._fit_optical_cache_chis
        chi0 = self._fit_optical_cache_chi0
        masks = self._fit_optical_cache_masks

        jac = np.zeros((len(chi0), len(params)), dtype=float)

        # Array of the start/stop indices in chi0 for each star:
        indx = np.zeros(len(chis)+1, dtype=int)
        indx[1:] = np.cumsum([len(c) for c in chis])

        # First the ones that don't require updating opt_params
        for i, star in enumerate(stars):
            opt_param_i = opt_params[i]

            # get profile; modify based on flux and shifts
            prof = self.getProfile(star, opt_param_i)

            j_u = n_opt + 2*i
            j_v = n_opt + 2*i + 1

            # Do derivatives for each of u and v params:
            duv = 1.e-5
            image, weight, image_pos = star.data.getImage()
            mask = masks[i]
            data = image.array.ravel()[mask]
            weight = weight.array.ravel()[mask]
            image_flux = np.sum(data * weight)

            # dchi/duc
            cen = (params[j_u] + duv, params[j_v])
            model_image = prof.shift(cen).drawImage(image.copy(), method='auto', center=image_pos)
            model = model_image.array.ravel()[mask]
            model_flux = np.sum(model * weight)
            model *= image_flux/model_flux
            chi = np.sqrt(weight) * (model - data)
            jac[indx[i]:indx[i+1],j_u] = (chi-chi0[indx[i]:indx[i+1]]) / duv

            # dchi/dvc
            cen = (params[j_u], params[j_v] + duv)
            prof.shift(cen).drawImage(model_image, method='auto', center=image_pos)
            model = model_image.array.ravel()[mask]
            model_flux = np.sum(model * weight)
            model *= image_flux/model_flux
            chi = np.sqrt(weight) * (model - data)
            jac[indx[i]:indx[i+1],j_v] = (chi-chi0[indx[i]:indx[i+1]]) / duv

        # For the rest, just call the residual function, since all of opt_params will change
        dp = 1.e-5
        for j in range(n_opt):
            p = params.copy()
            p[j] += dp
            jac[:,j] = (self._fit_optics_pixel_residual(p, stars, fit_keys) - chi0) / dp
        return jac


    def _fit_model_residual(self, params, star, optical_profile, L0, logger=None):
        """Residual function for fitting individual profile parameters to observed pixels.

        :param params:          numpy array of fit parameters: [logflux, du, dv, logsize, g1, g2]
        :param star:            A Star instance.
        :param optical_profile: The optical part of the profile.
        :param L0               L0 (== -1 for Kolmogorov)
        :param logger:          A logger object for logging debug info.  [default: None]

        :returns chi: Chi of observed pixels to model pixels
        """
        logger = LoggerWrapper(logger)

        logflux, du, dv, logsize, g1, g2 = params
        flux = np.exp(logflux)
        size = np.exp(logsize)

        if L0 < 0:
            atmo = galsim.Kolmogorov(gsparams=self.gsparams, **self.kolmogorov_kwargs)
            atmo = atmo.dilate(size)
        else:
            kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                      'r0': self.kolmogorov_kwargs['r0'] / size,
                      'L0': L0,}
            atmo = galsim.VonKarman(gsparams=self.gsparams, **kwargs)
        atmo = atmo.shear(g1=g1, g2=g2)

        # convolve together
        prof = galsim.Convolve([optical_profile, atmo], gsparams=self.gsparams)
        prof = prof.shift(du, dv) * flux

        # calculate chi
        image, weight, image_pos = star.data.getImage()
        image_model = prof.drawImage(image.copy(), method='auto', center=star.image_pos)
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

    def _create_caches(self, stars, logger=None):
        """Save aberrations from reference wavefronts. This is useful if we want
        to keep calling getParams but we aren't changing the positions of the
        stars. We save the results, so we can call up the same aberrations from
        the reference wavefronts quickly.
        :param stars:   A list of stars
        :param logger:  A logger object for logging debug info [default: None]
        """
        cacheless_counter = 0
        # obtain reference wavefront zernike values for all stars
        if self.reference_wavefront:
            logger.debug('Caching reference aberrations')
            self._caches = True
            clean_stars = [Star(star.data, None) for star in stars]
            interp_stars = self.reference_wavefront.interpolateList(clean_stars)
            aberrations_reference_wavefront = np.array(
                [star_interpolated.fit.params for star_interpolated in interp_stars])
        else:
            logger.debug('Cache called, but no reference wavefront. Skipping')
            cacheless_counter = cacheless_counter + 1
            aberrations_reference_wavefront = None
        # obtain higher order reference wavefront zernike values for all stars
        if self.higher_order_reference_wavefront:
            logger.debug('Caching higher order reference aberrations')
            self._caches = True
            aberrations_higher_order_reference_wavefront = np.zeros([len(stars), 34])
            for s, star in enumerate(stars):
                aberrations_higher_order_reference_wavefront[s] = \
                    self.get_aberrations_higher_order_reference_wavefront_for_one_star(star)
        else:
            logger.debug('Higher order cache called, but no higher order reference wavefront.'
                         'Skipping')
            cacheless_counter = cacheless_counter + 1
            aberrations_higher_order_reference_wavefront = None
        # cache zernike values for all stars from both reference wavefront and higher order
        # reference wavefront
        if cacheless_counter == 2:
            # only set self._caches and self._aberrations_reference_wavefronts to False if have
            # neither reference wavefront nor higher order reference wavefront
            self._caches = False
            self._aberrations_reference_wavefronts = None
            highest_zernike = 0
        else:
            if (aberrations_reference_wavefront is not None and
                aberrations_higher_order_reference_wavefront is not None):
                highest_zernike = int(np.max(self._reference_wavefront_zernikes_list +
                                             self._higher_order_reference_wavefront_zernikes_list))
            elif aberrations_reference_wavefront is not None:
                highest_zernike = np.max(self._reference_wavefront_zernikes_list)
            elif aberrations_higher_order_reference_wavefront is not None:
                highest_zernike = np.max(self._higher_order_reference_wavefront_zernikes_list)
            self._aberrations_reference_wavefronts = np.zeros([len(stars), highest_zernike - 3])
            # the shape of this is [number of stars, number of zernikes up to highest zernike
            # requested from one of the reference wavefronts]; note that the 3 is here because we
            # start with defocus (z4)
            # first, fill in zernikes from the reference wavefront; which zernikes you want from
            # the reference wavefront are can be specified in the yaml file
            if aberrations_reference_wavefront is not None:
                for reference_wavefront_zernike in self._reference_wavefront_zernikes_list:
                    max_zern = reference_wavefront_zernike - 4
                    # the 4 is here because we start with defocus (z4)
                    self._aberrations_reference_wavefronts[:,max_zern] = \
                        aberrations_reference_wavefront[:,max_zern]
            # second, fill in zernikes from the higher order reference wavefront; which zernikes
            # you want from the higher order reference wavefront can also be specified in the yaml
            # file
            if aberrations_higher_order_reference_wavefront is not None:
                for higher_order_reference_wavefront_zernike in \
                        self._higher_order_reference_wavefront_zernikes_list:
                    max_zern = higher_order_reference_wavefront_zernike - 4
                    # the 4 is here because we start with defocus (z4)
                    self._aberrations_reference_wavefronts[:,max_zern] = \
                        aberrations_higher_order_reference_wavefront[:,max_zern]

    def _delete_caches(self, logger=None):
        """Delete reference wavefront cache.
        :param logger:  A logger object for logging debug info [default: None]
        """
        if self.reference_wavefront or self.higher_order_reference_wavefront:
            logger.debug('Clearing caches of reference aberrations for all reference wavefronts')
        else:
            logger.debug('Delete caches called, but no reference wavefronts. Skipping')
        self._caches = False
        self._aberrations_reference_wavefronts = None
