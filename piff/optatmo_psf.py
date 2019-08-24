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
import copy
import os
from sklearn.ensemble import RandomForestRegressor
import sklearn
#import _pickle as cPickle
import pickle
from scipy.interpolate import Rbf

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .outliers import Outliers
from .model import ModelFitError
# from .gsobject_model import GSObjectModel, Kolmogorov, Gaussian
from .star import Star, StarFit, StarData
from .util import hsm_error, hsm_third_moments, hsm_error_third_moments, hsm_fourth_moments, hsm_error_fourth_moments, hsm_orthogonal, hsm_error_orthogonal, measure_snr, write_kwargs, read_kwargs
from .config import LoggerWrapper

def horner_from_galsim1(x, coef, dtype=None):
    """Evaluate univariate polynomial using Horner's method.

    I.e., take A + Bx + Cx^2 + Dx^3 and evaluate it as
    A + x(B + x(C + x(D)))

    @param x        A numpy array of values at which to evaluate the polynomial.
    @param coef     Polynomial coefficients of increasing powers of x.
    @param dtype    Optionally specify the dtype of the return array. [default: None]

    @returns a numpy array of the evaluated polynomial.  Will be the same shape as x.
    """
    coef = np.trim_zeros(coef, trim='b')
    result = np.zeros_like(x, dtype=dtype)
    if len(coef) == 0: return result
    result += coef[-1]
    for c in coef[-2::-1]:
        result *= x
        if c != 0: result += c
    #np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval(x,coef))
    return result

def horner2d_from_galsim1(x, y, coefs, dtype=None):
    """Evaluate bivariate polynomial using nested Horner's method.

    @param x        A numpy array of the x values at which to evaluate the polynomial.
    @param y        A numpy array of the y values at which to evaluate the polynomial.
    @param coefs    2D array-like of coefficients in increasing powers of x and y.
                    The first axis corresponds to increasing the power of y, and the second to
                    increasing the power of x.
    @param dtype    Optionally specify the dtype of the return array. [default: None]

    @returns a numpy array of the evaluated polynomial.  Will be the same shape as x and y.
    """
    result = horner_from_galsim1(y, coefs[-1], dtype=dtype)
    for coef in coefs[-2::-1]:
        result *= x
        result += horner_from_galsim1(y, coef, dtype=dtype)
    # Useful when working on this... (Numpy method is much slower, btw.)
    #np.testing.assert_almost_equal(result, np.polynomial.polynomial.polyval2d(x,y,coefs))
    return result

class wavefrontmap(object):
    """ wavefrontmap is a class used to build and access a Wavefront map - zernike coefficients vs. X,Y

    Aaron Roodman (C) SLAC National Accelerator Laboratory, Stanford University 2018.
    """

    def __init__(self,file):
        # init contains all initializations which are done only once for all fits

        
        self.file = file
        mapdict = pickle.load(open(self.file,'rb'))
        self.x = mapdict['x']
        self.y = mapdict['y']
        self.zcoeff = mapdict['zcoeff']

        self.interpDict = {}
        for iZ in range(3,37):    # numbering is such that iZ=3 is zern4
            self.interpDict[iZ] = Rbf(self.x, self.y, self.zcoeff[:,iZ])        

    def get(self,x,y,nZernikeFirst=12,nZernikeLast=37):
        # fill an array with Zernike coefficients for this x,y in the Map

        zout = np.zeros((nZernikeLast-nZernikeFirst+1))
        for iZactual in range(nZernikeFirst,nZernikeLast+1):
            iZ = iZactual-1
            zout[iZactual-nZernikeFirst] = self.interpDict[iZ](x,y)

        return zout

class OptAtmoPSF(PSF):

    """Combine Optical and Atmospheric PSFs together
    """

    def __init__(self, atmo_interp=None, outliers=None, optatmo_psf_kwargs={}, optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None, n_optfit_stars=0, fov_radius=4500., jmax_pupil=11, jmax_focal=10, min_optfit_snr=0, fit_optics_mode='shape', higher_order_reference_wavefront_file=None, random_forest_shapes_model_pickles_location=None, fit_atmosphere_mode='pixel', atmosphere_model='vonkarman', atmo_mad_outlier=False, shape_weights=[], logger=None, **kwargs):
        """
        Fit Combined Atmosphere and Optical PSF in two stage process.
        :param atmo_interp:                                     Piff Interpolant object that represents
                                                                the atmospheric interpolation
        :param outliers:                                        Optionally, an Outliers instance used
                                                                to remove outliers during atmosphere
                                                                fit.  [default: None]
        :param optatmo_psf_kwargs:                              Terms that set the state of the PSF,
                                                                excepting the atmospheric interpolant
        :param optical_psf_kwargs:                              Arguments to pass into galsim
                                                                opticalpsf object
        :param kolmogorov_kwargs:                               Arguments to pass into galsim
                                                                kolmogorov or vonkarman object
        :param reference_wavefront:                             Reference interpolator for the optical
                                                                wavefront. Takes in stars, returns
                                                                aberrations. Default is to not include.
        :param n_optfit_stars:                                  [default: 0] If > 0, randomly sample
                                                                only n_optfit_stars for the optical fit. Only use n_optfit_stars if doing a test run, not a serious fit.
        :param fov_radius:                                      [Default: 1.] Radius of telescope in
                                                                u,v coordinates
        :param jmax_pupil:                                      Number of pupil-basis zernikes in
                                                                Optical model. Inclusive and in Noll
                                                                convention. [default: 11]
        :param jmax_focal:                                      Number of focal-basis zernikes in
                                                                Optical model. Inclusive and in Noll
                                                                convention. [default: 11]
        :param min_optfit_snr:                                  minimum snr from star property required
                                                                for optical portion of fit. If 0,
                                                                ignored. [default: 0]
        :param fit_optics_mode:                                 Choose ['random_forest', 'shape', 'pixel']
                                                                for optics fitting mode. [default:
                                                                'shape']
        :param higher_order_reference_wavefront_file:           A string with the path and filename of the 
                                                                pickle containing the higher order
                                                                reference wavefront. Note: the code should
                                                                not work if this file is not specified. 
        :random_forest_shapes_model_pickles_location:           A string with the path to the folder
                                                                containing the random forest model pickles
                                                                for the random_forest fit. Note: the code
                                                                should not work if this file is not
                                                                specified.
        :param fit_atmosphere_mode:                             Choose ['shape', 'pixel']
                                                                for atmosphere fitting mode. [default:
                                                                'pixel']
        :param atmosphere_model:                                Choose ['kolmogorov', 'vonkarman']. Selects the galsim object used for the atmospheric piece.
                                                                Note that the default is vonkarman and to use kolmogorov would require some changes to input.py.
        :param atmo_mad_outlier:                                Boolean. If true, when computing atmosphere interps remove 5 sigma outliers from a MAD cut
        :shape_weights:                                         A list of weights for the different moments to be used in the chisq fit
        :param logger:                                          A logger object for logging debug info.
                                                                [default: None]
        Notes
        -----
        Our model of the PSF is the convolution of a sheared VonKarman
        with an optics model. Note than L0 is the VonKarman outer scale:
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
        """#TODO: either stop code from working if lower order reference wavefront not specified or allow code to work if higher order reference wavefront specified
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
        self.higher_order_reference_wavefront = wavefrontmap(file=self.higher_order_reference_wavefront_file)

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
        #if galsim.__version__ >= 2.0:
        self._noll_coef_field = galsim.zernike._noll_coef_array(self.jmax_focal, 0.0)
        #else:
        #    self._noll_coef_field = galsim.phase_screens._noll_coef_array(self.jmax_focal, 0.0, False)

        min_sizes = {'kolmogorov': 0.45, 'vonkarman': 0.7}
        if atmosphere_model == 'vonkarman':
            self.optatmo_psf_kwargs = {
                    'L0':   25.0,    'fix_L0':   False, 'min_L0': 5.0, 'max_L0': 100.0,
                    'size': 1.0,  'fix_size': False, 'min_size': min_sizes[atmosphere_model], 'max_size': 3.0,
                    'g1':   0,    'fix_g1':   False, 'min_g1': -0.4, 'max_g1': 0.4,
                    'g2':   0,    'fix_g2':   False, 'min_g2': -0.4, 'max_g2': 0.4,
                }
            self.keys = [ 'size', 'g1', 'g2', 'L0', ]
        else:
            self.optatmo_psf_kwargs = {
                    'size': 1.0,  'fix_size': False, 'min_size': min_sizes[atmosphere_model], 'max_size': 3.0,
                    'g1':   0,    'fix_g1':   False, 'min_g1': -0.4, 'max_g1': 0.4,
                    'g2':   0,    'fix_g2':   False, 'min_g2': -0.4, 'max_g2': 0.4,
                }
            self.keys = [ 'size', 'g1', 'g2',]
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
                    self.optatmo_psf_kwargs[zkey] = 0
                else:
                    initial_value = np.random.random() * (0.1 - -0.1) + -0.1
                    logger.debug('Setting initial {0} to randomly generated value {1}'.format(zkey, initial_value))
                    self.optatmo_psf_kwargs[zkey] = initial_value
                    # self.optatmo_psf_kwargs[zkey] = 0
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

        # Set up hardcoded gsparams for _considerable_ speedup. This is likely not necessary because optical_model.py already does this but this is here just in case.
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
        # Decrease pad_factor and oversampling for speedup in optical modeling. This is likely not necessary because optical_model.py already does this but this is here just in case.
        if 'pad_factor' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['pad_factor'] = 0.5
        if 'oversampling' not in self.optical_psf_kwargs:
            self.optical_psf_kwargs['oversampling'] = 0.5

        # max size of shapes allowed
        self._max_shapes = np.array([1.5, 0.12, 0.12, 0.15, 0.15, 0.15, 0.15, 1.5, 5.0, 50.0])
        # weighting of shapes
        self._shape_weights = np.array([0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        if len(shape_weights) > 0:
            if len(shape_weights) != len(self._shape_weights):
                raise ValueError('Specified {0} shape weights, but need to specify {1}!'.format(len(shape_weights), len(self._shape_weights)))
            for i, si in enumerate(shape_weights):
                self._shape_weights[i] = si
            

        self.fit_optics_mode = fit_optics_mode
        self.random_forest_shapes_model_pickles_location = random_forest_shapes_model_pickles_location
        self.fit_atmosphere_mode = fit_atmosphere_mode
        if atmosphere_model not in ['kolmogorov', 'vonkarman']:
            raise KeyError('Atmosphere model {0} not allowed! choose either kolmogorov or vonkarman'.format(atmosphere_model))
        self.atmosphere_model = atmosphere_model
        if self.atmosphere_model == 'kolmogorov':
            self.n_params_atmosphere = 3
            self.n_params_constant_atmosphere = 3
            self.n_params_constant_atmosphere_and_atmosphere = 6
        elif self.atmosphere_model == 'vonkarman':
            self.n_params_atmosphere = 3
            self.n_params_constant_atmosphere = 4
            self.n_params_constant_atmosphere_and_atmosphere = 7

        self.atmo_mad_outlier = atmo_mad_outlier

        # kwargs
        self.kwargs = {'fov_radius': self.fov_radius,
                       'shape_weights': self._shape_weights,
                       'jmax_pupil': self.jmax_pupil,
                       'jmax_focal': self.jmax_focal,
                       'min_optfit_snr': self.min_optfit_snr,
                       'n_optfit_stars': self.n_optfit_stars,
                       'fit_optics_mode': self.fit_optics_mode,
                       'higher_order_reference_wavefront_file': self.higher_order_reference_wavefront_file,
                       'random_forest_shapes_model_pickles_location': self.random_forest_shapes_model_pickles_location,
                       'fit_atmosphere_mode': self.fit_atmosphere_mode,
                       'atmosphere_model': self.atmosphere_model,
                       'atmo_mad_outlier': self.atmo_mad_outlier,
                       # junk entries to be overwritten in _finish_read function
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
        self._cache_higher_order = False
        self._aberrations_higher_order_reference_wavefront = None

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
        kwargs.pop('type',None)

        # do processing as appropriate
        # set up optical and atmosphere psf kwargs using the optical model
        optical_psf_kwargs = config_psf.pop('optical_psf_kwargs', {})

        optical = Optical(logger=logger, **optical_psf_kwargs)
        kwargs['optical_psf_kwargs'] = optical.optical_psf_kwargs
        kolmogorov_kwargs = optical.kolmogorov_kwargs
        if 'kolmogorov_kwargs' in config_psf:
            kolmogorov_kwargs.update(config_psf['kolmogorov_kwargs'])
        # if we only have lam (which we expect from Optical models), then put in a placeholder half_light_radius
        # Also, let r0=0 or None indicate that there is no kolmogorov component
        if kolmogorov_kwargs.keys() == ['lam'] or ('r0' in kolmogorov_kwargs and not kolmogorov_kwargs['r0']):
            # kolmogorov_kwargs = {'half_light_radius': 1.0}
            kolmogorov_kwargs = {'fwhm': 1.0}
        kwargs['kolmogorov_kwargs'] = kolmogorov_kwargs
       
       #custom shape weights for the moments used in fitting     
        if 'optatmo_psf_kwargs' in config_psf:
            kwargs['optatmo_psf_kwargs'] = config_psf['optatmo_psf_kwargs']

        # atmo interp may be skipped for the purposes of zeroing in on the optics model
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

    def fit(self, train_stars, test_stars, wcs, pointing,
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

        self.higher_order_reference_wavefront = wavefrontmap(file=self.higher_order_reference_wavefront_file) #here we save the specified higher order reference wavefront as an instance of the wavefrontmap class. The non-higher order reference wavefront goes up to z11 and this one goes from z12 to z37
        self.wcs = wcs
        self.pointing = pointing


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
                shape = self.measure_shape_orthogonal(star, logger=logger) #shapes measured here include flux, center, 2nd, 3rd, and orthogonal radial moments up to eighth moments
                error = self.measure_error_orthogonal(star, logger=logger) #errors measured here include flux, center, 2nd, 3rd, and orthogonal radial moments up to eighth moments
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
                shape = self.measure_shape_orthogonal(star, logger=logger) #shapes measured here include flux, center, 2nd, 3rd, and orthogonal radial moments up to eighth moments
                error = self.measure_error_orthogonal(star, logger=logger) #errors measured here include flux, center, 2nd, 3rd, and orthogonal radial moments up to eighth moments
                star = Star(star.data, StarFit(None, flux=shape[0], center=(shape[1], shape[2])))
                star.data.properties['shape'] = shape
                star.data.properties['shape_error'] = error
                snr = self.measure_snr(star)

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
        test_conds_shape = (np.all(np.abs(self.test_star_shapes[:, 3:]) <= self._max_shapes, axis=1))


        # also a MAD cut
        med = np.nanmedian(                        np.concatenate([self.star_shapes[:, 3:], self.test_star_shapes[:, 3:]], axis=0)                                     , axis=0)
        mad = np.nanmedian(           np.abs(      np.concatenate([self.star_shapes[:, 3:], self.test_star_shapes[:, 3:]], axis=0)            - med[None])             , axis=0)
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


        # do an snr cut for the stars for the fit and record how many have been cut and why so far in the logger
        self.fit_optics_indices = np.arange(len(self.stars))
        conds_snr = (self.star_snrs >= self.min_optfit_snr)
        self.fit_optics_indices = self.fit_optics_indices[conds_snr]
        logger.info('Cutting to {0} stars for fitting the optics based on SNR > {1} ({2} stars) on maximum shapes ({3} stars) and on a 5 sigma outlier cut ({4} stars)'.format(len(self.fit_optics_indices), self.min_optfit_snr, len(conds_snr) - np.sum(conds_snr), len(conds_shape) - np.sum(conds_shape), len(conds_mad) - np.sum(conds_mad)))


        # cut further if we have more stars for fit than n_optfit_stars.
        # Warning: only use n_optfit_stars if doing a test run, not a serious fit. This limits the ability to get the 500 highest SNR stars for the fit.
        if self.n_optfit_stars and self.n_optfit_stars < len(self.fit_optics_indices) and self.n_optfit_stars <= 500:
            logger.info('Cutting from {0} to {1} stars for the fit, as requested in n_optfit_stars. Warning: at least 500 (highest SNR) stars recommended for the optical fit. Only use n_optfit_stars if doing a test run, not a serious fit.'.format(len(self.fit_optics_indices), self.n_optfit_stars))
            max_stars = self.n_optfit_stars
            np.random.shuffle(self.fit_optics_indices)
            self.fit_optics_indices = self.fit_optics_indices[:max_stars]
        else:
            if len(self.fit_optics_indices) < self.n_optfit_stars and self.n_optfit_stars > 0:
                logger.info("{0} stars remaining after cuts instead of the {1} requested using n_optfit_stars. Of these, the (at most) 500 highest SNR stars will be passed on to the optical fit. Note: only use n_optfit_stars if doing a test run, not a serious fit.".format(max_stars, self.n_optfit_stars))
            if self.n_optfit_stars and self.n_optfit_stars < len(self.fit_optics_indices) and self.n_optfit_stars > 500:
                logger.info('{0} stars remaining after cuts. Of these, the (at most) 500 highest SNR stars will be passed on to the optical fit. Cutting down to {1} stars has been requested using n_optfit_stars; however, since this number is more than 500 this will be ignored. Only use n_optfit_stars if doing a test run, not a serious fit.'.format(len(self.fit_optics_indices), self.n_optfit_stars))

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
                snrs.append(-self.measure_snr(fit_optics_star))
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

        # perform initial fit in "random_forest" mode, which uses a random forest model (this model is trained to return shapes based on what fit parameters you give it)
        # the fit parameters here are the optical fit parameters and the average of the atmospheric fit parameters
        self.fit_optics(self.fit_optics_stars, self.fit_optics_star_shapes, self.fit_optics_star_errors, mode='random_forest', logger=logger, **kwargs)

        # first just fit the optical size parameter to correct size offset
        # the size parameter is proportional to 1/r0, where r0 is the Fried parameter         
        # the "optics" size is the average of this across the focal plane, whereas
        # the "atmospheric" size is the deviation from this average at different points in the focal plane.
        # only use the e0 moment to fit
        # fit_size() is used before the full optical fit because it makes that fit faster
        moments_list = ["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2", "orth4", "orth6", "orth8"]
        self.length_of_moments_list = len(moments_list)
        self.fit_size(self.fit_optics_stars, self.fit_optics_star_shapes, self.fit_optics_star_errors, logger=logger, **kwargs)

        # do a fit to moments ("shape" mode) or pixels ("pixel" mode), whichever is specified in the yaml file. Nothing happens here if "random_forest" mode (which is the default) is chosen
        # this is the "optical" fit; despite being called that the fit parameters here are the optical fit parameters and the average of the atmospheric fit parameters
        self.total_redchi_across_iterations = []
        if self.fit_optics_mode in ['shape', 'pixel']:
            self.fit_optics(self.fit_optics_stars, self.fit_optics_star_shapes, self.fit_optics_star_errors, mode=self.fit_optics_mode, logger=logger, ftol=1.e-3, **kwargs) #looser convergence criteria used than default of ftol=1.e-7
        elif self.fit_optics_mode == 'random_forest':
            # already did it, so can pass
            pass
        else:
            # an unrecognized mode is simply ignored
            logger.warning('Found unrecognized fit_optics_mode {0}. Ignoring'.format(self.fit_optics_mode))

        print("len(self.stars): {0}".format(len(self.stars)))
        # one extra round of outlier rejection using the pull from the moments (only up to third moments)
        for s, stars in enumerate([self.stars, self.test_stars, self.fit_optics_stars]):
            if s == 0:
                print("now prepping pull cuts for self.stars; in other words the train stars")
            if s == 1:
                print("now prepping pull cuts for self.test_stars")
            if s == 2:
                print("now prepping pull cuts for self.fit_optics_stars; in other words the subset train stars specifically used in the optical fit")
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
            pull_all_stars = (data_shapes_all_stars - model_shapes_all_stars) / data_errors_all_stars #pull is (data-model)/error
            print("data_shapes_all_stars: {0}".format(data_shapes_all_stars))
            print("model_shapes_all_stars: {0}".format(model_shapes_all_stars))
            print("data_errors_all_stars: {0}".format(data_errors_all_stars))
            print("pull_all_stars: {0}".format(pull_all_stars))
            conds_pull = (np.all(np.abs(pull_all_stars) <= 4.0, axis=1)) #all stars with more than 4.0 pull are thrown out
            conds_pull_e0 = (np.abs(pull_all_stars[:,0]) <= 4.0)
            conds_pull_e1 = (np.abs(pull_all_stars[:,1]) <= 4.0)
            conds_pull_e2 = (np.abs(pull_all_stars[:,2]) <= 4.0)
            if s == 0:
                self.stars = np.array(self.stars)[conds_pull].tolist()
            if s == 1:
                self.test_stars = np.array(self.test_stars)[conds_pull].tolist()
            if s == 2:
                self.number_of_outliers_optical = np.array([len(self.fit_optics_stars) - np.sum(conds_pull_e0), len(self.fit_optics_stars) - np.sum(conds_pull_e1), len(self.fit_optics_stars) - np.sum(conds_pull_e2)])
                self.number_of_stars_pre_cut_optical = len(self.fit_optics_stars)
                self.fit_optics_stars = np.array(self.fit_optics_stars)[conds_pull].tolist()
                self.number_of_stars_post_cut_optical = len(self.fit_optics_stars)
                self.pull_mean_optical = np.nanmean(pull_all_stars[:,:3], axis=0) #the mean pull (only second moments) for stars used in the fit is later used to find outliers among exposures
                self.pull_rms_optical = np.sqrt(np.nanmean(np.square(pull_all_stars[:,:3]),axis=0))
                self.pull_all_stars_optical = pull_all_stars


        number_of_stars_used_in_optical_chi = len(self.final_optical_chi)//self.length_of_moments_list 
        print("total chisq for optical chi: {0}".format(np.sum(np.square(self.final_optical_chi))))
        for tm, test_moment in enumerate(moments_list):
            print("total chisq for optical chi for {0}: {1}".format(test_moment,np.sum(np.square(self.final_optical_chi)[tm::self.length_of_moments_list])))
        print("total dof for optical chi: {0}".format(len(self.final_optical_chi)))
        print("number_of_stars_used_in_optical_chi: {0}".format(number_of_stars_used_in_optical_chi))

        # record the chi
        self.chisq_all_stars_optical = np.empty(number_of_stars_used_in_optical_chi)
        for s in range(0,number_of_stars_used_in_optical_chi):
            self.chisq_all_stars_optical[s] = np.sum(np.square(self.final_optical_chi[s*self.length_of_moments_list:s*self.length_of_moments_list+self.length_of_moments_list]))
        print("len(self.stars): {0}".format(len(self.stars)))

        # this is the "atmospheric" fit.
        # we start here with the optical fit parameters and the average values of the atmospheric parameters found in the optical fit and hold those fixed. 
        # we float only the deviation of these atmospheric parameters from the average here.
        # this fit can be skipped
        if self.atmo_interp in ['skip', 'Skip', None, 'none', 'None', 0]:
            pass
        else:
            print("len(self.stars): {0}".format(len(self.stars)))
            stars_fit_atmosphere, stars_fit_atmosphere_stripped = self.fit_atmosphere(self.stars, chisq_threshold=chisq_threshold, max_iterations=max_iterations, logger=logger, **kwargs)
            self.stars = stars_fit_atmosphere  # keeps all fit params and vars, but does NOT include any removed with outliers

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
        #aberrations_pupil = np.array([galsim.utilities.horner2d(rsqr, r, ca, dtype=complex).real
        #import ipdb; ipdb.set_trace()
        aberrations_pupil = np.array([horner2d_from_galsim1(rsqr, r, ca, dtype=complex).real
                               for ca in self._coef_arrays_field]).T  # (nstars, ncoefs)

        return aberrations_pupil

    def getParamsList(self, stars, logger=None):
        """Get params for a list of stars.
        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :returns:           Params  [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...] for each star
                            where all params that are not "z_number" are atmospheric params. Those labelled
                            "opt_something" are the averages of these atmospheric params across the focal plane
                            and those labelled "atm_something" are the deviations from these averages for stars
                            at different points in the focal plane.
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
        params = np.zeros((len(stars), self.jmax_pupil + self.n_params_constant_atmosphere), dtype=np.float64)

        logger.debug('Getting aberrations from optical / mean system')
        aberrations_pupil = self._getParamsList_aberrations_field(stars)
        params[:, self.n_params_constant_atmosphere:] += aberrations_pupil

        if self.reference_wavefront:
            if self._cache:
                logger.debug('Getting cached reference wavefront aberrations')
                # use precomputed cache. assumes stars are the same as in cache!
                aberrations_reference_wavefront = self._aberrations_reference_wavefront
            else:
                logger.debug('Getting reference wavefront aberrations')
                stars = [Star(star.data, None) for star in stars]
                stars = self.reference_wavefront.interpolateList(stars)
                aberrations_reference_wavefront = np.array([star_interpolated.fit.params for star_interpolated in stars])
            if self._cache_higher_order:
                logger.debug('Getting cached higher order reference wavefront aberrations')
                # use precomputed higher order cache. assumes stars are the same as in cache!
                aberrations_higher_order_reference_wavefront = self._aberrations_higher_order_reference_wavefront
            else:
                logger.debug('Getting higher_order_reference wavefront aberrations')
                aberrations_higher_order_reference_wavefront = np.empty([len(stars),26])
                for s, star in enumerate(stars):
                    star_data = star.data
                    star_data.local_wcs = star_data.image.wcs.local(star_data.image_pos)
                    x_value = star_data.local_wcs._x(star_data['u'],star_data['v'])
                    y_value = star_data.local_wcs._y(star_data['u'],star_data['v'])
                    x_value = x_value * (15.0/1000.0)
                    y_value = y_value * (15.0/1000.0) # wavefront map class expects units of mm, rather than pixels
                    zout_camera = self.higher_order_reference_wavefront.get(x=x_value, y=y_value) #zout_camera[0] is for z12
                    zout_sky = np.array([-zout_camera[0], zout_camera[1], zout_camera[2], -zout_camera[3], zout_camera[5], zout_camera[4], -zout_camera[7], -zout_camera[6], zout_camera[9], zout_camera[8], zout_camera[10], zout_camera[11], -zout_camera[12], -zout_camera[13], zout_camera[14], zout_camera[15], -zout_camera[16], zout_camera[18], zout_camera[17], -zout_camera[20], -zout_camera[19], zout_camera[22], zout_camera[21], -zout_camera[24], -zout_camera[23], zout_camera[25]]) #conversion from zout_camera (AOS system) to zout_sky (Galsim) inspired by thesis of Chris Davis
                    aberrations_higher_order_reference_wavefront[s] = zout_sky
            # put aberrations_reference_wavefront
            # reference wavefront starts at z4 but may not span full range of aberrations used
            n_reference_aberrations = aberrations_reference_wavefront.shape[1]
            n_higher_order_reference_aberrations = 26
            # the 3 seen here below is because z4 starts at index 3

            aberrations_reference_wavefront = aberrations_reference_wavefront[:,:8] #limit reference wavefront to up to z11 to make room for higher order reference wavefront, which starts at z12
            if n_reference_aberrations + 3 < self.jmax_pupil:
                params[:, self.n_params_constant_atmosphere_and_atmosphere: n_reference_aberrations + self.n_params_constant_atmosphere_and_atmosphere] += aberrations_reference_wavefront
                if n_reference_aberrations + n_higher_order_reference_aberrations + 3 < self.jmax_pupil:
                    params[:, self.n_params_constant_atmosphere_and_atmosphere + n_reference_aberrations: n_reference_aberrations + n_higher_order_reference_aberrations + self.n_params_constant_atmosphere_and_atmosphere] += aberrations_higher_order_reference_wavefront
                else:
                    params[:, self.n_params_constant_atmosphere_and_atmosphere + n_reference_aberrations:] += aberrations_higher_order_reference_wavefront[:, :self.jmax_pupil - 11]
            else:
                # we have more jmax_pupil than reference wavefront
                params[:, self.n_params_constant_atmosphere_and_atmosphere:] += aberrations_reference_wavefront[:, :self.jmax_pupil - 3]
        print("loaded reference wavefront")
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
                params[:, 0:self.n_params_atmosphere] += aberrations_atmo_star
        if self.atmosphere_model == 'vonkarman':
            # set the vonkarman outer scale
            params[:, 3] = self.optatmo_psf_kwargs['L0']

        return params

    def getParams(self, star):
        """Get params for a given star.
        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :returns:           Params  [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...]
                            where all params that are not "z_number" are atmospheric params. Those labelled
                            "opt_something" are the averages of these atmospheric params across the focal plane
                            and those labelled "atm_something" are the deviations from these averages for stars
                            at different points in the focal plane.
        """
        return self.getParamsList([star])[0]

    def getProfile(self, params, logger=None):
        """Get galsim profile for a given params
        :param params:      [atm_size, atm_g1, atm_g2, opt_L0, opt_size, opt_g1, opt_g2, z4, z5...]. 
                            where all params that are not "z_number" are atmospheric params. Those labelled
                            "opt_something" are the averages of these atmospheric params across the focal plane
                            and those labelled "atm_something" are the deviations from these averages for stars
                            at different points in the focal plane. Note how this means that, for example, 
                            atm_size and opt_size are added together for the Kolmogorov model
        :returns:           Galsim profile
        """
        logger = LoggerWrapper(logger)

        # optics
        aberrations = np.zeros(4 + len(params[self.n_params_constant_atmosphere_and_atmosphere:]))  # fill piston etc with 0
        aberrations[4:] = params[self.n_params_constant_atmosphere_and_atmosphere:]
        aberrations = aberrations * (700.0/self.optical_psf_kwargs['lam'])
        opt = galsim.OpticalPSF(aberrations=aberrations,
                                gsparams=self.gsparams,
                                **self.optical_psf_kwargs)

        # atmosphere
        # add stochastic (labelled "atm") and constant (labelled "opt") pieces together
        if self.atmosphere_model == 'kolmogorov':
            size = params[0] + params[3]
            g1 = params[1] + params[4]
            g2 = params[2] + params[5]
            atmo = galsim.Kolmogorov(gsparams=self.gsparams,
                                     **self.kolmogorov_kwargs
                                     ).dilate(size).shear(g1=g1, g2=g2)
        elif self.atmosphere_model == 'vonkarman':
            size = params[0] + params[4]
            g1 = params[1] + params[5]
            g2 = params[2] + params[6]
            L0 = params[3]
            kolmogorov_kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                                 'r0': self.kolmogorov_kwargs['r0'] / size,
                                 'L0': L0,}
            atmo = galsim.VonKarman(gsparams=self.gsparams,
                                    **kolmogorov_kwargs).shear(g1=g1, g2=g2)

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
        :returns:           Star instance with its image filled with rendered
                            PSF
        """
        # use flux and center properties
        if use_fit:
            prof = prof.shift(star.fit.center) * star.fit.flux
        image, weight, image_pos = star.data.getImage()
        if copy_image:
            image_model = image.copy()
        else:
            image_model = image
        prof.drawImage(image_model, method='auto', offset=(star.image_pos-image_model.true_center))
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

    def draw_fitted_star_given_fitted_image_and_flux(self, x, y, fitted_image, pointing, flux):
        """Creates the appropriate Star instance for a given image (usually a fitted image), position, pointing, and flux.

        :param x:               x coordinate of the star's position
        :param y:               y coordinate of the star's position
        :param fitted_image:    Image of star
        :param pointing:        Pointing of star
        :param flux:            Flux of star

        :returns:               Star instance with its image filled
        """
        star = Star.makeTarget(x=x, y=y, image=fitted_image, pointing=pointing, flux=flux)
        return star

    def drawStar(self, star, copy_image=True):
        """Generate PSF image for a given star.
        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.
        :returns:           Star instance with its image filled with rendered
                            PSF
        """

        params = self.getParams(star)
        prof = self.getProfile(params)
        star = self.drawProfile(star, prof, params, copy_image=copy_image)
        return star

    def drawStarList(self, stars, copy_image=True):
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
        stars_drawn = []
        for param, star in zip(params, stars):
            try:
                stars_drawn.append(self.drawProfile(star, self.getProfile(param), param, copy_image=copy_image))
            except:
                stars_draw.append(None)
        # 
        # stars_drawn = [self.drawProfile(star, self.getProfile(param), param, copy_image=copy_image) for param, star in zip(params, stars)]
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

            # size, g1, g2, L0 mean constant atmospheric terms. These are called "opt_size", etc. elsewhere as opposed to "atm_size," etc. which are the deviations from these means.
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
                    raise ValueError('Not allowed to fit pupil zernike {0} less than {2}, key {1}!'.format(pupil_index, key, 4))
                elif focal_index < 1:
                    raise ValueError('Not allowed to fit focal zernike {0} less than {2} !, key {1}!'.format(focal_index, key, 1))
                elif pupil_index > self.jmax_pupil:
                    raise ValueError('Not allowed to fit pupil zernike {0}, greater than {2}, key {1}!'.format(pupil_index, key, self.jmax_pupil))
                elif focal_index > self.jmax_focal:
                    raise ValueError('Not allowed to fit focal zernike {0} greater than {2} !, key {1}!'.format(focal_index, key, self.jmax_focal))

            if key != 'L0':
                old_value = self.aberrations_field[pupil_index - 1, focal_index - 1]
            else:
                old_value = self.optatmo_psf_kwargs['L0']
            new_value = optatmo_psf_kwargs[key]           

            # figure out if we really need to recompute the coef arrays
            if old_value != new_value:
                if 'fix_' + key in optatmo_psf_kwargs:
                    if optatmo_psf_kwargs['fix_' + key]:
                        logger.warning('Warning! Changing key {0} which is designated as fixed from {1} to {2}!'.format(key, old_value, new_value))
                logger.debug('Updating Zernike parameter {0} from {1:+.4e} + {3:+.4e} = {2:+.4e}'.format(key, old_value, new_value, new_value - old_value))
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

        # values = flux, u0, v0, e0, e1, e2, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2
        values = hsm_error(star, return_debug=False, logger=logger, return_error=return_error)

        shape = np.array(values[:6])
        if np.any(shape != shape):
            raise ModelFitError

        # flux is underestimated empirically
        shape[0] = shape[0] / 0.92

        logger.debug('Measured Shape is {0}'.format(str(shape)))
        if return_error:
            error = np.array(values[6:])
            logger.debug('Measured Error is {0}'.format(str(error)))
            return shape, error
        else:
            return shape

    def measure_shape_third_moments(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm. Goes up to third moments. Does not return error.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape in unnormalized basis. Goes up
                                    to third moments.
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2
        values = hsm_third_moments(star, logger=logger)

        shape = np.array(values)

        # flux is underestimated empirically
        shape[0] = shape[0] / 0.92

        return shape

    def measure_error_third_moments(self, star, logger=None):
        """Measure the shape error of a star using the HSM algorithm. Goes up to third moments.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape Error in unnormalized basis. Goes up
                                    to third moments.
        """
        logger = LoggerWrapper(logger)

        # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2
        values = hsm_error_third_moments(star, logger=logger)

        error = np.array(values)

        return error

    def measure_shape_fourth_moments(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.  Goes up to fourth moments. Does not return error.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape in unnormalized basis. Goes up
                                    to fourth moments.
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, xi, eta1, eta2, lambda1, lambda2
        values = hsm_fourth_moments(star, logger=logger)

        shape = np.array(values)

        # flux is underestimated empirically
        shape[0] = shape[0] / 0.92

        return shape

    def measure_error_fourth_moments(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.  Goes up to fourth moments.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape Error in unnormalized basis. Goes up
                                    to fourth moments.
        """
        logger = LoggerWrapper(logger)

        # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_xi, sigma_eta1, sigma_eta2, sigma_lambda1, sigma_lambda2
        values = hsm_error_fourth_moments(star, logger=logger)

        error = np.array(values)

        return error

    def measure_shape_orthogonal(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.  Goes up to third moments plus orthogonal radial moments up to eighth moments. Does not return error.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape in unnormalized basis. Goes up to third moments plus orthogonal radial moments up to eighth moments
        """
        logger = LoggerWrapper(logger)

        # values = flux, u0, v0, e0, e1, e2, zeta1, zeta2, delta1, delta2, orth4, orth6, orth8
        values = hsm_orthogonal(star, logger=logger)

        shape = np.array(values)

        # flux is underestimated empirically
        shape[0] = shape[0] / 0.92

        return shape

    def measure_error_orthogonal(self, star, logger=None):
        """Measure the shape of a star using the HSM algorithm.  Goes up to third moments plus orthogonal radial moments up to eighth moments.

        :param star:                Star we want to measure
        :param logger:              A logger object for logging debug info
        :returns:                   Shape Error in unnormalized basis. Goes up to third moments plus orthogonal radial moments up to eighth moments.
                                    to fourth moments.
        """
        logger = LoggerWrapper(logger)

        # values = sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_orth4, sigma_orth6, sigma_orth8
        values = hsm_error_orthogonal(star, logger=logger)

        error = np.array(values)

        return error

    @staticmethod
    def measure_snr(star):
        """Calculate the signal-to-noise of a given star. Calls util
        measure_snr function
        :param star:    Input star, with stamp, weight
        :returns:       signal to noise ratio
        """
        return measure_snr(star)

    def fit_optics(self, stars, shapes, errors, mode, logger=None, ftol=1.e-7, **kwargs):
        """Fit interpolated PSF model to star shapes. It is important to note that although this fit is referred to as 
        the "optical" fit we still fit the average of the atmospheric parameters across the focal plane here. Finding 
        the deviation of these atmospheric parameters from the average is then done later in the fit_atmosphere() 
        function. For example, there is an atmospheric parameter known as the "size" parameter (which is proportional 
        to 1/r0 with r0 being the Fried parameter) whose average we fit in this function. Finding the deviation of this
        size parameter from the average is then done later in the fit_atmosphere() function.
        :param stars:       A list of Stars
        :param shapes:      A list of premeasured Star shapes
        :param errors:      A list of premeasured Star shape errors
        :param mode:        Parameter mode ['random_forest', 'shape', 'pixel']. Dictates which residual function we use.
        :param logger:      A logger object for logging debug info.
                            [default: None]
        :param ftol:        One of the convergence criteria for the optical fit. Based on relative change in the 
                            chi after an iteration. Smaller ftol is stricter and takes longer to converge. Not 
                            used in "random_forest" or "pixel" mode.
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
        logger = LoggerWrapper(logger)
        import lmfit
        logger.info("Start fitting Optical in {0} mode for {1} stars".format(mode, len(stars)))

        # save reference wavefront values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_cache(stars, logger=logger)
        if self.reference_wavefront: self._create_cache_higher_order(stars, logger=logger)

        lmparams = self._fit_optics_lmparams(self.optatmo_psf_kwargs, self.keys)
        if mode == 'random_forest':
            residual = self._fit_random_forest_residual
        elif mode == 'shape':
            residual = self._fit_optics_residual
        elif mode == 'pixel': #fitting optics in pixel mode not necessarily set up to work with vonkarman atmosphere
            residual = self._fit_optics_pixel_residual
            # fix size, g1, g2 here
            lmparams['size'].set(vary=False)
            lmparams['g1'].set(vary=False)
            lmparams['g2'].set(vary=False)
            # and update the optatmopsf accordingly
            self.optatmo_psf_kwargs['fix_size'] = True
            self.optatmo_psf_kwargs['fix_g1'] = True
            self.optatmo_psf_kwargs['fix_g2'] = True

            # fill with initial guesses. Make sure they are treated as floats!
            self._fit_pixel_fluxes = np.array([star.image.array.sum() * 1. for star in stars])
            self._fit_pixel_centers = [(0.0, 0.0) for star in stars]
            self._fit_pixel_sizes = np.array([0.0 for star in stars])
            self._fit_pixel_g1s = np.array([0.0 for star in stars])
            self._fit_pixel_g2s = np.array([0.0 for star in stars])
            self._fit_pixel_sizes_vars = np.array([0.0 for star in stars])
            self._fit_pixel_g1s_vars = np.array([0.0 for star in stars])
            self._fit_pixel_g2s_vars = np.array([0.0 for star in stars])
        else:
            raise KeyError('Unrecognized fit mode: {0}'.format(mode))

        # load up random forest model (used only in "random_forest" mode)
        regr_dictionary = {}
        for m, moment in enumerate(np.array(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"])):
            with open("{0}/random_forest_shapes_model_{1}.pickle".format(self.random_forest_shapes_model_pickles_location, moment), 'rb') as f:

                try:
                    regr = pickle.load(f)
                except:
                    raise AttributeError('A random forest model pickle failed to load. This likely means the pickle has not been placed in the correct directory such that the file {0}/random_forest_shapes_model_{1}.pickle does not exist.'.format(self.random_forest_shapes_model_pickles_location, moment))
                version = regr.__getstate__()['_sklearn_version']
                if version != sklearn.__version__:
                    raise AttributeError('A random forest model does not have the same sklearn version as the one currently being used. This likely requires you to remake the model from the data using the version of sklearn you currently have. This can be done by going to the data directory inside PIFF and running: python make_new_random_forest_model_pickles.py')
                regr_dictionary[moment] = regr             
            
        # do fit!     
        if mode == 'random_forest':
            results = lmfit.minimize(residual, lmparams, args=(stars, shapes, errors, regr_dictionary, logger,), epsfcn=1e-5)
            print("finished random_forest fit")
            logger.info("finished random_forest fit")
        elif mode == 'shape':
            results = lmfit.minimize(residual, lmparams, args=(stars, shapes, errors, logger,), epsfcn=1e-5, ftol=ftol)
            print("finished full optics fit")
            logger.info("finished full optics fit")         
        else:
            results = lmfit.minimize(residual, lmparams, args=(stars, shapes, errors, logger,), epsfcn=1e-5)
        nfev = results.nfev
        nvarys = results.nvarys
        maxfev = 2000*(nvarys+1)
        print("nfev: {0}".format(nfev))
        print("nvarys: {0}".format(nvarys))
        print("maxfev: {0}".format(maxfev))
        logger.info("nfev: {0}".format(nfev))
        logger.info("nvarys: {0}".format(nvarys))
        logger.info("maxfev: {0}".format(maxfev))
        if nfev >=maxfev:
            print("nfev >=maxfev; fit likely failed; aborting")
            logger.info("nfev >=maxfev; fit likely failed; aborting")
            raise AttributeError("nfev >=maxfev; fit likely failed; aborting")

        if mode == 'pixel': #fitting optics in pixel mode not necessarily set up to work with code since vonkarman atmosphere started being favored
            # smooth vars
            self._fit_pixel_sizes_vars = self._fit_pixel_sizes_vars + 1e-4 ** 2
            self._fit_pixel_g1s_vars = self._fit_pixel_g1s_vars + 1e-4 ** 2
            self._fit_pixel_g2s_vars = self._fit_pixel_g2s_vars + 1e-4 ** 2

            # update pixel mode with size, g1, g2 fits
            size_avg = np.average(self._fit_pixel_sizes, weights=1. / (self._fit_pixel_sizes_vars))
            var_size_avg = np.average((self._fit_pixel_sizes - size_avg) ** 2, weights = 1. / (self._fit_pixel_sizes_vars))
            size = size_avg + self.optatmo_psf_kwargs['size']
            error_size = np.sqrt(var_size_avg + self.optatmo_psf_kwargs.pop('error_size', 0) ** 2)

            g1_avg = np.average(self._fit_pixel_g1s, weights=1. / (self._fit_pixel_g1s_vars))
            var_g1_avg = np.average((self._fit_pixel_g1s - g1_avg) ** 2, weights = 1. / (self._fit_pixel_g1s_vars))
            g1 = g1_avg + self.optatmo_psf_kwargs['g1']
            error_g1 = np.sqrt(var_g1_avg + self.optatmo_psf_kwargs.pop('error_g1', 0) ** 2)

            g2_avg = np.average(self._fit_pixel_g2s, weights=1. / (self._fit_pixel_g2s_vars))
            var_g2_avg = np.average((self._fit_pixel_g2s - g2_avg) ** 2, weights = 1. / (self._fit_pixel_g2s_vars))
            g2 = g2_avg + self.optatmo_psf_kwargs['g2']
            error_g2 = np.sqrt(var_g2_avg + self.optatmo_psf_kwargs.pop('error_g2', 0) ** 2)

            # because we fixed size, g1, g2, they will not appear in the below bit, so we put them into optatmo psf kwargs here
            self.optatmo_psf_kwargs['size'] = size
            self.optatmo_psf_kwargs['g1'] = g1
            self.optatmo_psf_kwargs['g2'] = g2
            self.optatmo_psf_kwargs['error_size'] = error_size
            self.optatmo_psf_kwargs['error_g1'] = error_g1
            self.optatmo_psf_kwargs['error_g2'] = error_g2

            # put them in results anyways so that we can see their values in the fit_report
            results.params['size'].set(value=size)
            results.params['g1'].set(value=g1)
            results.params['g2'].set(value=g2)

        # update PSF parameters with fit results
        # TODO: can I go through this for loop from lmparams directly without blindly hoping key_i lines up?
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

        logger.info('{0} optical fit from lmfit parameters:'.format(mode))
        logger.info(lmfit.fit_report(results, min_correl=0.5))

        # save results for debugging purposes
        self._fit_optics_results = results

        # remove saved values from the reference wavefront caches when we are done with the fit
        if self.reference_wavefront: self._delete_cache(logger=logger)
        if self.reference_wavefront: self._delete_cache_higher_order(logger=logger)

    def fit_size(self, stars, shapes, shape_errors, logger=None, **kwargs):
        """Adjusts the optics size parameter found in the random forest fit by doing forced search of 501 steps +-
        0.1 about the result found in the random forest fit. The size parameter is proportional to 1/r0, r0 
        being the Fried parameter. The "optics" size is the average of this across the focal plane, whereas 
        "atmospheric" size is the deviation from this average at different points in the focal plane.
        :param stars:           A list of Star instances.
        :param shapes:          A list of premeasured Star shapes
        :param shape_errors:    A list of premeasured Star shape errors
        :param logger:          A logger object for logging debug info.
                                [default: None]
        """
        logger = LoggerWrapper(logger)
        import lmfit
        logger.info("Start fitting Optical fit of size alone")
        # fit_size() is used before the full optical fit and makes that fit faster

        # save reference wavefront values so we don't keep calling it during fit
        if self.reference_wavefront: self._create_cache(stars, logger=logger)
        if self.reference_wavefront: self._create_cache_higher_order(stars, logger=logger)

        # make kwargs with only size
        Ns = kwargs.pop('Ns', 501)
        dparam = kwargs.pop('dparam', 0.3)  # search +- this range
        param = self.optatmo_psf_kwargs['size']
        fit_size_kwargs = {'size': param, 'min_size': param - dparam, 'max_size': param + dparam}
        # make sure we don't go too crazy
        min_size = self.optatmo_psf_kwargs['min_size']
        if fit_size_kwargs['min_size'] < min_size:
            fit_size_kwargs['min_size'] = min_size
        max_size = self.optatmo_psf_kwargs['max_size']
        if fit_size_kwargs['max_size'] < max_size:
            fit_size_kwargs['max_size'] = max_size

        lmparams = self._fit_optics_lmparams(fit_size_kwargs, ['size'])

        # do fit
        results = lmfit.minimize(self._fit_size_residual, lmparams, args=(stars, shapes, shape_errors, logger,), method='brute', Ns=Ns)  # 1e-3 steps
        print("finished optics size fit")
        logger.info("finished optics size fit")   
        nfev = results.nfev
        nvarys = results.nvarys
        maxfev = 2000*(nvarys+1)
        print("nfev: {0}".format(nfev))
        print("nvarys: {0}".format(nvarys))
        print("maxfev: {0}".format(maxfev))
        logger.info("nfev: {0}".format(nfev))
        logger.info("nvarys: {0}".format(nvarys))
        logger.info("maxfev: {0}".format(maxfev))
        if nfev >=maxfev:
            print("nfev >=maxfev; fit likely failed; aborting")
            logger.info("nfev >=maxfev; fit likely failed; aborting")
            raise AttributeError("nfev >=maxfev; fit likely failed; aborting")

        # set final fit
        logger.info('Optical fit from lmfit parameters:')
        logger.info(lmfit.fit_report(results, min_correl=0.5))

        self.optatmo_psf_kwargs['size'] = results.params.valuesdict()['size']
        if results.errorbars:
            err = np.sqrt(results.covar[0, 0])
            self.optatmo_psf_kwargs['error_size'] = err
        else:
            logger.warning('No error calculated for size in fit_size')
        self._update_optatmopsf(self.optatmo_psf_kwargs, logger=logger)

        # save this for debugging purposes
        self._fit_size_results = results

        # remove saved values from the reference wavefront caches when we are done with the fit
        if self.reference_wavefront: self._delete_cache(logger=logger)
        if self.reference_wavefront: self._delete_cache_higher_order(logger=logger)

    def fit_atmosphere(self, stars,
                       chisq_threshold=0.1, max_iterations=30, logger=None): # Note: This is not currently being used. Instead, the atmospheric fitting is currently being done in the PIFF fitting pipeline itself. As a result, it has not been updated in a while and it is not known if it is compatible with the current version of the PIFF fitting pipeline.
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
            logger.info("Setting _enable_atmosphere == False. Was {0}".format(self._enable_atmosphere))
            self._enable_atmosphere = False

        # fit models
        logger.info("Initial Fitting atmo model")
        params = self.getParamsList(stars)
        model_fitted_stars = []
        for star_i, star in zip(range(len(stars)), stars):
            try:
                model_fitted_star, results = self.fit_model(star, params=params[star_i], vary_shape=True, vary_optics=False, mode=self.fit_atmosphere_mode, logger=logger)
                model_fitted_stars.append(model_fitted_star)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.warning('{0}'.format(str(e)))
                logger.warning('Warning! Failed to fit atmosphere model for star {0}. Ignoring star in atmosphere fit'.format(star_i))

        logger.debug("Stripping star fit params down to just atmosphere params for fitting with the atmo_interp")
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
                    logger.debug('Removing star {0} based on MAD. params are {1}'.format(si, str(params[si])))
            if len(mad_stars) != len(stars):
                logger.info('Stripped stars from {0} to {1} based on 5sig MAD cut'.format(len(stars), len(mad_stars)))
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
                logger.warning("Iteration %d: Fitting %d stars", iteration+1, len(stars))

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
                params[:, 0:self.n_params_atmosphere] += aberrations_atmo_star

                # refluxing star and get chisq
                refluxed_stars = []
                for param, star in zip(params, stars_interp):
                    refluxed_star, res = self.fit_model(star, param, vary_shape=False, vary_optics=False, logger=logger)
                    refluxed_stars.append(refluxed_star)

                # put back into the refluxed stars the fitted model params. This way when outliers returns the new list, we won't have to refit those parameters (which will be the same as earlier)
                reparam_stars = []
                for params, params_var, star in zip(fitted_model_params, fitted_model_params_var, refluxed_stars):
                    new_star = Star(star.data, StarFit(params, params_var=params_var, flux=star.fit.flux, center=star.fit.center, chisq=star.fit.chisq, dof=star.fit.dof, alpha=star.fit.alpha, beta=star.fit.beta, worst_chisq=star.fit.worst_chisq))
                    reparam_stars.append(new_star)

                # Perform outlier rejection
                logger.debug("             Looking for outliers")
                nonoutlier_stars, nremoved1 = self.outliers.removeOutliers(reparam_stars, logger=logger)
                if nremoved1 == 0:
                    logger.debug("             No outliers found")
                else:
                    logger.info("             Removed %d outliers", nremoved1)
                nremoved += nremoved1

                stars = nonoutlier_stars

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

        return model_fitted_stars, stars

    def fit_model(self, star, params, vary_shape=True, vary_optics=False,  mode='pixel', minimize_kwargs={}, logger=None, estimated_errorbars_not_required=False):
        """Fit model to star's pixel data. Always vary flux and center, but also can selectively vary atmospheric terms and Zernike coefficients
        :param star:        A Star instance
        :param params:      An array of initial star parameters like one would
                            get from getParams
        :param vary_shape:  Boolean. If true, will vary Kolmogorov size and
                            ellipticity in fit [default: True]
        :param vary_optics: Boolean. If true, will vary Zernike coefficients
                            during fit [default: False]. This is only
                            moderately useful in most in focus cases.
        :param mode:        Parameter mode ['shape', 'pixel']. Dictates which residual function we use. Default is pixel
        :param minimize_kwargs: A set of parameters to pass in for changing the
                            way lmfit does the minimization.
        :param logger:      A logger object for logging debug info. [default: None]
        :returns:           New Star instance and results, with updated flux,
                            center, chisq, dof, and fit params and params_var
        """
        logger = LoggerWrapper(logger)
        import lmfit
        # create lmparameters
        # put in initial guesses for flux, du, dv if they exist
        flux = star.fit.flux
        if flux == 1.:
            # a pretty reasonable first guess is to just take the sum of the pixels
            flux = star.image.array.sum()
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

        # cache optical profile if vary_optics is set to False
        if vary_optics == False:
            aberrations = np.zeros(4 + len(params[self.n_params_constant_atmosphere_and_atmosphere:]))  # fill piston etc with 0
            aberrations[4:] = params[self.n_params_constant_atmosphere_and_atmosphere:]
            aberrations = aberrations * (700.0/self.optical_psf_kwargs['lam'])
            optical_profile = galsim.OpticalPSF(aberrations=aberrations,
                                gsparams=self.gsparams,
                                **self.optical_psf_kwargs)
        else:
            optical_profile = None

        # measure the shape and shape error of the data star 
        shape = self.measure_shape_orthogonal(star)
        error = self.measure_error_orthogonal(star)

        # optical residuals are used to get the initial starting values for atmo_size, atmo_g1, and atmo_g2 if vary_optics is set to False
        if vary_optics == False:
            # calculate the second moment residuals for a model star made only with values from an optical fit
            full_profile = self.getProfile(params)
            star_model = self.drawProfile(star, full_profile, params)
            shape_model = self.measure_shape_orthogonal(star_model)
            shape_difference = shape - shape_model
            optical_de0 = shape_difference[3]
            optical_de1 = shape_difference[4]
            optical_de2 = shape_difference[5]
            # linear relations between between the optical residals of the second moments and atmo_size/atmo_g1/atmo_g2 taken by comparing the two against each other for a couple dozen exposures
            fit_size = 1.143 * optical_de0 + 0.001
            fit_g1 = 1.075 * optical_de1 + 0.001
            fit_g2 = 1.033 * optical_de2
        else:
            # getParams is used to get the initial starting values for atmo_size, atmo_g1, and atmo_g2 if vary_optics is set to True
            fit_size = params[0]
            fit_g1 = params[1]
            fit_g2 = params[2]

        # acquire the values of opt_size, opt_g1, opt_g2, and, possibly, opt_L0 if using vonkarman atmosphere
        if self.atmosphere_model == 'vonkarman':
            opt_L0 = params[3]
        opt_size = params[self.n_params_constant_atmosphere + 0]
        opt_g1 = params[self.n_params_constant_atmosphere + 1]
        opt_g2 = params[self.n_params_constant_atmosphere + 2]      

        # finish putting in atmo_size, atmo_g1, and atmo_g2 terms
        lmparams.add('atmo_size', value=fit_size, vary=vary_shape, min=min_size - opt_size, max=max_size - opt_size)
        lmparams.add('atmo_g1', value=fit_g1, vary=vary_shape, min=-max_g - opt_g1, max=max_g - opt_g1)
        lmparams.add('atmo_g2', value=fit_g2, vary=vary_shape, min=-max_g - opt_g2, max=max_g - opt_g2)

        # put in the other params to the params model
        if self.atmosphere_model == 'vonkarman':
            lmparams.add('optics_L0', value=opt_L0, vary=False) # we NEVER vary the opt_size, opt_g1, opt_g2, or, if we use it, opt_L0
        lmparams.add('optics_size', value=opt_size, vary=False)
        lmparams.add('optics_g1', value=opt_g1, vary=False)
        lmparams.add('optics_g2', value=opt_g2, vary=False)
        for i, pi in enumerate(params[self.n_params_constant_atmosphere_and_atmosphere:]):
            lmparams.add('optics_zernike_{0}'.format(i + 4), value=pi, vary=vary_optics, min=-5, max=5) # we do allow zernikes to vary if vary_optics is set to True

        # set up the residual to either use moments or pixels, as desired
        if mode == 'shape':
            residual = self._fit_model_shape_residual
        elif mode == 'pixel':
            residual = self._fit_model_residual

        # prepare minimize_kwargs_in for fit
        minimize_kwargs_in = {'method': 'leastsq', 'epsfcn': 1e-5, 'maxfev': 1000}
        minimize_kwargs_in.update(minimize_kwargs)

        # do fit
        if mode == 'shape':
            results = lmfit.minimize(residual, lmparams,
                                 args=(star, optical_profile, shape, error, logger,),
                                 **minimize_kwargs_in)
        else:
            results = lmfit.minimize(residual, lmparams,
                                 args=(star, optical_profile, logger,),
                                 **minimize_kwargs_in)
        nfev = results.nfev
        nvarys = results.nvarys
        maxfev = 2000*(nvarys+1)
        if nfev >=maxfev:
            print("nfev: {0}".format(nfev))
            print("nvarys: {0}".format(nvarys))
            print("maxfev: {0}".format(maxfev))
            logger.info("nfev: {0}".format(nfev))
            logger.info("nvarys: {0}".format(nvarys))
            logger.info("maxfev: {0}".format(maxfev))
            print("nfev >=maxfev; fit likely failed; aborting")
            logger.info("nfev >=maxfev; fit likely failed; aborting")
            raise AttributeError("nfev >=maxfev; fit likely failed; aborting")

        logger.debug(lmfit.fit_report(results, min_correl=0.5))
        if not results.success:
            raise AttributeError('Not successful fit')

        # errors can be zero if the chisq is close to perfect
        if ((not results.errorbars) * (results.chisqr > 1e-8)):
            if estimated_errorbars_not_required:
                print("Warning: No estimated errorbars")
                logger.info('Warning: No estimated errorbars')
            else:
                raise AttributeError('No estimated errorbars')

        # subtract 3 for the flux, du, dv
        fit_params = np.zeros(len(results.params) - 3)
        params_var = np.zeros(len(fit_params))
        for i, key in enumerate(results.params):
            indx = i - 3
            if key in ['flux', 'du', 'dv']:
                continue
            param = results.params[key]
            fit_params[indx] = param.value
            if param.vary:
                var = 0.0
                if hasattr(param, 'stderr'):
                    if type(param.stderr) == "NoneType":
                        if estimated_errorbars_not_required:  # if estimated errorbars not required param variances likely also not required; this is likely if you are using fit_model() just to do a refluxing for a star
                            print("Warning: param.stderr is NoneType")
                            logger.info('Warning: param.stderr is NoneType')
                        else:
                            raise AttributeError('param.stderr is NoneType')
                    else:
                        var = param.stderr ** 2
                params_var[indx] = var

        flux = results.params['flux'].value
        du = results.params['du'].value
        dv = results.params['dv'].value
        center = (du, dv)
        chisq = results.chisqr
        dof = results.nfree
        fit = StarFit(fit_params, params_var=params_var, flux=flux, center=center,
                      chisq=chisq, dof=dof)
        star_fit = Star(star.data, fit)
        return star_fit, results

    def stripStarList(self, stars, logger=None):
        """take star fits and strip fit params to just the first three
        parameters, which correspond to the atmospheric terms. Keep flux and
        center but get rid of everything else
        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info.
                                [default: None]
        :returns:               A list of stars with only num_keep fit params
        """
        num_keep = self.n_params_atmosphere
        new_stars = []
        for star_i, star in enumerate(stars):
            try:
                fit_params = star.fit.params
                new_fit_params = fit_params[:num_keep]
            except AttributeError:
                logger.debug("Star {0} has no fit params".format(star_i))
                new_fit_params = None
            try:
                fit_params_var = star.fit.params_var
                new_fit_params_var = fit_params_var[:num_keep]
            except AttributeError:
                logger.debug("Star {0} has no fit params_var".format(star_i))
                new_fit_params_var = None
            new_fit = StarFit(new_fit_params, params_var=new_fit_params_var,
                              flux=star.fit.flux, center=star.fit.center)
            new_star = Star(star.data, new_fit)
            new_stars.append(new_star)
        return new_stars

    def adjustStarList(self, stars, logger=None):
        """Fit the Model to the star's data, varying only the flux and center.
        :param stars:       A list of Stars
        :param logger:      A logger object for logging debug info. [default: None]
        :returns:           New Star instances, with updated flux, center, chisq, dof
        """
        logger = LoggerWrapper(logger)
        params = self.getParamsList(stars)
        stars_adjusted = []
        for star, param in zip(stars, params):
            star_adjusted, results = self.fit_model(star, param, vary_shape=False, vary_optics=False, logger=logger)
            stars_adjusted.append(star_adjusted)
        return stars_adjusted

    def adjustStar(self, star, logger=None):
        """Fit the Model to the star's data, varying only the flux and center.
        :param star:        A Star instance
        :param logger:      A logger object for logging debug info. [default: None]
        :returns:           New Star instance, with updated flux, center, chisq, dof
        """
        return self.adjustStarList([star], logger=logger)[0]

    def reflux(self, star, logger=None):
        """Fit the Model to the star's data, varying only the flux and center. This puts one of the options for fit_model into the regular Piff syntax.
        :param star:        A Star instance
        :param logger:      A logger object for logging debug info. [default: None]
        :returns:           New Star instance, with updated flux, center, chisq, dof
        Notes
        -----
        This is just adjustStar but with a name more like other Piff models
        """
        return self.adjustStar(star, logger=logger)

    def _fit_optics_lmparams(self, optatmo_psf_kwargs, keys):
        """turns optatmo_psf_kwargs and set of keys to fit into an lmparams object
        :param optatmo_psf_kwargs:  Dictionary with keys like (for parameter
                                    "size") size [value to start fit at],
                                    fix_size [do not allow parameter to vary],
                                    min_,max_size [min and maximum values
                                    allowed for size during fit] (in this example
                                    "size" is proportional to the average of 1/r0 
                                    across the focal plane, r0 being the Fried 
                                    these are specified, will fill with guessed
                                    values.
        :param keys:                List of keys we want to add to the lmfit
                                    parameters.
        :returns lmparams:          An lmfit Parameters object
        """
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

    def _fit_random_forest_residual(self, lmparams, stars, shapes, shape_errors, regr_dictionary, logger=None):
        """Residual function for fitting optics via random forest model. 
        This is what is done in "random_forest" mode.
        :param lmparams:        LMFit Parameters object
        :param stars:           A list of Stars
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
        self._update_optatmopsf(lmparams.valuesdict(), logger)

        # get star params
        params_all = self.getParamsList(stars)
        param_values_all_stars = params_all[:,self.n_params_constant_atmosphere:self.n_params_constant_atmosphere+11]
        number_of_rows, number_of_columns = param_values_all_stars.shape
        if number_of_columns < 11:
            param_values_all_stars_copy = copy.deepcopy(param_values_all_stars)
            param_values_all_stars = np.zeros((number_of_rows,11))
            param_values_all_stars[:,:number_of_columns] = param_values_all_stars_copy

        # generate the stars' moments using random forest model and the fit parameters of the stars
        #note: only up to third moments used for the random forest fit
        shapes_model_list = []
        for m, moment in enumerate(np.array(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"])):    
            regr = regr_dictionary[moment]    
            shapes_model_list.append(regr.predict(param_values_all_stars))
        shapes_model = np.column_stack(tuple(shapes_model_list))
        shape_weights = self._shape_weights[:7]

        # calculate chi. Exclude measurements of flux and centroids
        shapes = shapes[:, 3:10]
        errors = shape_errors[:, 3:10]
        chi = (shape_weights[None] * (shapes_model - shapes) / errors).flatten() #chi is
        logger.debug('Current Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))

        # chi is a one-dimensional numpy array, containing moment_weight*((moment_model-moment)/moment_error) for up to all moments up to third moments, for all stars
        # the model moments in this case are based on what the random forest model returns for a model star with a given set of fit parameters
        return chi

    def _fit_size_residual(self, lmparams, stars, shapes, shape_errors, logger=None):
        """Residual function for fitting the optics size parameter to the
        observed e0 moment. The size parameter is proportional to 1/r0, r0 
        being the Fried parameter. The "optics" size is the average of this
        across the focal plane, whereas "atmospheric" size is the deviation
        from this average at different points in the focal plane. This 
        function calls _fit_optics_residual and then limits the chi to only 
        use the e0 moment.
        :param lmparams:        LMFit Parameters object
        :param stars:           A list of Stars
        :param shapes:          A list of premeasured Star shapes
        :param shape_errors:    A list of premeasured Star shape errors
        :param logger:          A logger object for logging debug info.
                                [default: None]
        :returns chi:           Chi of observed e0 to model e0
        Notes
        -----
        This is done by forward modeling the PSF and measuring its shape via HSM
        """
        logger = LoggerWrapper(logger)
        # this residual is only used to find the optics size offset when using fit_size()
        # fit_size() is used before the full optical fit and makes that fit faster
        chi = self._fit_optics_residual(lmparams, stars, shapes, shape_errors, logger, only_size=True)
        chi = chi[0::self.length_of_moments_list] / self._shape_weights[0::self.length_of_moments_list]
        logger.debug('Current Total Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))
        # chi is a one-dimensional numpy array, containing e0_weight*((e0_model-e0)/e0_error) for all stars
        return chi

    def _fit_optics_residual(self, lmparams, stars, shapes, shape_errors, logger=None, only_size=False):
        """Residual function for fitting the optical fit parameters and the average values of the atmospheric
        fit parameters to the observed shapes. Fitting is done via lmfit.
        :param lmparams:          LMFit Parameters object
        :param stars:             A list of Stars
        :param shapes:            A list of premeasured Star shapes
        :param shape_errors:      A list of premeasured Star shape errors
        :param logger:            A logger object for logging debug info.
                                  [default: None]
        :param only_size:         Boolean. If False, record the reduced
                                  chisq at each iteration
                                  [default: False]                                  
        :returns chi:             Chi of observed shapes to model shapes
        Notes
        -----
        This is done by forward modeling the PSF and measuring its shape via HSM
        """
        logger = LoggerWrapper(logger)
        # update psf
        self._update_optatmopsf(lmparams.valuesdict(), logger)

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
                profile = self.getProfile(params)

                # measure final shape
                star_model = self.drawProfile(star, profile, params)
                shape_model = self.measure_shape_orthogonal(star_model)
                if np.any(shape_model != shape_model):
                    logger.warning('Star {0} returned nan shape'.format(i))
                    logger.warning('Parameters are {0}'.format(str(params)))
                    logger.warning('Input parameters are {0}'.format(str(lmparams.valuesdict())))
                    logger.warning('Filling with zero chi')
                    shape_model = shape
            except (ModelFitError, RuntimeError) as e:
                logger.warning(str(e))
                logger.warning('Star {0}\'s model failed to be drawn and measured.'.format(i))
                logger.warning('Parameters are {0}'.format(str(params)))
                logger.warning('Input parameters are {0}'.format(str(lmparams.valuesdict())))
                logger.warning('Filling with zero chi')
                shape_model = shape

            # don't care about flux, du, dv here
            chi_i = self._shape_weights * (((shape_model - shape) / error)[3:])
            chi = np.hstack((chi, chi_i))
        
        if only_size == False:
            logger.debug('Current Total Chi2 / dof is {0:.4e} / {1}'.format(np.sum(np.square(chi)), len(chi)))
            self.total_redchi_across_iterations.append(np.sum(np.square(chi))/len(chi))
        self.final_optical_chi = chi
        # chi is a one-dimensional numpy array, containing moment_weight*((moment_model-moment)/moment_error) for all moments, for all stars
        return chi

    def _fit_optics_pixel_residual(self, lmparams, stars, shapes, errors, logger=None): #not necessarily set up to work with vonkarman atmosphere; also not currently used because too slow
        """Residual function for fitting all stars. The only difference between this 
        function and _fit_optics_residual is that pixels instead of shapes are used here.
        :param lmparams:    LMFit Parameters object
        :param stars:       A list of Stars
        :param shapes:      A list of premeasured Star shapes
        :param errors:      A list of premeasured Star shape errors
        :param logger:      A logger object for logging debug info.
                            [default: None]
        :returns chi:       Chi of observed pixels of all stars to model pixels after fitting for flux, centering, and atmospheric size / ellipticity
        """
        logger = LoggerWrapper(logger)
        # update psf
        self._update_optatmopsf(lmparams.valuesdict(), logger)

        # get optical params
        opt_params = self.getParamsList(stars)

        chi = np.array([])
        for i, star in enumerate(stars):
            params = opt_params[i]
            image, weight, image_pos = star.data.getImage()

            try:

                # put in fit pixel values as first guesses for the fit_model
                star.fit.flux = self._fit_pixel_fluxes[i]
                star.fit.center = self._fit_pixel_centers[i]
                params[0] = self._fit_pixel_sizes[i]
                params[1] = self._fit_pixel_g1s[i]
                params[2] = self._fit_pixel_g2s[i]
                # TODO: because I hardcoded the numbers, I start L0 from whatever my initial guess was, if we do vonkarman

                # do fit marginalizing over the atmosphere shape
                fitted_star, fitted_results = self.fit_model(star, params, vary_shape=True, vary_optics=False, mode='pixel', logger=logger)

                # draw star for evaluating the chi2
                prof = self.getProfile(fitted_star.fit.params).shift(fitted_star.fit.center) * fitted_star.fit.flux
                image_model = self.drawProfile(fitted_star, prof, fitted_star.fit.params, use_fit=False).image

                # update fit pixel values
                self._fit_pixel_fluxes[i] = fitted_star.fit.flux
                self._fit_pixel_centers[i] = fitted_star.fit.center
                self._fit_pixel_sizes[i] = fitted_star.fit.params[0]
                self._fit_pixel_g1s[i] = fitted_star.fit.params[1]
                self._fit_pixel_g2s[i] = fitted_star.fit.params[2]
                self._fit_pixel_sizes_vars[i] = fitted_star.fit.params_var[0]
                self._fit_pixel_g1s_vars[i] = fitted_star.fit.params_var[1]
                self._fit_pixel_g2s_vars[i] = fitted_star.fit.params_var[2]

            except (ModelFitError, RuntimeError) as e:
                logger.warning(str(e))
                logger.warning('Star {0}\'s model failed to be drawn and measured.'.format(i))
                logger.warning('Parameters are {0}'.format(str(params)))
                logger.warning('Input parameters are {0}'.format(str(lmparams.valuesdict())))
                logger.warning('Filling with zero chi')

                image_model = image

            chi_i = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
            chi = np.hstack((chi, chi_i))
        chi2 = np.sum(chi * chi)
        logger.debug('fit optics residual chi2 / dof = {0:.3f} / {1} = {2:.3f}'.format(chi2, len(chi), chi2 * 1. / len(chi)))

        return chi

    def _fit_model_shape_residual(self, lmparams, star, optical_profile, shape, error, logger=None): 
        # Note: Indications are that this alternative to _fit_model_residual() is slower. As a result, it has not been updated in a while and it is not known if it is compatible with the current version of the PIFF fitting pipeline.
        """Residual function for fitting individual profile parameters to observed shapes.
        :param lmparams:    lmfit Parameters object
        :param star:        A Star instance.
        :param logger:      A logger object for logging debug info.
                            [default: None]
        :returns chi:       Chi of observed shape params to model params
        """
        logger = LoggerWrapper(logger)

        all_params = np.array(list(lmparams.valuesdict().values()))
        flux, du, dv = all_params[:3]
        params = all_params[3:]
        if optical_profile == None:
            # if no optical profile, make the entire profile from scratch
            profile = self.getProfile(params).shift(du, dv) * flux
        else:
            # if there is an optical profile, just make the atmospheric profile
            # add stochastic (labelled "atm") and constant (labelled "opt") pieces together
            if self.atmosphere_model == 'kolmogorov':
                size = params[0] + params[3]
                g1 = params[1] + params[4]
                g2 = params[2] + params[5]
                atmo = galsim.Kolmogorov(gsparams=self.gsparams,
                                         **self.kolmogorov_kwargs
                                         ).dilate(size).shear(g1=g1, g2=g2)
            elif self.atmosphere_model == 'vonkarman':
                size = params[0] + params[4]
                g1 = params[1] + params[5]
                g2 = params[2] + params[6]
                L0 = params[3]
                kolmogorov_kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                                     'r0': self.kolmogorov_kwargs['r0'] / size,
                                     'L0': L0,}
                atmo = galsim.VonKarman(gsparams=self.gsparams,
                                        **kolmogorov_kwargs).shear(g1=g1, g2=g2)

            # convolve together
            profile = galsim.Convolve([optical_profile, atmo], gsparams=self.gsparams).shift(du, dv) * flux

        star_model = self.drawProfile(star, profile, params, use_fit=False)
        shape_model = self.measure_shape_orthogonal(star_model) # maybe don't include flux and center in the chi?; maybe don't even float flux and center if using shape mode for fit_model()

        chi = (shape_model - shape) / error

        return chi

    def _fit_model_residual(self, lmparams, star, optical_profile, logger=None):
        """Residual function for fitting individual profile parameters to observed pixels.
        :param lmparams:    lmfit Parameters object
        :param star:        A Star instance.
        :param logger:      A logger object for logging debug info.
                            [default: None]
        :returns chi:       Chi of observed pixels to model pixels
        """
        logger = LoggerWrapper(logger)

        all_params = np.array(list(lmparams.valuesdict().values()))
        flux, du, dv = all_params[:3]
        params = all_params[3:]
        if optical_profile == None:
            # if no optical profile, make the entire profile from scratch
            prof = self.getProfile(params).shift(du, dv) * flux
        else:
            # if there is an optical profile, just make the atmospheric profile
            # add stochastic (labelled "atm") and constant (labelled "opt") pieces together
            if self.atmosphere_model == 'kolmogorov':
                size = params[0] + params[3]
                g1 = params[1] + params[4]
                g2 = params[2] + params[5]
                atmo = galsim.Kolmogorov(gsparams=self.gsparams,
                                         **self.kolmogorov_kwargs
                                         ).dilate(size).shear(g1=g1, g2=g2)
            elif self.atmosphere_model == 'vonkarman':
                size = params[0] + params[4]
                g1 = params[1] + params[5]
                g2 = params[2] + params[6]
                L0 = params[3]
                kolmogorov_kwargs = {'lam': self.kolmogorov_kwargs['lam'],
                                     'r0': self.kolmogorov_kwargs['r0'] / size,
                                     'L0': L0,}
                atmo = galsim.VonKarman(gsparams=self.gsparams,
                                        **kolmogorov_kwargs).shear(g1=g1, g2=g2)

            # convolve together
            prof = galsim.Convolve([optical_profile, atmo], gsparams=self.gsparams).shift(du, dv) * flux

        # calculate chi
        image, weight, image_pos = star.data.getImage()
        image_model = self.drawProfile(star, prof, params, use_fit=False).image
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

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
            
    def _create_cache_higher_order(self, stars, logger=None):
        """Save aberrations from the higher order reference wavefront. The purpose
        of this is the same as with the _create_cache() function except that
        _create_cache() saves up to z11, while this saves from z22 to z37.
        :param stars:   A list of stars
        :param logger:  A logger object for logging debug info [default: None]
        """
        if self.reference_wavefront: #if no reference wavefront, assume no higher order reference wavefront either
            logger.debug('Caching higher order reference aberrations')
            self._cache_higher_order = True
            aberrations_higher_order_reference_wavefront = np.empty([len(stars),26])
            for s, star in enumerate(stars):
                star_data = star.data
                star_data.local_wcs = star_data.image.wcs.local(star_data.image_pos)
                x_value = star_data.local_wcs._x(star_data['u'], star_data['v'])
                y_value = star_data.local_wcs._y(star_data['u'], star_data['v'])
                x_value = x_value * (15.0/1000.0)
                y_value = y_value * (15.0/1000.0)  
                zout_camera = self.higher_order_reference_wavefront.get(x=x_value, y=y_value)
                zout_sky = np.array([-zout_camera[0], zout_camera[1], zout_camera[2], -zout_camera[3], zout_camera[5], zout_camera[4], -zout_camera[7], -zout_camera[6], zout_camera[9], zout_camera[8], zout_camera[10], zout_camera[11], -zout_camera[12], -zout_camera[13], zout_camera[14], zout_camera[15], -zout_camera[16], zout_camera[18], zout_camera[17], -zout_camera[20], -zout_camera[19], zout_camera[22], zout_camera[21], -zout_camera[24], -zout_camera[23], zout_camera[25]]) #conversion from zout_camera (AOS system) to zout_sky (Galsim) inspired by thesis of Chris Davis
                aberrations_higher_order_reference_wavefront[s] = zout_sky
            self._aberrations_higher_order_reference_wavefront = aberrations_higher_order_reference_wavefront
        else:
            logger.debug('Higher Order Cache called, but no reference wavefront. Skipping') #if no reference wavefront, assume no higher order reference wavefront either
            self._cache_higher_order = False
            self._aberrations_higher_order_reference_wavefront = None

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
        
    def _delete_cache_higher_order(self, logger=None):
        """Delete higher order reference wavefront cache.
        :param logger:  A logger object for logging debug info [default: None]
        """
        if self.reference_wavefront: #if no reference wavefront, assume no higher order reference wavefront either
            logger.debug('Clearing cache_higher_order of reference aberrations')
        else:
            logger.debug('Delete cache_higher_order called, but no reference wavefront. Skipping') #if no reference wavefront, assume no higher order reference wavefront either
        self._cache_higher_order = False
        self._aberrations_higher_order_reference_wavefront = None
