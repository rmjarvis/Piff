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
.. module:: phase_psf
"""

from __future__ import print_function

import galsim
import numpy as np

from .psf import PSF
from .optical_model import Optical
from .interp import Interp
from .gsobject_model import Kolmogorov
from .star import Star, StarFit, StarData

# TODO: incorporate weights_moment_fit into fitting
class PhasePSF(PSF):
    """Combine Optical and Atmospheric PSFs together
    """

    def __init__(self, atmo_interp=None, analytic_coefs=None, phase_psf_kwargs={}, optical_psf_kwargs={}, kolmogorov_kwargs={}, reference_wavefront=None, n_optfit_stars=0, weights_moment_fit=[0.5, 1, 1], fov_radius=1., jmax_pupil=11, jmax_focal=10,  min_optfit_snr=0, optfit_optimize='analytic', logger=None):
        """
        Instead of arbitrarily putting different models in and dealing with
        convolutions of models, put into single piece here.

        Fitting is done with lmfit, which also lets us specificy which params are fixed
        as we desire. This is akin to the gsobject, but we can explicitly fit parameters.

        Further, the interpolation and model are integrated together.

        Things I am thinking about right now that we will need and I am likely to forget:
            - A function to strip star parameters for the atmosphere interpolant to fit
            - A function to go from u,v position to zernike polynomials. Could one day be replaced by galsim implementation
            - make sure when we fit the constant piece (which includes r0, g1, g2) that the atmosphere piece does NOT go beyond constant component

        In the end, this will completely replace the Optical Model

        Fit Combined Atmosphere and Optical PSF in two stage process

        :param atmo_interp:             Piff Interpolant object that represents
                                        the atmospheric interpolation
        :param analytic_coefs:          Terms in analytic breakdown of zernike
                                        to shape transformation. Only used in
                                        when optfit_optimize = analytic
        :param phase_psf_kwargs:        Terms that set the state of the PSF,
                                        excepting the atmospheric interpolant
        :param optical_psf_kwargs:      Arguments to pass into galsim opticalpsf object
        :param kolmogorov_kwargs:       Arguments to pass into galsim kolmogorov object
        :param reference_wavefront:     Reference interpolator for the optical
                                        wavefront. Takes in stars, resturns
                                        aberrations. Default is to not include it.
        :param n_optfit_stars:          [default: 0] If > 0, randomly sample
                                        only n_optfit_stars for the fit
        :param weights_moment_fit:      Array or list of weights_moment_fit for
                                        comparing gaussian shapes in fit
                                        [default: [0.5, 1, 1], so downweight
                                        size]
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
        self.jmax_pupil = jmax_pupil
        self.jmax_focal = jmax_focal

        self.fov_radius = fov_radius

        # Field-of-view does not have obscuration, so obscuration=0 and annular=False here.
        self._noll_coef_field = galsim.phase_screens._noll_coef_array(self.jmax_focal, 0.0, False)

        self.phase_psf_kwargs = {
            'r0': 0.15, 'fix_r0': False, 'error_r0': 1e-2,
            'g1': 0,    'fix_g1': False, 'error_g1': 1e-2,
            'g2': 0,    'fix_g2': False, 'error_g2': 1e-2,
            }
        self.keys = [ 'r0', 'g1', 'g2', ]
        # throw in default zernike parameters
        for zi in range(4, self.jmax_pupil + 1):  # only fit zernikes starting at 4 / defocus
            for dxy in range(1, self.jmax_focal + 1):
                zkey = 'zUV{0:03d}_zXY{1:03d}'.format(zi, dxy)
                self.keys.append(zkey)
                # initial value
                self.phase_psf_kwargs[zkey] = 0
                # fix if greater than max focal zernike
                if dxy > self.jmax_focal:
                    self.phase_psf_kwargs['fix_' + zkey] = True
                else:
                    self.phase_psf_kwargs['fix_' + zkey] = False
                # # we shall not specify in advance a zernike limit
                # self.phase_psf_kwargs['limit_' + zkey] = (-2, 2)
                # initial placeholder for error in parameter
                zerror = 10000
                self.phase_psf_kwargs['error_' + zkey] = zerror
        try:
            self.phase_psf_kwargs.update(phase_psf_kwargs)
        except TypeError:
            # should just be from loading, and is fixed by _finish_read
            pass

        # create initial aberrations_field from phase_psf_kwargs
        logger.info("Initializing optpsf state")
        self.aberrations_field = np.zeros((self.jmax_pupil, self.jmax_focal), dtype=float)
        self._update_optpsf(self.phase_psf_kwargs, logger)

        # since we haven't fit the interpolator, yet, disable atmosphere
        logger.info('Disabling Varying Atmosphere')
        self._enable_atmosphere = False

        # set up gsobject model to measure shapes
        logger.info("Loading Kolmogorov for modeling shapes")
        self.shape_modeller = Kolmogorov(fastfit=False, force_model_center=False, include_pixel=True, unnormalized_basis=True, logger=logger)

        self.weights_moment_fit = np.array(weights_moment_fit)
        # normalize weights_moment_fit
        self.weights_moment_fit /= self.weights_moment_fit.sum()

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
                       # junk entries to be overwritten in _finish_read function
                       'phase_psf_kwargs': 0,
                       'atmo_interp': 0,
                       'reference_wavefront': 0,
                       'weights_moment_fit': 0,

                       }

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        config_psf = config_psf.copy()  # Don't alter the original dict.

        kwargs = {}

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
        if 'phase_psf_kwargs' in config_psf:
            kwargs['phase_psf_kwargs'] = config_psf['phase_psf_kwargs']

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

        # write the atmo interp
        self.atmo_interp.write(fits, extname + '_atmo_interp', logger)

        # write reference wavefront
        self.reference_wavefront.write(fits, extname + '_reference_wavefront', logger)

        # write the final fitted state
        dtypes = [('weights_moment_fit', '3f4')]
        for key in self.phase_psf_kwargs:
            if 'fix_' in key:
                dtypes.append((key, bool))
            else:
                dtypes.append((key, float))
        data = np.zeros(1, dtype=dtypes)
        data['weights_moment_fit'] = self.weights_moment_fit
        for key in self.phase_psf_kwargs:
            data[key][0] = self.phase_psf_kwargs[key]

        fits.write_table(data, extname=extname + '_solution')
        logger.info('Wrote optpsf state')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # read the atmo interp
        self.atmo_interp = Interp.read(fits, extname + '_atmo_interp', logger)

        # read reference wavefront
        self.reference_wavefront = Interp.read(fits, extname + '_reference_wavefront', logger)

        # read the final state, update the psf
        data = fits[extname + '_solution'].read()
        for key in data.dtype.names:
            if key == 'weights_moment_fit':
                self.weights_moment_fit = data['weights_moment_fit'][0]
            else:
                self.phase_psf_kwargs[key] = data[key][0]
        logger.info('Reloading optpsf state')
        self._update_optpsf(self.phase_psf_kwargs, logger)

        # enable atmosphere
        logger.info('Enabling Varying Atmosphere')
        self._enable_atmosphere = True

    def fit(self, stars, wcs, pointing, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the
                                telescope pointing.
                                [Note: pointing should be None if the WCS is
                                not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing

        # fit optical interpolant piece including constant atmosphere to
        # either: analytic moments, actual moments, pixels. Mode is decided in
        # __init__ of PSF
        logger.info('Fitting optics')
        self.fit_optics(stars, wcs, pointing, logger=logger, **kwargs)

        # fit atmosphere
        logger.info('Fitting atmosphere')
        self.fit_atmosphere(stars, wcs, pointing, logger=logger, **kwargs)

        # enable atmosphere interpolation now that we have solved the interp
        logger.debug('Enabling Atmosphere')
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

        # TODO: are the aberrations here ordered in j or j-1? ie if I go aberrations_pupil[4] do I get z4, or do I get z5? I _think_ it is the former.

        return aberrations_pupil

    def getParamsList(self, stars):
        """Get params for a list of stars.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           Params  [size, g1, g2, z4, z5...] for each star
        """
        params = np.zeros((len(stars), 3 + self.jmax_pupil), dtype=float)

        aberrations_pupil = self._getParamsList_aberrations_field(stars)
        params += aberrations_pupil

        if self.reference_wavefront:
            # TODO: this could be remade to just be for u, v coordinates centered about the center instead of pixel_to_focal stuff
            stars_interpolated = self.reference_wavefront.interpolateList(stars)
            aberrations_reference_wavefront = np.array([star_interpolated.fit.params for star_interpolated in stars_interpolated])
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
            atmo_stars = [Star(star.data, None) for star in stars]
            interp_atmo_stars = self.atmo_interp.interpolateList(atmo_stars)
            aberrations_atmo_star = np.array([interp_atmo_star.fit.params for interp_atmo_star in interp_atmo_stars])
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
        if len(params) >= 3:
            aberrations = [0,0,0,0] + list(params[3:])
            opt = galsim.OpticalPSF(aberrations=aberrations, **self.optical_psf_kwargs)

        # atmosphere
        size = params[0]
        g1 = params[1]
        g2 = params[2]
        atmo = galsim.Kolmogorov(**self.kolmogorov_kwargs).dilate(size).shear(g1=g1, g2=g2)

        # convolve together!
        prof = galsim.Convolve([opt, atmo])
        return prof

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """

        # get profile and params
        params = self.getParams(star)
        prof = self._profile(params)
        star = self._drawProfile(star, prof, params)
        return star

    @classmethod
    def _drawProfile(cls, star, prof, params):
        """Generate PSF image for a given star and profile

        :param profile:     A galsim profile
        :param star:        Star instance holding information needed for
                            interpolation as well as an image/WCS into which
                            PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # since I do this in a couple different places, let's do it in one place
        image = star.image.copy()
        image = prof.drawImage(image, method='auto', offset=(star.image_pos-image.trueCenter()))
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
        return Star(data, StarFit(params))

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           List of Star instances with its image filled with rendered PSF
        """
        # get all params at once
        params = self.getParamsList(stars)
        # now step through to make the stars
        stars_drawn = [self._drawProfile(star, self._profile(param), param) for param, star in zip(params, stars)]
        return stars_drawn

    def _update_optpsf(self, phase_psf_kwargs={}, logger=None):
        """Update aberrations_field attribute, recompute coef arrays for the field

        :param _coef_arrays_field:      The arrays that take an input complex number r = u + iv and its square abs(u + iv) ** 2 to compute the Zernike aberrations
        :param aberrations_field:      A two-dimensional array of coefficients.  First index is over nm, i.e. the
                            pupil aberrations.  The second index is over rs, i.e., the field dependence. Both are defined according to Noll convention
        :param logger:      A logger object for logging debug info
        """
        if len(phase_psf_kwargs) == 0:
            phase_psf_kwargs = self.phase_psf_kwargs
            keys = self.keys
        else:
            keys = phase_psf_kwargs.keys()

        aberrations_field = np.zeros((self.jmax_pupil, self.jmax_focal), dtype=float)
        for key in keys:
            # some checks
            if 'limit_' in key:
                continue
            elif 'error_' in key:
                continue
            elif 'fix_' in key:
                continue

            # r0, g1, g2 mean constant terms
            if key == 'r0':
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
                    raise Exception('Not allowed to fit pupil zernike {0}, key {1}!'.format(uv, key))
            aberrations_field[uv - 1, xy - 1] = phase_psf_kwargs[key]
            old_value = self.aberrations_field[uv - 1, xy - 1]
            new_value = aberrations_field[uv - 1, xy - 1]
            if old_value != new_value:
                logger.debug('Updating Zernike parameter {0} from {1:+.2e} to {2:+.2e}'.format(key, old_value, new_value))

        self.aberrations_field = aberrations_field
        # One coef_array for each wavefront aberration
        # shape (jmax_pupil, maxn_focal, maxm_focal)
        self._coef_arrays_field = np.array([np.dot(self._noll_coef_field, a)
                                            for a in self.aberrations_field])

    @staticmethod
    def measure_shape(star, gsobject_model, logger=None):
        import lmfit
        # measure shape of star using a piff gsobject_model
        # in order to get to the covariance, use model lmfit directly
        if not hasattr(star.data.properties, 'hsm'):
            star = gsobject_model.initialize(star)
        # TODO: could redo fit here in favored unnormalized moment basis (vs T, g's)
        params = gsobject_model._lmfit_params(star)
        results = gsobject_model._lmfit_minimize(params, star, logger=logger)
        logger.debug(lmfit.fit_report(results))
        flux, du, dv, scale, g1, g2 = results.params.valuesdict().values()
        shape = np.array([scale, g1, g2])
        error = np.sqrt(np.diag(results.covar)[3:])
        return shape, error

    def fit_optics(self, stars, wcs, pointing, logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
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

        indices = np.arange(len(stars))
        if self.n_optfit_stars and self.n_optfit_stars < len(stars):
            np.random.shuffle(indices)
            max_stars = self.n_optfit_stars
        else:
            max_stars = len(stars)

        for indx in indices:
            if len(self._opt_stars) > max_stars:
                if logger:
                    logger.debug("Finishing opt_star measurements at {0} stars".format(len(self._opt_stars)))
                break

            if logger:
                logger.log(5, "Measuring shape of star {0}".format(indx))
            star = stars[indx]
            if self.min_optfit_snr > 0:
                snr = self.measure_snr(star)
                if snr < self.min_optfit_snr:
                    if logger:
                        logger.log(5, "Skipping star {0} because SNR {1} < {2}".format(indx, snr, self.min_optfit_snr))
                    continue

            shape, error = self.measure_shape(star, self.shape_modeller, logger)
            if np.any(shape != shape):
                if logger:
                    logger.log(5, "Skipping star {0} because of NaN in shape")
                continue
            if np.any(error != error):
                if logger:
                    logger.log(5, "Skipping star {0} because of NaN in shape error")
                continue

            self._opt_stars.append(star)
            self._opt_snrs.append(snr)
            self._opt_shapes.append(shape)
            self._opt_shape_errors.append(error)

        if logger:
            if len(self._opt_stars) < max_stars:
                logger.debug("Using {0} stars instead of desired {1} out of {2}".format(len(self._opt_stars), max_stars, len(stars)))

        # create lmparameters
        params = lmfit.Parameters()
        # step through keys
        for key in self.keys:
            value = self.phase_psf_kwargs[key]
            vary = not self.phase_psf_kwargs['fix_' + key]
            if 'limit_' + key in self.phase_psf_kwargs:
                min, max = self.phase_psf_kwargs['limit_' + key]
            else:
                min, max = (None, None)
            params.add(key, value=value, vary=vary, min=min, max=max)

        # do fit
        results = lmfit.minimize(resid, params, args=(logger,), maxfev=self.kwargs['max_iterations'])

        # set final fit
        if logger:
            logger.info('Optical fit from lmfit parameters:')
        key_i = 0
        for key in self.keys:
            if not self.phase_psf_kwargs['fix_' + key]:
                val = results.params.valuesdict()[key]
                self.phase_psf_kwargs[key] = val
                logstring = '{0}:\t{1:+.2e}'.format(key, val)

                err = np.sqrt(results.covar[key_i, key_i])
                self.phase_psf_kwargs['error_' + key] = err
                logstring += '\t{0:.2e}'.format(err)

                if logger:
                    logger.info(logstring)

                key_i += 1

    @staticmethod
    def measure_snr(star):
        """Calculate the signal-to-noise of a given star. Copied from input.py

        :param star:        Input star, with stamp, weight

        :returns: the SNR value.
        """
        # The S/N value that we use will be the weighted total flux where the weight function
        # is the star's profile itself.  This is the maximum S/N value that any flux measurement
        # can possibly produce, which will be closer to an in-practice S/N than using all the
        # pixels equally.
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
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optpsf(params, logger)

        # generate stars
        stars_model = self.drawStarList(self._opt_stars)

        # calculate chi
        chi = np.array([])
        for star, star_model in zip(self.stars, stars_model):
            image, weight, image_pos = star.data.getImage()
            image_model, weight_model, image_pos_model = star_model.data.getImage()
            chi_i = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
            chi = np.hstack((chi, chi_i))

        return chi

    def _fit_optics_residual_moments(self, lmparams, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optpsf(params, logger)

        # generate stars
        stars_model = self.drawStarList(self._opt_stars)

        # measure their shapes and calculate chi
        chi = []
        for i, star, shape, error in zip(np.arange(len(stars_model)), stars_model, self._opt_shapes, self._opt_errors):
            shape_model, error_model = self.measure_shape(star, self.shape_modeller, logger)
            if np.any(shape_model != shape_model):
                if logger:
                    logger.debug('Warning! Shape measurement failed for model of star {0}'.format(i))
                continue
            chi.append(self.weights_moment_fit * (shape_model - shape) / error)
        chi = np.array(chi).flatten()

        return chi

    def _fit_optics_residual_analytic(self, lmparams, logger=None):
        # convert lmparams instance
        params = lmparams.valuesdict()

        # update psf
        self._update_optpsf(params, logger)

        # get star params
        zernikes = self.getParamsList(self._opt_stars)

        # generate analytic star moments
        shapes_model = self.analytic_zernike_shapes(zernikes, self.analytic_coefs)

        # calculate chi
        chi = (self.weights_moment_fit[None] * (shapes_model - self._opt_shapes) / self._opt_errors).flatten()

        return chi

    def fit_atmosphere(self, stars, logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        if self._enable_atmosphere:
            logger.debug("Setting _enable_atmosphere == False. Was {0}".format(self._enable_atmosphere))
            self._enable_atmosphere = False

        logger.debug("Initializing atmo interpolator")
        initialized_stars = self.atmo_interp.initialize(stars, logger=logger)

        # fit models
        logger.debug("Fitting atmo model")
        model_fitted_stars = [self._fit_atmosphere_model(star, logger=logger)
                              for star in initialized_stars]

        # fit interpolant
        logger.debug("Fitting atmo interpolant")
        self.atmo_interp.solve(model_fitted_stars, logger=logger)

    def _fit_atmosphere_model(self, star, logger=None):
        import lmfit
        # we also fit flux, du, dv but those are less important

        # create lmparameters
        params = lmfit.Parameters()
        # Order of params is important!
        params.add('flux', value=1.0, vary=True, min=0.0)
        params.add('du', value=0, vary=True)
        params.add('dv', value=0, vary=True)
        # scale can be negative here
        # params.add('scale', value=1.0, vary=True, min=0.0)
        params.add('scale', value=0, vary=True, min=-2., max=2.)
        # Limits of +/- 0.7 is definitely a hack to avoid |g| > 1, but if the PSF is ever actually
        # this elliptical then we have more serious problems to worry about than hacky code!
        params.add('g1', value=0, vary=True, min=-0.7, max=0.7)
        params.add('g2', value=0, vary=True, min=-0.7, max=0.7)

        # do fit
        results = lmfit.minimize(self._fit_atmosphere_model_residual, params, args=(star, logger,), maxfev=self.kwargs['max_iterations'])
        flux, du, dv, scale, g1, g2 = results.params.valuesdict().values()
        params = np.array([ scale, g1, g2 ])
        center = (du, dv)
        fit = StarFit(params, flux=flux, center=center)
        return Star(star.data, fit)

    def _fit_atmosphere_model_residual(self, lmparams, star, logger=None):
        # convert lmparams instance
        params = self.getParams(star)

        # modify params with lmparams
        flux, du, dv, scale, g1, g2 = lmparams.valuesdict().values()
        params[0] = params[0] + scale
        params[1] = params[1] + g1
        params[2] = params[2] + g2

        prof = self._profile(params).shift(du, dv) * flux

        # calculate chi2
        image, weight, image_pos = star.data.getImage()
        image_model = image.copy()
        image_model = prof.drawImage(image_model, method='auto', offset=(star.image_pos-image_model.trueCenter()))
        chi = (np.sqrt(weight.array) * (image_model.array - image.array)).flatten()
        return chi

    @staticmethod
    def analytic_zernike_shapes(zernikes, coefs):
        # transform into full index
        params_onehot = np.vstack((np.ones(len(zernikes)).T, zernikes.T)).T
        nsample, ndim = params_onehot.shape
        ppoly_full = (params_onehot[:, :, None, None] * params_onehot[:, None, :, None] * params_onehot[:, :2][:, None, None, :])
        ppoly = ppoly_full.reshape(nsample, 2 * ndim * ndim)

        # apply model
        shapes = np.array([np.dot(ppoly, coef) for coef in coefs]).T
        return shapes
