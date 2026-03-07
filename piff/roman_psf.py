# Copyright (c) 2026 by Mike Jarvis and the other collaborators on GitHub at
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
.. module:: roman_psf
"""

import galsim
import galsim.roman
import numpy as np
import scipy.linalg

from .interp import Interp
from .model import Model
from .outliers import Outliers
from .psf import PSF
from .star import Star


def _get_sca(star):
    # We allow the SCA to be specified either as an sca property (preferred) or as the
    # more generic chipnum property.  This helper function encapsulates that for the
    # several places we access this information about a star.
    if 'sca' in star.data.properties:
        return int(star.data.properties['sca'])
    if 'chipnum' in star.data.properties:
        return int(star.data.properties['chipnum'])
    raise ValueError("RomanOptics requires an explicit 'sca' property for each star")


class RomanSCAInterp(Interp):
    """Interpolate Roman aberration vectors as constants per SCA."""
    _type_name = 'RomanSCA'

    def __init__(self, per_sca=True, logger=None):
        self.per_sca = per_sca
        self.degenerate_points = False
        self.kwargs = {'per_sca': self.per_sca}
        self.sca_mean = {}
        self.global_mean = None
        self.set_num(None)

    @property
    def property_names(self):
        return ('sca',)

    def solve(self, stars, logger=None):
        params = np.array([star.fit.get_params(self._num) for star in stars])
        sca = np.array([_get_sca(star) for star in stars], dtype=int)
        self.global_mean = np.mean(params, axis=0)
        self.sca_mean = {}
        if self.per_sca:
            for s in np.unique(sca):
                self.sca_mean[s] = np.mean(params[sca == s], axis=0)

    def set_sca_solution(self, sca_mean):
        self.global_mean = np.mean(list(sca_mean.values()), axis=0)
        self.sca_mean = dict(sca_mean) if self.per_sca else {}

    def interpolate(self, star, logger=None, inplace=False):
        sca = _get_sca(star)
        if self.per_sca and sca in self.sca_mean:
            # The second clause should rarely be false.
            # But if we need to interpolate onto an SCA that did not have any PSF stars,
            # this prevents a runtime error.  Using the global mean is probably as good
            # as anything for that case.
            params = self.sca_mean[sca]
        else:
            params = self.global_mean
        if params is None:
            # This can happen during initialization.  sca_mean/global_mean are not set yet.
            return star

        if inplace:
            star.fit.updateParams(params, num=self._num)
            return star
        else:
            return Star(star.data, star.fit.newParams(params, num=self._num))

    def _finish_write(self, writer):
        if self.global_mean is None:
            return
        sca = sorted(self.sca_mean)
        mean = np.array([self.sca_mean[s] for s in sca])
        data = np.array(
            list(zip(sca, mean)),
            dtype=[('sca', int), ('mean', float, (len(self.global_mean),))],
        )
        writer.write_table('solution', data)
        writer.write_table(
            'global',
            np.array(
                [(self.global_mean,)],
                dtype=[('mean', float, (len(self.global_mean),))],
            ),
        )

    def _finish_read(self, reader):
        data = reader.read_table('solution')
        self.sca_mean = {}
        if data is not None:
            for row in data:
                self.sca_mean[row['sca']] = np.array(row['mean'], dtype=float)
        global_data = reader.read_table('global')
        self.global_mean = (
            None if global_data is None else np.array(global_data['mean'][0], dtype=float)
        )


class Roman(Model):
    """Model a Roman PSF using GalSim's built-in Roman optical model.

    The expensive GalSim Roman PSF construction is approximated as a bilinear function across
    each SCA. For each ``(sca, params)`` state we build corner PSFs at the four SCA corners and
    interpolate between them at each star position. This keeps the optical model much faster when
    many stars on a chip share similar parameter vectors.
    """

    _type_name = 'Roman'
    _method = 'auto'
    _centered = False
    _model_can_be_offset = False

    def __init__(
        self,
        filter,
        chromatic=True,
        max_zernike=22,
        aberration_prior_sigma=1.0,
        logger=None,
    ):
        self.logger = logger
        self.filter = filter
        self.chromatic = chromatic
        self.max_zernike = int(max_zernike)
        self.set_num(None)

        if self.max_zernike < 4 or self.max_zernike > 22:
            raise ValueError("max_zernike must be in the range 4..22")

        # Notation: filter is the string name of the filter.
        #           bandpass is the galsim.Bandpass object with the transmission function.
        bandpasses = galsim.roman.getBandpasses()
        if self.filter not in bandpasses:
            raise ValueError("Roman filter %r is not a valid GalSim Roman bandpass" % self.filter)
        self.bandpass = bandpasses[self.filter]
        self.prior_sigma = self._parse_prior_sigma(aberration_prior_sigma)
        self.prior_invsigsq = 1.0/self.prior_sigma**2 if self.prior_sigma is not None else None

        self.kwargs = {
            'filter': self.filter,
            'chromatic': self.chromatic,
            'max_zernike': self.max_zernike,
            'aberration_prior_sigma': self.prior_sigma if self.prior_sigma is not None else None,
        }
        self.sca_size = float(galsim.roman.n_pix)
        self.clear_cache()

    @property
    def param_len(self):
        return self.max_zernike - 3

    def initialize_iteration(self):
        # No per-iteration setup needed for Roman currently.
        pass

    def clear_cache(self):
        self._corner_cache = {}
        self._sca_wcs = {}

    def _draw_profile_to_image(self, prof, image, center):
        if self.chromatic:
            prof.drawImage(
                image,
                bandpass=self.bandpass,
                method=self._method,
                center=center,
            )
        else:
            prof.drawImage(
                image,
                method=self._method,
                center=center
            )

    def _solve_params(self, aw, bw, params):
        nparam = len(params)
        ridge = 1.0e-6  # Mild ridge regression for regularization
        ata = aw.T.dot(aw)
        atb = aw.T.dot(bw)
        ata.flat[::nparam + 1] += ridge
        self._apply_prior(ata, atb, params)
        dparams = scipy.linalg.lstsq(ata, atb)[0]
        new_params = params + dparams
        cov = np.linalg.pinv(ata)
        var = np.diag(cov)
        return new_params, var

    def _parse_prior_sigma(self, aberration_prior_sigma):
        if aberration_prior_sigma is None:
            return None
        sigma = np.array(aberration_prior_sigma, dtype=float).ravel()
        if sigma.size == 0:
            raise ValueError("aberration_prior_sigma may not be empty")
        if sigma.size == 1:
            sigma = np.full(self.param_len, sigma[0], dtype=float)
        elif sigma.size != self.param_len:
            raise ValueError(
                "aberration_prior_sigma must be a scalar or length %d"
                % self.param_len
            )
        if np.any(sigma <= 0):
            raise ValueError("aberration_prior_sigma values must all be > 0")
        return sigma

    def _apply_prior(self, ata, atb, params):
        if self.prior_invsigsq is None:
            return
        nparam = len(params)
        ata.flat[::nparam + 1] += self.prior_invsigsq
        atb -= self.prior_invsigsq * params

    def initialize(self, star, logger=None, default_init=None):
        params = np.zeros(self.param_len, dtype=float)
        params_var = np.zeros_like(params)
        fit = star.fit.newParams(params, params_var=params_var, num=self._num)
        return Star(star.data, fit)

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        # We usually batch the stars in groups by SCA and do the fitting calculation
        # together for efficiency of building the adjusted corner profiles.  So this
        # function is rarely used.  However, it's part of the Interp API, so just
        # farm this out to the fit_many function with a list of one star.
        return self.fit_many(
            [star],
            logger=logger,
            convert_funcs=[convert_func],
            draw_method=draw_method,
        )[0]

    def fit_many(self, stars, logger=None, convert_funcs=None, draw_method=None):
        if convert_funcs is None:
            convert_funcs = [None] * len(stars)
        elif len(convert_funcs) != len(stars):
            raise ValueError("len(convert_funcs) must match len(stars)")

        grouped = {}
        for i, (star, convert_func) in enumerate(zip(stars, convert_funcs)):
            sca = _get_sca(star)
            if sca not in grouped:
                grouped[sca] = {'indices': [], 'stars': [], 'convert_funcs': []}
            grouped[sca]['indices'].append(i)
            grouped[sca]['stars'].append(star)
            grouped[sca]['convert_funcs'].append(convert_func)

        out = [None] * len(stars)
        sca_mean = {}
        for sca, group in grouped.items():
            fit_group, mean_params = self._fit_sca_group(
                sca,
                group['stars'],
                logger=logger,
                draw_method=draw_method,
                convert_funcs=group['convert_funcs'],
            )
            sca_mean[sca] = mean_params
            # Output the stars in the same order as input, so they stay matched with
            # convert_funcs array if there is one.
            for i, star in zip(group['indices'], fit_group):
                out[i] = star
        self._last_sca_mean = sca_mean
        return out

    def _fit_sca_group(self, sca, stars, logger=None, draw_method=None, convert_funcs=None):
        ref = stars[0]
        params = ref.fit.get_params(self._num)

        if convert_funcs is None:
            convert_funcs = [None] * len(stars)

        # We're about to change params, so if the corner profiles aren't yet in the cache,
        # there is not reason now to add them.
        corner_base_profiles = self._get_corner_profiles(ref, params, cache=False, sca=sca)

        # First draw the base image for each star.
        # Also set up an empty Jacobian array for each to be filled in below.
        nparam = len(params)
        group_data = []
        for star, convert_func in zip(stars, convert_funcs):
            image = star.image.array
            weight = star.weight.array
            base_image = self._draw_model_image_from_corners(
                star,
                corner_base_profiles,
                convert_func=convert_func,
            )
            resid = image - base_image
            sw = np.sqrt(weight.ravel())
            bw = resid.ravel() * sw
            jac = np.empty((image.size, nparam), dtype=float)
            group_data.append((star, convert_func, weight, sw, bw, jac, base_image))

        # Build the Jacobians using finite differences.
        steps = 0.05 * np.maximum(np.abs(params), 0.02)  # min step = 1.e-3
        for i, step in enumerate(steps):
            p1 = params.copy()
            p1[i] += step
            corner_plus_profiles = self._get_corner_profiles(ref, p1, cache=False, sca=sca)
            scale = 1.0 / step
            # Use these adjusted corner profiles for all stars on the SCA.
            for star, convert_func, _, _, _, jac, base_image in group_data:
                im1 = self._draw_model_image_from_corners(
                    star,
                    corner_plus_profiles,
                    convert_func=convert_func,
                )
                jac[:, i] = ((im1 - base_image) * scale).ravel()

        # Finish the fits now that all the Jacobians are calculated.
        solved = []
        solved_params = []
        for star, convert_func, weight, sw, bw, jac, _ in group_data:
            aw = jac * sw[:, np.newaxis]
            new_params, var = self._solve_params(aw, bw, params)
            solved.append((star, convert_func, weight, var))
            solved_params.append(new_params)

        # Use one common parameter vector per SCA for the final model/chisq pass.
        sca_params = np.mean(solved_params, axis=0)
        corner_sca_profiles = self._get_corner_profiles(ref, sca_params, cache=True, sca=sca)

        # Compute the final model and chisq for each star.
        out = []
        for star, convert_func, weight, var in solved:
            model_image = self._draw_model_image_from_corners(
                star, corner_sca_profiles, convert_func=convert_func
            )
            chisq = np.sum(weight * (star.image.array - model_image) ** 2)
            dof = np.count_nonzero(weight) - 3
            out.append(
                Star(
                    star.data,
                    star.fit.newParams(
                        sca_params, params_var=var, num=self._num, chisq=chisq, dof=dof
                    ),
                )
            )
        return out, sca_params

    def draw(self, star, copy_image=True):
        params = star.fit.get_params(self._num)
        prof = self.getProfile(params, star=star).shift(star.fit.center) * star.fit.flux
        image = star.image.copy() if copy_image else star.image
        self._draw_profile_to_image(prof, image, center=star.image_pos)
        return Star(star.data.withNew(image=image), star.fit)

    def getProfile(self, params=None, star=None, cache=True):
        if star is None:
            raise ValueError("Roman.getProfile requires the star argument")
        if params is None:
            params = np.zeros(self.param_len, dtype=float)
        corner_profiles = self._get_corner_profiles(star, params, cache=cache)
        return self._interpolate_corners(star, corner_profiles)

    def _draw_model_image(self, star, params, cache=True, convert_func=None):
        corner_profiles = self._get_corner_profiles(star, params, cache=cache)
        return self._draw_model_image_from_corners(
            star, corner_profiles, convert_func=convert_func
        )

    def _draw_model_image_from_corners(self, star, corner_profiles, convert_func=None):
        prof = self._interpolate_corners(star, corner_profiles)
        prof = prof.shift(star.fit.center) * star.fit.flux
        if convert_func is not None:
            prof = convert_func(prof)
        image = star.image.copy()
        self._draw_profile_to_image(prof, image, star.image_pos)
        return image.array

    def _get_corner_profiles(self, star, params, cache=True, sca=None):
        if sca is None:
            sca = _get_sca(star)
        if sca in self._corner_cache:
            cached_params, cached_profiles = self._corner_cache[sca]
            if np.array_equal(cached_params, params):
                return cached_profiles

        if sca not in self._sca_wcs:
            self._sca_wcs[sca] = star.data.local_wcs
        wcs = self._sca_wcs[sca]
        wavelength = None if self.chromatic else self.bandpass.effective_wavelength
        corners = (
            galsim.PositionD(0.0, 0.0),
            galsim.PositionD(self.sca_size, 0.0),
            galsim.PositionD(0.0, self.sca_size),
            galsim.PositionD(self.sca_size, self.sca_size),
        )
        profiles = tuple(
            galsim.roman.getPSF(
                sca,
                self.filter,
                SCA_pos=corner,
                wcs=wcs,
                extra_aberrations=params,
                wavelength=wavelength,
            )
            for corner in corners
            )
        if cache:
            self._corner_cache[sca] = (params, profiles)
        return profiles

    def _interpolate_corners(self, star, corner_profiles):
        w_ll, w_lr, w_ul, w_ur = self._corner_weights(star)
        ll, lr, ul, ur = corner_profiles
        return w_ll * ll + w_lr * lr + w_ul * ul + w_ur * ur

    def _corner_weights(self, star):
        x = np.clip(star.image_pos.x, 0.0, self.sca_size)
        y = np.clip(star.image_pos.y, 0.0, self.sca_size)
        fx = x / self.sca_size
        fy = y / self.sca_size
        return (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        )


class RomanOptics(PSF):
    """A PSF wrapper for Roman optical fits with per-SCA or global aberration interpolation."""

    _type_name = 'RomanOptics'

    def __init__(self, model, interp, outliers=None, chisq_thresh=0.1, min_iter=2, max_iter=30):
        self.model = model
        self.interp = interp
        self.outliers = outliers
        self.chisq_thresh = chisq_thresh
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.kwargs = {
            'model': 0,
            'interp': 0,
            'outliers': 0,
            'chisq_thresh': self.chisq_thresh,
            'min_iter': self.min_iter,
            'max_iter': self.max_iter,
        }
        self.chisq = 0.0
        self.last_delta_chisq = 0.0
        self.dof = 0
        self.nremoved = 0
        self.niter = 0
        self.degenerate_points = False
        self.set_num(None)

    def set_num(self, num):
        self._num = num
        # During read, model and interp might not be real.
        # Only call set_num if they are actually built.
        if isinstance(self.model, Model):
            self.model.set_num(num)
        if isinstance(self.interp, Interp):
            self.interp.set_num(num)

    @property
    def interp_property_names(self):
        return self.interp.property_names

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        kwargs = dict(config_psf)
        kwargs.pop('type', None)

        per_sca = kwargs.pop('per_sca', True)
        outliers = kwargs.pop('outliers', None)
        chisq_thresh = kwargs.pop('chisq_thresh', 0.1)
        min_iter = kwargs.pop('min_iter', 2)
        max_iter = kwargs.pop('max_iter', 30)

        model = Roman(logger=logger, **kwargs)
        interp = RomanSCAInterp(per_sca=per_sca)

        parsed = {
            'model': model,
            'interp': interp,
            'chisq_thresh': chisq_thresh,
            'min_iter': min_iter,
            'max_iter': max_iter,
        }
        if outliers is not None:
            parsed['outliers'] = Outliers.process(outliers, logger=logger)
        return parsed

    def initialize_params(self, stars, logger=None, default_init=None):
        nremoved = 0

        # Start each new fit with a fresh corner-profile cache.
        self.model.clear_cache()

        logger.debug("Initializing models")
        new_stars = []
        for star in stars:
            try:
                star = self.model.initialize(star, logger=logger, default_init=default_init)
            except Exception as e:
                logger.warning("Failed initializing star at %s. Excluding it.", star.image_pos)
                logger.warning("  -- Caught exception: %s", e)
                nremoved += 1
                star = star.flag_if(True)
            new_stars.append(star)
        if nremoved == 0:
            logger.debug("No stars removed in initialize step")
        else:
            logger.verbose("Removed %d stars in initialize", nremoved)

        logger.debug("Initializing interpolator")
        stars = self.interp.initialize(new_stars, logger=logger)

        return stars, nremoved

    def single_iteration(self, stars, logger, convert_funcs, draw_method):
        self.model.initialize_iteration()
        all_stars = list(stars)

        fit_indices = [
            k for k, star in enumerate(stars) if not star.is_flagged and not star.is_reserve
        ]
        fit_stars = [stars[k] for k in fit_indices]
        fit_convert_funcs = (
            [None] * len(fit_stars)
            if convert_funcs is None
            else [convert_funcs[k] for k in fit_indices]
        )

        new_fit_stars = self.model.fit_many(
            fit_stars,
            logger=logger,
            draw_method=draw_method,
            convert_funcs=fit_convert_funcs,
        )
        for k, new_star in zip(fit_indices, new_fit_stars):
            all_stars[k] = new_star

        if not new_fit_stars:
            raise RuntimeError("No stars left to fit.  Cannot find PSF model.")

        logger.debug("             Calculating the interpolation")
        self.interp.set_sca_solution(self.model._last_sca_mean)
        all_stars = self.interp.interpolateList(all_stars)
        return all_stars, 0

    @property
    def fit_center(self):
        return self.model._centered

    @property
    def include_model_centroid(self):
        return self.model._centered and self.model._model_can_be_offset

    def interpolateStarList(self, stars, inplace=False):
        return self.interp.interpolateList(stars, inplace=inplace)

    def interpolateStar(self, star, inplace=False):
        return self.interp.interpolate(star, inplace=inplace)

    def _drawStar(self, star):
        return self.model.draw(star)

    def _getRawProfile(self, star):
        params = star.fit.get_params(self._num)
        return self.model.getProfile(params, star=star), self.model._method

    def _finish_write(self, writer, logger):
        from .config import LoggerWrapper
        logger = LoggerWrapper(logger)
        chisq_dict = {
            'chisq': self.chisq,
            'last_delta_chisq': self.last_delta_chisq,
            'dof': self.dof,
            'nremoved': self.nremoved,
            'niter': self.niter,
        }
        writer.write_struct('chisq', chisq_dict)
        logger.debug("Wrote the chisq info to %s", writer.get_full_name('chisq'))
        self.model.write(writer, 'model')
        logger.debug("Wrote the PSF model to %s", writer.get_full_name('model'))
        self.interp.write(writer, 'interp')
        logger.debug("Wrote the PSF interp to %s", writer.get_full_name('interp'))
        if self.outliers:
            Outliers.write_all(writer, 'outliers', self.outliers)
            logger.debug("Wrote the PSF outliers to %s", writer.get_full_name('outliers'))

    def _finish_read(self, reader, logger):
        chisq_dict = reader.read_struct('chisq')
        for key in chisq_dict:
            setattr(self, key, chisq_dict[key])
        self.model = Model.read(reader, 'model')
        self.interp = Interp.read(reader, 'interp')
        self.outliers = Outliers.read_all(reader, 'outliers')
