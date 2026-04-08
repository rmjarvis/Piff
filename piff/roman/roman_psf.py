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
import copy

from ..interp import Interp
from ..model import Model
from ..outliers import Outliers
from ..psf import PSF
from ..star import Star
from ..config import LoggerWrapper
from ..util import make_flat
from ..util import run_multi

# Global control of GalSim Roman pupil-plane resolution in getPSF calls.
# Kept module-level so tests can override to faster values (e.g. 8 or 16).
pupil_bin = 4


def _get_sca(star):
    # We allow the SCA to be specified either as an sca property (preferred) or as the
    # more generic chipnum property.  This helper function encapsulates that for the
    # several places we access this information about a star.
    if 'sca' in star.data.properties:
        return int(star.data.properties['sca'])
    elif 'chipnum' in star.data.properties:
        return int(star.data.properties['chipnum'])
    else:
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
        # Writes an n_param length array 'mean' for each 'sca'
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


class RomanOpticalModel(Model):
    """Model a Roman PSF using GalSim's built-in Roman optical model.

    The expensive GalSim Roman PSF construction is approximated across each SCA by evaluating
    PSFs at a small set of fixed sample points and interpolating between them at each star
    position. ``nominal_interp='bilinear'`` uses the four SCA corners; ``'five_point'`` adds
    the SCA center and uses a five-term basis ``[1, x, y, x*y, (x^2+y^2)]`` in normalized
    coordinates.

    The ``aberration_interp`` option controls how the fitted extra aberrations vary across an
    SCA. ``'global'`` and ``'constant'`` both use one fitted aberration vector per SCA in the
    model, with ``'global'`` only changing how the outer interpolator shares those vectors across
    SCAs. ``'linear'`` fits a planar extra-aberration field on each SCA, modeled as
    ``a + b x + c y`` in normalized SCA coordinates.
    """

    _type_name = 'Roman'
    _method = 'auto'
    _centered = True
    _model_can_be_offset = False

    def __init__(
        self,
        chromatic=True,
        filter_name=None,
        max_zernike=22,
        aberration_interp='constant',
        nominal_interp='bilinear',
        aberration_prior_sigma=0.05,
        nproc=1,
        logger=None,
    ):
        self.logger = logger
        self.chromatic = chromatic
        self.max_zernike = int(max_zernike)
        self.aberration_interp = str(aberration_interp)
        self.nominal_interp = str(nominal_interp)
        self.nproc = int(nproc)
        self.bandpass = None
        self.flat_bandpass = None
        self.filter_name = filter_name
        self.set_num(None)

        if self.max_zernike < 4 or self.max_zernike > 22:
            raise ValueError("max_zernike must be in the range 4..22")
        if self.aberration_interp not in ('global', 'constant', 'linear'):
            raise ValueError("aberration_interp must be one of 'global', 'constant', 'linear'")
        if self.nominal_interp not in ('bilinear', 'five_point'):
            raise ValueError("nominal_interp must be one of 'bilinear', 'five_point'")
        # Save the original for serialization.
        self.orig_prior_sigma = self._parse_prior_sigma(aberration_prior_sigma)
        self.prior_sigma = self._expand_prior_sigma(self.orig_prior_sigma)
        self.prior_invsigsq = 1.0/self.prior_sigma**2 if self.prior_sigma is not None else None

        self.kwargs = {
            'chromatic': self.chromatic,
            'filter_name': self.filter_name,
            'max_zernike': self.max_zernike,
            'aberration_interp': self.aberration_interp,
            'nominal_interp': self.nominal_interp,
            'aberration_prior_sigma': self.orig_prior_sigma,
            'nproc': self.nproc,
        }
        self.sca_size = float(galsim.roman.n_pix)
        self._roman_five_point_data = {}
        self.clear_cache()

    def __getstate__(self):
        # Do not pickle logger instances for multiprocessing jobs.
        state = dict(self.__dict__)
        state['logger'] = None
        return state

    @property
    def param_len(self):
        if self.aberration_interp == 'linear':
            return 3 * self._single_point_param_len
        else:
            return self._single_point_param_len

    @property
    def _single_point_param_len(self):
        return self.max_zernike - 3

    def initialize_iteration(self):
        # No per-iteration setup needed for Roman currently.
        pass

    def clear_cache(self):
        self._corner_cache = {}

    def set_bandpass(self, bandpass, filter_name=None):
        if self.filter_name is None:
            if filter_name is not None:
                self.filter_name = filter_name
            elif getattr(bandpass, 'name', None) is not None:
                self.filter_name = bandpass.name
            else:
                raise ValueError(
                    "RomanOptics requires filter_name when bandpass has no name attribute"
                )
        self.bandpass = bandpass
        self.flat_bandpass = make_flat(bandpass)

    def _require_bandpass(self):
        if self.bandpass is None:
            raise ValueError("RomanOptics requires bandpass to be set before use")
        return self.bandpass

    def _get_sed_eff(self, star):
        sed_eff = star.data.properties.get('sed_eff')
        if sed_eff is None:
            raise ValueError(
                "RomanOptics with chromatic=True requires each star to have an 'sed_eff' property"
            )
        return sed_eff

    def _get_roman_five_point_data(self, sca):
        if sca in self._roman_five_point_data:
            return self._roman_five_point_data[sca]

        aberrations, x_pos, y_pos = galsim.roman.roman_psfs._read_aberrations(sca)
        center = galsim.PositionD(x_pos[0], y_pos[0])

        # The project aberrations data has aberration values at 5 points on each SCA:
        # the four corners plus the center.  The center is always at position 0, but
        # the others aren't necessarily in the same order each time.  So figure out
        # which one is which.
        # cf. _interp_aberrations_bilinear in galsim.roman.roman_psfs.py
        ll = ul = lr = ur = None
        for i in range(1, 5):
            if x_pos[i] < x_pos[0] and y_pos[i] < y_pos[0]:
                ll = i
            if x_pos[i] < x_pos[0] and y_pos[i] > y_pos[0]:
                ul = i
            if x_pos[i] > x_pos[0] and y_pos[i] < y_pos[0]:
                lr = i
            if x_pos[i] > x_pos[0] and y_pos[i] > y_pos[0]:
                ur = i
        assert None not in (ll, ul, lr, ur)

        # Order as (ll, lr, ul, ur, center) to match the rest of this class.
        idx = (ll, lr, ul, ur, 0)
        points = tuple(galsim.PositionD(x_pos[i], y_pos[i]) for i in idx)

        # Build the solver matrix for basis [1, x, y, xy, x^2+y^2].
        basis = np.empty((5, 5), dtype=float)
        px = np.array([x_pos[i] for i in idx], dtype=float)
        py = np.array([y_pos[i] for i in idx], dtype=float)
        basis[0,:] = 1.0
        basis[1,:] = px
        basis[2,:] = py
        basis[3,:] = px * py
        basis[4,:] = px * px + py * py
        # If f(x,y) = c·phi(x,y), where phi(x,y) = [1, x, y, xy, x^2+y^2],
        # then sampled values satisfy s = B c, where B[:,k] = phi(x_k,y_k).
        # For any target point, weights w that interpolate from
        # sample values are w = B^{-1} phi(target), so that f(target) = w·s.
        solver = np.linalg.inv(basis)

        # GalSim uses bilinear nominal interpolation, so correct center by adding this delta.
        bilinear_center = galsim.roman.roman_psfs._interp_aberrations_bilinear(
            aberrations, x_pos, y_pos, center
        )
        delta = np.array(aberrations[0] - bilinear_center, dtype=float)
        center_delta = np.zeros(self.max_zernike + 1, dtype=float)
        center_delta[4:] = delta[4:self.max_zernike + 1]

        out = {'points': points, 'solver': solver, 'center_delta': center_delta}
        self._roman_five_point_data[sca] = out
        return out

    def _make_extra_aberrations(self, params):
        # GalSim expects extra_aberrations indexed by Zernike number.  Our parameter vector
        # corresponds to z4..z_max, so pad indices 0..3.
        extra_aberrations = np.zeros(self.max_zernike + 1, dtype=float)
        extra_aberrations[4:] = params
        return extra_aberrations

    def _draw_profile_to_image(self, prof, image, center, star):
        if self.chromatic:
            prof = prof * self._get_sed_eff(star)
            prof.drawImage(self.flat_bandpass, image=image, method=self._method, center=center)
        else:
            prof.drawImage(image=image, method=self._method, center=center)

    def _solve_params(self, aw, bw, params):
        ata = aw.T.dot(aw)
        atb = aw.T.dot(bw)
        return self._solve_normal_equations(ata, atb, params)

    def _solve_normal_equations(self, ata, atb, params, ridge=1.0e-6):
        nparam = len(params)
        # Mild ridge regression for regularization
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
        nprior = self._single_point_param_len
        if sigma.size == 1:
            sigma = np.full(nprior, sigma[0], dtype=float)
        elif sigma.size != nprior:
            raise ValueError(
                "aberration_prior_sigma must be a scalar or length %d" % nprior
            )
        if np.any(sigma <= 0):
            raise ValueError("aberration_prior_sigma values must all be > 0")
        return sigma

    def _expand_prior_sigma(self, prior_sigma):
        # Possibly tile the prior_sigma array for the (a, b, c) coefficient blocks.
        if self.aberration_interp == 'linear' and prior_sigma is not None:
            return np.tile(prior_sigma, 3)
        else:
            return prior_sigma

    def _evaluate_linear_params(self, params, points):
        coeffs = params.reshape(3, self._single_point_param_len)
        a, b, c = coeffs
        sample_params = np.empty((len(points), self._single_point_param_len), dtype=float)
        for i, pt in enumerate(points):
            xh = pt.x / self.sca_size
            yh = pt.y / self.sca_size
            sample_params[i] = a + b * xh + c * yh
        return sample_params

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
        # together for efficiency of building the adjusted sample-point profiles.  So this
        # function is rarely used.  However, it's part of the Interp API, so just
        # farm this out to the fit_many function with a list of one star.
        return self.fit_many(
            [star],
            logger=logger,
            convert_funcs=[convert_func],
            draw_method=draw_method,
        )[0]

    @staticmethod
    def _fit_sca_group_worker(model, sca, stars, convert_funcs, draw_method, logger):
        fit_group, mean_params, sample_profiles = model._fit_sca_group(
            sca,
            stars,
            convert_funcs=convert_funcs,
            logger=logger,
            draw_method=draw_method,
        )
        return sca, fit_group, mean_params, sample_profiles

    def fit_many(self, stars, logger=None, convert_funcs=None, draw_method=None):
        logger = LoggerWrapper(logger)

        if len(stars) == 0:
            self._last_sca_mean = {}
            return []

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

        args = []
        for sca, group in grouped.items():
            # Keep multiprocessing payload small by only sending the relevant cache entry
            # for this SCA to each worker.
            worker_model = copy.copy(self)
            worker_model._corner_cache = (
                {sca: self._corner_cache[sca]} if sca in self._corner_cache else {}
            )
            args.append(
                (worker_model, sca, group['stars'], group['convert_funcs'], draw_method)
            )

        fit_results = run_multi(
            self._fit_sca_group_worker,
            self.nproc,
            raise_except=True,
            args=args,
            logger=logger,
        )

        out = [None] * len(stars)
        sca_mean = {}
        for fit_result in fit_results:
            sca, fit_group, mean_params, sample_profiles = fit_result
            indices = grouped[sca]['indices']
            sca_mean[sca] = mean_params
            wcs = stars[indices[0]].image.wcs
            self._corner_cache[sca] = (mean_params, wcs, sample_profiles)
            # Output the stars in the same order as input, so they stay matched with
            # convert_funcs array if there is one.
            for i, star in zip(indices, fit_group):
                out[i] = star
        self._last_sca_mean = sca_mean
        return out

    def _fit_sca_group(self, sca, stars, convert_funcs, logger=None, draw_method=None):
        ref = stars[0]
        params = ref.fit.get_params(self._num)

        # We're about to change params, so if the sample profiles aren't yet in the cache,
        # there is not reason now to add them.
        sample_base_profiles = self._get_sample_profiles(ref, params, cache=False, sca=sca)

        # First draw the base image for each star.
        # Also set up an empty Jacobian array for each to be filled in below.
        nparam = len(params)
        group_data = []
        for star, convert_func in zip(stars, convert_funcs):
            image = star.image.array
            weight = star.weight.array
            base_image = self._draw_model_image_from_samples(
                    star, sample_base_profiles, convert_func)
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
            sample_plus_profiles = self._get_sample_profiles(ref, p1, cache=False, sca=sca)
            scale = 1.0 / step
            # Use these adjusted sample profiles for all stars on the SCA.
            for star, convert_func, _, _, _, jac, base_image in group_data:
                im1 = self._draw_model_image_from_samples(star, sample_plus_profiles, convert_func)
                jac[:, i] = ((im1 - base_image) * scale).ravel()

        # Build one SCA-level normal equation from all stars.
        ata_sum = np.zeros((nparam, nparam), dtype=float)
        atb_sum = np.zeros(nparam, dtype=float)
        solved = []
        for star, convert_func, weight, sw, bw, jac, _ in group_data:
            aw = jac * sw[:, np.newaxis]
            ata_sum += aw.T.dot(aw)
            atb_sum += aw.T.dot(bw)
            solved.append((star, convert_func, weight))

        # Solve once for a single SCA parameter vector.
        sca_params, var = self._solve_normal_equations(ata_sum, atb_sum, params)
        sample_sca_profiles = self._get_sample_profiles(ref, sca_params, cache=False, sca=sca)

        # Compute the final model and chisq for each star.
        out = []
        for star, convert_func, weight in solved:
            model_image = self._draw_model_image_from_samples(
                    star, sample_sca_profiles, convert_func)
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
        return out, sca_params, sample_sca_profiles

    def draw(self, star, copy_image=True):
        params = star.fit.get_params(self._num)
        prof = self.getProfile(params, star=star).shift(star.fit.center) * star.fit.flux
        image = star.image.copy() if copy_image else star.image
        self._draw_profile_to_image(prof, image, star.image_pos, star)
        return Star(star.data.withNew(image=image), star.fit)

    def getProfile(self, params=None, star=None, cache=True):
        if star is None:
            raise ValueError("Roman.getProfile requires the star argument")
        if params is None:
            params = np.zeros(self.param_len, dtype=float)
        else:
            params = np.asarray(params)
            if params.size != self.param_len:
                raise ValueError(
                    "Roman params must have length %d (got %d)" % (self.param_len, params.size)
                )
        sample_profiles = self._get_sample_profiles(star, params, cache=cache)
        return self._interpolate_samples(star, sample_profiles)

    def _draw_model_image_from_samples(self, star, sample_profiles, convert_func):
        prof = self._interpolate_samples(star, sample_profiles)
        prof = prof.shift(star.fit.center) * star.fit.flux
        if convert_func is not None:
            prof = convert_func(prof)
        image = star.image.copy()
        self._draw_profile_to_image(prof, image, star.image_pos, star)
        return image.array

    def _get_sample_profiles(self, star, params, cache=True, sca=None):
        if sca is None:
            sca = _get_sca(star)
        wcs = star.image.wcs
        if sca in self._corner_cache:
            cached_params, cached_wcs, cached_profiles = self._corner_cache[sca]
            same_wcs = cached_wcs is wcs
            if not same_wcs:
                same_wcs = (cached_wcs == wcs)
            if same_wcs and np.array_equal(cached_params, params):
                return cached_profiles

        bandpass = self._require_bandpass()
        wavelength = None if self.chromatic else bandpass.effective_wavelength
        if self.nominal_interp == 'five_point':
            points = self._get_roman_five_point_data(sca)['points']
        else:
            points = (
                galsim.PositionD(0.0, 0.0),
                galsim.PositionD(self.sca_size, 0.0),
                galsim.PositionD(0.0, self.sca_size),
                galsim.PositionD(self.sca_size, self.sca_size),
            )
        if self.aberration_interp == 'linear':
            sample_params = self._evaluate_linear_params(params, points)
        else:
            npts = 5 if self.nominal_interp == 'five_point' else 4
            # In constant/global mode, the same per-SCA vector applies everywhere.
            sample_params = np.tile(params, npts).reshape(npts, self._single_point_param_len)
        profiles = []
        for i, (pt, p) in enumerate(zip(points, sample_params)):
            extra = self._make_extra_aberrations(p)
            if self.nominal_interp == 'five_point' and i == 4:
                extra = extra + self._get_roman_five_point_data(sca)['center_delta']
            profiles.append(
                galsim.roman.getPSF(
                    sca,
                    self.filter_name,
                    SCA_pos=pt,
                    pupil_bin=pupil_bin,
                    wcs=wcs,
                    extra_aberrations=extra,
                    wavelength=wavelength,
                )
            )
        profiles = tuple(profiles)
        if cache:
            self._corner_cache[sca] = (params, wcs, profiles)
        return profiles

    def _interpolate_samples(self, star, sample_profiles):
        if self.nominal_interp == 'five_point':
            w_ll, w_lr, w_ul, w_ur, w_c = self._five_point_weights(star)
            ll, lr, ul, ur, c = sample_profiles
            return w_ll * ll + w_lr * lr + w_ul * ul + w_ur * ur + w_c * c
        else:
            w_ll, w_lr, w_ul, w_ur = self._corner_weights(star)
            ll, lr, ul, ur = sample_profiles
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

    def _five_point_weights(self, star):
        sca = _get_sca(star)
        solver = self._get_roman_five_point_data(sca)['solver']
        x = star.image_pos.x
        y = star.image_pos.y
        phi = np.array([1.0, x, y, x * y, x * x + y * y], dtype=float)
        # Convert basis values at this star position into interpolation weights on the
        # 5 sample locations (LL, LR, UL, UR, C).
        return tuple(solver.dot(phi))


class RomanOpticsPSF(PSF):
    """A PSF wrapper for Roman optical fits with configurable aberration interpolation."""

    _type_name = 'RomanOptics'

    def __init__(self, model, interp, outliers=None, chisq_thresh=0.1, min_iter=2, max_iter=30):
        self.model = model
        self.interp = interp
        if isinstance(model, Model):
            self.bandpass = model.bandpass
        else:
            # During read, model is temporarily a placeholder integer until _finish_read.
            self.bandpass = None
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
            self.interp.set_num(num)

    def set_context(self, wcs, pointing=None, bandpass=None):
        super().set_context(wcs, pointing, bandpass)
        # During read, model is temporarily a placeholder integer until _finish_read.
        if isinstance(self.model, Model):
            self.model.set_bandpass(self.bandpass)

    @property
    def interp_property_names(self):
        return self.interp.property_names

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        kwargs = dict(config_psf)
        kwargs.pop('type', None)

        outliers = kwargs.pop('outliers', None)
        chisq_thresh = kwargs.pop('chisq_thresh', 0.1)
        min_iter = kwargs.pop('min_iter', 2)
        max_iter = kwargs.pop('max_iter', 30)
        aberration_interp = kwargs.pop('aberration_interp', 'constant')

        model_interp = 'constant' if aberration_interp == 'global' else aberration_interp
        interp_per_sca = (aberration_interp != 'global')

        model = RomanOpticalModel(aberration_interp=model_interp, logger=logger, **kwargs)
        interp = RomanSCAInterp(per_sca=interp_per_sca)

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

        # Start each new fit with a fresh sample-profile cache.
        self.model.clear_cache()

        logger.debug("Initializing models")
        new_stars = [
            self.model.initialize(star, logger=logger, default_init=default_init)
            for star in stars
        ]
        logger.debug("No stars removed in initialize step")

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
        if self.bandpass is not None:
            # Now that model is built, we can update its bandpass to the correct value.
            self.model.set_bandpass(self.bandpass)
        self.interp = Interp.read(reader, 'interp')
        self.outliers = Outliers.read_all(reader, 'outliers')
