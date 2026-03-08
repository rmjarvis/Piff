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

from __future__ import print_function

from contextlib import contextmanager
import os
import tempfile
from unittest import mock

import numpy as np
import piff
import galsim
import pytest

from piff.roman.roman_psf import RomanOpticsPSF, RomanOpticalModel, RomanSCAInterp
from piff_test_helper import timer


@contextmanager
def fast_pupil_bin(value=16):
    """Temporarily raise Roman `pupil_bin` to speed up the getPSF calls in unit tests.
    Using a coarser pupil sampling makes those calls fast enough for routine test runs,
    so most of the tests below are run entirely in this context.
    """
    from piff.roman import roman_psf
    original = roman_psf.pupil_bin
    roman_psf.pupil_bin = value
    try:
        yield
    finally:
        roman_psf.pupil_bin = original


@timer
def test_roman_optics():
    """Check RomanOptics basic construction, drawing, and SCA/chipnum property handling.
    """
    with fast_pupil_bin():
        # Check basic construction.
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'filter': 'H158', 'chromatic': False, 'max_zernike': 6}
        )
        assert isinstance(psf, RomanOpticsPSF)
        logger = piff.config.setup_logger()
        assert psf.interp_property_names == ('sca',)
        assert psf.fit_center is False
        assert psf.include_model_centroid is False
        assert psf.model.aberration_interp == 'constant'
        assert psf.interp.per_sca is True

        # Global mode uses one aberration vector for full focal plane.
        psf_global = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'filter': 'H158',
                'chromatic': False,
                'max_zernike': 6,
                'aberration_interp': 'global',
            }
        )
        assert psf_global.model.aberration_interp == 'constant'
        assert psf_global.interp.per_sca is False

        # Check the core RomanOptics flow: initialize, draw, and profile retrieval.
        star = piff.Star.makeTarget(
            x=123.4,
            y=456.7,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5},
        ).withFlux(1.0, (0.0, 0.0))

        stars, nremoved = psf.initialize_params([star], logger=logger)
        assert nremoved == 0
        psf.wcs = {5: galsim.PixelScale(0.11)}
        psf.pointing = None
        psf.interp.set_sca_solution({5: np.zeros(psf.model.param_len)})
        prof, method = psf.get_profile(x=123.4, y=456.7, sca=5)
        assert prof is not None
        assert method == psf.model._method
        image = psf.draw(x=123.4, y=456.7, sca=5)
        assert image.array.shape == (48, 48)  # Default stamp_size=48 in draw function.
        assert np.isclose(image.array.sum(), 1.0, rtol=0.05)

        # Helper function to use as a side-effect of getPSF to record what sca argument is used.
        used_sca = [] # Use a list so we can modify in place.
        real_get_psf = galsim.roman.getPSF  # Save this here to avoid recursion when patched.
        def traced_get_psf(*args, **kwargs):
            sca = args[0] if args else kwargs['SCA']
            # Clear list first, so it always has just a single element.
            used_sca.clear()
            used_sca.append(sca)
            return real_get_psf(*args, **kwargs)

        # If both chipnum and sca are present, sca takes precedence.
        star = piff.Star.makeTarget(
            x=12.3,
            y=45.6,
            stamp_size=25,
            scale=0.11,
            properties={'chipnum': 7, 'sca': 4},
        ).withFlux(1.0, (0.0, 0.0))
        with mock.patch('galsim.roman.getPSF', side_effect=traced_get_psf):
            psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])
        assert used_sca[0] == 4

        # If only chipnum is present, assume that is the sca number.
        star = piff.Star.makeTarget(
            x=78.9,
            y=10.1,
            stamp_size=25,
            scale=0.11,
            properties={'chipnum': 8},
        ).withFlux(1.0, (0.0, 0.0))
        with mock.patch('galsim.roman.getPSF', side_effect=traced_get_psf):
            psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])
        assert used_sca[0] == 8

        # If neither is present, raise an exception.
        star = piff.Star.makeTarget(x=11.1, y=22.2, stamp_size=25, scale=0.11)
        star = star.withFlux(1.0, (0.0, 0.0))
        with pytest.raises(ValueError) as err:
            psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])
        assert "explicit 'sca' property" in str(err.value)

        # Check some simple interpolation machinery.
        star = piff.Star.makeTarget(
            x=30.0,
            y=40.0,
            stamp_size=15,
            scale=0.11,
            properties={'sca': 2},
        ).withFlux(1.0, (0.0, 0.0))
        stars, _ = psf.initialize_params([star], logger=logger)
        psf.interp.set_sca_solution({2: np.zeros(psf.model.param_len)})
        interp_star = psf.interpolateStar(stars[0], inplace=False)
        assert interp_star is not stars[0]

        # Check that it fails gracefully when no stars are left unflagged.
        flagged = stars[0].flag_if(True)
        with pytest.raises(RuntimeError) as err:
            psf.single_iteration([flagged], logger=logger, convert_funcs=None, draw_method=None)
        assert "No stars left to fit" in str(err.value)

        # Check outliers round trip through file
        psf1 = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'filter': 'H158',
                'chromatic': False,
                'max_zernike': 6,
                'outliers': [
                    {'type': 'Chisq', 'nsigma': 5.0},
                    {'type': 'Centroid', 'max_offset': 0.2},
                ],
            }
        )
        assert psf1.outliers is not None
        assert isinstance(psf1.outliers, list)
        fn = os.path.join('output', 'roman_outliers_write_test.piff')
        psf1.write(fn)
        psf2 = piff.read(fn)
        assert isinstance(psf2.outliers, list)
        assert len(psf2.outliers) == len(psf1.outliers)
        assert psf2.outliers[0]._type_name == 'Chisq'
        assert psf2.outliers[1]._type_name == 'Centroid'

    # max_zernike must be >= 4
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=3)
    assert "range 4..22" in str(err.value)

    # max_zernike must be <= 22
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=23)
    assert "range 4..22" in str(err.value)

    # Check invalid filter string
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='NotAFilter', chromatic=False, max_zernike=6)
    assert "not a valid GalSim Roman bandpass" in str(err.value)

    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6, aberration_interp='bad')
    assert "must be one of" in str(err.value)

@timer
def test_roman_corner_cache():
    """Verify corner-profile caching reuses one 4-corner set for same SCA and params.
    """
    with fast_pupil_bin():
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'filter': 'H158', 'chromatic': False, 'max_zernike': 6}
        )
        logger = piff.config.setup_logger()
        stars = [
            piff.Star.makeTarget(
                x=100.0,
                y=200.0,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
            piff.Star.makeTarget(
                x=800.0,
                y=600.0,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
        ]
        stars, _ = psf.initialize_params(stars, logger=logger)
        psf.drawStar(stars[0])
        psf.drawStar(stars[1])

        params = stars[0].fit.params
        profiles1 = psf.model._get_corner_profiles(stars[0], params, cache=True)
        profiles2 = psf.model._get_corner_profiles(stars[1], params, cache=True)
        assert profiles1 is profiles2
        assert len(psf.model._corner_cache) == 1
        assert 5 in psf.model._corner_cache
        assert len(psf.model._corner_cache[5][2]) == 4


@timer
def test_roman_fit():
    """Check local convergence of single-star Roman fits in constant mode.
    """
    with fast_pupil_bin():
        model = RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_prior_sigma=1.0e6,
        )
        star = piff.Star.makeTarget(
            x=64.0,
            y=64.0,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5},
        ).withFlux(1.0, (0.0, 0.0))

        init_star = model.initialize(star)
        # Keep injected extra aberrations modest relative to baseline Roman optics.
        # Typical built-in |z4..z22| values are around 1e-3 to a few e-2, so 4-5e-3 is
        # realistic while still providing measurable signal in this unit test.
        truth_params = np.array([0.004, -0.003, 0.005])
        truth_fit = init_star.fit.newParams(
            truth_params,
            params_var=np.zeros_like(truth_params),
        )
        truth_star = model.draw(piff.Star(init_star.data, truth_fit))

        fit_star = piff.Star(
            truth_star.data,
            init_star.fit.newParams(
                np.zeros_like(truth_params),
                params_var=np.zeros_like(truth_params)
            ),
        )
        fitted = model.fit(fit_star)
        print('fit = ',fitted.fit.params)
        # 1 pass isn't great, but after 2 passes, the agreement is sub percent.
        fitted = model.fit(fitted)
        print('fit => ',fitted.fit.params)
        # And 3 is within 0.1% agreement.
        fitted = model.fit(fitted)
        print('fit => ',fitted.fit.params)
        fitted_params = fitted.fit.params
        fitted_var = fitted.fit.params_var

        assert fitted.fit.chisq >= 0
        assert fitted.fit.dof > 0
        print('final fit = ',fitted_params)
        print('    truth = ',truth_params)
        np.testing.assert_allclose(fitted_params, truth_params, atol=0.0, rtol=1.e-3)
        assert np.all(fitted_var >= 0)

        # Sanity tests about getting profile and drawing it.
        prof = model.getProfile(star=fit_star)
        assert prof is not None
        drawn_star = model.draw(fit_star)
        assert drawn_star.image.array.shape == fit_star.image.array.shape

        # Error if no star argument in getProfile.  (Allowed by some other classes.)
        with pytest.raises(ValueError) as err:
            model.getProfile(params=np.zeros(model.param_len), star=None)
        assert "requires the star argument" in str(err.value)

        # Error if params is given and has wrong length.
        with pytest.raises(ValueError) as err:
            model.getProfile(star=fit_star, params=np.zeros(model.param_len+2))
        assert "params must have length 3" in str(err.value)

        # Error if convert_funcs is given but has different length than stars.
        with pytest.raises(ValueError) as err:
            model.fit_many([fit_star], convert_funcs=[])
        assert "len(convert_funcs) must match len(stars)" in str(err.value)


@timer
def test_roman_aberration_prior():
    """Validate the use of priors on the aberration values.
    """
    # Check that fitted aberrations stay closer to 0 when strong prior is applied.
    weak_prior = RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_prior_sigma=1.0e6,
    )
    strong_prior = RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_prior_sigma=[0.02],  # List with 1 element treated as scalar.
    )
    aw = np.zeros((10, weak_prior.param_len))
    aw[:, 0] = 12.3
    aw[:, 1] = 1.7
    aw[:, 2] = 2.4
    bw = np.ones(10)
    p0 = np.zeros(weak_prior.param_len)
    new_weak, _ = weak_prior._solve_params(aw, bw, p0)
    new_strong, _ = strong_prior._solve_params(aw, bw, p0)
    assert abs(new_strong[0]) < abs(new_weak[0])

    # None means no prior, which is equivalent to infinite prior.
    no_prior = RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_prior_sigma=None,
    )
    ata = np.eye(no_prior.param_len)
    atb = np.ones(no_prior.param_len)
    p = np.zeros(no_prior.param_len)
    # Internally this means that the apply_prior doesn't change the inputs.
    ata0 = ata.copy()
    atb0 = atb.copy()
    no_prior._apply_prior(ata, atb, p)
    np.testing.assert_allclose(ata, ata0)
    np.testing.assert_allclose(atb, atb0)

    # Compare the solve to what you get with an actually infinite prior.
    inf_prior = RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_prior_sigma=np.inf
    )
    no_prior_result, _ = no_prior._solve_params(aw, bw, p0)
    inf_prior_result, _ = inf_prior._solve_params(aw, bw, p0)
    print('no prior',no_prior_result)
    print('inf prior',inf_prior_result)
    np.testing.assert_allclose(no_prior_result, inf_prior_result)

    # prior length must match number of zernikes used
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6,
                          aberration_prior_sigma=[1.0, 2.0])
    assert "scalar or length 3" in str(err.value)

    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6,
                          aberration_prior_sigma=[])
    assert "scalar or length 3" in str(err.value)

    # Same for linear mode (in particular the allowed length is 3 here, not 12,
    # which is the full param_len).
    with pytest.raises(ValueError) as err:
        piff.roman.RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6,
                                     aberration_interp='linear',
                                     aberration_prior_sigma=[1.0] * 12)
    assert "scalar or length 3" in str(err.value)

    # In linear mode, scalar and one-corner vectors tile to 4 corner blocks.
    linear_scalar = piff.roman.RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_interp='linear',
        aberration_prior_sigma=0.05,
    )
    np.testing.assert_allclose(linear_scalar.prior_sigma, np.full(12, 0.05))

    linear_vec = piff.roman.RomanOpticalModel(
        filter='H158',
        chromatic=False,
        max_zernike=6,
        aberration_interp='linear',
        aberration_prior_sigma=[0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(linear_vec.prior_sigma, np.tile([0.1, 0.2, 0.3], 4))

    # priors cannot be <= 0
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6,
                          aberration_prior_sigma=[1.0, 0.0, 1.0])
    assert "must all be > 0" in str(err.value)

    with pytest.raises(ValueError) as err:
        RomanOpticalModel(filter='H158', chromatic=False, max_zernike=6,
                          aberration_prior_sigma=[1.0, -1.0, 1.0])
    assert "must all be > 0" in str(err.value)


@timer
def test_roman_fit_many():
    """Check accuracy of fitting multiple stars using fit_many.
    """
    with fast_pupil_bin():
        model = RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_prior_sigma=1.0e6,
        )
        stars = [
            piff.Star.makeTarget(
                x=64.2,
                y=64.1,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
            piff.Star.makeTarget(
                x=71.8,
                y=62.7,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
        ]
        stars = [model.initialize(s) for s in stars]
        # Use the same realistic small-amplitude extra-aberration vector as test_roman_fit.
        truth_params = np.array([0.004, -0.003, 0.005])
        truth = [
            model.draw(
                piff.Star(
                    s.data,
                    s.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                    ),
                )
            )
            for s in stars
        ]
        fit_stars = [
            piff.Star(
                s.data,
                stars[i].fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params),
                ),
            )
            for i, s in enumerate(truth)
        ]

        # 1 pass isn't great, but after 2 passes, the agreement is sub percent.
        # And 3 is within 0.1% agreement.
        for _ in range(3):
            fit_stars = model.fit_many(fit_stars)
            print('fit[0] => ',fit_stars[0].fit.params)

        p0 = fit_stars[0].fit.params
        p1 = fit_stars[1].fit.params
        print('final p0 = ',p0)
        print('final p1 = ',p1)
        print('   truth = ',truth_params)
        np.testing.assert_allclose(p0, p1, atol=1.0e-12, rtol=0.0)
        np.testing.assert_allclose(p0, truth_params, atol=0.0, rtol=1.e-3)


@timer
def test_roman_fit_many_nproc():
    """Check `fit_many` multiprocessing path and accuracy with `nproc > 1`.
    """
    with fast_pupil_bin():
        model = piff.roman.RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_prior_sigma=1.0e6,
            nproc=2,
        )
        stars = [
            piff.Star.makeTarget(
                x=64.2,
                y=64.1,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
            piff.Star.makeTarget(
                x=171.8,
                y=162.7,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
        ]
        stars = [model.initialize(s) for s in stars]
        truth_params = np.array([0.004, -0.003, 0.005])
        truth = [
            model.draw(
                piff.Star(
                    s.data,
                    s.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                    ),
                )
            )
            for s in stars
        ]
        fit_stars = [
            piff.Star(
                s.data,
                stars[i].fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params),
                ),
            )
            for i, s in enumerate(truth)
        ]

        for _ in range(3):
            fit_stars = model.fit_many(fit_stars)

        assert [int(s['sca']) for s in fit_stars] == [5, 5]
        for s in fit_stars:
            np.testing.assert_allclose(s.fit.params, truth_params, atol=0.0, rtol=2.e-3)
        assert list(model._corner_cache.keys()) == [5]


@timer
def test_roman_fit_linear():
    """Check linear-mode convergence with stars spanning the SCA geometry.
    """
    with fast_pupil_bin():
        model = piff.roman.RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_interp='linear',
            aberration_prior_sigma=1.0e6,
        )
        # Use stars spread across the SCA so all four corner aberration blocks are constrained.
        pos = [
            (512.0, 512.0),
            (2044.0, 512.0),
            (3576.0, 512.0),
            (512.0, 2044.0),
            (2044.0, 2044.0),
            (3576.0, 2044.0),
            (512.0, 3576.0),
            (2044.0, 3576.0),
            (3576.0, 3576.0),
        ]
        stars = [
            piff.Star.makeTarget(
                x=x,
                y=y,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0))
            for x, y in pos
        ]
        init_stars = [model.initialize(star) for star in stars]
        base_truth = np.array([0.004, -0.003, 0.005])
        truth_params = np.tile(base_truth, 4)
        truth_stars = [
            model.draw(
                piff.Star(
                    init_star.data,
                    init_star.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                    ),
                )
            )
            for init_star in init_stars
        ]

        fit_stars = [
            piff.Star(
                truth_star.data,
                init_star.fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params)
                ),
            )
            for truth_star, init_star in zip(truth_stars, init_stars)
        ]

        for _ in range(4):
            fit_stars = model.fit_many(fit_stars)

        fitted_params = fit_stars[0].fit.params
        fitted_var = fit_stars[0].fit.params_var

        for fit_star in fit_stars:
            assert fit_star.fit.chisq >= 0
            assert fit_star.fit.dof > 0
            np.testing.assert_allclose(fit_star.fit.params, fitted_params, atol=1.e-12, rtol=0.0)
        print('linear fit = ', fitted_params)
        print('    truth = ', truth_params)
        np.testing.assert_allclose(fitted_params, truth_params, atol=0.0, rtol=1.e-3)
        assert np.all(fitted_var >= 0)

        for fit_star, truth_star in zip(fit_stars, truth_stars):
            model_star = model.draw(fit_star)
            np.testing.assert_allclose(
                model_star.image.array, truth_star.image.array, atol=5.e-5, rtol=0.0
            )


@timer
def test_roman_fit_linear_gradient():
    """Check linear-mode recovery when corner aberration vectors are genuinely different.
    """
    with fast_pupil_bin():
        model = piff.roman.RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_interp='linear',
            aberration_prior_sigma=1.0e6,
        )
        pos = [
            (512.0, 512.0),
            (2044.0, 512.0),
            (3576.0, 512.0),
            (512.0, 2044.0),
            (2044.0, 2044.0),
            (3576.0, 2044.0),
            (512.0, 3576.0),
            (2044.0, 3576.0),
            (3576.0, 3576.0),
        ]
        stars = [
            piff.Star.makeTarget(
                x=x,
                y=y,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0))
            for x, y in pos
        ]
        init_stars = [model.initialize(star) for star in stars]

        # Define each Zernike truth as a + b*(x/4088) + c*(y/4088).
        # Build corner values and per-star values analytically from this formula.
        abc = np.array(
            [
                [0.004, 2.e-5, -7.e-5],
                [-0.003, 6.e-5, -2.e-5],
                [0.005, -3.e-5, -5.e-5],
            ]
        )

        def eval_truth(x, y):
            xh = x / model.sca_size
            yh = y / model.sca_size
            return abc[:, 0] + abc[:, 1] * xh + abc[:, 2] * yh

        # Corner order used by Roman is LL, LR, UL, UR.
        corner_truth = np.array(
            [
                eval_truth(0.0, 0.0),
                eval_truth(model.sca_size, 0.0),
                eval_truth(0.0, model.sca_size),
                eval_truth(model.sca_size, model.sca_size),
            ]
        )
        truth_params = corner_truth.ravel()
        assert not np.any(np.isclose(corner_truth[0], corner_truth[1]))
        assert not np.any(np.isclose(corner_truth[0], corner_truth[2]))
        assert not np.any(np.isclose(corner_truth[0], corner_truth[3]))

        # Verify independently that bilinear interpolation of corners reproduces the
        # per-star truth function (for linear functions without an xy term).
        ll, lr, ul, ur = corner_truth
        for x, y in pos:
            fx = x / model.sca_size
            fy = y / model.sca_size
            wll = (1.0 - fx) * (1.0 - fy)
            wlr = fx * (1.0 - fy)
            wul = (1.0 - fx) * fy
            wur = fx * fy
            local_from_corners = wll * ll + wlr * lr + wul * ul + wur * ur
            np.testing.assert_allclose(
                local_from_corners, eval_truth(x, y), atol=1.e-12, rtol=0.0
            )

        # Draw truth_stars using the class's bilinear interpolation given truth_params
        truth_stars = [
            model.draw(
                piff.Star(
                    init_star.data,
                    init_star.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                    ),
                )
            )
            for init_star in init_stars
        ]

        # Draw the stars to use for fitting using the aberration array from eval_truth directly.
        fit_stars = [
            piff.Star(
                truth_star.data,
                init_star.fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params)
                ),
            )
            for truth_star, init_star in zip(truth_stars, init_stars)
        ]
        for star in fit_stars:
            aber = eval_truth(star.image_pos.x, star.image_pos.y)
            prof = galsim.roman.getPSF(5, 'H158', star.image_pos,
                                       pupil_bin=piff.roman.roman_psf.pupil_bin,
                                       wcs=star.image.wcs,
                                       extra_aberrations=model._make_extra_aberrations(aber),
                                       wavelength=model.bandpass.effective_wavelength)
            prof.drawImage(star.image, method='auto', center=star.image_pos)

        for _ in range(5):
            fit_stars = model.fit_many(fit_stars)

        # Note: these are not expected to match exactly.  The real images include the
        # natural variation within the SCA from the roman aberration pattern, fully separate
        # from the extra_aberrations we're fitting for.  This variation is not quite linear,
        # so the bilinear approximation is a small model mismatch.  However, it's relatively
        # close in the fitted extra aberrations, and the drawn images are very close.
        fitted_params = fit_stars[0].fit.params
        print('linear gradient fit = ', fitted_params)
        print('           truth = ', truth_params)
        np.testing.assert_allclose(fitted_params, truth_params, atol=1.e-4, rtol=0.3)
        for fit_star in fit_stars:
            np.testing.assert_allclose(fit_star.fit.params, fitted_params, atol=1.e-4, rtol=0.01)
        for fit_star, truth_star in zip(fit_stars, truth_stars):
            model_star = model.draw(fit_star)
            np.testing.assert_allclose(
                model_star.image.array, truth_star.image.array, atol=1.e-4, rtol=0.01)


@timer
def test_roman_fit_linear_nproc():
    """Check linear-mode fit_many behavior with multiprocessing enabled.
    """
    with fast_pupil_bin():
        model = piff.roman.RomanOpticalModel(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_interp='linear',
            aberration_prior_sigma=1.0e6,
            nproc=2,
        )
        pos = [
            (512.0, 512.0),
            (2044.0, 512.0),
            (3576.0, 512.0),
            (512.0, 3576.0),
            (2044.0, 3576.0),
            (3576.0, 3576.0),
        ]
        stars = [
            piff.Star.makeTarget(
                x=x,
                y=y,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0))
            for x, y in pos
        ]
        init_stars = [model.initialize(star) for star in stars]
        truth_params = np.tile([0.004, -0.003, 0.005], 4)
        truth_stars = [
            model.draw(
                piff.Star(
                    init_star.data,
                    init_star.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                    ),
                )
            )
            for init_star in init_stars
        ]
        fit_stars = [
            piff.Star(
                truth_star.data,
                init_star.fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params)
                ),
            )
            for truth_star, init_star in zip(truth_stars, init_stars)
        ]

        for _ in range(4):
            fit_stars = model.fit_many(fit_stars)
        for fit_star in fit_stars:
            np.testing.assert_allclose(fit_star.fit.params, truth_params, atol=0.0, rtol=1.e-3)



@timer
def test_roman_optics_convert_funcs():
    """Check aberration recovery when fitting with a nontrivial convert_func (profile shear).
    """
    with fast_pupil_bin():
        psf = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'filter': 'H158',
                'chromatic': False,
                'max_zernike': 6,
                'aberration_prior_sigma': 1.0e6,
            }
        )
        logger = piff.config.setup_logger()
        stars = [
            piff.Star.makeTarget(
                x=64.2,
                y=64.1,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
            piff.Star.makeTarget(
                x=71.8,
                y=62.7,
                stamp_size=25,
                scale=0.11,
                properties={'sca': 5},
            ).withFlux(1.0, (0.0, 0.0)),
        ]
        stars, _ = psf.initialize_params(stars, logger=logger)

        truth_params = np.array([0.004, -0.003, 0.005])

        def apply_shear(prof):
            return prof.shear(g1=0.01, g2=-0.005)

        convert_funcs = [apply_shear] * len(stars)
        fit_stars = []
        for star in stars:
            truth_fit = star.fit.newParams(
                truth_params,
                params_var=np.zeros_like(truth_params),
            )
            truth_star = piff.Star(star.data, truth_fit)
            # True profile is sheared version of optical PSF.
            prof = psf.model.getProfile(truth_params, star=truth_star).shear(g1=0.01, g2=-0.005)
            image = star.image.copy()
            psf.model._draw_profile_to_image(prof, image, star.image_pos)
            fit_stars.append(piff.Star(star.data.withNew(image=image), star.fit))

        for _ in range(3):
            fit_stars, nremoved = psf.single_iteration(
                fit_stars,
                logger=logger,
                convert_funcs=convert_funcs,
                draw_method=None,
            )
            assert nremoved == 0
            print('params[0] => ',fit_stars[0].fit.params)

        assert len(fit_stars) == len(stars)
        print('truth = ',truth_params)
        for i, star in enumerate(fit_stars):
            print(f'star {i} params = ',star.fit.params)
            np.testing.assert_allclose(star.fit.params, truth_params, atol=0.0, rtol=1.e-3)


@timer
def test_roman_sca_interp():
    """Test per-SCA/global interpolation behavior and RomanSCAInterp serialization round-trip.
    """
    with fast_pupil_bin():
        # Start with some basic exercises of interp machinery.
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'filter': 'H158', 'chromatic': False, 'max_zernike': 6}
        )
        assert type(psf.interp).__name__ == 'RomanSCAInterp'

        logger = piff.config.setup_logger()
        p1 = np.array([0.1, 0.0, -0.1])
        p2 = np.array([-0.2, 0.3, 0.0])

        # Solve just computes the mean on each sca as well as a global mean.
        s1 = piff.Star.makeTarget(
            x=10.0, y=20.0, stamp_size=25, scale=0.11, properties={'sca': 2}
        ).withFlux(1.0, (0.0, 0.0))
        s2 = piff.Star.makeTarget(
            x=30.0, y=40.0, stamp_size=25, scale=0.11, properties={'sca': 5}
        ).withFlux(1.0, (0.0, 0.0))

        stars, _ = psf.initialize_params([s1, s2], logger=logger)
        stars[0] = piff.Star(
            stars[0].data,
            stars[0].fit.newParams(p1, params_var=np.zeros_like(p1)),
        )
        stars[1] = piff.Star(
            stars[1].data,
            stars[1].fit.newParams(p2, params_var=np.zeros_like(p2)),
        )
        psf.interp.solve(stars)
        np.testing.assert_equal(psf.interp.sca_mean[2], p1)
        np.testing.assert_equal(psf.interp.sca_mean[5], p2)
        global_mean = np.mean([p1,p2], axis=0)
        np.testing.assert_equal(psf.interp.global_mean, global_mean)

        # interpolate assigns the corresponding sca_mean to the params,
        # or the global mean if the sca wasn't used in the fit (e.g. no stars on an SCA)
        t1 = piff.Star.makeTarget(x=12.0, y=126.0, stamp_size=25, scale=0.11, properties={'sca': 2})
        t2 = piff.Star.makeTarget(x=33.0, y=114.0, stamp_size=25, scale=0.11, properties={'sca': 5})
        t3 = piff.Star.makeTarget(x=98.0, y=453.0, stamp_size=25, scale=0.11, properties={'sca': 7})
        t1 = t1.withFlux(1.0, (0.0, 0.0))
        t2 = t2.withFlux(1.0, (0.0, 0.0))
        t3 = t3.withFlux(1.0, (0.0, 0.0))
        tstars, _ = psf.initialize_params([t1, t2, t3], logger=logger)
        tstars = psf.interpolateStarList(tstars)
        np.testing.assert_allclose(tstars[0].fit.params, p1)
        np.testing.assert_allclose(tstars[1].fit.params, p2)
        np.testing.assert_allclose(tstars[2].fit.params, global_mean)

        # Check round trip through file.
        fn = os.path.join('output', 'roman_sca_test.piff')
        psf.write(fn)
        psf2 = piff.read(fn)
        tstars2, _ = psf2.initialize_params([t1, t2, t3], logger=logger)
        tstars2 = psf2.interpolateStarList(tstars2)
        np.testing.assert_allclose(tstars2[0].fit.params, p1)
        np.testing.assert_allclose(tstars2[1].fit.params, p2)
        np.testing.assert_allclose(tstars2[2].fit.params, global_mean)

        # Global mode always uses global_mean.
        psf_global = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'filter': 'H158',
                'chromatic': False,
                'max_zernike': 6,
                'aberration_interp': 'global',
            }
        )
        psf_global.interp.solve(stars)
        np.testing.assert_equal(psf_global.interp.global_mean, np.mean([p1,p2],axis=0))
        assert psf_global.interp.sca_mean == {}

        tstars3, _ = psf_global.initialize_params([t1, t2, t3], logger=logger)
        tstars3 = psf_global.interpolateStarList(tstars)
        np.testing.assert_allclose(tstars3[0].fit.params, global_mean)
        np.testing.assert_allclose(tstars3[1].fit.params, global_mean)
        np.testing.assert_allclose(tstars3[2].fit.params, global_mean)

        # Round trip unsolved interp through real I/O path.  This exercises the
        # _finish_write early-return logic when no solve() has populated means yet.
        fn = os.path.join('output', 'roman_interp_unsolved.fits')
        psf = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'filter': 'H158',
            }
        )
        psf.write(fn)
        psf2 = piff.read(fn)
        assert isinstance(psf2.interp, RomanSCAInterp)
        assert psf2.interp.global_mean is None
        assert psf2.interp.sca_mean == {}


if __name__ == '__main__':
    test_roman_optics()
    test_roman_corner_cache()
    test_roman_fit()
    test_roman_aberration_prior()
    test_roman_fit_many()
    test_roman_fit_many_nproc()
    test_roman_fit_linear()
    test_roman_fit_linear_gradient()
    test_roman_fit_linear_nproc()
    test_roman_optics_convert_funcs()
    test_roman_sca_interp()
