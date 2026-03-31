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
import fitsio

from piff.util import make_flat
from piff.roman import RomanOpticsPSF, RomanOpticalModel, RomanSCAInterp
from piff_test_helper import timer


def make_roman_model(**kwargs):
    model = RomanOpticalModel(**kwargs)
    model.set_bandpass(galsim.roman.getBandpasses()['H158'])
    return model


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
        bandpass = galsim.roman.getBandpasses()['H158']

        # Check basic construction.
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'chromatic': False, 'max_zernike': 6}
        )
        assert isinstance(psf, RomanOpticsPSF)
        logger = piff.config.setup_logger()
        assert psf.interp_property_names == ('sca',)
        assert psf.fit_center is False
        assert psf.include_model_centroid is False
        assert psf.model.aberration_interp == 'constant'
        assert psf.model.nominal_interp == 'bilinear'
        assert psf.interp.per_sca is True

        # Global mode uses one aberration vector for full focal plane.
        psf_global = piff.PSF.process(
            {
                'type': 'RomanOptics',
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
        psf.set_context({5: galsim.PixelScale(0.11)}, None, bandpass)
        psf.interp.set_sca_solution({5: np.zeros(psf.model.param_len)})
        prof, method = psf.get_profile(x=123.4, y=456.7, sca=5)
        assert prof is not None
        assert method == psf.model._method
        image = psf.draw(x=123.4, y=456.7, sca=5)
        assert image.array.shape == (48, 48)  # Default stamp_size=48 in draw function.
        assert np.isclose(image.array.sum(), 1.0, rtol=0.05)

        chromatic_psf = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'chromatic': True,
                'max_zernike': 6,
            }
        )
        chromatic_psf.set_context({5: galsim.PixelScale(0.11)}, None, bandpass)
        sed = galsim.SED(lambda w: 1.0, wave_type='nm', flux_type='fphotons')
        chromatic_star = piff.Star.makeTarget(
            x=123.4,
            y=456.7,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5, 'sed_eff': sed},
        ).withFlux(1.0, (0.0, 0.0))
        chromatic_stars, _ = chromatic_psf.initialize_params([chromatic_star], logger=logger)
        chromatic_psf.interp.set_sca_solution({5: np.zeros(chromatic_psf.model.param_len)})
        chromatic_model_star = chromatic_psf.drawStar(chromatic_stars[0])
        assert np.isfinite(chromatic_model_star.image.array).all()
        assert chromatic_model_star.image.array.sum() > 0.0
        chromatic_psf.drawStar(chromatic_stars[0])
        assert chromatic_stars[0].data.properties['sed_eff'] is sed

        missing_sed_star = piff.Star.makeTarget(
            x=123.4,
            y=456.7,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5},
        ).withFlux(1.0, (0.0, 0.0))
        missing_sed_stars, _ = chromatic_psf.initialize_params([missing_sed_star], logger=logger)
        with pytest.raises(ValueError) as err:
            chromatic_psf.drawStar(missing_sed_stars[0])
        assert "requires each star to have an 'sed_eff' property" in str(err.value)

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
                'chromatic': False,
                'max_zernike': 6,
                'outliers': [
                    {'type': 'Chisq', 'nsigma': 5.0},
                    {'type': 'Centroid', 'max_offset': 0.2},
                ],
            }
        )
        psf1.set_context(None, None, bandpass)
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
        RomanOpticalModel(chromatic=False, max_zernike=3)
    assert "range 4..22" in str(err.value)

    # max_zernike must be <= 22
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(chromatic=False, max_zernike=23)
    assert "range 4..22" in str(err.value)

    # A custom bandpass can be used as long as the Roman filter_name is given explicitly.
    model = RomanOpticalModel(chromatic=False, max_zernike=6)
    custom_bandpass = galsim.Bandpass(lambda wave: 1.0, 'nm', blue_limit=1000, red_limit=2000)
    with pytest.raises(ValueError) as err:
        model.getProfile(params=np.zeros(model.param_len), star=star)
    assert "requires bandpass to be set before use" in str(err.value)
    with pytest.raises(ValueError) as err:
        model.set_bandpass(custom_bandpass)
    assert "requires filter_name" in str(err.value)
    model.set_bandpass(custom_bandpass, filter_name='H158')
    assert model.bandpass == custom_bandpass
    assert model.filter_name == 'H158'

    # Invalid filter_name fails when GalSim is actually asked to build the Roman PSF.
    model.filter_name = None  # Lets set_bandpass actually work.
    model.set_bandpass(
        galsim.Bandpass(lambda wave: 1.0, 'nm', blue_limit=1000, red_limit=2000),
        filter_name='NotAFilter',
    )
    with pytest.raises(ValueError) as err:
        model.getProfile(params=np.zeros(model.param_len), star=star)
    assert "valid Roman bandpass" in str(err.value)

    # Invalid aberration_interp
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(chromatic=False, max_zernike=6, aberration_interp='bad')
    assert "must be one of" in str(err.value)

    # Invalid nominal_interp
    with pytest.raises(ValueError) as err:
        RomanOpticalModel(chromatic=False, max_zernike=6, nominal_interp='bad')
    assert "must be one of" in str(err.value)

@timer
def test_corner_cache():
    """Verify corner-profile caching reuses one 4-corner set for same SCA and params.
    """
    with fast_pupil_bin():
        bandpass = galsim.roman.getBandpasses()['H158']
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'chromatic': False, 'max_zernike': 6}
        )
        psf.set_context(None, None, bandpass)
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
        profiles1 = psf.model._get_sample_profiles(stars[0], params, cache=True)
        profiles2 = psf.model._get_sample_profiles(stars[1], params, cache=True)
        assert profiles1 is profiles2
        assert len(psf.model._corner_cache) == 1
        assert 5 in psf.model._corner_cache
        assert len(psf.model._corner_cache[5][2]) == 4

        # In five-point mode, the cache stores four corners plus center.
        psf5 = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'chromatic': False,
                'max_zernike': 6,
                'nominal_interp': 'five_point',
            }
        )
        psf5.set_context(None, None, bandpass)
        stars5, _ = psf5.initialize_params(stars, logger=logger)
        psf5.drawStar(stars5[0])
        params5 = stars5[0].fit.params
        profiles5 = psf5.model._get_sample_profiles(stars5[0], params5, cache=True)
        assert len(profiles5) == 5
        assert len(psf5.model._corner_cache[5][2]) == 5


@timer
def test_five_point_weights():
    """Check five-point interpolation weights for corner/center and quadratic basis terms.
    """
    model = make_roman_model(chromatic=False, max_zernike=6, nominal_interp='five_point')
    sca = 7
    size = model.sca_size

    def make_star(x, y):
        return piff.Star.makeTarget(
            x=x,
            y=y,
            stamp_size=25,
            scale=0.11,
            properties={'sca': sca},
        ).withFlux(1.0, (0.0, 0.0))

    fp = model._get_roman_five_point_data(sca)
    points = fp['points']

    # Sampling locations used by the interpolation should have a single w=1.
    w_ll = model._five_point_weights(make_star(points[0].x, points[0].y))
    w_lr = model._five_point_weights(make_star(points[1].x, points[1].y))
    w_ul = model._five_point_weights(make_star(points[2].x, points[2].y))
    w_ur = model._five_point_weights(make_star(points[3].x, points[3].y))
    w_c = model._five_point_weights(make_star(points[4].x, points[4].y))
    np.testing.assert_allclose(w_ll, [1, 0, 0, 0, 0], atol=1.e-11, rtol=0.0)
    np.testing.assert_allclose(w_lr, [0, 1, 0, 0, 0], atol=1.e-11, rtol=0.0)
    np.testing.assert_allclose(w_ul, [0, 0, 1, 0, 0], atol=1.e-11, rtol=0.0)
    np.testing.assert_allclose(w_ur, [0, 0, 0, 1, 0], atol=1.e-11, rtol=0.0)
    np.testing.assert_allclose(w_c, [0, 0, 0, 0, 1], atol=1.e-11, rtol=0.0)

    # Verify exact reproduction of [1, x, y, xy, x^2+y^2] basis under this interpolation.
    sample_xy = np.array([(p.x, p.y) for p in points], dtype=float)

    def basis_values(x, y):
        return np.array([1.0, x, y, x * y, x * x + y * y], dtype=float)

    sample_basis = np.array([basis_values(x, y) for x, y in sample_xy])
    test_positions = [
        (0.13 * size, 0.27 * size),
        (0.82 * size, 0.39 * size),
        (0.61 * size, 0.91 * size),
    ]
    for x, y in test_positions:
        star = make_star(x, y)
        w = np.array(model._five_point_weights(star))
        expected = basis_values(x, y)
        interp = w.dot(sample_basis)
        np.testing.assert_allclose(interp, expected, atol=1.e-8, rtol=0.0)


@timer
def test_fit():
    """Check local convergence of single-star Roman fits in constant mode.
    """
    with fast_pupil_bin():
        model = make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=1.0e6)
        star = piff.Star.makeTarget(
            x=64.0,
            y=64.0,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5},
        ).withFlux(1.0, (0.0, 0.0))

        star = model.initialize(star)
        # Keep injected extra aberrations modest relative to baseline Roman optics.
        # Typical built-in |z4..z22| values are around 1e-3 to a few e-2, so 4-5e-3 is
        # realistic while still providing measurable signal in this unit test.
        truth_params = np.array([0.004, -0.003, 0.005])
        prof = model.getProfile(truth_params, star=star)
        model._draw_profile_to_image(prof * star.fit.flux, star.image, star.image_pos, star)
        fitted = model.fit(star)
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
        prof = model.getProfile(star=star)
        assert prof is not None
        drawn_star = model.draw(star)
        assert drawn_star.image.array.shape == star.image.array.shape

        # Error if no star argument in getProfile.  (Allowed by some other classes.)
        with pytest.raises(ValueError) as err:
            model.getProfile(params=np.zeros(model.param_len), star=None)
        assert "requires the star argument" in str(err.value)

        # Error if params is given and has wrong length.
        with pytest.raises(ValueError) as err:
            model.getProfile(star=star, params=np.zeros(model.param_len+2))
        assert "params must have length 3" in str(err.value)

        # Error if convert_funcs is given but has different length than stars.
        with pytest.raises(ValueError) as err:
            model.fit_many([star], convert_funcs=[])
        assert "len(convert_funcs) must match len(stars)" in str(err.value)


@timer
def test_aberration_prior():
    """Validate the use of priors on the aberration values.
    """
    # Check that fitted aberrations stay closer to 0 when strong prior is applied.
    weak_prior = make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=1.0e6)
    strong_prior = make_roman_model(
        chromatic=False, max_zernike=6, aberration_prior_sigma=[0.02]
    )  # List with 1 element treated as scalar.
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
    no_prior = make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=None)
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
    inf_prior = make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=np.inf)
    no_prior_result, _ = no_prior._solve_params(aw, bw, p0)
    inf_prior_result, _ = inf_prior._solve_params(aw, bw, p0)
    print('no prior',no_prior_result)
    print('inf prior',inf_prior_result)
    np.testing.assert_allclose(no_prior_result, inf_prior_result)

    # prior length must match number of zernikes used
    with pytest.raises(ValueError) as err:
        make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=[1.0, 2.0])
    assert "scalar or length 3" in str(err.value)

    with pytest.raises(ValueError) as err:
        make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=[])
    assert "scalar or length 3" in str(err.value)

    # Same for linear mode (in particular the allowed length is 3 here, not 9,
    # which is the full param_len).
    with pytest.raises(ValueError) as err:
        make_roman_model(
            chromatic=False,
            max_zernike=6,
            aberration_interp='linear',
            aberration_prior_sigma=[1.0] * 12,
        )
    assert "scalar or length 3" in str(err.value)

    # In linear mode, scalar and one-point vectors tile to the (a, b, c) coefficient blocks.
    linear_scalar = make_roman_model(
        chromatic=False,
        max_zernike=6,
        aberration_interp='linear',
        aberration_prior_sigma=0.05,
    )
    np.testing.assert_allclose(linear_scalar.prior_sigma, np.full(9, 0.05))

    linear_vec = make_roman_model(
        chromatic=False,
        max_zernike=6,
        aberration_interp='linear',
        aberration_prior_sigma=[0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(linear_vec.prior_sigma, np.tile([0.1, 0.2, 0.3], 3))

    # priors cannot be <= 0
    with pytest.raises(ValueError) as err:
        make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=[1.0, 0.0, 1.0])
    assert "must all be > 0" in str(err.value)

    with pytest.raises(ValueError) as err:
        make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=[1.0, -1.0, 1.0])
    assert "must all be > 0" in str(err.value)


@timer
def test_linear_prior_io():
    """Ensure linear-mode aberration priors serialize compactly and round-trip correctly.
    """
    psf = piff.PSF.process(
        {
            'type': 'RomanOptics',
            'chromatic': False,
            'max_zernike': 6,
            'aberration_interp': 'linear',
            'aberration_prior_sigma': [0.11, 0.22, 0.33],
        }
    )

    fn = os.path.join('output', 'roman_linear_prior_io.piff')
    psf.write(fn)
    psf2 = piff.read(fn)

    model = psf2.model
    np.testing.assert_allclose(model.orig_prior_sigma,
                               [0.11, 0.22, 0.33], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        model.prior_sigma,
        np.tile([0.11, 0.22, 0.33], 3),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        model.kwargs['aberration_prior_sigma'],
        [0.11, 0.22, 0.33],
        atol=0.0,
        rtol=0.0,
    )

    # Also check scalar prior in input config.
    psf_scalar = piff.PSF.process(
        {
            'type': 'RomanOptics',
            'chromatic': False,
            'max_zernike': 6,
            'aberration_interp': 'linear',
            'aberration_prior_sigma': 0.05,
        }
    )
    fn_scalar = os.path.join('output', 'roman_linear_prior_io_scalar.piff')
    psf_scalar.write(fn_scalar)
    psf_scalar2 = piff.read(fn_scalar)
    model_scalar = psf_scalar2.model
    np.testing.assert_allclose(model_scalar.orig_prior_sigma,
                               [0.05, 0.05, 0.05], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        model_scalar.prior_sigma,
        np.full(9, 0.05),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        model_scalar.kwargs['aberration_prior_sigma'],
        [0.05, 0.05, 0.05],
        atol=0.0,
        rtol=0.0,
    )


@timer
def test_fit_many():
    """Check accuracy of fitting multiple stars using fit_many.
    """
    with fast_pupil_bin():
        model = make_roman_model(chromatic=False, max_zernike=6, aberration_prior_sigma=1.0e6)
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
        # Use the same realistic small-amplitude extra-aberration vector as test_fit.
        truth_params = np.array([0.004, -0.003, 0.005])
        stars = [
            piff.Star(
                s.data,
                s.fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params),
                ),
            )
            for s in stars
        ]
        for star in stars:
            prof = model.getProfile(truth_params, star=star)
            model._draw_profile_to_image(prof * star.fit.flux, star.image, star.image_pos, star)

        # 1 pass isn't great, but after 2 passes, the agreement is sub percent.
        # And 3 is within 0.1% agreement.
        for _ in range(3):
            stars = model.fit_many(stars)
            print('fit[0] => ',stars[0].fit.params)

        p0 = stars[0].fit.params
        p1 = stars[1].fit.params
        print('final p0 = ',p0)
        print('final p1 = ',p1)
        print('   truth = ',truth_params)
        np.testing.assert_allclose(p0, p1, atol=1.0e-12, rtol=0.0)
        np.testing.assert_allclose(p0, truth_params, atol=0.0, rtol=1.e-3)


@timer
def test_fit_many_nproc():
    """Check `fit_many` multiprocessing path and accuracy with `nproc > 1`.
    """
    with fast_pupil_bin():
        model = make_roman_model(
            chromatic=False, max_zernike=6, aberration_prior_sigma=1.0e6, nproc=2
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
        stars = [
            piff.Star(
                s.data,
                s.fit.newParams(
                    np.zeros_like(truth_params),
                    params_var=np.zeros_like(truth_params),
                ),
            )
            for s in stars
        ]
        for star in stars:
            prof = model.getProfile(truth_params, star=star)
            model._draw_profile_to_image(prof * star.fit.flux, star.image, star.image_pos, star)

        for _ in range(3):
            stars = model.fit_many(stars)

        assert [int(s['sca']) for s in stars] == [5, 5]
        for s in stars:
            np.testing.assert_allclose(s.fit.params, truth_params, atol=0.0, rtol=2.e-3)
        assert list(model._corner_cache.keys()) == [5]


@timer
def test_bilinear_vs_five_point():
    """Exercise practical fit behavior and quantify bilinear vs five-point differences.
    """
    with fast_pupil_bin():
        # Make 5 stars at various points around an SCA using GalSim.
        filt = 'H158'
        sca = 8
        size = galsim.roman.n_pix

        def make_star(x, y):
            return piff.Star.makeTarget(
                x=x,
                y=y,
                stamp_size=25,
                scale=0.11,
                properties={'sca': sca},
            ).withFlux(1.0, (0.0, 0.0))

        positions = [
            (0.12 * size, 0.18 * size),
            (0.33 * size, 0.74 * size),
            (0.71 * size, 0.21 * size),
            (0.83 * size, 0.66 * size),
            (0.49 * size, 0.52 * size),
        ]
        stars = [make_star(x, y) for x, y in positions]
        truth_params = np.array([0.004, -0.003, 0.005])
        extra_truth = np.concatenate(([0,0,0,0], truth_params))
        wavelength = galsim.roman.getBandpasses()[filt].effective_wavelength
        for st in stars:
            prof = galsim.roman.getPSF(
                int(st['sca']), filt,
                SCA_pos=st.image_pos,
                pupil_bin=piff.roman.roman_psf.pupil_bin,
                wcs=st.image.wcs,
                extra_aberrations=extra_truth,
                wavelength=wavelength,
            )
            prof.drawImage(st.image, center=st.image_pos)

        # Fit these stars with both models in Piff.  Bilinear and 5-point.
        # Note: GalSim internally does bilinear interpolation of the nominal aberrations.
        # This is not the same bilinear that we do in Piff -- we do either bilinear or
        # 5-point interpolation of the profiles.  These are not equivalent.
        model_bilinear = make_roman_model(
            chromatic=False,
            max_zernike=6,
            nominal_interp='bilinear',
            aberration_prior_sigma=1.0e6,
        )
        model_five = make_roman_model(
            chromatic=False,
            max_zernike=6,
            nominal_interp='five_point',
            aberration_prior_sigma=1.0e6,
        )
        fit_bilinear = [model_bilinear.initialize(st) for st in stars]
        fit_five = [model_five.initialize(st) for st in stars]
        for it in range(3):
            fit_bilinear = model_bilinear.fit_many(fit_bilinear)
            fit_five = model_five.fit_many(fit_five)
            print(f'iter {it+1} bilinear fit[0] = ', fit_bilinear[0].fit.params)
            print(f'iter {it+1} five-point fit[0] = ', fit_five[0].fit.params)
            print(
                f'iter {it+1} chisq (bilinear, five-point) = ',
                sum(st.fit.chisq for st in fit_bilinear),
                sum(st.fit.chisq for st in fit_five),
            )

        # Compare the results.
        chisq_bilinear = sum(st.fit.chisq for st in fit_bilinear)
        chisq_five = sum(st.fit.chisq for st in fit_five)
        params_bilinear = fit_bilinear[0].fit.params
        params_five = fit_five[0].fit.params
        print('bilinear chisq = ', chisq_bilinear)
        print('five-point chisq = ', chisq_five)
        print('final bilinear fit[0] = ', params_bilinear)
        print('final five-point fit[0] = ', params_five)
        print('truth params = ', truth_params)

        # With direct GalSim truth generation, both nominal interpolation options should converge
        # to reasonable fits, but neither is expected to recover truth_params exactly
        # for the reason mentioned above.  (Profile interpolation != aberration interpolation.)
        # Interesting that it turns out that bilinear gets the model parameters somewhat
        # closer to the truth, but the chisq for five_point is smaller.
        np.testing.assert_allclose(params_bilinear, truth_params, atol=1.5e-3)
        np.testing.assert_allclose(params_five, truth_params, atol=4.e-3)
        assert chisq_bilinear < 1.0e-6
        assert chisq_five < 4.0e-7
        # Quantify that the two nominal interpolation choices are observably different.
        assert abs(chisq_five - chisq_bilinear) > 1.e-7

        # Quantify direct model-image difference at the same position and params.
        sample = make_star(0.63 * size, 0.37 * size)
        image_b = model_bilinear.draw(model_bilinear.initialize(sample)).image.array
        image_f = model_five.draw(model_five.initialize(sample)).image.array
        mean_abs_diff = np.mean(np.abs(image_b - image_f))
        print('mean abs image difference = ', mean_abs_diff)
        assert mean_abs_diff > 1.e-6


@timer
def test_fit_linear():
    """Check linear-mode convergence with stars spanning the SCA geometry.
    """
    with fast_pupil_bin():
        # Use stars spread across the SCA so the constant and slope terms are constrained.
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
        truth_params = np.array([
            0.004, -0.003, 0.005,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])
        for nominal_interp in ['bilinear', 'five_point']:
            model = make_roman_model(
                chromatic=False,
                max_zernike=6,
                aberration_interp='linear',
                nominal_interp=nominal_interp,
                aberration_prior_sigma=1.0e6,
            )
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
            stars = [model.initialize(star) for star in stars]
            for star in stars:
                prof = model.getProfile(truth_params, star=star)
                model._draw_profile_to_image(prof * star.fit.flux, star.image, star.image_pos, star)

            for _ in range(4):
                stars = model.fit_many(stars)

            fitted_params = stars[0].fit.params
            fitted_var = stars[0].fit.params_var

            for star in stars:
                assert star.fit.chisq >= 0
                assert star.fit.dof > 0
                np.testing.assert_allclose(star.fit.params, fitted_params, atol=1.e-12, rtol=0.0)
            print(f'linear {nominal_interp} fit = ', fitted_params)
            print('                truth = ', truth_params)
            np.testing.assert_allclose(fitted_params[:3], truth_params[:3], atol=0.0, rtol=1.e-3)
            np.testing.assert_allclose(fitted_params[3:], truth_params[3:], atol=5.e-9, rtol=0.0)
            assert np.all(fitted_var >= 0)

            for star in stars:
                model_star = model.draw(star)
                np.testing.assert_allclose(
                    model_star.image.array, star.image.array, atol=5.e-5, rtol=0.0
                )


@timer
def test_fit_linear_gradient():
    """Check linear-mode recovery when aberrations vary as a + b x + c y across the SCA.
    """
    with fast_pupil_bin():
        model = make_roman_model(
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
        stars = [model.initialize(star) for star in stars]

        # Define each Zernike truth as a + b*(x/4088) + c*(y/4088).
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

        truth_params = np.concatenate([abc[:, 0], abc[:, 1], abc[:, 2]])
        corner_truth = np.array([
            eval_truth(0.0, 0.0),
            eval_truth(model.sca_size, 0.0),
            eval_truth(0.0, model.sca_size),
            eval_truth(model.sca_size, model.sca_size),
        ])
        assert not np.any(np.isclose(corner_truth[0], corner_truth[1]))
        assert not np.any(np.isclose(corner_truth[0], corner_truth[2]))
        assert not np.any(np.isclose(corner_truth[0], corner_truth[3]))

        # Verify independently that bilinear interpolation of corners reproduces the
        # per-star truth function for a purely planar field.
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

        # Draw the stars to use for fitting using direct GalSim Roman PSFs at each star
        # position, with the local extra-aberration vector from eval_truth.
        for star in stars:
            aber = eval_truth(star.image_pos.x, star.image_pos.y)
            prof = galsim.roman.getPSF(5, 'H158', star.image_pos,
                                       pupil_bin=piff.roman.roman_psf.pupil_bin,
                                       wcs=star.image.wcs,
                                       extra_aberrations=model._make_extra_aberrations(aber),
                                       wavelength=model.bandpass.effective_wavelength)
            prof.drawImage(star.image, method='auto', center=star.image_pos)

        for _ in range(5):
            stars = model.fit_many(stars)

        # Note: these are not expected to match exactly.  The real images include the
        # natural variation within the SCA from the roman aberration pattern, fully separate
        # from the extra_aberrations we're fitting for.  This variation is not quite linear,
        # so the linear approximation is a small model mismatch.  However, it's relatively
        # close in the fitted extra aberrations, and the drawn images are very close.
        fitted_params = stars[0].fit.params
        print('linear gradient fit = ', fitted_params)
        print('           truth = ', truth_params)
        np.testing.assert_allclose(fitted_params, truth_params, atol=3.e-4, rtol=0.3)
        for star in stars:
            np.testing.assert_allclose(star.fit.params, fitted_params)
        for star in stars:
            model_star = model.draw(star)
            np.testing.assert_allclose(
                model_star.image.array, star.image.array, atol=1.e-4, rtol=0.002)


@timer
def test_fit_linear_nproc():
    """Check linear-mode fit_many behavior with multiprocessing enabled.
    """
    with fast_pupil_bin():
        model = make_roman_model(
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
        stars = [model.initialize(star) for star in stars]
        truth_params = np.array([
            0.004, -0.003, 0.005,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])
        for star in stars:
            prof = model.getProfile(truth_params, star=star)
            model._draw_profile_to_image(prof * star.fit.flux, star.image, star.image_pos, star)

        for _ in range(4):
            stars = model.fit_many(stars)
        for star in stars:
            np.testing.assert_allclose(star.fit.params[:3], truth_params[:3], atol=0.0, rtol=1.e-3)
            np.testing.assert_allclose(star.fit.params[3:], truth_params[3:], atol=5.e-9, rtol=0.0)



@timer
def test_optics_convert_funcs():
    """Check aberration recovery when fitting with a nontrivial convert_func (profile shear).
    """
    with fast_pupil_bin():
        bandpass = galsim.roman.getBandpasses()['H158']
        psf = piff.PSF.process(
            {
                'type': 'RomanOptics',
                'chromatic': False,
                'max_zernike': 6,
                'aberration_prior_sigma': 1.0e6,
            }
        )
        psf.set_context(None, None, bandpass)
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
        for star in stars:
            # True profile is sheared version of optical PSF.
            prof = psf.model.getProfile(truth_params, star=star).shear(g1=0.01, g2=-0.005)
            psf.model._draw_profile_to_image(prof, star.image, star.image_pos, star)

        for _ in range(3):
            stars, nremoved = psf.single_iteration(
                stars,
                logger=logger,
                convert_funcs=convert_funcs,
                draw_method=None,
            )
            assert nremoved == 0
            print('params[0] => ',stars[0].fit.params)

        assert len(stars) == len(convert_funcs)
        print('truth = ',truth_params)
        for i, star in enumerate(stars):
            print(f'star {i} params = ',star.fit.params)
            np.testing.assert_allclose(star.fit.params, truth_params, atol=0.0, rtol=1.e-3)


@timer
def test_sca_interp():
    """Test per-SCA/global interpolation behavior and RomanSCAInterp serialization round-trip.
    """
    with fast_pupil_bin():
        # Start with some basic exercises of interp machinery.
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'chromatic': False, 'max_zernike': 6}
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
            }
        )
        psf.write(fn)
        psf2 = piff.read(fn)
        assert isinstance(psf2.interp, RomanSCAInterp)
        assert psf2.interp.global_mean is None
        assert psf2.interp.sca_mean == {}


if __name__ == '__main__':
    test_roman_optics()
    test_corner_cache()
    test_five_point_weights()
    test_fit()
    test_aberration_prior()
    test_fit_many()
    test_fit_many_nproc()
    test_bilinear_vs_five_point()
    test_fit_linear()
    test_fit_linear_gradient()
    test_fit_linear_nproc()
    test_optics_convert_funcs()
    test_sca_interp()
