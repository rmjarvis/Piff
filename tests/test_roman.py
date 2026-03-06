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
import numpy as np
import os
import piff
import galsim
import tempfile
from unittest import mock

from piff_test_helper import timer


@timer
def test_roman_optics():
    calls = []

    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        calls.append(
            {
                'SCA': SCA,
                'bandpass': bandpass,
                'SCA_pos': SCA_pos,
                'extra_aberrations': np.array(extra_aberrations, copy=True),
                'wavelength': wavelength,
            }
        )
        return galsim.Gaussian(sigma=0.2)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
        psf = piff.PSF.process(
            {'type': 'RomanOptics', 'filter': 'H158', 'chromatic': False, 'max_zernike': 6}
        )
        assert isinstance(psf, piff.RomanOptics)
        logger = piff.config.setup_logger()

        star = piff.Star.makeTarget(
            x=123.4,
            y=456.7,
            stamp_size=25,
            scale=0.11,
            properties={'chipnum': 3, 'sca': 5},
        )
        star = star.withFlux(1.0, (0.0, 0.0))

        stars, nremoved = psf.initialize_params([star], logger=logger)
        assert nremoved == 0
        model_star = psf.drawStar(stars[0])
        assert model_star.image.array.shape == (25, 25)

        # If both are present, prefer the explicit sca value.
        star = piff.Star.makeTarget(
            x=12.3,
            y=45.6,
            stamp_size=25,
            scale=0.11,
            properties={'chipnum': 7, 'sca': 4},
        )
        star = star.withFlux(1.0, (0.0, 0.0))
        psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])

        # If sca is absent, chipnum acts as an alias.
        star = piff.Star.makeTarget(
            x=78.9,
            y=10.1,
            stamp_size=25,
            scale=0.11,
            properties={'chipnum': 8},
        )
        star = star.withFlux(1.0, (0.0, 0.0))
        psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])

        # If neither is present, fail with the preferred sca-centric message.
        star = piff.Star.makeTarget(
            x=11.1,
            y=22.2,
            stamp_size=25,
            scale=0.11,
        )
        star = star.withFlux(1.0, (0.0, 0.0))
        try:
            psf.drawStar(psf.initialize_params([star], logger=logger)[0][0])
            assert False
        except ValueError as e:
            assert "explicit 'sca' property" in str(e)

    assert len(calls) == 12
    sca_calls = [call['SCA'] for call in calls]
    assert sca_calls.count(5) == 4
    assert sca_calls.count(4) == 4
    assert sca_calls.count(8) == 4
    assert all(call['bandpass'] == 'H158' for call in calls)
    assert all(call['SCA_pos'] is not None for call in calls)
    assert all(np.allclose(call['extra_aberrations'], np.zeros(3)) for call in calls)
    assert all(
        np.isclose(call['wavelength'], psf.model.bandpass.effective_wavelength)
        for call in calls
    )


@timer
def test_roman_corner_cache():
    calls = []

    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        calls.append((SCA, SCA_pos.x, SCA_pos.y))
        return galsim.Gaussian(sigma=0.2)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
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

    # Same SCA and params should reuse the same four corner PSFs.
    assert len(calls) == 4


@timer
def test_roman_fit():
    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        sigma = 0.2 + extra_aberrations[0]
        return galsim.Gaussian(sigma=sigma)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
        model = piff.Roman(
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
        )
        star = star.withFlux(1.0, (0.0, 0.0))

        init_star = model.initialize(star)
        truth_params = np.array([0.03, 0.0, 0.0])
        truth_fit = init_star.fit.newParams(
            truth_params,
            params_var=np.zeros_like(truth_params),
            num=model._num,
        )
        truth_star = model.draw(piff.Star(init_star.data, truth_fit))

        fit_star = piff.Star(truth_star.data, init_star.fit)
        fitted = model.fit(fit_star)
        fitted_params = fitted.fit.get_params(model._num)
        fitted_var = fitted.fit.get_params_var(model._num)

    assert fitted.fit.chisq >= 0
    assert fitted.fit.dof > 0
    assert fitted_params[0] > 0
    np.testing.assert_allclose(fitted_params[0], truth_params[0], atol=5.0e-3)
    np.testing.assert_allclose(fitted_params[1:], truth_params[1:], atol=5.0e-3)
    assert np.all(fitted_var >= 0)


@timer
def test_roman_aberration_prior():
    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        sigma = 0.2 + extra_aberrations[0]
        return galsim.Gaussian(sigma=sigma)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
        weak_prior = piff.Roman(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_prior_sigma=1.0e6,
        )
        strong_prior = piff.Roman(
            filter='H158',
            chromatic=False,
            max_zernike=6,
            aberration_prior_sigma=0.02,
        )
        star = piff.Star.makeTarget(
            x=64.0,
            y=64.0,
            stamp_size=25,
            scale=0.11,
            properties={'sca': 5},
        )
        star = star.withFlux(1.0, (0.0, 0.0))

        init_weak = weak_prior.initialize(star)
        truth_params = np.array([0.08, 0.0, 0.0])
        truth_fit = init_weak.fit.newParams(
            truth_params,
            params_var=np.zeros_like(truth_params),
            num=weak_prior._num,
        )
        truth_star = weak_prior.draw(piff.Star(init_weak.data, truth_fit))

        fit_weak = weak_prior.fit(piff.Star(truth_star.data, init_weak.fit))
        init_strong = strong_prior.initialize(star)
        fit_strong = strong_prior.fit(piff.Star(truth_star.data, init_strong.fit))

        a_weak = fit_weak.fit.get_params(weak_prior._num)[0]
        a_strong = fit_strong.fit.get_params(strong_prior._num)[0]
        assert abs(a_strong) < abs(a_weak)

        try:
            piff.Roman(
                filter='H158',
                chromatic=False,
                max_zernike=6,
                aberration_prior_sigma=[1.0, 2.0],
            )
            assert False
        except ValueError as e:
            assert "scalar or length 3" in str(e)

        try:
            piff.Roman(
                filter='H158',
                chromatic=False,
                max_zernike=6,
                aberration_prior_sigma=[1.0, 0.0, 1.0],
            )
            assert False
        except ValueError as e:
            assert "must all be > 0" in str(e)


@timer
def test_roman_fit_many_matches_fit():
    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        sigma = 0.2 + extra_aberrations[0]
        return galsim.Gaussian(sigma=sigma)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
        model = piff.Roman(
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
        truth_params = np.array([0.04, 0.0, 0.0])
        truth = [
            model.draw(
                piff.Star(
                    s.data,
                    s.fit.newParams(
                        truth_params,
                        params_var=np.zeros_like(truth_params),
                        num=model._num,
                    ),
                )
            )
            for s in stars
        ]
        fit_stars = [piff.Star(s.data, stars[i].fit) for i, s in enumerate(truth)]

        fit_many = model.fit_many(fit_stars)
        fit_one = [model.fit(s) for s in fit_stars]

        for s_many, s_one in zip(fit_many, fit_one):
            np.testing.assert_allclose(
                s_many.fit.get_params(model._num),
                s_one.fit.get_params(model._num),
                atol=1.0e-6,
                rtol=1.0e-6,
            )


@timer
def test_roman_optics_convert_funcs_batch():
    def fake_get_psf(
        SCA,
        bandpass,
        SCA_pos=None,
        pupil_bin=4,
        wcs=None,
        n_waves=None,
        extra_aberrations=None,
        wavelength=None,
        gsparams=None,
        logger=None,
        high_accuracy=None,
        approximate_struts=None,
    ):
        sigma = 0.2 + extra_aberrations[0]
        return galsim.Gaussian(sigma=sigma)

    with mock.patch('galsim.roman.getPSF', side_effect=fake_get_psf):
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

        truth_params = np.array([0.04, 0.0, 0.0])
        truth_stars = []
        for star in stars:
            truth_fit = star.fit.newParams(
                truth_params,
                params_var=np.zeros_like(truth_params),
                num=psf._num,
            )
            truth_stars.append(psf.model.draw(piff.Star(star.data, truth_fit)))

        convert_funcs = [lambda prof: prof.shear(g1=0.01, g2=-0.005)] * len(truth_stars)
        fitted, nremoved = psf.single_iteration(
            truth_stars,
            logger=logger,
            convert_funcs=convert_funcs,
            draw_method=None,
        )
        assert nremoved == 0
        assert len(fitted) == len(truth_stars)
        for star in fitted:
            assert star.fit.get_params(psf._num) is not None


@timer
def test_roman_sca_interp():
    psf = piff.PSF.process(
        {'type': 'RomanOptics', 'filter': 'H158', 'chromatic': False, 'max_zernike': 6}
    )
    assert type(psf.interp).__name__ == 'RomanSCAInterp'

    logger = piff.config.setup_logger()
    p1 = np.array([0.1, 0.0, -0.1])
    p2 = np.array([-0.2, 0.3, 0.0])

    s1 = piff.Star.makeTarget(x=10.0, y=20.0, stamp_size=25, scale=0.11, properties={'sca': 2})
    s2 = piff.Star.makeTarget(x=30.0, y=40.0, stamp_size=25, scale=0.11, properties={'sca': 5})
    s1 = s1.withFlux(1.0, (0.0, 0.0))
    s2 = s2.withFlux(1.0, (0.0, 0.0))

    stars, _ = psf.initialize_params([s1, s2], logger=logger)
    stars[0] = piff.Star(
        stars[0].data,
        stars[0].fit.newParams(p1, params_var=np.zeros_like(p1), num=psf._num),
    )
    stars[1] = piff.Star(
        stars[1].data,
        stars[1].fit.newParams(p2, params_var=np.zeros_like(p2), num=psf._num),
    )
    psf.interp.solve(stars)

    t1 = piff.Star.makeTarget(x=1.0, y=2.0, stamp_size=25, scale=0.11, properties={'sca': 2})
    t2 = piff.Star.makeTarget(
        x=3.0, y=4.0, stamp_size=25, scale=0.11, properties={'chipnum': 5}
    )
    t1 = t1.withFlux(1.0, (0.0, 0.0))
    t2 = t2.withFlux(1.0, (0.0, 0.0))
    tstars, _ = psf.initialize_params([t1, t2], logger=logger)
    tstars = psf.interpolateStarList(tstars)

    np.testing.assert_allclose(tstars[0].fit.get_params(psf._num), p1)
    np.testing.assert_allclose(tstars[1].fit.get_params(psf._num), p2)

    # Check that write/read preserves the per-SCA interpolation solution.
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'roman_sca_test.piff')
        psf.write(fn)
        psf2 = piff.read(fn)
        tstars2, _ = psf2.initialize_params([t1, t2], logger=logger)
        tstars2 = psf2.interpolateStarList(tstars2)
        np.testing.assert_allclose(tstars2[0].fit.get_params(psf2._num), p1)
        np.testing.assert_allclose(tstars2[1].fit.get_params(psf2._num), p2)
