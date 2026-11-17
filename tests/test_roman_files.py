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

import os
import shutil
import numpy as np
import galsim
import fitsio
import pytest
import piff

from piff_test_helper import timer


@pytest.fixture(scope='module', autouse=True)
def setup():
    """Write the Roman image, catalog, sky, and reference calibration used by these tests."""
    asdf = pytest.importorskip('asdf')
    gwcs = pytest.importorskip('gwcs')
    from astropy import units
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs.coordinate_frames import Frame2D, CelestialFrame

    # Build an image with sloping sky and 3 stars in it.
    y, x = np.indices((128, 128))
    sky = 5 + 0.001 * x + 0.002 * y
    signal = galsim.ImageD(128, 128, scale=0.11)
    positions = [(33.2, 35.7), (93.4, 42.2), (65.1, 93.8)]
    for pos, flux in zip(positions, [3000, 6000, 4000]):
        profile = galsim.Moffat(beta=3.5, fwhm=0.3, flux=flux).shear(g1=0.04, g2=-0.02)
        profile.drawImage(image=signal, center=galsim.PositionD(*pos), add_to_image=True)
    science = (sky + signal.array).astype(np.float32)

    # Make the variances realistic given a specific set of read times.
    pattern = [[1], [2, 3], [4, 5, 6], [7, 8], [9, 10, 11, 12]]
    t_read = np.arange(1, 13) * 2.0  # Individual read times in seconds.
    n = np.array([len(reads) for reads in pattern])  # Number of reads per resultant.

    # R averages individual reads into resultants.
    # C_r and C_p are the resultant covariances for the read noise and poisson noise respectively,
    # per unit read variance (g) and photon arrival rate (f).
    # t gives the mean resultant times.
    R = np.zeros((len(pattern), len(t_read)))
    for i, reads in enumerate(pattern):
        for read_number in reads:
            R[i, read_number - 1] = 1.0 / n[i]
    C_r = R @ R.T
    P = np.array([[min(t1, t2) for t2 in t_read] for t1 in t_read])
    C_p = R @ P @ R.T
    t = R @ t_read

    # Compute the optimal up-the-ramp estimates with optimal weights the way the project
    # computes these.
    g = 12.5  # Read-noise variance per individual read.
    gain = 3.7  # Actual detector gain; the photon covariance amplitude is f = signal/gain.
    A = np.array([[1.0, ti] for ti in t])  # Design matrix for fitting the ramp.
    f = science / gain 

    # The first two axes index pixels; the last two hold each pixel's covariance matrix.
    C = g * C_r + f[:, :, None, None] * C_p

    # Fit intercept and slope to resultants r via
    # beta = F r, where F = (A.T C^-1 A)^-1 A.T C^-1.
    # Row 1 of each pixel's F gives its slope weights w.
    Cinv = np.linalg.inv(C)
    F = np.linalg.inv(A.T @ Cinv @ A) @ A.T @ Cinv
    w = F[:, :, 1, :]

    # Propagate each covariance contribution: w.T C w. Summing over the last axis
    # contracts the resultant index, leaving read- and photon-variance images V_r and V_p.
    V_r = g * np.sum((w @ C_r) * w, axis=-1)
    V_p = f * np.sum((w @ C_p) * w, axis=-1)

    # Uniform fitting uses resultant weights W=diag(n). Solve the weighted normal
    # equations directly, independently of the closed-form coefficients in RomanFiles.
    # As above, F maps resultants to intercept and slope; u is the uniform slope vector.
    W = np.diag(n)
    F = np.linalg.inv(A.T @ W @ A) @ A.T @ W
    u = F[1]

    # Propagating the uniform weights gives read variance g u.T C_r u and photon variance
    # (signal/gain) u.T C_p u. Thus the effective gain is gain/(u.T C_p u).
    uniform_read = g * (u @ C_r @ u)
    uniform_gain = gain / (u @ C_p @ u)

    # Make a TAN WCS in GWCS format.
    wcs = gwcs.WCS(
        (models.Shift(-63.5) & models.Shift(-63.5))
        | (models.Scale(-0.11 / 3600) & models.Scale(0.11 / 3600))
        | models.Pix2Sky_TAN() | models.RotateNative2Celestial(53, -27, 180),
        input_frame=Frame2D(name='detector', unit=(units.pix, units.pix)),
        output_frame=CelestialFrame(reference_frame=ICRS(), name='world',
                                   unit=(units.deg, units.deg)),
    )

    # Include two allowed DQ bits and one disallowed bit for the masking tests below.
    dq = np.zeros(science.shape, dtype=np.uint32)
    dq[10, 10] = 8192
    dq[11, 11] = 4
    dq[12, 12] = 16777216
    dq[35, 32] = 4

    # A 30-ERR DUMO excursion at an isolated pixel should be rejected by the optional
    # spike screen; the same excursion beside a bright source should be preserved.
    dumo = np.full(science.shape, 0.01, dtype=np.float32)
    dumo[11, 11] = 30 * np.sqrt(V_r[11, 11] + V_p[11, 11])
    dumo[35, 32] = 30 * np.sqrt(V_r[35, 32] + V_p[35, 32])

    tree = {
        'roman': {
            'data': science,
            'err': np.sqrt(V_r + V_p),
            'var_poisson': V_p,
            'dq': dq,
            'dumo': dumo,
            'meta': {
                'wcs': wcs,
                'exposure': {'frame_time': 2.0, 'read_pattern': pattern},
            },
        },
    }
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    with asdf.AsdfFile(tree) as af:
        af.write_to(os.path.join(output_dir, 'roman_input.asdf'))
    catalog_rows = []
    for (x, y), sky_value in zip(positions, [4.0, 5.0, 6.0]):
        catalog_rows.append((x, y, sky_value))
    catalog = np.array(catalog_rows, dtype=[('x', float), ('y', float), ('sky', float)])
    fitsio.write(os.path.join(output_dir, 'roman_cat.fits'), catalog, clobber=True)
    fitsio.write(os.path.join(output_dir, 'roman_sky.fits'), sky.astype(np.float32), clobber=True)
    np.savez(os.path.join(output_dir, 'roman_truth.npz'),
             uniform_read=uniform_read, uniform_gain=uniform_gain)


@timer
def test_roman_input():
    """Test the normal path using the optimal-ramp values."""
    # Default optimal fitting uses DATA, rejects nonzero DQ, and removes photon variance.
    asdf = pytest.importorskip('asdf')
    file_name = os.path.join('output', 'roman_input.asdf')
    cat_name = os.path.join('output', 'roman_cat.fits')
    with asdf.open(file_name) as af:
        roman = {}
        for key in ['data', 'err', 'var_poisson', 'dq']:
            roman[key] = np.array(af['roman'][key])
    config = {
        'type': 'RomanFiles',
        'image_file_name': file_name,
        'cat_file_name': cat_name,
        'stamp_size': 25,
    }
    data = piff.roman.RomanFiles(config).getRawImageData(0)
    good = roman['dq'] == 0
    expected_gain = roman['data'] / roman['var_poisson']
    expected_variance = roman['err']**2 - roman['var_poisson']
    np.testing.assert_array_equal(data.image.array, roman['data'])
    np.testing.assert_allclose(data.gain_image.array[good], expected_gain[good])
    np.testing.assert_allclose(data.weight.array[good], 1 / expected_variance[good], rtol=3.e-6)
    np.testing.assert_array_equal(data.weight.array[~good], 0)

    # Allowing selected DQ bits makes those pixels usable without allowing other flags.
    config['good_dq_bits'] = 4 | 16777216
    data = piff.roman.RomanFiles(config).getRawImageData(0)
    good = roman['dq'] != 8192
    np.testing.assert_allclose(data.weight.array[good], 1 / expected_variance[good], rtol=3.e-6)
    np.testing.assert_array_equal(data.weight.array[~good], 0)

    # A supplied sky image is subtracted from DATA and its photon variance is retained.
    sky_name = os.path.join('output', 'roman_sky.fits')
    sky_image = fitsio.read(sky_name)
    config['sky_file_name'] = sky_name
    data = piff.roman.RomanFiles(config).getRawImageData(0)
    expected_variance += sky_image / expected_gain
    np.testing.assert_allclose(data.image.array, roman['data'] - sky_image)
    np.testing.assert_allclose(data.weight.array[good], 1 / expected_variance[good], rtol=3.e-6)
    np.testing.assert_array_equal(data.weight.array[~good], 0)

    # Check that stars get the right image, wcs, gain, weight.
    stars, wcs, _, _ = piff.Input.process(config)
    assert len(stars) == len(fitsio.read(cat_name))
    assert isinstance(wcs[0], piff.roman.RomanGWCS)
    for star in stars:
        bounds = star.image.bounds
        np.testing.assert_allclose(star.image.array, data.image[bounds].array)
        np.testing.assert_allclose(star.weight.array, data.weight[bounds].array)
        np.testing.assert_allclose(star.data.gain_image.array, data.gain_image[bounds].array)
        assert star.image.wcs == wcs[0]

    # remove_signal_from_weight defaults to true, but can be turned off.
    config.pop('sky_file_name')
    config['remove_signal_from_weight'] = False
    data = piff.roman.RomanFiles(config).getRawImageData(0)
    np.testing.assert_allclose(data.weight.array[good], 1 / roman['err'][good]**2)

    # An explicit WCS overrides the native Roman GWCS.
    config['wcs'] = {'type': 'PixelScale', 'scale': 0.2}
    assert piff.roman.RomanFiles(config).getWCS()[0] == galsim.PixelScale(0.2)

    # Nonpositive science/ERR/photon variance and negative residual variance are masked.
    invalid_file = os.path.join('output', 'roman_invalid.asdf')
    shutil.copyfile(file_name, invalid_file)
    with asdf.open(invalid_file, mode='rw') as af:
        roman = af['roman']
        roman['data'][20, 20] = 0
        roman['err'][21, 20] = 0
        roman['err'][21, 21] = np.sqrt(0.5 * roman['var_poisson'][21, 21])
        roman['var_poisson'][22, 20] = 0
        af.update()
    invalid_config = dict(config, image_file_name=invalid_file, remove_signal_from_weight=True)
    data = piff.roman.RomanFiles(invalid_config).getRawImageData(0)
    assert data.weight.array[20, 20] == 0
    assert data.weight.array[21, 20] == 0
    assert data.weight.array[21, 21] == 0  # Total non-poisson variance is negative.
    assert data.weight.array[22, 20] == 0

    # Overriding the locations of weight, gain, badpix, noise are not currently allowed.
    for option in ['weight_ext', 'gain_ext', 'badpix_ext', 'noise']:
        invalid_config = dict(config)
        invalid_config[option] = 1
        with np.testing.assert_raises(ValueError) as e:
            piff.roman.RomanFiles(invalid_config)
        assert option in str(e.exception)

    # Check invalid ramp_fit.
    invalid_config = dict(config, ramp_fit='other')
    with np.testing.assert_raises(ValueError) as e:
        piff.roman.RomanFiles(invalid_config).getRawImageData(0)
    assert 'ramp_fit' in str(e.exception)

    # good_dq_bits must be a valid uint32 number.
    for value in [-1, 2**32]:
        invalid_config = dict(config, good_dq_bits=value)
        with np.testing.assert_raises(ValueError) as e:
            piff.roman.RomanFiles(invalid_config).getRawImageData(0)
        assert 'good_dq_bits' in str(e.exception)

@timer
def test_roman_uniform():
    """Test the ramp_fit=uniform option."""
    asdf = pytest.importorskip('asdf')

    # The uniform image is computed from data + dumo.
    file_name = os.path.join('output', 'roman_input.asdf')
    cat_name = os.path.join('output', 'roman_cat.fits')
    with asdf.open(file_name) as af:
        roman = {}
        for key in ['data', 'dumo', 'dq']:
            roman[key] = np.array(af['roman'][key])
    config = {
        'type': 'RomanFiles',
        'image_file_name': file_name,
        'cat_file_name': cat_name,
        'ramp_fit': 'uniform',
        'good_dq_bits': 4 | 16777216,
    }
    data = piff.roman.RomanFiles(config).getRawImageData(0)
    good = roman['dq'] != 8192
    np.testing.assert_array_equal(data.image.array, roman['data'] + roman['dumo'])

    # setup saved the correct uniform read noise and gain in roman_truth.npz so we can
    # compare what RomanFiles calculates from the asdf image.
    with np.load(os.path.join('output', 'roman_truth.npz')) as truth:
        uniform_read = truth['uniform_read']
        uniform_gain = truth['uniform_gain']

    # The global approximation recovers gain within 0.5% and variance within 1%.
    np.testing.assert_allclose(data.gain_image.array, uniform_gain, rtol=0.005)
    np.testing.assert_allclose(data.weight.array[good], 1 / uniform_read, rtol=0.01)
    np.testing.assert_array_equal(data.weight.array[~good], 0)

    # remove_signal_from_weight defaults to true, but can be turned off.
    total = piff.roman.RomanFiles(dict(config, remove_signal_from_weight=False)).getRawImageData(0)
    expected_variance = uniform_read + (roman['data'] + roman['dumo']) / uniform_gain
    np.testing.assert_allclose(total.weight.array[good], 1 / expected_variance[good], rtol=0.01)

    # If sky is given, it contributes sky/gain to the uniform variance.
    data = piff.roman.RomanFiles(dict(config, sky=5.0)).getRawImageData(0)
    np.testing.assert_allclose(data.image.array, roman['data'] + roman['dumo'] - 5.0)
    np.testing.assert_allclose(data.weight.array[good],
                               1 / (uniform_read + 5.0 / uniform_gain), rtol=0.01)

    # A sky image contributes its local sky/gain to the uniform variance.
    sky_name = os.path.join('output', 'roman_sky.fits')
    sky_image = fitsio.read(sky_name)
    data = piff.roman.RomanFiles(dict(config, sky_file_name=sky_name)).getRawImageData(0)
    expected_variance = uniform_read + sky_image / uniform_gain
    np.testing.assert_allclose(data.weight.array[good], 1 / expected_variance[good], rtol=0.01)

    # If sky is given per star, the mean catalog sky sets the background variance.
    data = piff.roman.RomanFiles(dict(config, sky_col='sky')).getRawImageData(0)
    np.testing.assert_allclose(data.weight.array[good],
                               1 / (uniform_read + 5.0 / uniform_gain), rtol=0.01)

    # The mask_dumo_spikes options masks isolated spikes, but keeps any connected to a source.
    data = piff.roman.RomanFiles(dict(config, mask_dumo_spikes=True)).getRawImageData(0)
    assert data.weight.array[11, 11] == 0
    assert data.weight.array[12, 12] != 0
    assert data.weight.array[35, 32] != 0

    # smoothing_scale must be positive.
    with np.testing.assert_raises(ValueError) as e:
        piff.roman.RomanFiles(dict(config, smoothing_scale=0)).getRawImageData(0)
    assert 'smoothing_scale must be positive' in str(e.exception)

    # The uniform calculation only uses DQ=0 pixels, even if those are considered "good_dq_bits".
    # So if the whole image is marked as JUMP_DET, the uniform calculation will fail.
    alljump_file = os.path.join('output', 'roman_no_background.asdf')
    shutil.copyfile(file_name, alljump_file)
    with asdf.open(alljump_file, mode='rw') as af:
        af['roman']['dq'][:] = 4
        af.update()
    with np.testing.assert_raises(ValueError) as e:
        piff.roman.RomanFiles(dict(config, image_file_name=alljump_file)).getRawImageData(0)
    assert 'No clean background samples' in str(e.exception)


if __name__ == '__main__':
    test_roman_input()
    test_roman_uniform()
