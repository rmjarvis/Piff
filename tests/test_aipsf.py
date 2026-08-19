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

import numpy as np
import galsim
import piff
import os
import pytest

from piff_test_helper import timer

# torch is an optional dependency of Piff.  All these tests need it.
try:
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")

GRID_SIZE = 25


def make_checkpoint(file_name, latent_dim=4, hidden_channels=2, seed=1234):
    """Make a small random-weight autoencoder and save it as a checkpoint file.
    """
    torch.manual_seed(seed)
    net = piff.aimodels.Conv2dAutoEncoder(grid_size=GRID_SIZE, latent_dim=latent_dim,
                                          hidden_channels=hidden_channels)
    net.eval()
    piff.aimodels.save_checkpoint(net, file_name)
    return net


def make_gaussian_star(sigma=1.2, flux=100., du=0.26, fpu=0., fpv=0., nside=GRID_SIZE,
                       noise=0., rng=None):
    """Make a Star instance filled with a Gaussian profile.
    """
    g = galsim.Gaussian(sigma=sigma, flux=flux)
    if noise == 0.:
        var = 1.e-6
    else:
        var = noise**2
    weight = galsim.Image(nside, nside, dtype=float, init_value=1./var, scale=du)
    star = piff.Star.makeTarget(x=nside/2, y=nside/2, u=fpu, v=fpv, scale=du,
                                stamp_size=nside, weight=weight)
    g.drawImage(star.image, method='no_pixel', use_true_center=False)
    if noise != 0.:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


@requires_torch
@timer
def test_checkpoint_roundtrip():
    """Test that save_checkpoint/load_autoencoder preserve the network exactly.
    """
    os.makedirs('output', exist_ok=True)
    file_name = os.path.join('output', 'test_aipsf_ckpt.pth')

    latent_dim = 4
    hidden_channels = 2
    net = make_checkpoint(file_name, latent_dim=latent_dim, hidden_channels=hidden_channels)

    net2 = piff.aimodels.load_autoencoder(file_name)
    assert net2.grid_size == GRID_SIZE
    assert net2.latent_dim == latent_dim
    assert net2.hidden_channels == hidden_channels

    # The decoder outputs should be identical for the same latent vector.
    torch.manual_seed(42)
    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        image1 = net.decoder(z).numpy()
        image2 = net2.decoder(z).numpy()
    np.testing.assert_array_equal(image1, image2)

    # And the encoder outputs should be identical for the same stamp.
    stamp = torch.rand(1, 1, GRID_SIZE, GRID_SIZE)
    stamp /= stamp.sum()
    with torch.no_grad():
        z1 = net.encoder(stamp).numpy()
        z2 = net2.encoder(stamp).numpy()
    np.testing.assert_array_equal(z1, z2)

    # The decoder output is normalized (SpatialSoftmax + ZeroFloor), so it
    # sums to 1 and has an exactly zero floor.
    np.testing.assert_allclose(np.sum(image1), 1., rtol=1.e-6)
    np.testing.assert_allclose(np.min(image1), 0., atol=1.e-12)
    assert net2.zero_floor is True
    assert net2.latent_norm is True

    # The zero_floor=False / latent_norm=False variant (the pre-flag
    # architecture): strictly positive floor (softmax only).
    file_name2 = os.path.join('output', 'test_aipsf_ckpt_nofloor.pth')
    torch.manual_seed(1234)
    net3 = piff.aimodels.Conv2dAutoEncoder(grid_size=GRID_SIZE, latent_dim=latent_dim,
                                           hidden_channels=hidden_channels,
                                           zero_floor=False, latent_norm=False)
    net3.eval()
    piff.aimodels.save_checkpoint(net3, file_name2)
    net4 = piff.aimodels.load_autoencoder(file_name2)
    assert net4.zero_floor is False
    assert net4.latent_norm is False
    with torch.no_grad():
        image3 = net4.decoder(z).numpy()
    assert np.min(image3) > 0.
    np.testing.assert_allclose(np.sum(image3), 1., rtol=1.e-6)

    # A checkpoint without the 'zero_floor'/'latent_norm' keys (written before
    # the flags existed) loads as the pre-flag architecture.
    legacy = torch.load(file_name2, map_location='cpu')
    del legacy['zero_floor']
    del legacy['latent_norm']
    file_name3 = os.path.join('output', 'test_aipsf_ckpt_legacy.pth')
    torch.save(legacy, file_name3)
    net5 = piff.aimodels.load_autoencoder(file_name3)
    assert net5.zero_floor is False
    assert net5.latent_norm is False


@requires_torch
@timer
def test_grid_sizes():
    """Test that the autoencoder geometry works for any odd grid_size.

    The two stride-2 encoder stages and the matching transposed convolutions
    of the decoder must map grid_size -> (grid_size-1)/2 -> ceil((grid_size-1)/4)
    and exactly back, for both parities of the intermediate size.
    """
    os.makedirs('output', exist_ok=True)

    for grid_size in [5, 7, 15, 17, 19, 21, 25, 31]:
        torch.manual_seed(1234)
        net = piff.aimodels.Conv2dAutoEncoder(grid_size=grid_size, latent_dim=4,
                                              hidden_channels=2)
        net.eval()

        # Forward pass roundtrips the stamp shape through the bottleneck.
        stamp = torch.rand(1, 1, grid_size, grid_size)
        stamp /= stamp.sum()
        with torch.no_grad():
            z = net.encoder(stamp)
            recon = net.decoder(z)
        assert z.shape == (1, 4)
        assert recon.shape == (1, 1, grid_size, grid_size)
        np.testing.assert_allclose(recon.numpy().sum(), 1., rtol=1.e-6)
        # Non-negative with an exactly zero floor (ZeroFloor projection).
        assert recon.min().item() >= 0.
        np.testing.assert_allclose(recon.numpy().min(), 0., atol=1.e-12)

    # And the full AIPSF path works with a non-default grid size.
    grid_size = 19
    file_name = os.path.join('output', 'test_aipsf_grid19.pth')
    torch.manual_seed(1234)
    net = piff.aimodels.Conv2dAutoEncoder(grid_size=grid_size, latent_dim=4,
                                          hidden_channels=2)
    net.eval()
    piff.aimodels.save_checkpoint(net, file_name)

    mod = piff.AIPSF(scale=0.26, model_file=file_name)
    assert mod.grid_size == grid_size
    star = make_gaussian_star(nside=grid_size)
    star = mod.initialize(star)
    star = mod.fit(star, draw_method='no_pixel')
    assert star.fit.params.shape == (4,)
    assert np.isfinite(star.fit.chisq)
    prof = mod.getProfile(star.fit.params)
    np.testing.assert_allclose(prof.flux, 1., rtol=1.e-6)


@requires_torch
@timer
def test_aipsf_model():
    """Test the basic AIPSF Model API: initialize, fit, getProfile, draw.
    """
    os.makedirs('output', exist_ok=True)
    file_name = os.path.join('output', 'test_aipsf_model.pth')

    latent_dim = 4
    make_checkpoint(file_name, latent_dim=latent_dim)

    du = 0.26
    mod = piff.AIPSF(scale=du, model_file=file_name, device='cpu')
    assert mod.grid_size == GRID_SIZE
    assert mod.latent_dim == latent_dim

    star = make_gaussian_star(du=du)
    input_flux = np.sum(star.image.array)

    star = mod.initialize(star)
    assert star.fit.params.shape == (latent_dim,)
    assert np.all(np.isfinite(star.fit.params))
    np.testing.assert_allclose(star.fit.flux, input_flux, rtol=1.e-8)

    # initialize stores the measured latents as scalar star properties, equal
    # to the initial fit params.  (This covers reserve stars, which only go
    # through initialize, never fit.)
    zmeas = np.array([star.data.properties['aipsf_zmeas_%d' % i] for i in range(latent_dim)])
    np.testing.assert_array_equal(zmeas, star.fit.params)

    star = mod.fit(star, draw_method='no_pixel')
    assert star.fit.params.shape == (latent_dim,)
    assert np.isfinite(star.fit.chisq)
    assert star.fit.chisq >= 0.
    assert star.fit.dof == GRID_SIZE**2 - latent_dim

    # fit() stores the per-star amplitude/background nuisance parameters.
    a = star.data.properties['aipsf_a']
    b = star.data.properties['aipsf_b']
    assert np.isfinite(a)
    assert np.isfinite(b)
    # They solve the weighted normal equations: the residual of a*psf + b is
    # orthogonal to the psf and to a constant.
    drawn = mod.draw(star)
    psf_unit = drawn.image.array.ravel() / star.fit.flux
    data, weight, _, _ = star.data.getDataVector()
    resid = data - a*psf_unit - b
    scale = np.sqrt(np.sum(weight * data**2) * np.sum(weight * psf_unit**2))
    assert abs(np.sum(weight * resid * psf_unit)) < 1.e-8 * scale
    assert abs(np.sum(weight * resid)) < 1.e-8 * np.sqrt(np.sum(weight * data**2) * np.sum(weight))

    # The encoder is deterministic, so fit after initialize gives the same params.
    star2 = mod.initialize(star)
    np.testing.assert_array_equal(star.fit.params, star2.fit.params)

    # getProfile returns a unit-flux GSObject.
    prof = mod.getProfile(star.fit.params)
    assert isinstance(prof, galsim.GSObject)
    np.testing.assert_allclose(prof.flux, 1., rtol=1.e-6)

    # draw uses the same profile, scaled by the fit flux.
    star3 = mod.draw(star)
    image = star.image.copy()
    prof.shift(star.fit.center).withFlux(star.fit.flux).drawImage(
        image, method='no_pixel', center=star.image_pos)
    np.testing.assert_allclose(star3.image.array, image.array, rtol=1.e-6)

    # background_fit_mode='normalized': one-parameter constrained fit, with
    # the amplitude tied to the background by the stamp sum.
    mod2 = piff.AIPSF(scale=du, model_file=file_name, background_fit_mode='normalized')
    star_n = mod2.initialize(make_gaussian_star(du=du))
    star_n = mod2.fit(star_n, draw_method='no_pixel')
    a2 = star_n.data.properties['aipsf_a']
    b2 = star_n.data.properties['aipsf_b']
    data_n, weight_n, _, _ = star_n.data.getDataVector()
    np.testing.assert_allclose(a2 + data_n.size*b2, np.sum(data_n), rtol=1.e-6)
    # The constrained residual is orthogonal to the basis q = 1 - N*psf.
    drawn_n = mod2.draw(star_n)
    psf_unit_n = drawn_n.image.array.ravel() / star_n.fit.flux
    q = 1. - data_n.size*psf_unit_n
    resid_n = data_n - a2*psf_unit_n - b2
    scale_q = np.sqrt(np.sum(weight_n*data_n**2) * np.sum(weight_n*q**2))
    assert abs(np.sum(weight_n*resid_n*q)) < 1.e-8 * scale_q


@requires_torch
@timer
def test_single_image():
    """Test a full fit through the config-driven path: AIPSF model + Polynomial interp,
    followed by a write/read round trip.
    """
    os.makedirs('output', exist_ok=True)
    file_name = os.path.join('output', 'test_aipsf_single.pth')
    psf_file = os.path.join('output', 'test_aipsf_psf.fits')

    latent_dim = 4
    make_checkpoint(file_name, latent_dim=latent_dim)

    du = 0.26
    rng = galsim.BaseDeviate(1234)
    np_rng = np.random.RandomState(1234)

    # A grid of stars with slowly varying sigma across the field.
    stars = []
    for fpu in np.linspace(-1., 1., 3):
        for fpv in np.linspace(-1., 1., 3):
            sigma = 1.2 + 0.1*fpu - 0.05*fpv
            flux = 100. + 100*np_rng.rand()
            stars.append(make_gaussian_star(sigma=sigma, flux=flux, du=du, fpu=fpu, fpv=fpv,
                                            noise=0.1, rng=rng))

    config = {
        'type': 'Simple',
        'model': {
            'type': 'AIPSF',
            'scale': du,
            'model_file': file_name,
        },
        'interp': {
            'type': 'Polynomial',
            'order': 1,
        },
        'max_iter': 3,
    }
    psf = piff.PSF.process(config)
    assert type(psf.model) is piff.AIPSF
    assert type(psf.interp) is piff.Polynomial

    psf.set_context(wcs={0: galsim.PixelScale(du)})
    psf.fit(stars, logger=None)

    # After the fit, star.fit.params holds the *interpolated* latents (the
    # interpolation step replaces the measured values), while the measured
    # (encoder) latents are preserved as the 'aipsf_zmeas_{i}' properties.
    for s in psf.stars:
        zmeas = np.array([s.data.properties['aipsf_zmeas_%d' % i] for i in range(latent_dim)])
        zenc, _ = psf.model._encode(s)
        np.testing.assert_array_equal(zmeas, zenc)
        # With noisy stars and a field-varying PSF, the order-1 polynomial
        # cannot pass exactly through the measured latents of every star.
        assert not np.array_equal(zmeas, s.fit.params)

    # Draw the PSF at a new location.
    target = piff.Star.makeTarget(x=GRID_SIZE/2, y=GRID_SIZE/2, u=0.3, v=-0.2, scale=du,
                                  stamp_size=GRID_SIZE)
    test_star = psf.drawStar(target)
    assert np.all(np.isfinite(test_star.image.array))
    assert test_star.fit.params.shape == (latent_dim,)

    # Round trip through a file.
    # Note: the checkpoint file is not embedded in the output file, so reading the
    # PSF back requires the checkpoint file to still exist at the same path.
    psf.write(psf_file)
    psf2 = piff.read(psf_file)
    assert type(psf2.model) is piff.AIPSF
    assert type(psf2.interp) is piff.Polynomial
    assert psf2.model.grid_size == GRID_SIZE
    assert psf2.model.latent_dim == latent_dim

    test_star2 = psf2.drawStar(target)
    np.testing.assert_allclose(test_star2.fit.params, test_star.fit.params, rtol=1.e-6)
    np.testing.assert_allclose(test_star2.image.array, test_star.image.array, rtol=1.e-6)
    # The background_fit_mode kwarg survives the write/read roundtrip.
    assert psf2.model.background_fit_mode == 'free'


@requires_torch
@timer
def test_errors():
    """Test that invalid inputs raise appropriate errors.
    """
    os.makedirs('output', exist_ok=True)
    file_name = os.path.join('output', 'test_aipsf_err.pth')
    make_checkpoint(file_name)

    # model_file is required.
    with np.testing.assert_raises(ValueError):
        piff.AIPSF(scale=0.26, model_file=None)

    # Missing file.
    with np.testing.assert_raises(FileNotFoundError):
        piff.AIPSF(scale=0.26, model_file='no_such_file.pth')

    # A checkpoint without the architecture metadata is invalid.
    torch.manual_seed(1234)
    net = piff.aimodels.Conv2dAutoEncoder(grid_size=GRID_SIZE, latent_dim=4, hidden_channels=2)
    bad_file = os.path.join('output', 'test_aipsf_bad1.pth')
    torch.save({'model_state_dict': net.state_dict()}, bad_file)
    with np.testing.assert_raises(ValueError):
        piff.AIPSF(scale=0.26, model_file=bad_file)

    # A raw state_dict is also invalid.
    bad_file2 = os.path.join('output', 'test_aipsf_bad2.pth')
    torch.save(net.state_dict(), bad_file2)
    with np.testing.assert_raises(ValueError):
        piff.AIPSF(scale=0.26, model_file=bad_file2)

    # Only odd grid sizes >= 5 are supported.
    with np.testing.assert_raises(ValueError):
        piff.aimodels.Conv2dAutoEncoder(grid_size=24, latent_dim=4, hidden_channels=2)
    with np.testing.assert_raises(ValueError):
        piff.aimodels.Conv2dAutoEncoder(grid_size=3, latent_dim=4, hidden_channels=2)

    # Invalid background_fit_mode.
    with np.testing.assert_raises(ValueError):
        piff.AIPSF(scale=0.26, model_file=file_name, background_fit_mode='banana')

    mod = piff.AIPSF(scale=0.26, model_file=file_name)

    # Wrong stamp size.
    star = make_gaussian_star(nside=21)
    with np.testing.assert_raises(ValueError):
        mod.initialize(star)

    # Non-finite pixel values.
    star = make_gaussian_star()
    star.image.array[12, 12] = np.nan
    with np.testing.assert_raises(ValueError):
        mod.initialize(star)

    # Non-positive total flux.
    star = make_gaussian_star()
    star.image.array[:, :] = 0.
    with np.testing.assert_raises(ValueError):
        mod.initialize(star)


if __name__ == '__main__':
    if torch is None:
        print('torch is not installed.  Skipping all aipsf tests.')
    else:
        test_checkpoint_roundtrip()
        test_grid_sizes()
        test_aipsf_model()
        test_single_image()
        test_errors()
