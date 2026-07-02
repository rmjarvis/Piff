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
import shutil
import pickle
import pytest

from piff_test_helper import timer

# torch is an optional dependency of Piff.  All these tests need it.
try:
    import torch
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")

GRID_SIZE = 25


def make_training_dict(nstars=100, noise=0.03, seed=1234):
    """Make a synthetic training data dict of noisy Gaussian stamps.

    Follows the schema of the training pickles: each record has a 'star' stamp
    normalized to sum to 1 and a 'starPiff' reference stamp.
    """
    np_rng = np.random.RandomState(seed)
    data = {}
    for i in range(nstars):
        sigma = 1.0 + 0.5*np_rng.rand()
        image = galsim.Image(GRID_SIZE, GRID_SIZE, scale=0.26)
        galsim.Gaussian(sigma=sigma, flux=1.).drawImage(image, method='no_pixel',
                                                        use_true_center=False)
        clean = image.array.copy()
        clean /= np.sum(clean)
        noisy = clean + noise*np.std(clean)*np_rng.randn(GRID_SIZE, GRID_SIZE)
        noisy /= np.sum(noisy)
        data['12345_%d_r_%d' % (i % 3, i)] = {
            'star': noisy.astype(np.float32),
            'starPiff': clean.astype(np.float32),
        }
    return data


def write_training_files(data, dir_name):
    """Write a training dict both as one merged pickle and split across 3 pickles
    in a directory (as the per-detector collection would).

    :returns: (merged_file_name, split_dir_name)
    """
    os.makedirs(dir_name, exist_ok=True)
    merged_file = os.path.join(dir_name, 'merged.pkl')
    with open(merged_file, 'wb') as f:
        pickle.dump(data, f)

    split_dir = os.path.join(dir_name, 'split')
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)
    keys = list(data.keys())
    for det in range(3):
        subset = {k: data[k] for i, k in enumerate(keys) if i % 3 == det}
        with open(os.path.join(split_dir, '12345_%d_r.pkl' % det), 'wb') as f:
            pickle.dump(subset, f)
    return merged_file, split_dir


@requires_torch
@timer
def test_load_training_data():
    """Test that loading a merged file and a directory of split files are equivalent.
    """
    os.makedirs('output', exist_ok=True)
    data = make_training_dict()
    merged_file, split_dir = write_training_files(data, os.path.join('output', 'aipsf_train'))

    data1 = piff.aimodels.load_training_data(merged_file)
    data2 = piff.aimodels.load_training_data(split_dir)

    assert set(data1.keys()) == set(data.keys())
    assert set(data2.keys()) == set(data.keys())
    for k in data:
        np.testing.assert_array_equal(data1[k]['star'], data2[k]['star'])
        np.testing.assert_array_equal(data1[k]['starPiff'], data2[k]['starPiff'])

    # Errors for bad paths.
    with np.testing.assert_raises(FileNotFoundError):
        piff.aimodels.load_training_data('no_such_path')
    empty_dir = os.path.join('output', 'aipsf_train', 'empty')
    os.makedirs(empty_dir, exist_ok=True)
    with np.testing.assert_raises(ValueError):
        piff.aimodels.load_training_data(empty_dir)


@requires_torch
@timer
def test_train_api():
    """Test the python training API: dataloaders + train_autoencoder + AIPSF loading.
    """
    os.makedirs('output', exist_ok=True)
    data = make_training_dict()
    checkpoint_file = os.path.join('output', 'test_aipsf_trained.pth')

    torch.manual_seed(1234)
    train_loader, val_loader = piff.aimodels.create_dataloaders(
        data, batch_size=32, val_fraction=0.2, seed=42, num_workers=0)
    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20

    model = piff.aimodels.Conv2dAutoEncoder(grid_size=GRID_SIZE, latent_dim=4,
                                            hidden_channels=2)
    history = piff.aimodels.train_autoencoder(
        model, train_loader, val_loader, epochs=5, initial_lr=1e-3, device='cpu',
        checkpoint_file=checkpoint_file)

    for key in ['loss_ae_train', 'loss_ae_val', 'loss_piff_train', 'loss_piff_val']:
        assert len(history[key]) > 0
        assert np.all(np.isfinite(history[key]))

    # The training loss should decrease from the first to the last epoch.
    steps_per_epoch = len(history['loss_ae_train']) // 5
    first_epoch = np.mean(history['loss_ae_train'][:steps_per_epoch])
    last_epoch = np.mean(history['loss_ae_train'][-steps_per_epoch:])
    print('first epoch loss = ', first_epoch, ', last epoch loss = ', last_epoch)
    assert last_epoch < first_epoch

    # The checkpoint is loadable by the low-level loader and by AIPSF.
    net = piff.aimodels.load_autoencoder(checkpoint_file)
    assert net.grid_size == GRID_SIZE
    assert net.latent_dim == 4
    assert net.hidden_channels == 2

    mod = piff.AIPSF(scale=0.26, model_file=checkpoint_file)
    assert mod.latent_dim == 4

    # Regression check: after the training submodule has been imported (by the
    # attribute accesses above), piff.aimodels.train must still resolve to the
    # config-driven function, not to a submodule shadowing it.
    assert callable(piff.aimodels.train)


@requires_torch
@timer
def test_train_config():
    """Test the config-driven train() entry point (what trainify runs), then use the
    resulting checkpoint in a full AIPSF fit.
    """
    os.makedirs('output', exist_ok=True)
    data = make_training_dict()
    merged_file, split_dir = write_training_files(data, os.path.join('output', 'aipsf_train'))
    checkpoint_file = os.path.join('output', 'test_aipsf_config.pth')
    history_file = os.path.join('output', 'test_aipsf_history.pkl')

    torch.manual_seed(1234)
    config = {
        'input': {
            'file_name': split_dir,
            'batch_size': 32,
            'val_fraction': 0.2,
            'seed': 42,
            'num_workers': 0,
        },
        'model': {
            'type': 'Conv2dAutoEncoder',
            'grid_size': GRID_SIZE,
            'latent_dim': 4,
            'hidden_channels': 2,
        },
        'training': {
            'epochs': 2,
            'initial_lr': 1.e-3,
            'device': 'cpu',
        },
        'output': {
            'file_name': checkpoint_file,
            'history_file': history_file,
        },
        'verbose': 0,
    }
    history = piff.aimodels.train(config)
    assert os.path.exists(checkpoint_file)
    assert os.path.exists(history_file)
    with open(history_file, 'rb') as f:
        history2 = pickle.load(f)
    np.testing.assert_array_equal(history['loss_ae_train'], history2['loss_ae_train'])

    # Config validation errors.
    with np.testing.assert_raises(ValueError):
        piff.aimodels.train({'input': {'file_name': split_dir}, 'verbose': 0})
    with np.testing.assert_raises(ValueError):
        piff.aimodels.train({'input': {'file_name': split_dir},
                             'output': {'file_name': checkpoint_file},
                             'model': {'type': 'SomeOtherNet'}, 'verbose': 0})

    # Use the trained checkpoint in a full AIPSF + Polynomial fit.
    psf_config = {
        'type': 'Simple',
        'model': {'type': 'AIPSF', 'scale': 0.26, 'model_file': checkpoint_file},
        'interp': {'type': 'Polynomial', 'order': 1},
        'max_iter': 3,
    }
    psf = piff.PSF.process(psf_config)

    rng = galsim.BaseDeviate(1234)
    stars = []
    for fpu in np.linspace(-1., 1., 3):
        for fpv in np.linspace(-1., 1., 3):
            g = galsim.Gaussian(sigma=1.2, flux=100.)
            weight = galsim.Image(GRID_SIZE, GRID_SIZE, dtype=float, init_value=100., scale=0.26)
            star = piff.Star.makeTarget(x=GRID_SIZE/2, y=GRID_SIZE/2, u=fpu, v=fpv, scale=0.26,
                                        stamp_size=GRID_SIZE, weight=weight)
            g.drawImage(star.image, method='no_pixel', use_true_center=False)
            stars.append(star)

    psf.set_context(wcs={0: galsim.PixelScale(0.26)})
    psf.fit(stars, logger=None)

    target = piff.Star.makeTarget(x=GRID_SIZE/2, y=GRID_SIZE/2, u=0.1, v=0.2, scale=0.26,
                                  stamp_size=GRID_SIZE)
    test_star = psf.drawStar(target)
    assert np.all(np.isfinite(test_star.image.array))
    # The drawn PSF has the decoder normalization, so it should sum close to the fit flux.
    assert test_star.image.array.sum() > 0.


if __name__ == '__main__':
    if torch is None:
        print('torch is not installed.  Skipping all aipsf training tests.')
    else:
        test_load_training_data()
        test_train_api()
        test_train_config()
