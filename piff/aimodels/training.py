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
.. module:: training

Training tools for the AIPSF autoencoder.

Note: this module is named ``training`` (not ``train``) so that the
config-driven entry point can be exposed as ``piff.aimodels.train`` without
the imported submodule shadowing the function.

The training data are pickle files containing a dict of star records:

    { star_id: { 'star':     numpy array (N, N), the stamp normalized to sum to 1,
                 'weight':   numpy array (N, N), inverse variance of the normalized
                             stamp, zero for masked pixels (None in older
                             training sets),
                 'starPiff': numpy array (N, N), a reference PSF model prediction
                             at the star position (used as a diagnostic baseline),
                 ... }, ... }

Such files are produced per (visit, detector, band) by the training-sample
collection in the LSST meas_extensions_piff package.  `load_training_data`
accepts either a single merged pickle file or a directory of such files.

This module requires PyTorch, which is an optional dependency of Piff.
"""

import os
import glob
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .models import Conv2dAutoEncoder, save_checkpoint
from ..config import LoggerWrapper


class PSFDataset(Dataset):
    """A torch Dataset of PSF star stamps.

    Each item is a dict with keys 'star' and 'star_piff', both tensors of shape
    (1, N, N), plus 'weight' (same shape) if use_weights is True.

    :param data_dict:   A dict mapping star_id -> star record (see module docstring).
    :param use_weights: Whether to include the per-pixel weight maps (inverse
                        variance of the normalized stamps) in the samples.
                        [default: False]
    :param transform:   An optional callable to apply to each sample. [default: None]
    """
    def __init__(self, data_dict, use_weights=False, transform=None):
        self.ids = list(data_dict.keys())
        self.data = data_dict
        self.use_weights = use_weights
        self.transform = transform
        if use_weights and self.ids:
            first = self.data[self.ids[0]]
            if first.get('weight') is None:
                raise ValueError(
                    "use_weights=True, but the training data has no weight maps. "
                    "(Older training sets stored weight=None; re-generate them "
                    "with a current meas_extensions_piff.)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        entry = self.data[self.ids[idx]]
        star = torch.from_numpy(entry['star']).float().unsqueeze(0)            # [1, H, W]
        star_piff = torch.from_numpy(entry['starPiff']).float().unsqueeze(0)   # [1, H, W]
        sample = {'star': star, 'star_piff': star_piff}
        if self.use_weights:
            sample['weight'] = torch.from_numpy(entry['weight']).float().unsqueeze(0)
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_training_data(path, logger=None):
    """Load a training data dict from a pickle file or a directory of pickle files.

    If path is a directory, all `*.pkl` files in it are loaded and merged into a
    single dict.  The star ids are expected to be unique across files (they
    normally encode visit, detector, and band); duplicates are logged as
    warnings and the last value wins.

    :param path:    A pickle file name, or a directory containing pickle files.
    :param logger:  A logger object for logging progress. [default: None]

    :returns: a dict mapping star_id -> star record.
    """
    logger = LoggerWrapper(logger)

    if os.path.isdir(path):
        file_names = sorted(glob.glob(os.path.join(path, '*.pkl')))
        if len(file_names) == 0:
            raise ValueError("No .pkl files found in directory %s" % path)
        data_dict = {}
        for file_name in file_names:
            with open(file_name, 'rb') as f:
                d = pickle.load(f)
            duplicates = set(d.keys()) & set(data_dict.keys())
            if duplicates:
                logger.warning("Found %d duplicate star ids in %s; keeping the last value.",
                               len(duplicates), file_name)
            data_dict.update(d)
            logger.info("Loaded %d stars from %s", len(d), file_name)
        logger.warning("Loaded %d stars total from %d files in %s",
                       len(data_dict), len(file_names), path)
        return data_dict
    elif os.path.isfile(path):
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        logger.warning("Loaded %d stars from %s", len(data_dict), path)
        return data_dict
    else:
        raise FileNotFoundError("Training data path not found: %s" % path)


def create_dataloaders(data, batch_size=8192, val_fraction=0.1, shuffle=True, seed=None,
                       num_workers=4, use_weights=False, logger=None):
    """Build training and validation DataLoaders from PSF training data.

    :param data:            A training data dict, a pickle file name, or a directory of
                            pickle files (see `load_training_data`).
    :param batch_size:      The batch size. [default: 8192]
    :param val_fraction:    The fraction of the data reserved for validation. [default: 0.1]
    :param shuffle:         Whether to shuffle the training data. [default: True]
    :param seed:            An optional seed for the train/validation split. [default: None]
    :param num_workers:     The number of DataLoader worker processes. [default: 4]
    :param use_weights:     Whether to include the per-pixel weight maps in the batches
                            (required for the weighted chi2 loss). [default: False]
    :param logger:          A logger object for logging progress. [default: None]

    :returns: (train_loader, val_loader)
    """
    logger = LoggerWrapper(logger)

    if isinstance(data, str):
        data = load_training_data(data, logger=logger)

    full_ds = PSFDataset(data, use_weights=use_weights)
    total = len(full_ds)
    n_val = int(total * val_fraction)
    n_train = total - n_val
    logger.warning("Training: %d | Validation: %d", n_train, n_val)

    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=persistent)
    return train_loader, val_loader


def fit_amplitude_background(profile, star, weight, mode='free'):
    """Solve, per star, for the amplitude and constant background in
    star ~ a * profile + b, by weighted least squares.

    Two modes:

    - 'free': a and b are both free parameters; the normal equations for
      chi2 = sum_pixels w * (star - a*profile - b)^2 are::

          [S_pp  S_p] [a]   [S_py]
          [S_p   S_w] [b] = [S_y ]

      with S_w = sum(w), S_p = sum(w*p), S_pp = sum(w*p^2), S_y = sum(w*y),
      S_py = sum(w*p*y).

    - 'normalized': one free parameter.  If the data stamp is the model stamp
      plus a constant background, the stamp sums tie the amplitude to the
      background: sum(star) = a*sum(profile) + N*b with sum(profile) = 1, so
      a = S - N*b with S = sum(star) and N the number of pixels per stamp
      (for the sum-normalized training stamps, S = 1 exactly).  Substituting
      leaves star - S*profile = b*(1 - N*profile), solved by weighted least
      squares on the basis q = 1 - N*profile::

          b = sum(w*(star - S*profile)*q) / sum(w*q^2),   a = S - N*b

    The profile is detached, so (a, b) are constants for a backward pass
    (envelope theorem: exact at the (a, b) optimum).

    :param profile:     Tensor of shape (B, 1, N, N), the model stamps
                        (summing to 1).
    :param star:        Tensor of shape (B, 1, N, N), the data stamps.
    :param weight:      Tensor of shape (B, 1, N, N), the per-pixel weights
                        (inverse variance; zero for masked pixels).
    :param mode:        'free' or 'normalized'. [default: 'free']

    :returns: (a, b), tensors of shape (B,).
    """
    p = profile.detach()
    dims = (1, 2, 3)
    if mode == 'free':
        S_w = weight.sum(dim=dims)
        S_p = (weight * p).sum(dim=dims)
        S_pp = (weight * p * p).sum(dim=dims)
        S_y = (weight * star).sum(dim=dims)
        S_py = (weight * p * star).sum(dim=dims)
        det = (S_pp * S_w - S_p * S_p).clamp(min=1e-30)
        a = (S_w * S_py - S_p * S_y) / det
        b = (S_pp * S_y - S_p * S_py) / det
    elif mode == 'normalized':
        n_pix = p[0].numel()
        S = star.sum(dim=dims)
        q = 1. - n_pix * p
        resid = star - S.view(-1, 1, 1, 1) * p
        b = ((weight * resid * q).sum(dim=dims)
             / (weight * q * q).sum(dim=dims).clamp(min=1e-30))
        a = S - n_pix * b
    else:
        raise ValueError("mode must be 'free' or 'normalized'; got %r" % (mode,))
    return a, b


def train_autoencoder(model, train_loader, val_loader, epochs=10, initial_lr=1e-3,
                      device=None, scheduler_on_plateau=False, use_weights=False,
                      fit_background=False, fit_background_mode='free',
                      scheduler_factor=0.1, scheduler_patience=10,
                      scheduler_threshold=1e-4, scheduler_min_lr=1e-6,
                      checkpoint_file='autoencoder.pth', logger=None):
    """Train an autoencoder on PSF stamps and save the result as a checkpoint file.

    By default the loss is the pixel-level MSE between the autoencoder output and
    its input, scaled by 1e6 for numerical convenience.

    With use_weights=True, the loss is instead the mean per-star reduced chi2,
    using the per-pixel inverse-variance maps from the training data::

        chi2_star = sum_pixels[ w * (model - star)^2 ] / N_good

    where N_good is the number of unmasked (w > 0) pixels of the star.  A model
    that describes the data at the noise level gives a loss around 1, and masked
    pixels are naturally excluded.  (The batches must contain weight maps, i.e.
    the DataLoaders must be built with use_weights=True as well.)

    With fit_background=True, the per-star model is a * psf + b, where psf is
    the decoded stamp and the amplitude a and constant background b are nuisance
    parameters solved analytically per star (a 2x2 weighted linear system) at
    the current network weights, re-evaluated at every step.  The solve is
    detached from the graph: at the (a, b) optimum the gradient with respect to
    the network equals the fixed-(a, b) gradient (envelope theorem).  This
    absorbs local background over/under-subtraction, which the strictly
    positive SpatialSoftmax output cannot represent (an over-subtracted
    background gives negative wing pixels).

    In all cases, the same loss evaluated for the reference PSF model prediction
    ('starPiff') is also tracked as a diagnostic baseline, but does not enter
    the training.  The nuisance fit is NOT applied to the baseline: the
    PixelGrid model is fit per CCD and absorbs local background into its
    pixel grid by construction, so refitting (a, b) on top of it would
    double-count the correction.

    The checkpoint is written with :func:`piff.aimodels.save_checkpoint`, so it can
    be used directly by the AIPSF model.

    :param model:                   A Conv2dAutoEncoder instance to train.
    :param train_loader:            The training DataLoader.
    :param val_loader:              The validation DataLoader.
    :param epochs:                  The number of training epochs. [default: 10]
    :param initial_lr:              The initial learning rate for Adam. [default: 1e-3]
    :param device:                  The torch device to train on.  [default: None, which
                                    means use 'cuda' if available, else 'cpu']
    :param scheduler_on_plateau:    Whether to use a ReduceLROnPlateau scheduler on the
                                    validation loss. [default: False]
    :param use_weights:             Whether to use the weighted (reduced chi2) loss
                                    instead of the scaled MSE. [default: False]
    :param fit_background:          Whether to fit a per-star amplitude and constant
                                    background (model = a * psf + b) as nuisance
                                    parameters in the loss. [default: False]
    :param fit_background_mode:     'free' (a and b both free) or 'normalized'
                                    (one parameter, a = 1 - Npix*b via the stamp-sum
                                    constraint; see `fit_amplitude_background`).
                                    [default: 'free']
    :param scheduler_factor:        ReduceLROnPlateau lr reduction factor. [default: 0.1]
    :param scheduler_patience:      ReduceLROnPlateau patience, in epochs without
                                    sufficient improvement. [default: 10]
    :param scheduler_threshold:     ReduceLROnPlateau relative improvement threshold;
                                    an epoch only counts as an improvement if the
                                    validation loss drops by more than this fraction.
                                    [default: 1e-4]
    :param scheduler_min_lr:        ReduceLROnPlateau lower bound on the lr.
                                    [default: 1e-6]
    :param checkpoint_file:         The output checkpoint file name. [default: 'autoencoder.pth']
    :param logger:                  A logger object for logging progress. [default: None]

    :returns: a dict with the per-step loss histories, with keys 'loss_ae_train',
              'loss_ae_val', 'loss_piff_train', 'loss_piff_val'.
    """
    logger = LoggerWrapper(logger)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.warning("Training on device %s for %d epochs (use_weights=%s, "
                   "fit_background=%s, fit_background_mode=%s)",
                   device, epochs, use_weights, fit_background, fit_background_mode)

    # Scale the MSE loss for numerical convenience: normalized 25x25 stamps have
    # typical pixel values of order 1e-3, so the raw MSE values are tiny.
    # (Not used for the weighted loss, which is naturally of order 1.)
    k = 1.e6

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.MSELoss()

    def with_nuisance(profile, star, w):
        """Return the per-star model a * profile + b, with (a, b) from
        `fit_amplitude_background` at the current network weights.
        """
        if not fit_background:
            return profile
        a, b = fit_amplitude_background(profile, star, w, mode=fit_background_mode)
        return a.view(-1, 1, 1, 1) * profile + b.view(-1, 1, 1, 1)

    def compute_losses(inputs, outputs, target_piff, batch):
        # Note: the nuisance fit is applied to the autoencoder output only.
        # The 'starPiff' baseline is compared as-is (PixelGrid absorbs local
        # background into its model by construction).
        if use_weights:
            w = batch['weight'].to(device)
            n_good = (w > 0).sum(dim=(1, 2, 3)).clamp(min=1)
            model_ae = with_nuisance(outputs, inputs, w)
            loss_ae = ((w * (model_ae - inputs)**2).sum(dim=(1, 2, 3)) / n_good).mean()
            loss_piff = ((w * (target_piff - inputs)**2).sum(dim=(1, 2, 3)) / n_good).mean()
        else:
            ones = torch.ones_like(inputs)
            model_ae = with_nuisance(outputs, inputs, ones)
            loss_ae = criterion(model_ae, inputs) * k
            loss_piff = criterion(target_piff, inputs) * k
        return loss_ae, loss_piff

    if scheduler_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor,
            patience=scheduler_patience, threshold=scheduler_threshold,
            min_lr=scheduler_min_lr)

    history = {
        'loss_ae_train': [],
        'loss_ae_val': [],
        'loss_piff_train': [],
        'loss_piff_val': [],
    }

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        epoch_loss = 0.
        n_train = 0
        for batch in train_loader:
            inputs = batch['star'].to(device)
            target_piff = batch['star_piff'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_ae, loss_piff = compute_losses(inputs, outputs, target_piff, batch)
            loss_ae.backward()
            optimizer.step()

            history['loss_ae_train'].append(loss_ae.item())
            history['loss_piff_train'].append(loss_piff.item())
            epoch_loss += loss_ae.item() * inputs.size(0)
            n_train += inputs.size(0)

        # ---- Validation ----
        model.eval()
        val_loss_ae = 0.
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['star'].to(device)
                target_piff = batch['star_piff'].to(device)

                outputs = model(inputs)
                loss_ae, loss_piff = compute_losses(inputs, outputs, target_piff, batch)

                val_loss_ae += loss_ae.item() * inputs.size(0)
                history['loss_ae_val'].append(loss_ae.item())
                history['loss_piff_val'].append(loss_piff.item())

        val_loss_ae /= len(val_loader.dataset)
        if scheduler_on_plateau:
            scheduler.step(val_loss_ae)

        logger.warning("Epoch %d/%d: train loss = %.6f, val loss = %.6f, lr = %.2e",
                       epoch, epochs, epoch_loss / max(n_train, 1), val_loss_ae,
                       optimizer.param_groups[0]['lr'])

    model.eval()
    save_checkpoint(model, checkpoint_file)
    logger.warning("Wrote checkpoint file %s", checkpoint_file)

    return history


def train(config, logger=None):
    """Train an AIPSF autoencoder as specified by a configuration dict.

    This is the function called by the `trainify` executable.  The config dict
    has the following structure::

        input:
            file_name: training/        # a merged .pkl file OR a directory of .pkl files
            batch_size: 1024            # [default: 8192]
            val_fraction: 0.05          # [default: 0.1]
            seed: 42                    # [default: None]
            num_workers: 4              # [default: 4]
        model:
            type: Conv2dAutoEncoder     # [default and currently only option]
            grid_size: 25               # [default: 25]
            latent_dim: 64              # [default: 64]
            hidden_channels: 16         # [default: 16]
            zero_floor: true            # end the decoder with the ZeroFloor
                                        # projection [default: True]
            latent_norm: true           # end the encoder with an affine-free
                                        # BatchNorm over the latents [default: True]
        training:
            epochs: 40                  # [default: 10]
            initial_lr: 1.e-3           # [default: 1e-3]
            scheduler_on_plateau: true  # [default: False]
            scheduler_factor: 0.1       # lr reduction factor [default: 0.1]
            scheduler_patience: 5       # epochs without improvement before
                                        # reducing the lr [default: 10]
            scheduler_threshold: 1.e-3  # relative improvement threshold for the
                                        # plateau detection [default: 1e-4]
            scheduler_min_lr: 1.e-6     # [default: 1e-6]
            use_weights: false          # weighted (reduced chi2) loss; needs weight
                                        # maps in the training data [default: False]
            fit_background: false       # per-star amplitude + constant background
                                        # nuisance (model = a*psf + b), solved
                                        # analytically per star [default: False]
            fit_background_mode: free   # 'free' (a, b both free) or 'normalized'
                                        # (one parameter, a = 1 - Npix*b)
                                        # [default: free]
            device: cuda                # [default: cuda if available, else cpu]
        output:
            file_name: Conv2dAutoEncoder.pth
            history_file: history.pkl   # optional; the loss history is saved here

    :param config:      The configuration dict that defines how to train the model.
    :param logger:      A logger object for logging progress. [default: None]

    :returns: the loss history dict from `train_autoencoder`.
    """
    from ..config import setup_logger

    if logger is None:
        verbose = config.get('verbose', 1)
        logger = setup_logger(verbose=verbose)

    for key in ['input', 'output']:
        if key not in config:
            raise ValueError("%s field is required in config dict" % key)
    for key in ['file_name']:
        if key not in config['input']:
            raise ValueError("%s field is required in config dict input" % key)
        if key not in config['output']:
            raise ValueError("%s field is required in config dict output" % key)

    input_config = config['input']
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    output_config = config['output']

    model_type = model_config.get('type', 'Conv2dAutoEncoder')
    if model_type != 'Conv2dAutoEncoder':
        raise ValueError("model type %s is not a valid AIPSF model type. "
                         "Only Conv2dAutoEncoder is currently supported." % model_type)

    use_weights = training_config.get('use_weights', False)

    train_loader, val_loader = create_dataloaders(
        input_config['file_name'],
        batch_size=input_config.get('batch_size', 8192),
        val_fraction=input_config.get('val_fraction', 0.1),
        shuffle=input_config.get('shuffle', True),
        seed=input_config.get('seed', None),
        num_workers=input_config.get('num_workers', 4),
        use_weights=use_weights,
        logger=logger)

    model = Conv2dAutoEncoder(
        grid_size=model_config.get('grid_size', 25),
        latent_dim=model_config.get('latent_dim', 64),
        hidden_channels=model_config.get('hidden_channels', 16),
        zero_floor=model_config.get('zero_floor', True),
        latent_norm=model_config.get('latent_norm', True))
    n_params = sum(p.numel() for p in model.parameters())
    logger.warning("Built Conv2dAutoEncoder with grid_size=%d, latent_dim=%d, "
                   "hidden_channels=%d, zero_floor=%s, latent_norm=%s (%d parameters)",
                   model.grid_size, model.latent_dim, model.hidden_channels,
                   model.zero_floor, model.latent_norm, n_params)

    history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=training_config.get('epochs', 10),
        initial_lr=training_config.get('initial_lr', 1e-3),
        device=training_config.get('device', None),
        scheduler_on_plateau=training_config.get('scheduler_on_plateau', False),
        use_weights=use_weights,
        fit_background=training_config.get('fit_background', False),
        fit_background_mode=training_config.get('fit_background_mode', 'free'),
        scheduler_factor=training_config.get('scheduler_factor', 0.1),
        scheduler_patience=training_config.get('scheduler_patience', 10),
        scheduler_threshold=training_config.get('scheduler_threshold', 1e-4),
        scheduler_min_lr=training_config.get('scheduler_min_lr', 1e-6),
        checkpoint_file=output_config['file_name'],
        logger=logger)

    if 'history_file' in output_config:
        with open(output_config['history_file'], 'wb') as f:
            pickle.dump(history, f)
        logger.warning("Wrote loss history to %s", output_config['history_file'])

    return history
