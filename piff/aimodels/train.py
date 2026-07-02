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
.. module:: train

Training tools for the AIPSF autoencoder.

The training data are pickle files containing a dict of star records:

    { star_id: { 'star':     numpy array (N, N), the stamp normalized to sum to 1,
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
    (1, N, N).

    :param data_dict:   A dict mapping star_id -> star record (see module docstring).
    :param transform:   An optional callable to apply to each sample. [default: None]
    """
    def __init__(self, data_dict, transform=None):
        self.ids = list(data_dict.keys())
        self.data = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        entry = self.data[self.ids[idx]]
        star = torch.from_numpy(entry['star']).float().unsqueeze(0)            # [1, H, W]
        star_piff = torch.from_numpy(entry['starPiff']).float().unsqueeze(0)   # [1, H, W]
        sample = {'star': star, 'star_piff': star_piff}
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
                       num_workers=4, logger=None):
    """Build training and validation DataLoaders from PSF training data.

    :param data:            A training data dict, a pickle file name, or a directory of
                            pickle files (see `load_training_data`).
    :param batch_size:      The batch size. [default: 8192]
    :param val_fraction:    The fraction of the data reserved for validation. [default: 0.1]
    :param shuffle:         Whether to shuffle the training data. [default: True]
    :param seed:            An optional seed for the train/validation split. [default: None]
    :param num_workers:     The number of DataLoader worker processes. [default: 4]
    :param logger:          A logger object for logging progress. [default: None]

    :returns: (train_loader, val_loader)
    """
    logger = LoggerWrapper(logger)

    if isinstance(data, str):
        data = load_training_data(data, logger=logger)

    full_ds = PSFDataset(data)
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


def train_autoencoder(model, train_loader, val_loader, epochs=10, initial_lr=1e-3,
                      device=None, scheduler_on_plateau=False,
                      checkpoint_file='autoencoder.pth', logger=None):
    """Train an autoencoder on PSF stamps and save the result as a checkpoint file.

    The loss is the pixel-level MSE between the autoencoder output and its input,
    scaled by 1e6 for numerical convenience.  The MSE between the reference PSF
    model prediction ('starPiff') and the star is also tracked as a diagnostic
    baseline, but does not enter the training.

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
                                    validation loss (factor=0.1, patience=10, min_lr=1e-6).
                                    [default: False]
    :param checkpoint_file:         The output checkpoint file name. [default: 'autoencoder.pth']
    :param logger:                  A logger object for logging progress. [default: None]

    :returns: a dict with the per-step loss histories, with keys 'loss_ae_train',
              'loss_ae_val', 'loss_piff_train', 'loss_piff_val'.
    """
    logger = LoggerWrapper(logger)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.warning("Training on device %s for %d epochs", device, epochs)

    # Scale the MSE loss for numerical convenience: normalized 25x25 stamps have
    # typical pixel values of order 1e-3, so the raw MSE values are tiny.
    k = 1.e6

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.MSELoss()

    if scheduler_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

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
            loss_ae = criterion(outputs, inputs) * k
            loss_piff = criterion(target_piff, inputs) * k
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
                loss_ae = criterion(outputs, inputs) * k
                loss_piff = criterion(target_piff, inputs) * k

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
        training:
            epochs: 40                  # [default: 10]
            initial_lr: 1.e-3           # [default: 1e-3]
            scheduler_on_plateau: true  # [default: False]
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

    train_loader, val_loader = create_dataloaders(
        input_config['file_name'],
        batch_size=input_config.get('batch_size', 8192),
        val_fraction=input_config.get('val_fraction', 0.1),
        shuffle=input_config.get('shuffle', True),
        seed=input_config.get('seed', None),
        num_workers=input_config.get('num_workers', 4),
        logger=logger)

    model = Conv2dAutoEncoder(
        grid_size=model_config.get('grid_size', 25),
        latent_dim=model_config.get('latent_dim', 64),
        hidden_channels=model_config.get('hidden_channels', 16))
    n_params = sum(p.numel() for p in model.parameters())
    logger.warning("Built Conv2dAutoEncoder with grid_size=%d, latent_dim=%d, "
                   "hidden_channels=%d (%d parameters)",
                   model.grid_size, model.latent_dim, model.hidden_channels, n_params)

    history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=training_config.get('epochs', 10),
        initial_lr=training_config.get('initial_lr', 1e-3),
        device=training_config.get('device', None),
        scheduler_on_plateau=training_config.get('scheduler_on_plateau', False),
        checkpoint_file=output_config['file_name'],
        logger=logger)

    if 'history_file' in output_config:
        with open(output_config['history_file'], 'wb') as f:
            pickle.dump(history, f)
        logger.warning("Wrote loss history to %s", output_config['history_file'])

    return history
