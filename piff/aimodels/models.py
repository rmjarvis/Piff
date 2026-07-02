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
.. module:: models

Neural network architectures used by the AIPSF model.

This module requires PyTorch.  It is only imported when the AIPSF model or the
training tools are actually used, so torch is an optional dependency of Piff.
"""

import torch
import torch.nn as nn

from .._version import __version__


class ResidualBlockConv(nn.Module):
    """A residual block with two convolutional layers.

    Maintains the spatial dimensions (padding=1 for kernel_size=3).

    :param channels:    The number of input (and output) channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class SpatialSoftmax(nn.Module):
    """Apply a softmax over the spatial dimensions of the input.

    Assumes input is (Batch, Channels, Height, Width).  Flattens the spatial
    dimensions, applies a softmax, and reshapes back, so each output stamp is
    positive and sums to 1.  This enforces the PSF normalization.
    """
    def forward(self, x):
        b, c, h, w = x.shape
        # If channels > 1, this softmaxes over all channels*pixels.
        # Since the output is 1 channel, it is just pixels.
        x = x.view(b, -1)
        x = nn.functional.softmax(x, dim=1)
        x = x.view(b, c, h, w)
        return x


class Conv2dAutoEncoder(nn.Module):
    """A convolutional autoencoder for PSF stamps.

    The encoder maps a flux-normalized PSF stamp of shape (grid_size, grid_size)
    to a latent vector of dimension latent_dim.  The decoder maps a latent
    vector back to a stamp, normalized to sum to 1 by a final SpatialSoftmax.

    The two stride-2 stages of the encoder downsample the stamp as
    grid_size -> (grid_size-1)/2 -> ceil((grid_size-1)/4) (e.g. 25 -> 12 -> 6),
    and the decoder inverts this exactly, with the output_padding of the first
    transposed convolution chosen to compensate for the pixel lost to the
    floor division in the encoder.  This requires grid_size to be odd.

    :param grid_size:       The stamp size, i.e. the input and output images are
                            (grid_size, grid_size).  Must be an odd integer >= 5.
                            [default: 25]
    :param latent_dim:      The dimension of the latent space. [default: 3]
    :param hidden_channels: The number of channels after the first convolution.
                            Subsequent encoder stages use 2x and 4x this number.
                            [default: 32]
    """
    def __init__(self, grid_size=25, latent_dim=3, hidden_channels=32):
        super().__init__()

        if grid_size % 2 != 1 or grid_size < 5:
            raise ValueError("Conv2dAutoEncoder requires an odd grid_size >= 5; "
                             "got grid_size=%s" % grid_size)

        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Spatial sizes after the two stride-2 downsampling stages:
        #   down1: Conv2d(k=3, s=2, p=0):  n -> (n-1)/2      (n odd)
        #   down2: Conv2d(k=3, s=2, p=1):  n -> ceil(n/2)
        # e.g. grid_size=25: 25 -> 12 -> 6.
        down1_size = (grid_size - 1) // 2
        down2_size = (down1_size + 1) // 2

        # The decoder transposed convolutions must land exactly back on
        # (down1_size, grid_size):
        #   up1: ConvTranspose2d(k=3, s=2, p=1, output_padding=op):  n -> 2n - 1 + op
        #   up2: ConvTranspose2d(k=3, s=2, p=0, output_padding=0):   n -> 2n + 1
        # so up1 needs output_padding = down1_size - 2*down2_size + 1, which is
        # 1 when down1_size is even and 0 when it is odd, and up2 recovers any
        # odd grid_size with no correction.
        up1_output_padding = down1_size - 2 * down2_size + 1
        assert up1_output_padding in (0, 1)
        assert 2 * down1_size + 1 == grid_size

        # --- Encoder ---
        # 1. Conv -> Hidden (grid_size x grid_size)
        # 2. Downsample -> Hidden*2 (down1_size x down1_size)
        # 3. Downsample -> Hidden*4 (down2_size x down2_size)
        # 4. Flatten -> Linear -> Latent

        # Flatten size calculation:
        # down2_size x down2_size spatial * (hidden_channels*4) channels
        flatten_dim = (hidden_channels * 4) * down2_size * down2_size

        self.encoder = nn.Sequential(
            # Input: (B, 1, grid_size, grid_size)
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels),

            # Downsample 1: grid_size -> down1_size (25 -> 12)
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*2),

            # Downsample 2: down1_size -> down2_size (12 -> 6)
            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*4),

            # Flatten and Linear
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim)
        )

        # --- Decoder ---
        # 1. Linear -> Flattened size
        # 2. Unflatten -> (Hidden*4, down2_size, down2_size)
        # 3. Upsample -> Hidden*2 (down1_size x down1_size)
        # 4. Upsample -> Hidden (grid_size x grid_size)
        # 5. Output Conv -> 1 Channel
        # 6. Spatial Softmax

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim),
            nn.Unflatten(1, (hidden_channels*4, down2_size, down2_size)),

            # Upsample 1: down2_size -> down1_size (6 -> 12)
            nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, kernel_size=3, stride=2,
                               padding=1, output_padding=up1_output_padding, bias=False),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*2),

            # Upsample 2: down1_size -> grid_size (12 -> 25)
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=3, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels),

            # Final reconstruction
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            SpatialSoftmax()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def save_checkpoint(model, file_name):
    """Save a trained autoencoder to a checkpoint file.

    The checkpoint stores the network weights along with the hyperparameters
    needed to reconstruct the architecture, so `load_autoencoder` (and hence
    the AIPSF model) can rebuild the network without any external information.

    :param model:       A Conv2dAutoEncoder instance.
    :param file_name:   The file name to write the checkpoint to (conventionally .pth).
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'grid_size': model.grid_size,
        'latent_dim': model.latent_dim,
        'hidden_channels': model.hidden_channels,
        'model_type': 'Conv2dAutoEncoder',
        'piff_version': __version__,
    }, file_name)


def load_autoencoder(file_name, device='cpu', logger=None):
    """Load a trained autoencoder from a checkpoint file written by `save_checkpoint`.

    The architecture hyperparameters (grid_size, latent_dim, hidden_channels) are
    read from the checkpoint, so no architecture information needs to be supplied.

    :param file_name:   The checkpoint file name (.pth).
    :param device:      The torch device to put the network on ('cpu' or 'cuda').
                        [default: 'cpu']
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a Conv2dAutoEncoder instance in eval mode on the requested device.
    """
    # map_location ensures we can load a model trained on cuda onto a cpu.
    checkpoint = torch.load(file_name, map_location=torch.device(device))

    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            "Checkpoint file %s is not a valid AIPSF checkpoint. " % file_name +
            "Expecting a dict with a 'model_state_dict' key, as written by "
            "piff.aimodels.save_checkpoint.")

    missing = [key for key in ('grid_size', 'latent_dim', 'hidden_channels')
               if key not in checkpoint]
    if missing:
        raise ValueError(
            "Checkpoint file %s is missing the architecture keys %s. " % (file_name, missing) +
            "Re-save it with piff.aimodels.save_checkpoint.")

    model_type = checkpoint.get('model_type', 'Conv2dAutoEncoder')
    if model_type != 'Conv2dAutoEncoder':
        raise ValueError("Checkpoint file %s has unknown model_type %r." % (file_name, model_type))

    net = Conv2dAutoEncoder(grid_size=checkpoint['grid_size'],
                            latent_dim=checkpoint['latent_dim'],
                            hidden_channels=checkpoint['hidden_channels'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()

    if logger:
        logger.debug("Loaded Conv2dAutoEncoder from %s: grid_size=%d, latent_dim=%d, "
                     "hidden_channels=%d", file_name, net.grid_size, net.latent_dim,
                     net.hidden_channels)
    return net
