
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
.. module:: aipsf
"""

import numpy as np
import galsim
import torch
import os

from .model import Model
from .star import Star
from .aimodels.aimodels import Conv2dAutoEncoder

class AIPSF(Model):
    """A PSF model that uses a pre-trained Convolutional AutoEncoder.

    The PSF profile is defined by a latent vector in the autoencoder's latent space.
    The interpolation is done in this latent space.

    :param model_file:  The path to the trained PyTorch model file (.pth).
    :param device:      The device to run the model on ('cpu' or 'cuda'). [default: 'cpu']
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'AIPSF'

    def __init__(self, model_file, device='cpu', logger=None):
        self.model_file = model_file
        self.device = device
        self.kwargs = {
            'model_file': model_file,
            'device': device,
        }

        self.logger = logger
        
        if self.logger:
            self.logger.debug(f"Loading AIPSF model from {model_file}")
        
        self.net = Conv2dAutoEncoder(latent_dim = 16, grid_size = 25, hidden_channels=16)
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        # map_location ensures we can load a cuda model on cpu if needed
        checkpoint = torch.load(model_file, map_location=torch.device(device))
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            self.net.load_state_dict(checkpoint)
        else:
            self.net = checkpoint
             
        self.net.to(device)
        self.net.eval()
        
        self.grid_size = self.net.grid_size

                                  
    def initialize(self, star, logger=None, default_init=None):
        """Initialize a star to work with the current model.

        :param star:            A Star instance with the raw data.
        :param logger:          A logger object for logging debug info. [default: None]
        :param default_init:    The default initilization method if the user doesn't specify one.
                                [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        if logger:
            logger = logger
        
        image = star.data.image
        
        if image.array.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"Input star shape {image.array.shape} does not match"
                             f"model expected shape ({self.grid_size}, {self.grid_size})")

        # Normalize flux to 1
        stamp_data = image.array.copy()
        
        # Handle bad pixels.
        if np.any(~np.isfinite(stamp_data)):
             raise ValueError("Input star contains non-finite values.")

        flux = np.sum(stamp_data)     
        normalized_stamp = stamp_data / flux
        
        # Prepare for encoder
        # Model expects (Batch, Channel, Height, Width) -> (1, 1, grid_size, grid_size)
        input_tensor = torch.from_numpy(normalized_stamp).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Get latent vector
        with torch.no_grad():
            z = self.net.encoder(input_tensor)
            
        # z is (1, latent_dim)
        params = z.cpu().numpy().flatten()
        
        # Create fit object
        # PF Note: I don't know if flux needs to be initialized here.
        # To check. 
        fit = star.fit.withNew(params=params, flux=flux)

        return Star(star.data, fit)

    def fit(self, star, convert_func=None):
        """Fit the Model to the star's data.
        
        For the AutoEncoder model, 'fitting' is just running the encoder again
        to get the latent vector. Since the encoder is deterministic, 
        this yields the same result as initialize.
        
        :param star:            A Star instance
        :param convert_func:    (Ignored for this simple model)

        :returns:      New Star instance with updated fit information
        """
        return self.initialize(star)

    def getProfile(self, params):
        """Get the GalSim GSObject for the given parameters.

        :param params:  The latent vector (numpy array).
        :returns:       A GalSim GSObject.
        """
        # Convert params to tensor
        # Shape (1, latent_dim)
        z = torch.from_numpy(params).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.net.decoder(z)
            
        # Output is (1, 1, H, W) -> (H, W)
        output_image = output_tensor.squeeze().cpu().numpy()
        
        # Create GalSim InterpolatedImage
        gs_image = galsim.Image(output_image, scale=1.0)
        
        # The output of the decoder is normalized (SpatialSoftmax), so flux=1.
        prof = galsim.InterpolatedImage(gs_image, normalization='flux', flux=1.0)
        return prof
