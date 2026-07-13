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

import os
import numpy as np
import galsim

from .model import Model
from .star import Star

# torch is imported lazily by AIPSF.__init__, so that `import piff` works without
# torch installed (it is an optional dependency).  Once an AIPSF instance exists,
# the hot methods (_encode, getProfile) use this module-level reference directly.
torch = None


class AIPSF(Model):
    """A PSF model that uses a pre-trained convolutional autoencoder.

    The PSF at each star is described by the latent vector of the autoencoder:
    the encoder maps the flux-normalized star stamp to ``star.fit.params``, and
    the decoder maps a latent vector back to a PSF stamp (normalized to unit
    flux).  Spatial interpolation of the PSF is done in the latent space by the
    regular Piff interpolators.

    Notes:

    * This model requires PyTorch, which is an optional dependency of Piff.
    * The trained network is read from a checkpoint file written by
      :func:`piff.aimodels.save_checkpoint`, which stores the network weights
      together with the architecture hyperparameters (grid_size, latent_dim,
      hidden_channels).  The checkpoint file is *not* embedded in Piff output
      files; reading a serialized PSF requires the checkpoint file to be
      available at the same path.
    * The input stamps must be exactly (grid_size, grid_size) pixels, where
      grid_size comes from the checkpoint (an odd integer, 25 for the
      production networks).
    * The latent parameters can be interpolated by any interpolator that works
      on ``star.fit.params`` directly (e.g. Polynomial, KNearestNeighbors,
      GaussianProcess, Mean).  Basis-type interpolators (e.g. BasisPolynomial)
      are not supported, since this is only for PixelGrid.

    Use type name "AIPSF" in a config field to use this model.

    :param scale:       The pixel scale of the PSF stamps in arcsec.
    :param model_file:  The path to the trained checkpoint file (.pth), written by
                        :func:`piff.aimodels.save_checkpoint`.
    :param device:      The torch device to run the network on ('cpu' or 'cuda').
                        [default: 'cpu']
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'AIPSF'
    _method = 'no_pixel'
    # The star position is trusted from the input star; the model does not fit a center.
    _centered = False

    def __init__(self, scale, model_file=None, device='cpu', logger=None):
        self.scale = scale
        self.model_file = model_file
        self.device = device
        self.kwargs = {
            'model_file': model_file,
            'scale': scale,
            'device': device,
        }

        if model_file is None:
            raise ValueError("model_file is required for the AIPSF model")
        if not os.path.exists(model_file):
            raise FileNotFoundError("Model file not found: %s" % model_file)

        if logger:
            logger.debug("Loading AIPSF model from %s", model_file)

        # These imports are delayed until here, so torch stays an optional dependency.
        # The aimodels import raises an informative ImportError if torch is not
        # available.  Note: torch is deliberately not stored on self (modules cannot
        # be pickled, and the fitted PSF gets pickled by the LSST middleware).
        from .aimodels import load_autoencoder
        global torch
        if torch is None:
            import torch
        self.net = load_autoencoder(model_file, device=device, logger=logger)

        self.grid_size = self.net.grid_size
        self.latent_dim = self.net.latent_dim
        self.set_num(None)

    def __setstate__(self, state):
        # The fitted PSF gets pickled by the LSST middleware, and unpickling
        # bypasses __init__, so bind the module-level torch reference here too.
        # (By this point torch is importable: unpickling self.net required it.)
        global torch
        if torch is None:
            import torch
        self.__dict__.update(state)

    def _encode(self, star):
        """Run the encoder on the star's stamp.

        :param star:    A Star instance with the raw data.

        :returns: (params, flux) where params is the latent vector as a numpy array
                  and flux is the sum of the input stamp.
        """
        stamp_data = star.data.image.array

        if stamp_data.shape != (self.grid_size, self.grid_size):
            raise ValueError("Input star shape %s does not match "
                             "model expected shape (%d, %d)" %
                             (stamp_data.shape, self.grid_size, self.grid_size))

        if np.any(~np.isfinite(stamp_data)):
            raise ValueError("Input star contains non-finite values.")

        flux = np.sum(stamp_data)
        if flux <= 0:
            raise ValueError("Input star has non-positive total flux (%s)." % flux)
        normalized_stamp = stamp_data / flux

        # The network expects (Batch, Channel, Height, Width) -> (1, 1, grid_size, grid_size)
        input_tensor = torch.from_numpy(normalized_stamp).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.net.encoder(input_tensor)

        # z is (1, latent_dim)
        params = z.cpu().numpy().flatten()
        return params, flux

    def initialize(self, star, logger=None, default_init=None):
        """Initialize a star to work with the current model.

        The encoder is run on the star's stamp to get the initial latent vector,
        and the flux is initialized to the sum of the stamp.

        :param star:            A Star instance with the raw data.
        :param logger:          A logger object for logging debug info. [default: None]
        :param default_init:    The default initialization method if the user doesn't specify
                                one.  (Ignored by this model.) [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        params, flux = self._encode(star)
        fit = star.fit.newParams(params, num=self._num, flux=flux)
        return Star(star.data, fit)

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        """Fit the model to the star's data.

        For this model, "fitting" is running the (deterministic) encoder on the
        star's stamp, so the resulting latent vector is the same as from
        `initialize`.  The flux and center are left unchanged; they are updated
        by the reflux step of the PSF fitting.  The chisq and dof of the fit are
        computed so outlier rejection and convergence bookkeeping work as usual.

        In addition, the per-star amplitude and local background of the model,
        i.e. (a, b) in data ~ a * psf + b with psf the unit-flux decoded model,
        are solved analytically (weighted least squares, same normal equations
        as :func:`piff.aimodels.fit_amplitude_background`) and stored in
        ``star.data.properties`` as 'aipsf_a' and 'aipsf_b', in image counts,
        for downstream diagnostics.  They are recomputed at each fit iteration
        (the final values persist); reserve stars never go through fit, so they
        do not get these properties.

        :param star:            A Star instance
        :param logger:          A logger object for logging debug info. [default: None]
        :param convert_func:    An optional function to apply to the profile being fit.
                                (Ignored by this model.) [default: None]
        :param draw_method:     The method to use with drawImage.  (This model requires
                                'no_pixel'.) [default: None]

        :returns:      New Star instance with updated fit information
        """
        assert draw_method in (None, 'no_pixel')

        params, _ = self._encode(star)

        # Compute the chisq of this model prediction, scaled by the current flux estimate.
        prof = self.getProfile(params).shift(star.fit.center) * star.fit.flux
        image = star.image.copy()
        prof.drawImage(image, method=self._method, center=star.image_pos)
        model = image.array.ravel()

        data, weight, u, v = star.data.getDataVector()
        chisq = np.sum(weight * (data - model)**2)
        dof = np.count_nonzero(weight) - self.latent_dim

        # Solve per star for the amplitude and local background in
        # data ~ a * psf + b, with psf the unit-flux drawn model, by weighted
        # least squares (same normal equations as
        # piff.aimodels.fit_amplitude_background).  Stored as star properties
        # (in image counts) for downstream diagnostics.
        psf_unit = model / star.fit.flux
        S_w = np.sum(weight)
        S_p = np.sum(weight * psf_unit)
        S_pp = np.sum(weight * psf_unit**2)
        S_y = np.sum(weight * data)
        S_py = np.sum(weight * psf_unit * data)
        det = max(S_pp * S_w - S_p * S_p, 1.e-30)
        star.data.properties['aipsf_a'] = float((S_w * S_py - S_p * S_y) / det)
        star.data.properties['aipsf_b'] = float((S_pp * S_y - S_p * S_py) / det)

        fit = star.fit.newParams(params, num=self._num, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    def getProfile(self, params):
        """Get a version of the model as a GalSim GSObject.

        The decoder is run on the latent vector to produce a PSF stamp, which is
        returned as an InterpolatedImage with unit flux.

        :param params:  The latent vector (numpy array).

        :returns:       A galsim.GSObject instance
        """
        # Shape (1, latent_dim)
        z = torch.from_numpy(np.asarray(params)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.net.decoder(z)

        # Output is (1, 1, H, W) -> (H, W)
        output_image = output_tensor.squeeze().cpu().numpy()

        gs_image = galsim.Image(output_image, scale=self.scale)

        # The output of the decoder is normalized (SpatialSoftmax + ZeroFloor),
        # so it has zero floor and flux=1.
        prof = galsim.InterpolatedImage(gs_image, normalization='flux', flux=1.0)
        return prof
