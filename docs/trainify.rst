The trainify executable
=======================

The trainify executable trains the convolutional autoencoder used by the :class:`AIPSF`
model from a collection of PSF star stamps, using a YAML configuration file::

    trainify config_file

.. note::

    trainify (and everything else related to the AIPSF model) requires PyTorch, which
    is an optional dependency of Piff.  Install it with e.g. ``pip install torch``.

The configuration file has four fields:

    :input:     Where to read the training data.
    :model:     The autoencoder architecture hyperparameters.
    :training:  How to run the training.
    :output:    Where to write the trained checkpoint file.

For example::

    input:
        # Either a single merged pickle file or a directory of pickle files,
        # e.g. as written per (visit, detector, band) by the LSST meas_extensions_piff
        # training-sample collection.
        file_name: training/
        batch_size: 1024
        val_fraction: 0.05
        seed: 42
        num_workers: 4
    model:
        type: Conv2dAutoEncoder
        grid_size: 25
        latent_dim: 64
        hidden_channels: 16
    training:
        epochs: 40
        initial_lr: 1.e-3
        scheduler_on_plateau: true
        scheduler_factor: 0.1
        scheduler_patience: 5
        scheduler_threshold: 1.e-3
        scheduler_min_lr: 1.e-6
        use_weights: false
        fit_background: false
        device: cuda
    output:
        file_name: Conv2dAutoEncoder.pth
        history_file: training_history.pkl

The training data pickle files contain a dict of star records::

    { star_id: { 'star':     numpy array (N, N), the stamp normalized to sum to 1,
                 'weight':   numpy array (N, N), inverse variance of the normalized
                             stamp, zero for masked pixels,
                 'starPiff': numpy array (N, N), a reference PSF model prediction
                             at the star position (diagnostic baseline),
                 ... }, ... }

Any extra keys in the star records (positions, visit, detector, band, etc.) are ignored
by the training.

By default the training loss is the pixel-level MSE between the autoencoder output
and its input (scaled by 1e6 for convenience).  With ``use_weights: true``, the loss
is instead the mean per-star reduced chi2, using the per-pixel inverse-variance maps:
``chi2_star = sum[w * (model - star)^2] / N_good``, where ``N_good`` is the number of
unmasked pixels.  A model that describes the data at the noise level then has a loss
around 1, and masked pixels are naturally excluded from the fit.  This requires the
training data to contain the weight maps (older training sets stored ``None``).

With ``fit_background: true``, the per-star model becomes ``a * psf + b``, where the
amplitude ``a`` and constant background ``b`` are nuisance parameters solved
analytically per star (a 2x2 weighted linear system, re-evaluated at every step at
the current network weights, detached from the gradient; see
:func:`piff.aimodels.fit_amplitude_background`).  This absorbs local background
over/under-subtraction in the stamps, which the strictly positive SpatialSoftmax
output could not represent otherwise.  The nuisance fit is applied to the
autoencoder only, not to the 'starPiff' diagnostic baseline: the PixelGrid model is
fit per CCD and absorbs local background into its pixel grid by construction, so
refitting (a, b) on top of it would double-count the correction.

The ``scheduler_*`` options control the ReduceLROnPlateau scheduler used when
``scheduler_on_plateau`` is true.  Note that the plateau detection uses a *relative*
improvement threshold: with the default ``scheduler_threshold: 1e-4``, epochs that
improve the validation loss by more than 0.01% still reset the patience counter, so
on slowly-improving runs the learning rate may never drop; increase the threshold
(e.g. ``1e-3``) and/or lower the patience to make it fire.

The output checkpoint file stores the network weights along with the architecture
hyperparameters, so it can be used directly as the ``model_file`` of an :class:`AIPSF`
model without repeating the architecture configuration.

The functionality of the trainify executable is also available from python via
:func:`piff.aimodels.train` and related functions.

.. autofunction:: piff.aimodels.train

.. autofunction:: piff.aimodels.train_autoencoder

.. autofunction:: piff.aimodels.create_dataloaders

.. autofunction:: piff.aimodels.load_training_data

.. autofunction:: piff.aimodels.fit_amplitude_background

.. autofunction:: piff.aimodels.save_checkpoint

.. autofunction:: piff.aimodels.load_autoencoder
