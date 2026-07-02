Changes from version 1.6 to 1.7
===============================

Output file changes
--------------------


API Changes
-----------



Performance improvements
------------------------



New features
------------

- Added AIPSF, a PSF model based on a convolutional autoencoder: the encoder maps each star
  to a latent vector, which is interpolated across the field by the regular Piff interpolators,
  and the decoder renders the PSF.  Includes training tools in ``piff.aimodels`` and a new
  ``trainify`` executable to train the network from collections of PSF stamps.  PyTorch is
  required to use this model, but remains an optional dependency of Piff.


Bug fixes
---------

