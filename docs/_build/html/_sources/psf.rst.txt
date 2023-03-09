PSF classes
===========

The `PSF` classes define the interaction between the models and the interpolants.  The
simplest version `SimplePSF` has one model and one interpolant, but it is possible to have
more complicated combinations of these.

.. autoclass:: piff.PSF
    :members:

The simple case of one model/interp
-----------------------------------

.. autoclass:: piff.SimplePSF
    :members:

Using a separate solution for each chip
---------------------------------------

.. autoclass:: piff.SingleChipPSF
    :members:

