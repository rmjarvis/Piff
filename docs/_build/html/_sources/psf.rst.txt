PSF classes
===========

The `PSF` classes define the interaction between the models and the interpolants.  The
simplest version `SimplePSF` has one model and one interpolant, but it is possible to have
more complicated combinations of these.

.. autoclass:: piff.PSF
    :members:

    .. automethod:: piff.PSF._write
    .. automethod:: piff.PSF._read

The simple case of one model/interp
-----------------------------------

.. autoclass:: piff.SimplePSF
    :members:

    .. automethod:: piff.SimplePSF._finish_write
    .. automethod:: piff.SimplePSF._finish_read

Using a separate solution for each chip
---------------------------------------

.. autoclass:: piff.SingleChipPSF
    :members:

    .. automethod:: piff.SingleChipPSF._finish_write
    .. automethod:: piff.SingleChipPSF._finish_read

Composite PSF types
-------------------

.. autoclass:: piff.SumPSF
    :members:

    .. automethod:: piff.SumPSF._finish_write
    .. automethod:: piff.SumPSF._finish_read

.. autoclass:: piff.ConvolvePSF
    :members:

    .. automethod:: piff.ConvolvePSF._finish_write
    .. automethod:: piff.ConvolvePSF._finish_read
