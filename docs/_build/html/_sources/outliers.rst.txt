Removing Outliers
=================

Piff can remove stars that it deems to be outliers from the set of stars used to
build the PSF model.  This option is specified via an ``outliers`` section of the
``psf`` field in the configuration file.

.. autoclass:: piff.Outliers
   :members:

    .. automethod:: piff.Outliers._finish_write
    .. automethod:: piff.Outliers._finish_read

.. autoclass:: piff.ChisqOutliers
   :members:

