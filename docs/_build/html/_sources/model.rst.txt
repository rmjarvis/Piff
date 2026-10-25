Models
======

Models govern what functional form to use for the PSF at a single location.

The Model base class
--------------------

.. autoclass:: piff.Model
   :members:

    .. automethod:: piff.Model._finish_write
    .. automethod:: piff.Model._finish_read
    .. automethod:: piff.Model._fix_kwargs


Models based on GalSim objects
------------------------------

.. autoclass:: piff.GSObjectModel
   :members:

    .. automethod:: piff.GSObjectModel._resid
    .. automethod:: piff.GSObjectModel._get_params

.. autoclass:: piff.Gaussian
   :members:

.. autoclass:: piff.Kolmogorov
   :members:

.. autoclass:: piff.Moffat
   :members:


Pixel grid model
----------------

.. autoclass:: piff.PixelGrid
   :members:


Optical model
-------------

.. autoclass:: piff.Optical
   :members:

