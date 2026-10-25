Interpolation Schemes
=====================

Interpolators goven how the model parameters are interpolated across the field of view.

The Interp base class
---------------------

.. autoclass:: piff.Interp
   :members:

    .. automethod:: piff.Interp._finish_write
    .. automethod:: piff.Interp._finish_read

Mean interpolation
------------------

.. autoclass:: piff.Mean
   :members:

    .. automethod:: piff.Mean._finish_write
    .. automethod:: piff.Mean._finish_read

Polynomial interpolation
------------------------

.. autoclass:: piff.Polynomial
   :members:

    .. automethod:: piff.Polynomial._setup_indices
    .. automethod:: piff.Polynomial._set_function
    .. automethod:: piff.Polynomial._generate_indices
    .. automethod:: piff.Polynomial._pack_coefficients
    .. automethod:: piff.Polynomial._unpack_coefficients
    .. automethod:: piff.Polynomial._interpolationModel
    .. automethod:: piff.Polynomial._initialGuess
    .. automethod:: piff.Polynomial._finish_write
    .. automethod:: piff.Polynomial._finish_read

Interpolation using basis functions
-----------------------------------

.. autoclass:: piff.BasisInterp
   :members:

    .. automethod:: piff.BasisInterp._solve_qr
    .. automethod:: piff.BasisInterp._solve_direct

.. autoclass:: piff.BasisPolynomial
   :members:

    .. automethod:: piff.BasisPolynomial._finish_write
    .. automethod:: piff.BasisPolynomial._finish_read

K-Nearest Neighbors
-------------------

.. autoclass:: piff.KNNInterp
   :members:

    .. automethod:: piff.KNNInterp._fit
    .. automethod:: piff.KNNInterp._predict
    .. automethod:: piff.KNNInterp._finish_write
    .. automethod:: piff.KNNInterp._finish_read

Gaussian process interpolation
------------------------------

.. autoclass:: piff.GPInterp
   :members:

    .. automethod:: piff.GPInterp._fit
    .. automethod:: piff.GPInterp._predict
    .. automethod:: piff.GPInterp._finish_write
    .. automethod:: piff.GPInterp._finish_read
