Stars
=====

The `Star` class is mostly an implementation detail that users won't need to know, but for
more advanced usage, it may be helpful to use the `Star` objects directly.

.. autoclass:: piff.Star
    :members:

    .. automethod:: piff.Star.__getitem__

.. autoclass:: piff.StarData
    :members:

.. autoclass:: piff.StarFit
    :members:

Size and shape measurements
---------------------------

Piff primarily fits PSF profiles by directly comparing the image data with the
drawn PSF at the same location. We don't directly target the size or shape of the
profile to match the size or shape of the data. This is merely a consequence of trying
to make the images close to equal.

However, there are quite a few places throughout the code where it is useful to
measure sizes and shapes. This includes the following:

* `SizeMagSelect`
* `SmallBrightSelect`
* ``select.size_reject``
* Model initialization for `PixelGrid` and `GSObjectModel`
* Most of the `Output Statistics`

For all of these uses, the default behavior is to use the HSM adaptive moments
algorithm implemented in GalSim and described in
`Hirata and Seljak, 2003 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.343..459H/abstract>`_.

This algorithm is usually a good choice for the kinds of sizes and shapes that we need
in Piff, so most users can just use the default. However, on undersampled images,
the HSM algorithm has a tendency to collapse to a degenerate solution. Even when not
fully collapsing, the measuements can become significantly biased.

We therefore have an option to change the back-end shape measurement algorithm to
a different choice if desired. The available shape measurement algorithms are:

* **hsm** (default) The HSM adaptive moments algorithm from GalSim.
* **gauss** A maximum likelihood fit to an elliptical Gaussian profile using the NGmix
  software package.
* **am** An adaptive moment algorithm similar to HSM, but implemented by NGmix.
* **wmom** Weighted moments using a fixed round Gaussian weight whose size is explicitly
  set by the user.

The choice of algorithm to use is a global setting in Piff. In the config file,
it's a top level parameter, ``shape_method``. If you choose the **wmom** algorithm,
then you also need to specify ``shape_weight_fwhm``. All algorithms other than
**hsm** require the ``ngmix`` package to be installed.

.. autofunction:: piff.set_shape_method

.. autofunction:: piff.get_shape_method

.. autofunction:: piff.get_shape_weight_fwhm
