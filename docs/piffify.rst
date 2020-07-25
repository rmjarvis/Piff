The piffify executable
======================

The normal way to construct a Piff PSF model is using the piffify executable program with
a YAML configuration file::

    piffify config_file


The configuration file should have three fields which define the different aspects of
the process:

    :input:     Where to read the input images and catalogs.
    :psf:       What kind of model and interpolation to use to describe the PSF.
                Typically, this would have two subfields: model and interp.

                * model defines the shape of the PSF at a single location
                * interp defines how the model parameters are interpolated across the FOV.

    :output:    Where to write the output file.

Each field is governed by a :type: parameter (although there are useful defaults for all three
primary top-level fields.
This corresponds to different classes in the Python code.
The other parameters in each field correspond to the initialization kwargs for the class.

For instance the following cofiguration file uses the :class:`PixelGrid` class for the model and
the :class:`Polynomial` class for interpolation.  It uses the default
:class:`InputFiles` and :class:`OutputFile` for I/O. and :class:`SimplePSF` for
the PSF.::

    input:
        images: some_exposure/image*.fits.fz
        cats: some_exposure/cat*.fits
        x_col: X_IMAGE
        y_col: Y_IMAGE
        weight_hdu: 3
    psf:
        model:
            type: PixelGrid
            pixel_scale: 0.2
            size: 64
        interp:
            type: Polynomial
            order: 3
    output:
        file_name: some_exposure/piff_solution.fits


The functionality of the piffify executable is also available from python via
:func:`piffify` and related functions.


.. autofunction:: piff.piffify

.. autofunction:: piff.read_config

.. autofunction:: piff.setup_logger

.. autofunction:: piff.parse_variables

