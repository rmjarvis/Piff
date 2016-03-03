The piffify executable
======================

The normal way to construct a Piff PSF model is using the piffify executable program with
a YAML configuration file::

    piffify config_file


The configuration file should have four fields which define the different aspects of
the process:

    :input:     Where to read the input images and catalogs.
    :model:     What kind of model to use to describe the PSF a single location.
    :interp:    How to interpolate the coefficients of the model across the field of view.
    :output:    Where to write the output file.

Each field is governed by a :type: parameter (although there are useful defaults for both
:input: and :output:.  This corresponds to different classes in the Python code.
The other parameters in each field correspond to the initialization kwargs for the class.

For instance the following cofiguration file uses the :class:`~piff.Pixel` class for the model and
the :class:`~piff.Polynomial` class for interpolation.  It uses the default
:class:`piff.InputFiles` and :class:`piff.OutputFile` for I/O.::

    input:
        # default type is 'InputFiles', which can be (and typically is) omitted
        images: some_exposure/image*.fits.fz
        cats: some_exposure/cat*.fits
        x_col: X_IMAGE
        y_col: Y_IMAGE
        weight_hdu: 3
    model:
        type: Pixel
        pixel_scale: 0.2
        size: 64
    interp:
        type: Polynomial
        order: 3
    output:
        # default type is 'InputFiles', which can be (and typically is) omitted
        file_name: some_exposure/piff_solution.fits


