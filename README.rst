PIFF: PSFs In the Full FOV
==========================

Piff is a Python software package for modeling the point-spread function (PSF)
across multiple detectors in the full field of view (FOV).

Aspirational features: (We're still working on the code!)

- Has multiple basis sets for the underlying PSF model, including pixel-based,
  shapelets, Gaussian mixture, maybe also Moffat and/or Kolmogorov.
- Can build the models in either chip or sky coordinates, properly accounting
  for the WCS of the image.
- Can interpolate across the full field-of-view, or across each chip separately,
  or a combination of both.
- Can do the fitting in either real or Fourier space.
- Has multiple interpolation functions including polynomials, kriging, and
  PSFEnt.
- Can take knowledge of the optical aberrations as input to convolve the model
  of the atmospheric PSF.
- Performs outlier rejection to detect and remove stars that are not good
  exemplars of the PSF.  Outputs the list of stars that were actually used
  to build the final model.
- Allows the centroid to be fixed or floating.
- In general, allow any value to be fixed rather than fit for.
- Uses highly readable YAML configuration files to set the various options.
- Includes Python code to read in the PSF files and use it to draw an image
  of the PSF at an arbitrary location.
- Currently, the lead developers are:
  - Mike Jarvis (mikejarvis17 at gmail)
  - Chris Davis (chris.pa.davis at gmail)
  - Josh Meyers (jmeyers314 at gmail)
  If you'd like to join the development effort, or if you have any other
  questions or comments about the code, feel free to contact us at the above
  email addresses.


Installation
------------

Eventually, we'd like this to be installable as::

    pip install Piff

But for now, only the setup.py installation is available::

    python setup.py install

Depending on your setup, you might prefer/need one of these variants::

    sudo python setup.py install
    python setup.py install --user
    python setup.py install --prefix=PREFIX


Running Tests
-------------

After installing Piff, you can run the unit tests by doing::

    cd tests
    nosetests


Reporting bugs
--------------

If you have any trouble installing or using the code, or if you find a bug,
please report it at:

https://github.com/rmjarvis/Piff/issues

Click "New Issue", which will open up a form for you to fill in with the
details of the problem you are having.


Requesting features
-------------------

If you would like to request a new feature, do the same thing.  Open a new
issue and fill in the details of the feature you would like added to Piff.
Or if there is already an issue for your desired feature, please add to the
discussion, describing your use case.  The more people who say they want a
feature, the more likely we are to get around to it sooner than later.

