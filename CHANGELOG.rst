Changes from version 1.6 to 1.7
===============================

This version mostly adds a bunch of things we needed to enable running Piff on Roman
Space Telescope images using an optical model based on the galSim.roman PSF model.
The new PSF type is called `RomanOptics`. See the docstring for details about the new
type. You can also see some example configurations that use it in the devel/roman directory.

Dependency Changes
------------------

- Added ``pyarrow`` dependency for reading Parquet input catalogs. It is only required
  if you attempt to read a parquet file. (#188)
- Removed support for Python 3.8, 3.9.
- Added support for Python 3.14, 3.15.

API Changes
-----------

- Deprecated ``input.nstars`` in favor of ``select.nstars``, which now works across all
  input files. The old option is deprecated and retains the per-image behavior. (#190)
- Renamed ``select.max_snr`` to ``select.max_snr_weight`` to clarify that this caps
  a star's effective fitting weight, rather than rejecting bright stars. The old
  name is still works but is deprecated. (#195)
- Deprecated the use of strings to mean FITS-header keywords for ``noise``, ``gain``,
  ``satur`` and ``sky`` in the ``input`` section. You should now use the new
  ``ImageHeaderValue`` type instead. (#207)
- Various minor signature changes to Python functions. If you are using the Python
  interface rather than config files, you may need to check the doc strings for changes
  to return values or parameter names.

Output file changes
--------------------

- Added the number of fitting iterations, ``niter``, to the saved fit diagnostics. (#196)
- Added the observing bandpass when one is supplied. (#194)
- Star SEDs are saved as file names and reconstruction metadata, rather than embedding
  the full spectra. If you do something that needs the SEDs after reading the psf from a
  file, those SED files must be accessible. (#194, #198)
- Renamed the WCS table's ``nchunks`` column to ``wcs_nchunks``. Files with the old name
  should continue to work properly. (#198)

New features
------------

- Added ``input.badpix_zeros`` to mask zero-valued science pixels. (#185)
- Added support for reading the noise variance from the FITS header. (#186)
- Added an option to compute the sky level from the median of the input image.
  Use ``input.sky: median``. (#187)
- Added support for Parquet input catalogs, using ``pyarrow``. (#188)
- Added the ability to read multiple image HDUs from a single FITS file to be processed as
  separate images. (#189)
- Changed ``nstars`` to mean the total number of stars to use across all input images when there
  are more than one, rather than each image separately. Also moved this to ``select`` where
  it fits more naturally. (#190)
- Added ``raw_snr`` attribute for stars to keep track of their real snr, rather than the
  possibly adjusted value in ``snr`` due to ``select.max_snr_weight``. (#190)
- Added the ability to compute star ``properties`` derived from one or more columns in the
  input catalog. (#191)
- Added ``Centroid`` outlier rejection with a ``max_offset`` threshold, and allow
  ``outliers`` to be a list combining multiple rejection criteria. (#192)
- Added ``select.min_sep`` to reject objects with a nearby neighbor. (#193)
- Added ``input.bandpass`` for specifying the bandpass used for the observations. (#194)
- Added support stellar SEDs in the input field for chromatic fitting. You can either
  specify a single spectrum to use for all stars (using ``input.sed_file_name``) or
  specify a specific spectrum for each star separately (using ``input.sed_col``).
  There are additional parameters that can control how these files are used. See docs
  for `InputFiles` for more details. (#194, #198)
- Added the ``RomanOptics`` PSF type, based on GalSim's Roman optical model, with
  fitted Zernike aberrations and Gaussian priors. (#195, #196, #197, #198)
- Changed ``StarImages`` plots to include chip number and image coordinates for each star. (#196)
- Expanded support for GalSim-style config value types in the config file, including
  ``Eval`` and ``Current`` expressions and their ``$`` and ``@`` shortcuts. (#207)

Bug fixes
---------

- Ensure input stamps use floating-point data, so sky subtraction and other
  processing work with integer-valued source images. (#187)
- Fixed composite PSF serialization when components have parameter-variance arrays
  of different sizes. (#195)
- Reject failed HSM measurements when using ``hsm_size_reject`` or ``max_pixel_cut``. (#196)
- Fixed ``StarImages`` when fewer eligible stars are available than requested. (#196)
- Fixed GSObject fitting to avoid nonpositive sizes. (#200)
- Added checks for excessively large FFTs when fitting and reject them rather than
  risking huge memory allocations or crashes. (#205)
- Fixed the ``TwoDHist`` model-size panel, which previously repeated the measured
  star sizes instead of showing model sizes. (#205)
- Added warning about zero-S/N candidates when no ``min_snr`` cut is set, with an additional
  suggestion to supply noise information when it is missing. (#200)
- Fixed chi-squared outlier rejection for already-flagged stars. (#208)
