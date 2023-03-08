Changes from version 1.2 to 1.3
===============================

Output file changes
--------------------

- Preserve the dtype of property columns read from an input catalog when they are subsequently
  written to an output file (e.g HSMCatalog) (#141)
- Changed the HSM file output column name from flag_truth to flag_data to match the other
  \*_data columns. (#142)
- Fixed type of reserve output column to be int rather than float. (#143)
- Automatically write any extra_properties from the input file into the output file. (#143)


API Changes
-----------

- Changed the name of the purity parameter of the SizeMag selector to impurity.  This value
  really describes the maximum allowed impurity of the sample, so calling it purity was
  confusing.  "Impurity" is closer to what this parameter is intended to mean.  (The old name
  is still allowed for now, but gives a deprecation warning.) (#146)


Performance improvements
------------------------

- Changed default interpolant for PixelGrid to Lanczos(7) rather than Lanczos(3), since we found
  some significant inaccuracies in some cases when using Lanczos(3), so this seems like a safer
  default. (#145)


New features
------------

- Added Wavefront class to allow for more general descriptions of an optical wavefront for
  use with the Optical model. (#134)
- Made chipnum optional whenever the input model only covers a single chip, regardless of whether
  the chipnum was specified in the config file. (#140)
- Added option to output fourth_order moments in HSM output file. (#142)
- Added option to output all raw moments in HSM output file. (#142)
- Added star.withProperties method. (#143)
- Added model_properties option for stats to use e.g. a specific color rather than the stars'
  own colors for the model measurements. (#143)
- PSF instances that are read from a file via `piff.read` will have an attribute ``piff_version``
  giving the version of Piff that created the file.  (Files created prior to version 1.3 will
  have ``piff_version = None``. (#146)


Bug fixes
---------

- Fixed bug in output stats that what we called T was really sigma.  Now, it's correctly
  T = Ixx + Iyy = 2*sigma^2. (#133, #142)
