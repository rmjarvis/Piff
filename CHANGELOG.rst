Changes from version 1.3 to 1.4
===============================

Output file changes
--------------------

- Stars that were rejected as outliers are now included in the output file, but flagged with
  flag_psf=1. (#153)


API Changes
-----------

- Changed the list of property names that a PSF uses for interpolation to
  ``psf.interp_property_names``. (#150)
- Deprecated some redundant or potentially unclear type names. (#151)
  * GSObjectModel -> GSObject
  * GPInterp -> GP or GaussianProcess
  * kNNInterp -> KNN or KNearestNeighbors
  * Star -> StarImages
- Deprecated the include_reserve=True option for outlier rejection.  Now that all objects are
  preserved in the output file, reserve stars that would have been rejected are marked as such,
  so you can choose whether or not to use them for any diagnostic tests. (#153)
- Deprecated copy_image=False option to psf.drawStar and psf.drawStarList. (#155)


Performance improvements
------------------------

- Updated the DES-specific parameters for the `Optical` PSF model to be more accurate. (#138)
- Improved the flux/centroid-finding step of each iteration to be slightly more accurate. (#154)


New features
------------

- Added ``mirror_figure`` options for the `Optical` PSF model. (#138)
- Added ``trust_pos`` option in the input field.  This tells Piff not to adjust the star positions
  when fitting the PSF. (#154)
- Added ``fit_flux`` option for GSObject and PixelGrid models. (#155)
- Added ``init`` option for all models. (#155)
- Added `SumPSF` PSF type. (#157)
- Added `ConvolvePSF` PSF type. (#162)
- Added ``dof`` as a column in the stars section of the output file. (#165)


Bug fixes
---------

- Fixed a number of minor errors and inefficiencies in the `Optical` class. (#156)
