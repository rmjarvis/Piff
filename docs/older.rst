
Below is a summary of the major changes with each new tagged version of Piff.
Each version may also include various other minor changes and bug fixes not
listed here for brevity.  See the CHANGELOG files associates with each version
for a more complete list.  Issue numbers related to each change are given in parentheses.

v1.6
----

*Performance improvements*

- Made various performance improvements, courtesy of Nate Lust, from profiling how Piff is used
  by the LSST DM stack. (#184)

*New features*

- Allowed the weight and/or badpix images to be from different files than the main image.
  (#147, #183)
- Changed some of the logging messages to be at a more appropriate level.  (#132, #182)
- Made it possible to exclude some cross terms in BasisPolynomial interpolation.  (#180)
- Made fit_flux=True the default when using init=zero.  (#179)

*Bug fixes*

- Added an appropriate logging message when the input images are too small to constrain a
  PixelGrid model. (#131, #181)


v1.5
----

*Dependency Changes*

- Added dependencies, Eigen and PyBind11.  (#173)
- Added official support for python 3.13.
- Stopped supporting a few old versions of dependencies. (#178)

*API Changes*

- Changed the default behavior of the StarImages plot to include the average star and model. (#167)

*Performance improvements*

- Added a JAX implementation of the BaisPolynomial math calculations. (#166)
- Added a C++ Eigen implemenation of the BasisPolynomial math calculations.  (#173)

*New features*

- Added an image of the average star and model in the StarImages output plot. (#167)
- Added new Reader/Writer classes (#168, #170)
- Added the ability to compute a weight image from a sky image. (#174)

*Bug fixes*

- Fixed some places where the stats were not correctly using the new flagging interface. (#174)
- Fixed a bug that could rarely cause a (nearly) infinite loop when recentering fails. (#178)


v1.4
----

*Output file changes*

- Stars that were rejected as outliers are now included in the output file, but flagged. (#153)
- Added an image of the average star and model in the StarImages output plot. (#167)


*API Changes*

- Changed the list of property names to ``psf.interp_property_names``. (#150)
- Deprecated some redundant or potentially unclear type names. (#151)
- Deprecated the include_reserve=True option for outlier rejection. (#153)
- Deprecated copy_image=False option to psf.drawStar and psf.drawStarList. (#155)
- Changed the default behavior of the StarImages plot to include the average star and model. (#167)

*Performance improvements*

- Updated the DES-specific parameters for the `Optical` PSF model to be more accurate. (#138)
- Improved the flux/centroid-finding step of each iteration to be slightly more accurate. (#154)

*New features*

- Added ``mirror_figure`` options for the `Optical` PSF model. (#138)
- Added ``trust_pos`` option in the input field.  This tells Piff not to adjust the star positions
  when fitting the PSF. (#154)
- Added ``fit_flux`` option for GSObject and PixelGrid models. (#155)
- Added ``init`` option for all models. (#155)
- Added `SumPSF` PSF type. (#157)
- Added `ConvolvePSF` PSF type. (#162)
- Added ``dof`` as a column in the stars section of the output file. (#165)
- Added an option to show large T candidate stars in SizeMag plot.

*Bug fixes*

- Fixed a number of minor errors and inefficiencies in the `Optical` class. (#156)
- Fixed a bug where reflux errors could cause the fit not to converge even though the
  offending stars had already been flagged and removed from consideration.
- Fixed an error in TwoDHist stats that T_model was incorrectly using the star data.
- Fixed an error that caused the candidate stars not to display correctly in SizeMag plot.


v1.3
----

*Output file changes*

- Preserve the dtype of property columns from input catalog to output file. (#141)
- Changed the HSM file output column name from flag_truth to flag_data.  (#142)
- Fixed type of reserve output column to be bool rather than float. (#143)
- Automatically write any extra_properties from the input file into the output file. (#143)
- Removed is_reserve column from the HSM output file.
- Added piff_version to the FITS header in the HSM output file. (#148)

*API Changes*

- Changed the name of the purity parameter of the SizeMag selector to impurity. (#146)

*Performance improvements*

- Changed default interpolant for PixelGrid to Lanczos(7) rather than Lanczos(3). (#145)
- Added guards against HSM failures when doing fourth_order/raw_moments in HSM output file. (#149)

*New features*

- Added Wavefront class. (#134)
- Made chipnum optional whenever the input model only covers a single chip. (#140)
- Added option to output fourth_order moments in HSM output file. (#142)
- Added option to output all raw moments in HSM output file. (#142)
- Added star.withProperties method. (#143)
- Added model_properties option for stats. (#143)
- Added a ``piff_version`` attibute to PSF instances that are read from a file. (#146)


v1.2
----

*Output file changes*

- Don't skip stars in HSM output file.  Rather just flag failed HSM measurements.

*New features*

- Added new select types: SizeMag, Properties, Flags, and SmallBright.
- Flag objects when HSM returns a negative flux or moves the centroid by more than 1 pixel. (#136)
- Use single_threaded context when doing multiprocessing.
- Turn off TreeCorr's multi-threading.

*Bug fixes*

- Fixed bug in output stats that what we called T had actually been sigma. (#133, #142)
- Fixed bug in making SizeMag stats plot when some flux values are negative. (#135)
- Fixed labeling error in SizeMag plot (model and data were reversed). (#136)
- Fixed bug in StarStat plot when nplot=0. (#136)
- Fixed bug in SizeMag selector that the reserve and reject steps were applied twice. (#136)
- Fix error that started in matplotlib v3.6.
- Fix several deprecation warnings from recent matplotlib and scipy versions.

v1.1
----

*Dependency Changes*

- Piff no longer supports Python 2.  We currently support Python 3.6 through 3.9.

*New features*

- Added ability to interpolate on properties, given in the input file.

v1.0
----

This is the first stable release.  It should have all the features desribed in the
Piff paper, Jarvis, et al, 2021.
