Changes from version 1.5 to 1.6
===============================


Performance improvements
------------------------

- Made various performance improvements, courtesy of Nate Lust, from profiling how Piff is used
  by the LSST DM stack. (#184)


New features
------------

- Allowed the weight and/or badpix images to be from different files than the main image.
  (#147, #183)
- Changed some of the logging messages to be at a more appropriate level.  Most things that
  had been WARNING are now INFO, and the things that were INFO are now VERBOSE, a new
  logging level that is more verbose than INFO, but less than DEBUG.  Messages that really
  are legit warnings are still at the WARNING level. (#132, #182)
- Made it possible to exclude some cross terms in BasisPolynomial interpolation.
  For instance, if x and y are both interpolated at 4th order, and color at 3rd order,
  it used to be the case that cross terms like x*color^2 and y^2*color would have to be
  included.  Now you can specify what maximum order you want for any cross-combinations. (#180)
- Made fit_flux=True the default when using init=zero.  This fixes some confusing behavior
  when using the Sum PSF type (the typical reason one would want to use init=zero), where it
  was not obvious that fit_flux=True was also required for it to work correctly. (#179)


Bug fixes
---------

- Added an appropriate logging message when the input images are too small to constrain a
  PixelGrid model. (#131, #181)
