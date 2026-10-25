Changes from version 1.7 to 1.8
===============================

Dependency Changes
------------------


API Changes
-----------


Output file changes
--------------------


New features
------------

- Added an API to compute the derivative of the PSF image with respect to the position of
  the PSF, as needed e.g. by scene-modeling photometry codes.  The new methods are
  ``PSF.draw_derivative`` (derivative with respect to the image (x,y) position),
  ``PSF.drawStarDerivative`` (with respect to the world (u,v) position of a star) and
  ``Model.drawDerivative``.  The derivative is computed analytically from the derivative of
  the interpolant kernel.  This is currently only implemented for the PixelGrid model; other
  models raise NotImplementedError.

Bug fixes
---------

