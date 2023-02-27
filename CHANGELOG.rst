Changes from version 1.2 to 1.3
===============================

Output file changes
--------------------

- Preserve the dtype of property columns read from an input catalog when they are subsequently
  written to an output file (e.g HSMCatalog) (#141)


Performance improvements
------------------------

- Changed default interpolant for PixelGrid to Lanczos(7) rather than Lanczos(3), since we found
  some significant inaccuracies in some cases when using Lanczos(3), so this seems like a safer
  default. (#145)


New features
------------


Bug fixes
---------

- Fixed bug in output stats that what we called T was really sigma.  Now, it's correctly
  T = Ixx + Iyy = 2*sigma^2. (#133)
