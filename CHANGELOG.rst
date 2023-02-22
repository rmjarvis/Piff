Changes from version 1.2 to 1.3
===============================

Output file changes
--------------------

- Preserve the dtype of property columns read from an input catalog when they are subsequently
  written to an output file (e.g HSMCatalog) (#141)


Output file changes
-------------------

- Changed the HSM file output column name from flag_truth to flag_data to match the other
  \*_data columns. (#142)

- Changed default interpolant for PixelGrid to Lanczos(7) rather than Lanczos(3), since we found
  some significant inaccuracies in some cases when using Lanczos(3), so this seems like a safer
  default. (#145)


New features
------------

- Made chipnum optional whenever the input model only covers a single chip, regardless of whether
  the chipnum was specified in the config file. (#140)



Bug fixes
---------

- Fixed bug in output stats that what we called T was really sigma.  Now, it's correctly
  T = Ixx + Iyy = 2*sigma^2. (#133, #142)
