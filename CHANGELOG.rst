Changes from version 1.3 to 1.4
===============================

Output file changes
--------------------



API Changes
-----------

- Deprecated some redundant or potentially unclear type names.
  * GSObjectModel -> GSObject
  * GPInterp -> GP or GaussianProcess
  * kNNInterp -> KNN or KNearestNeighbors
  * Star -> StarImages
- Deprecated the include_reserve=True option for outlier rejection.  Now that all objects are
  preserved in the output file, reserve stars that would have been rejected are marked as such,
  so you can choose whether or not to use them for any diagnostic tests.


Performance improvements
------------------------



New features
------------



Bug fixes
---------

