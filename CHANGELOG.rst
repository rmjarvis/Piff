Changes from version 1.1 to 1.2
===============================

This version adds new options on how to select the input stars.
Including new select types: SizeMag, Properties, Flags, and
SmallBright (the last of which is really only designed as an
inital pass for the SizeMag selector).

Changes from version 1.2.0 to 1.2.1
-----------------------------------

- Fixed bug in output stats that what we called T was really sigma.  Now, it's correctly
  T = Ixx + Iyy = 2*sigma^2.
