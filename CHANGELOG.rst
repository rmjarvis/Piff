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

Changes from version 1.2.1 to 1.2.2
-----------------------------------

- Fixed bug in making SizeMag stats plot when some flux values from hsm end up
  negative or if there are no stars found. (#1156)

Changes from version 1.2.2 to 1.2.3
-----------------------------------

- Don't skip stars in HSM output file.  Rather just mark failed HSM measurements with
  flag_truth=-1 or flag_model=-1 as appropriate.
