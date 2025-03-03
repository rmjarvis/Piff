Changes from version 1.4 to 1.5
===============================

A complete list of all new features and changes is given below.
`Relevant PRs and Issues,
<https://github.com/rmjarvis/Piff/milestone/9?closed=1>`_
whose issue numbers are listed below for the relevant items.


Dependency Changes
------------------

- Added dependencies, Eigen and PyBind11.  These were already implicit dependencies through
  GalSim, but now they are used explicitly in Piff.  (#173)


API Changes
-----------

- Changed the default behavior of the StarImages plot to include the average star and model.
  To recover the old version without these images, use ``include_ave = False``. (#167)


Performance improvements
------------------------

- Added a JAX implementation of the BaisPolynomial math calculations.  Use ``solver='jax'``
  to enable it. (#166)
- Added a C++ Eigen implemenation of the BasisPolynomial math calculations.  Use ``solver='cpp'``
  to enable it. (#173)


New features
------------

- Added an image of the average star and model in the StarImages output plot. (#167)
- Added new Reader/Writer classes for handling I/O to make it easier to write a different
  back end (primarily for LSST DM usage). (#168, #170)
- Added the ability to compute a weight image from a sky image. (#174)


Bug fixes
---------

- Fixed some places where the stats were not correctly using the new flagging interface. (#174)

Changes from version 1.5.0 to 1.5.1
-----------------------------------

- Fixed a bug that could rarely cause a (nearly) infinite loop when recentering fails. (#178)
- Stopped supporting a few old versions of dependencies. (#178)
- Added official support for python 3.13.
