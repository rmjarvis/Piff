Changes from version 1.0 to 1.1
===============================

Dependency Changes
------------------

Piff no longer supports Python 2.  We currently support Python 3.6 through 3.9.

New features
------------

The primary new feature in this release is the ability to include additional
properties in the interpolation aside from the usual u,v interpolation.
Technically, this was supposed to be possible before, but it was not tested
and consequently was buggy.  We fixed those bugs and simplified the interface.
The use case we have in mind for this feature is to add color dependence to the
PSF model, although the implementation is generic, so you can use any column
in the input file that you want.

It is possible to use extra interpolation properties for BasisPolynomial,
KNNInterpolant, and GPInterpolant, although BasisPolynomial has a substantial
unit test of the feature so far.  (If you find bugs using this feature for the
other two, please let us know with an issue on GitHub.)

The implementation comes in two parts.  First, in the input section of the config
file, you should add the item ``property_cols`` to specify which additional column
names should be read in and saved into the property list of each star.  E.g.
to add a g-i color (say), you might add the line::

    property_cols: ['color_gi']

Second, in the specification of the interpolant, you should explicitly specify
the ``keys`` to use for the interpolation.  E.g. to include the above g-i color
along with the usual u and v positions, you would specify::

    keys: ['u', 'v', 'color_gi']

Finally, when using the PSF solution, you need to provide the values of any
additional properties to the draw command, so it can interpolate properly.::

    >>> image = psf.draw(x=x, y=y, color_gi=color_gi, ...)
