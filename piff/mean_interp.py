# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: mean_interp
"""

from __future__ import print_function
import numpy

from .interp import Interp
from .starfit import Star, StarFit

class Mean(Interp):
    """The simplest possible interpolation scheme.  It just finds the mean of the parameter
    vectors and uses that at every position.
    """
    def __init__(self):
        self.degenerate_points = False
        self.kwargs = {}
        self.mean = None

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data.

        Here the "solution" is just the mean.

        :param stars:       A list of stars with fitted parameters to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.mean = numpy.mean([star.fit.params for star in stars], axis=0)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        if star.fit is None:
            fit = StarFit(self.mean)
        else:
            fit = star.fit.newParams(p)
        return Star(star.data, fit)

    def writeSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interpolator information.
        """
        cols = [ self.mean ]
        dtypes = [ ('mean', float) ]
        data = numpy.array(zip(*cols), dtype=dtypes)
        fits.write_table(data, extname=extname)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interpolator information.
        """
        data = fits[extname].read()
        self.mean = data['mean']

