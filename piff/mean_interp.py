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
.. module:: interp_mean
"""

from __future__ import print_function
import numpy

from .interp import Interp

class Mean(Interp):
    """The simplest possible interpolation scheme.  It just finds the mean of the parameter
    vectors and uses that at every position.
    """
    def __init__(self):
        self.kwargs = {}

    def solve(self, pos, vectors, logger=None):
        """Solve for the interpolation coefficients given some data.

        Here the "solution" is just the mean.

        :param pos:         A list of positions to use for the interpolation.
        :param vectors:     A list of parameter vectors (numpy arrays) for each star.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.mean = numpy.mean(vectors, axis=0)

    def interpolate(self, pos, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param pos:         The position to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the parameter vector (a numpy array) interpolated to the given position.
        """
        return self.mean

    def writeSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        cols = [ self.mean ]
        dtypes = [ ('mean', float) ]
        data = numpy.array(zip(*cols), dtype=dtypes)
        fits.write_table(data, extname=extname)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        data = fits[extname].read()
        self.mean = data['mean']

