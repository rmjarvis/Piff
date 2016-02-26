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

from .interp import Interp

class Mean(Interp):
    """The simplest possible interpolation scheme.  It just finds the mean of the data
    and uses that at every position.
    """
    def __init__(self):
        pass

    def fitData(self, data, pos):
        """Fit for the interpolation coefficients given some data.

        Here the "fit" is just the mean.

        :param data:        A list of lists of data vectors (numpy arrays) for each star
        :param pos:         A list of lists of positions of the stars
        """
        import numpy
        self.mean = numpy.mean(data, axis=(0,1))

    def getParameters(self):
        return [self.mean]

    def interpolate(self, image_num, pos):
        """Perform the interpolation to find the interpolated data vector at some position in
        some image.

        :param image_num:   The index of the image in the original list of data vectors.
        :param pos:         The position to which to interpolate.

        :returns: the data vector (a numpy array) interpolated to the given position.
        """
        return self.mean
