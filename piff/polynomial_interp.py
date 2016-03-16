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
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: interp_mean
"""

from __future__ import print_function

from .interp import Interp

class Polynomial(Interp):
    """
    Uses the scipy curve_fit command to fit a polynomial surface
    """
    def __init__(self, order):
        self.order=order
    
    def _fit_function(self, xy, *coeffs):
        # x is the data values - the coordinates of the stars
        # shape (2,nstar)
        import numpy
        x = xy[0]
        y = xy[1]
        px = coeffs[:self.order]
        py = coeffs[self.order:]
        f = numpy.polyval(px, x) * numpy.polyval(py, y)
        return f

    def fitData(self, data, pos):
        """Fit for the interpolation coefficients given some data.

        Here the "fit" is just the mean.

        :param data:        A list of lists of data vectors (numpy arrays) for each star
        :param pos:         A list of lists of positions of the stars
        """
        import numpy
        import scipy.optimize
        data = numpy.array(data).T
        P = []
        X = numpy.array([p.x for p in pos])
        Y = numpy.array([p.y for p in pos])
        positions = numpy.array([X,Y])
       
        #We fit a separate polynomial in each parameter
        for parameter in data:
            p0 = numpy.zeros(self.order*2)
            p0[0] = parameter.mean()
            p0[self.order] = parameter.mean()
            p,covmat=scipy.optimize.curve_fit(self._fit_function, positions, parameter, p0)
            P.append(p)
        self.coeffs = numpy.concatenate(P)

    def getParameters(self):
        return self.coeffs

    def interpolate(self, image_num, pos):
        raise ValueError("Not done this bit yet")
