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

import numpy as np

from .interp import Interp
from .star import Star

class Mean(Interp):
    """The simplest possible interpolation scheme.  It just finds the mean of the parameter
    vectors and uses that at every position.

    Use type name "Mean" in a config field to use this interpolant.

    """
    _type_name = 'Mean'

    def __init__(self, logger=None):
        self.degenerate_points = False
        self.kwargs = {}
        self.mean = None
        self.set_num(None)

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data.

        Here the "solution" is just the mean.

        :param stars:       A list of stars with fitted parameters to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.mean = np.mean([star.fit.get_params(self._num) for star in stars], axis=0)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance holding the interpolated parameters
        """
        if self.mean is None:
            return star
        else:
            fit = star.fit.newParams(self.mean, num=self._num)
        return Star(star.data, fit)

    def _finish_write(self, writer):
        """Write the solution.

        :param writer:      A writer object that encapsulates the serialization format.
        """
        cols = [ self.mean ]
        dtypes = [ ('mean', float) ]
        data = np.array(list(zip(*cols)), dtype=dtypes)
        writer.write_table('solution', data)

    def _finish_read(self, reader):
        """Read the solution.

        :param reader:      A reader object that encapsulates the serialization format.
        """
        data = reader.read_table('solution')
        assert data is not None
        self.mean = data['mean']
