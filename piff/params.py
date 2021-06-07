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
.. module:: params
"""

from __future__ import print_function
import numpy as np


class Params(object):
    """ A class to store parameters. Contains names, staring values, bounds,
    as well as fitted results.  Allows setting parameters from floated to fixed or vice-versa, as needed.
    Also yields an array of floating parameters for optimizers, or an array of all parameters as needed.
    """
    def __init__(self):
        self.names = []
        self.initvalues = {}  #TODO: use OrderedDict for old Python?
        self.floating = {}
        self.bounds = {}
        self.values = {}
        self.previousvalues = {}

    def register(self,name,initvalue,bounds=None,floating=True):
        """
        Register parameter

        :param name:            Name of the parameter
        :param initvalue:       Initialization Value
        :param bounds:          List with bounds, [lo,hi]
        :param floating:        True for floating, False for fixed
        """
        self.names.append(name)
        self.initvalues[name] = initvalue
        self.floating[name] = floating
        if bounds:
            self.bounds[name] = bounds
        else:
            self.bounds[name] = [-np.inf,np.inf]
        self.values[name] = initvalue
        self.previousvalues[name] = initvalue

    def fix(self,name):
        """
        Set parameter to fixed

        :param name:            Name of the parameter
        """
        self.floating[name] = False

    def float(self,name):
        """
        Set parameter to floating

        :param name:            Name of the parameter
        """
        self.floating[name] = True

    def setValue(self,name,value):
        """ Set parameter to new value

        :param name:            Name of the parameter
        :param value:           Value
        """
        self.previousvalues[name] = self.values[name]
        self.values[name] = value

    def setValues(self,values):
        """ Store values of all parameters

        :param values:          Array of values
        """
        for i,name in enumerate(self.names):
            self.previousvalues[name] = self.values[name]
            self.values[name] = values[i]

    def setFloatingValues(self,values):
        """ Store values of only floating parameters.

        :param values:          Array of values, including only floating parameters, in order
        """
        j = 0
        for name in self.names:
            if self.floating[name]:
                self.previousvalues[name] = self.values[name]
                self.values[name] = values[j]
                j = j + 1

    def get(self,name):
        """ Get value of parameter

        :return value:          Parameter value
        """
        return self.values[name]

    def getValues(self):
        """ Get values of all parameters

        :return values:          Array of values
        """
        values = np.array(list(self.values.values()))
        return values

    def getFloatingValues(self):
        """ Get values of floating parameters

        :return values:          Array of values
        """
        values = []
        for name in self.names:
            if self.floating[name]:
                values.append(self.values[name])
        return np.array(values)

    def getFloatingBounds(self):
        """ Get bounds of floating parameters

        :return bounds:          Tuple of lo,hi bound Arrays
        """
        lo = []
        hi = []
        for name in self.names:
            if self.floating[name]:
                lo.append(self.bounds[name][0])
                hi.append(self.bounds[name][1])

        bounds = (np.array(lo),np.array(hi))
        return bounds

    def getChanges(self):
        """ Get dictionary with changed parameters

        :return changes:          Dictionary with parameters and values
        """
        changes = {}
        for name in self.names:
            if self.values[name] != self.previousvalues[name]:
                changes[name] = self.values[name] - self.previousvalues[name]
        return changes

    def print(self):
        print(self.values)

    def printChanges(self):
        changes = self.getChanges()
        print(changes)
