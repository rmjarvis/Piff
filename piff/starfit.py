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
.. module:: model
"""

from __future__ import print_function
import numpy

class Star(object):
    """Structure that links the data for a star to the results of fitting it.
    **This class and its two members are expected to be invariant outside of the
    code that creates them.**

    We will expect it to have the following attributes:
    - data:   a StarData instance holding all information on the star coming from observation
    - fit:    a StarFit instance holding all information or useful intermediates from the fitting
              process.
    """
    def __init__(self, data, fit):
        """Constructor for Star instance.

        :param data: A StarData instance (invariant)
        :param fit:  A StarFit instance (invariant)
        """
        self.data = data
        self.fit = fit
        return

class StarFit(object):
    """Class to hold the results of fitting a Model to some StarData, or specify
    the PSF interpolated to an unmeasured location.

    **Class is intended to be invariant once created.**

    This class can be extended
    to carry information of use to a given Model instance (such as intermediate
    results), but interpolators will be looking for some subset of these properties:
    
    -params:      numpy vector of parameters of the PSF that apply to this star
    -flux:        flux of the star
    -center:      (u,v) tuple giving position of stellar center (relative
                  to data.image_pos)
    -chisq:       Chi-squared of  fit to the data (if any) with current params
    -dof:         Degrees of freedom in the fit (will depend on whether last fit had
                  parameters free or just the flux/center).
    -alpha, beta: matrix, vector, giving Taylor expansion of chisq wrt params about
                  their current values. The alpha matrix also is the inverse covariance
                  matrix of the params.

    The params and alpha,beta,chisq are assumed to be marginalized over flux (and over center,
    if it is free to vary).
    """
    def __init__(self, params, flux=1., center=(0.,0.), alpha=None, beta=None, chisq=None, dof=None):
        """Constructor for base version of StarFit

        :param params: A 1d numpy array holding estimated PSF parameters
        :param flux:   Estimated flux for this star
        :param center: Estimated or fixed center position (u,v) of this star relative to
                       the StarData.image_pos reference point.
        :param alpha:  Quadratic dependence of chi-squared on params about current values
        :param beta:   Linear dependence of chi-squared on params about current values
        :param chisq:  chi-squared value at current parameters.
        """
        
        self.params = params
        self.flux = flux
        self.center = center
        self.alpha = alpha
        self.beta = beta
        self.chisq = chisq
        self.dof = dof
        return

    def newParams(self, p):
        """Return new StarFit that has the array p installed as new parameters.

        :param params:  A 1d array holding new parameters; must match size of current ones

        :returns:  New StarFit object with altered parameters.  All chisq-related parameters
                   are set to None since they are no longer valid.
        """
        npp = numpy.array(p)
        if not npp.shape==self.params.shape:
            raise TypeError('new StarFit parameters do not match dimensions of old ones')
        return StarFit(npp, flux=self.flux, center=self.center)
    
