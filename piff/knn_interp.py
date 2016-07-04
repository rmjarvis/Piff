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
.. module:: knn_interp
"""

from __future__ import print_function

from .interp import Interp
from .starfit import Star, StarFit
from .stardata import StarData

import numpy
import fitsio


class kNNInterp(Interp):
    """
    An interpolator that uses sklearn KNeighborsRegressor to interpolate a
    single surface
    """
    def __init__(self, n_neighbors=15, weights='uniform', algorithm='auto',
                 p=2,logger=None, **kwargs):
        """Create the kNN interpolator

        :param n_neighbors: Number of neighbors used for interpolation. [default: 15]
        :param weights:     Weight function used in prediction. Possible values are 'uniform', 'distance', and a callable function which accepts an array of distances and returns an array of the same shape containing the weights. [default: 'uniform']
        :param algorithm:   Algorithm used to compute nearest neighbors. Possible values are 'ball_tree', 'kd_tree', 'brute', and 'auto', which tries to determine the best choice. [default: 'auto']
        :param p:           Power parameter of distance metrice. p=2 is default euclidean distance, p=1 is manhattan. [default: 2]
        :param logger:      A logger object for logging debug info. [default: None]
        """

        self.kwargs = {}
        self.kwargs.update(kwargs)

        self.knr_kwargs = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'p': p,
            }
        self.knn = {}

    def build(self, attr_interp, attr_target, logger=None):
        """Create the kNN interpolator by passing attributes to KNeighborsRegressor

        :param attr_interp: A list of star attributes to interpolate from
        :param attr_target: A list of star attributes to interpolate to
        :param logger:      A logger object for logging debug info. [default: None]
        """
        from sklearn.neighbors import KNeighborsRegressor
        self.attr_interp = numpy.array(attr_interp)
        self.attr_target = numpy.array(attr_target)
        for target in self.attr_target:
            self.knn[target] = KNeighborsRegressor(**self.knr_kwargs)

    def _fit(self, locations, targets, logger=None):
        """Update the Neighbors Regressor with data

        :param locations:   The locations for interpolating. (n_samples, n_features). In sklearn parlance, this is 'X'
        :param targets:   The target values. (n_samples, n_targets). In sklearn parlance, this is 'y'
        """
        for key, yi in zip(self.attr_target, targets.T):
            self.knn[key].fit(locations, yi)
        if logger:
            logger.debug('knn updated to keys: %s', self.knn.keys())
        self.locations = locations
        if logger:
            logger.debug('locations updated to shape: %s', self.locations.shape)
        self.targets = targets
        if logger:
            logger.debug('targets updated to shape: %s', self.targets.shape)

    def _predict(self, locations, logger=None):
        """Predict from knn.

        :param locations:   The locations for interpolating. (n_samples, n_features). In sklearn parlance, this is 'X'

        :returns:   Regressed parameters y (n_samples, n_targets)
        """
        regression = numpy.array([self.knn[key].predict(locations) for key in self.attr_target]).T
        if logger:
            logger.debug('Regression shape: %s', regression.shape)
        return regression

    def getProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the independent variables for the
        interpolation.

        Take self.attr_interp from star.data

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A numpy vector of these properties.
        """
        return numpy.array([star.data[key] for key in self.attr_interp])

    def getFitProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the dependent variables for the
        interpolation.

        Take self.attr_target from star.fit

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A numpy vector of these properties.
        """
        return numpy.array([star.fit[key] for key in self.attr_target])

    def initialize(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.solve(star_list, logger=logger)

    def solve(self, star_list, logger=None):
        """Solve for the interpolation coefficients given stars and attributes

        :param star_list:   A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        locations = numpy.array([self.getProperties(star) for star in star_list])
        targets = numpy.array([self.getFitProperties(star) for star in star_list])
        self._fit(locations, targets)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position. Calls interpolateList because sklearn prefers list input anyways

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        # because of sklearn formatting, call interpolateList and take 0th entry
        return self.interpolateList([star], logger=logger)[0]

    def interpolateList(self, star_list, logger=None):
        """Perform the interpolation for a list of stars.

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """

        locations = numpy.array([self.getProperties(star) for star in star_list])
        # targets = numpy.array([self.getFitProperties(star) for star in star_list])
        targets = self._predict(locations)
        star_list_fitted = []
        for yi, star in zip(targets, star_list):
            if star.fit is None:
                fit = StarFit(yi)
            else:
                try:
                    fit = star.fit.newParams(yi)
                except TypeError:
                    if logger:
                        logger.info('Warning, stars interpolated to fewer params! %s', len(fit))
                    fit = StarFit(yi)
            star_list_fitted.append(Star(star.data, fit))
        return star_list_fitted

    def writeSolution(self, fits, extname):
        """Write the solution to a FITS binary table.

        Save the knn params and the locations and targets arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        dtypes = [('LOCATIONS', self.locations.dtype, self.locations.shape), ('TARGETS', self.targets.dtype, self.targets.shape),
                  ('ATTR_TARGET', self.attr_target.dtype, self.attr_target.shape),
                  ('ATTR_INTERP', self.attr_interp.dtype, self.attr_interp.shape),]
        data = numpy.empty(1, dtype=dtypes)
        # assign
        data['LOCATIONS'] = self.locations
        data['TARGETS'] = self.targets
        data['ATTR_TARGET'] = self.attr_target
        data['ATTR_INTERP'] = self.attr_interp

        # put the knn params in the header?
        header = self.kwargs

        # write to fits
        fits.write_table(data, extname=extname, header=header)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        The extension should contain the same values as are saved
        in the writeSolution method.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        # header = fits[extname].read_header()
        data = fits[extname].read()

        # kwargs come from base class read()
        # self.kwargs = header

        # attr_target and attr_interp assigned in build
        self.build(data['ATTR_INTERP'][0], data['ATTR_TARGET'][0])
        # self.locations and self.targets assigned in _fit
        self._fit(data['LOCATIONS'][0], data['TARGETS'][0])

class DECamWavefront(kNNInterp):
    """
    An interpolator of the DECam Wavefront as measured by out-of-focus stars.
    If you specify the location of the fits file and the extension, this will
    take care of the rest for you.
    """

    def load_wavefront(self, file_name, extname, logger=None):
        """Load up a fits file containing the optics model and use it to build the wavefront interpolator
        :param file_name:   Fits file containing the wavefront
        :param extname:     Extension name
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.z_min = 4
        self.z_max = 11
        fits = fitsio.FITS(file_name)
        data = fits[extname].read()
        attr_interp = ['focal_x', 'focal_y']
        attr_target = ['z{0}'.format(zi) for zi in xrange(self.z_min, self.z_max + 1)]
        locations = numpy.array([data[attr] for attr in attr_interp]).T
        targets = numpy.array([data[attr] for attr in attr_target]).T

        self.build(attr_interp, range(0, self.z_max - self.z_min + 1), logger=logger)
        self._fit(locations, targets)

        # set misalignment as [[delta_i, thetax_i, thetay_i]] with i == 0 corresponding to defocus
        self.misalignment = numpy.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))

        # to get the ccd coords
        attr_save = ['x', 'y', 'ccdnum']
        Xpixel = numpy.array([data[attr] for attr in attr_save]).T
        self.Xpixel = Xpixel

    def misalign_wavefront(self, misalignment):
        """Pass along misalignment parameter

        :param misalignment:    Parameters for misaligning zernike coefficients
        """
        # if dictionary, translate terms to array
        if type(misalignment) == dict:
            nu_misalignment = numpy.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))
            for zi in xrange(self.z_min, self.z_max + 1):
                indx = zi - 4
                # delta
                key = 'z{0:02}d'.format(zi)
                if key in misalignment:
                    nu_misalignment[indx, 0] = misalignment[key]
                # thetax
                key = 'z{0:02}x'.format(zi)
                if key in misalignment:
                    nu_misalignment[indx, 1] = misalignment[key]
                # thetay
                key = 'z{0:02}y'.format(zi)
                if key in misalignment:
                    nu_misalignment[indx, 2] = misalignment[key]
            misalignment = nu_misalignment
        # if array, check that shape is right, and put it in
        if hasattr(self, 'misalignment'):
            assert misalignment.shape == self.misalignment.shape,"New misalignment shape must match old!"
        self.misalignment = misalignment

    def _predict(self, locations, targets=None, logger=None):
        """Predict from knn.

        :param locations:   The locations for interpolating. (n_samples, n_features). In sklearn parlance, this is 'X'
        :param targets:   The target values. (n_samples, n_targets). In sklearn parlance, this is 'y'. If given, then only apply misalignment

        :returns:   Regressed parameters targets (n_samples, n_targets)
        """
        if numpy.shape(targets) == ():
            # if no y, then interpolate
            targets = numpy.array([self.knn[key].predict(locations) for key in self.attr_target]).T
        if logger:
            logger.debug('Regression shape: %s', targets.shape)
        # add misalignment shape (n_targets, 3)
        # locations is (n_samples, 2)
        # targets is (n_samples, n_targets)
        targets = targets + self.misalignment[numpy.newaxis, :, 0] \
                + locations[:, 1, numpy.newaxis] * self.misalignment[numpy.newaxis, :, 1] \
                + locations[:, 0, numpy.newaxis] * self.misalignment[numpy.newaxis, :, 2]

        return targets

    def writeSolution(self, fits, extname):
        """Write the solution to a FITS binary table.

        Save the knn params and the X and y arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        dtypes = [('LOCATIONS', self.locations.dtype, self.locations.shape),
                  ('TARGETS', self.targets.dtype, self.targets.shape),
                  ('XPIXEL', self.Xpixel.dtype, self.Xpixel.shape),
                  ('ATTR_TARGET', self.attr_target.dtype, self.attr_target.shape),
                  ('ATTR_INTERP', self.attr_interp.dtype, self.attr_interp.shape),
                  ('MISALIGNMENT', self.misalignment.dtype, self.misalignment.shape)]
        data = numpy.empty(1, dtype=dtypes)
        # assign
        data['LOCATIONS'] = self.locations
        data['TARGETS'] = self.targets
        data['XPIXEL'] = self.Xpixel
        data['ATTR_TARGET'] = self.attr_target
        data['ATTR_INTERP'] = self.attr_interp
        data['MISALIGNMENT'] = self.misalignment

        # put the knn params in the header?
        header = self.kwargs

        # write to fits
        fits.write_table(data, extname=extname, header=header)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        The extension should contain the same values as are saved
        in the writeSolution method.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        # header = fits[extname].read_header()
        data = fits[extname].read()

        # kwargs come from base class read()
        # self.kwargs = header

        # attr_target and attr_interp assigned in build
        self.build(data['ATTR_INTERP'][0], data['ATTR_TARGET'][0])
        # self.locations and self.targets assigned in _fit
        self._fit(data['LOCATIONS'][0], data['TARGETS'][0])
        self.misalign_wavefront(data['MISALIGNMENT'][0])

        # other attributes
        self.Xpixel = data['XPIXEL'][0]


# pretty sure the correct way to do this is in the WCS framework, but let's
# move forward first:
def pixel_to_focal(stardata):
    """Take stardata and add focal plane position to properties

    :param stardata:    The stardata with property 'ccdnum'

    :returns stardata:  New stardata with updated properties
    """
    from .decamutil import decaminfo
    # stardata needs to have ccdnum as a property!
    focal_x, focal_y = decaminfo().getPosition_extnum([stardata['ccdnum']], [stardata['x']], [stardata['y']])
    properties = stardata.properties.copy()
    properties['focal_x'] = focal_x[0]
    properties['focal_y'] = focal_y[0]
    for key in ['x', 'y', 'u', 'v']:
        # Get rid of keys that constructor doesn't want to see:
        properties.pop(key,None)
    return StarData(image=stardata.image,
                    image_pos=stardata.image_pos,
                    weight=stardata.weight,
                    pointing=stardata.pointing,
                    values_are_sb=stardata.values_are_sb,
                    properties=properties)
