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
.. module:: decam_wavefront
"""

from ..knn_interp import kNNInterp
from .decaminfo import DECamInfo

import numpy as np
import fitsio

class DECamWavefront(kNNInterp):
    """
    An interpolator of the DECam Wavefront as measured by out-of-focus stars.
    If you specify the location of the fits file and the extension, this will
    take care of the rest for you.
    """

    def __init__(self, file_name, extname, n_neighbors=15, weights='uniform', algorithm='auto',
                 p=2, logger=None):
        """Load up a fits file containing the optics model and use it to build the wavefront
        interpolator

        :param file_name:   Fits file containing the wavefront
        :param extname:     Extension name
        :param n_neighbors: Number of neighbors used for interpolation. [default: 15]
        :param weights:     Weight function used in prediction. Possible values are 'uniform',
                            'distance', and a callable function which accepts an array of distances
                            and returns an array of the same shape containing the weights.
                            [default: 'uniform']
        :param algorithm:   Algorithm used to compute nearest neighbors. Possible values are
                            'ball_tree', 'kd_tree', 'brute', and 'auto', which tries to determine
                            the best choice. [default: 'auto']
        :param p:           Power parameter of distance metrice. p=2 is default euclidean distance,
                            p=1 is manhattan. [default: 2]
        :param logger:      A logger object for logging debug info. [default: None]
        """

        self.z_min = 4
        self.z_max = 11
        # these were kwargs in knn interp, but are no longer kwargs because they are fixed by our
        # wavefront model!
        self.keys = ['focal_x', 'focal_y']
        self.attr_target_wavefront = ['z{0}'.format(zi) for zi in range(self.z_min, self.z_max + 1)]

        self.kwargs = {
            'file_name': file_name,
            'extname': extname,
            }

        self.knr_kwargs = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'p': p,
            }
        self.kwargs.update(self.knr_kwargs)

        self.decaminfo = DECamInfo()

        from sklearn.neighbors import KNeighborsRegressor
        self.knn = KNeighborsRegressor(**self.knr_kwargs)
        if logger:
            logger.debug("Made regressor")

        #print("file_name 2: {0}".format(file_name))
        fits = fitsio.FITS(file_name)
        if logger:
            logger.debug("Made fits")
            logger.debug(fits)
            logger.debug(fits[1])
        data = fits[extname].read()
        if logger:
            logger.debug("read data from fits file")
        locations = np.array([data[attr] for attr in self.keys]).T
        if logger:
            logger.debug("locations shape = %s",locations.shape)
        targets = np.array([data[attr] for attr in self.attr_target_wavefront]).T
        if logger:
            logger.debug("targets shape = %s",targets.shape)

        self._fit(locations, targets)
        if logger:
            logger.debug("done fit")

        # set misalignment as [[delta_i, thetax_i, thetay_i]] with i == 0 corresponding to defocus
        self.misalignment = np.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))
        if logger:
            logger.debug("misalignement shape = %s",self.misalignment.shape)

        # to get the ccd coords
        attr_save = ['x', 'y', 'ccdnum']
        Xpixel = np.array([data[attr] for attr in attr_save]).T
        self.Xpixel = Xpixel
        if logger:
            logger.debug("Xpixel shape = %s",self.Xpixel.shape)

    def misalign_wavefront(self, misalignment):
        """Pass along misalignment parameter

        :param misalignment:    Parameters for misaligning zernike coefficients
        """
        # if dictionary, translate terms to array
        if type(misalignment) == dict:
            nu_misalignment = np.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))
            for zi in range(self.z_min, self.z_max + 1):
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
        if hasattr(self, 'misalignment') and misalignment.shape != self.misalignment.shape:
            raise ValueError("New misalignment shape must match old!")
        self.misalignment = misalignment

    def _predict(self, locations, targets=None, logger=None):
        """Predict from knn.

        :param locations:   The locations for interpolating. (n_samples, n_features). In sklearn
                            parlance, this is 'X'
        :param targets:     The target values. (n_samples, n_targets). In sklearn parlance, this is
                            'y'. If given, then only apply misalignment

        :returns:   Regressed parameters targets (n_samples, n_targets)
        """
        if np.shape(targets) == ():
            # if no y, then interpolate
            targets = self.knn.predict(locations)
        if logger:
            logger.debug('Regression shape: %s', targets.shape)
        # add misalignment shape (n_targets, 3)
        # locations is (n_samples, 2)
        # targets is (n_samples, n_targets)
        targets = targets + self.misalignment[np.newaxis, :, 0] \
                + locations[:, 1, np.newaxis] * self.misalignment[np.newaxis, :, 1] \
                + locations[:, 0, np.newaxis] * self.misalignment[np.newaxis, :, 2]

        return targets

    def _finish_write(self, fits, extname, logger=None):
        """Write the solution to a FITS binary table.

        Save the knn params and the locations and targets arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension with the interp information.
        """

        dtypes = [('LOCATIONS', self.locations.dtype, self.locations.shape),
                  ('TARGETS', self.targets.dtype, self.targets.shape),
                  ('XPIXEL', self.Xpixel.dtype, self.Xpixel.shape),
                  ('MISALIGNMENT', self.misalignment.dtype, self.misalignment.shape)]
        data = np.empty(1, dtype=dtypes)
        # assign
        data['LOCATIONS'] = self.locations
        data['TARGETS'] = self.targets
        data['XPIXEL'] = self.Xpixel
        data['MISALIGNMENT'] = self.misalignment

        # write to fits
        fits.write_table(data, extname=extname + '_solution')

    def _finish_read(self, fits, extname, logger=None):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        data = fits[extname + '_solution'].read()

        # self.locations and self.targets assigned in _fit
        self._fit(data['LOCATIONS'][0], data['TARGETS'][0])
        self.misalign_wavefront(data['MISALIGNMENT'][0])

        # other attributes
        self.Xpixel = data['XPIXEL'][0]

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position. First makes sure stars are in focal basis with call to decaminfo, and then super's to knn interpolate

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        return super(DECamWavefront, self).interpolate(self.decaminfo.pixel_to_focal(star), logger=logger)

    def interpolateList(self, star_list, logger=None):
        """Perform the interpolation for a list of stars. First makes sure stars are in focal basis, and then super's to knn interpolate

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """
        return super(DECamWavefront, self).interpolateList(self.decaminfo.pixel_to_focalList(star_list), logger=logger)
