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

import numpy as np
import fitsio

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
        locations = np.array([data[attr] for attr in attr_interp]).T
        targets = np.array([data[attr] for attr in attr_target]).T

        self.build(attr_interp, range(0, self.z_max - self.z_min + 1), logger=logger)
        self._fit(locations, targets)

        # set misalignment as [[delta_i, thetax_i, thetay_i]] with i == 0 corresponding to defocus
        self.misalignment = np.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))

        # to get the ccd coords
        attr_save = ['x', 'y', 'ccdnum']
        Xpixel = np.array([data[attr] for attr in attr_save]).T
        self.Xpixel = Xpixel

    def misalign_wavefront(self, misalignment):
        """Pass along misalignment parameter

        :param misalignment:    Parameters for misaligning zernike coefficients
        """
        # if dictionary, translate terms to array
        if type(misalignment) == dict:
            nu_misalignment = np.array([[0.0, 0.0, 0.0]] * (self.z_max - self.z_min + 1))
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
        if np.shape(targets) == ():
            # if no y, then interpolate
            targets = np.array([self.knn[key].predict(locations) for key in self.attr_target]).T
        if logger:
            logger.debug('Regression shape: %s', targets.shape)
        # add misalignment shape (n_targets, 3)
        # locations is (n_samples, 2)
        # targets is (n_samples, n_targets)
        targets = targets + self.misalignment[np.newaxis, :, 0] \
                + locations[:, 1, np.newaxis] * self.misalignment[np.newaxis, :, 1] \
                + locations[:, 0, np.newaxis] * self.misalignment[np.newaxis, :, 2]

        return targets

    def _finish_write(self, fits, extname):
        """Write the solution to a FITS binary table.

        Save the knn params and the locations and targets arrays

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension with the interp information.
        """

        dtypes = [('LOCATIONS', self.locations.dtype, self.locations.shape),
                  ('TARGETS', self.targets.dtype, self.targets.shape),
                  ('XPIXEL', self.Xpixel.dtype, self.Xpixel.shape),
                  ('ATTR_TARGET', self.attr_target.dtype, self.attr_target.shape),
                  ('ATTR_INTERP', self.attr_interp.dtype, self.attr_interp.shape),
                  ('MISALIGNMENT', self.misalignment.dtype, self.misalignment.shape)]
        data = np.empty(1, dtype=dtypes)
        # assign
        data['LOCATIONS'] = self.locations
        data['TARGETS'] = self.targets
        data['XPIXEL'] = self.Xpixel
        data['ATTR_TARGET'] = self.attr_target
        data['ATTR_INTERP'] = self.attr_interp
        data['MISALIGNMENT'] = self.misalignment

        # write to fits
        fits.write_table(data, extname=extname + '_solution')

    def _finish_read(self, fits, extname):
        """Read the solution from a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        data = fits[extname + '_solution'].read()

        # attr_target and attr_interp assigned in build
        self.build(data['ATTR_INTERP'][0], data['ATTR_TARGET'][0])
        # self.locations and self.targets assigned in _fit
        self._fit(data['LOCATIONS'][0], data['TARGETS'][0])
        self.misalign_wavefront(data['MISALIGNMENT'][0])

        # other attributes
        self.Xpixel = data['XPIXEL'][0]
