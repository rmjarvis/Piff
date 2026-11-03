# Copyright (c) 2026 by Mike Jarvis and the other collaborators on GitHub at
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

import galsim
import numpy as np
import os
from galsim.wcs import CelestialWCS
from ..util import get_asdf_value


class RomanGWCS(CelestialWCS):
    """Use a Roman GWCS object as a GalSim celestial WCS.

    This really just wraps a ``gwcs.WCS`` instance to work as a ``galsim.CelestialWCS``.
    It requires the gwcs module to be installed to work.

    By default, Piff pixel (1, 1) corresponds to GWCS pixel (0, 0). Local Jacobians use
    GalSim's numerical derivatives.

    The typical usage would be to let the GWCS be automatically loaded from the Roman
    asdf files. But you can also load the WCS separately using an explicit input.wcs field::

        input:
            wcs:
                type: RomanGWCS
                file_name: exposure_cal.asdf
                path: roman.meta.wcs  # This is the default.

    The optional ``path`` specifies the dot-separated path to the GWCS in the ASDF tree.
    The default is roman.meta.wcs. You can also specify a ``dir`` using the normal Piff
    pattern for file access.

    The default origin (``galsim.PositionD(1,1)``) accounts for the different convention
    between gwcs and GalSim.

    :param wcs:     A gwcs.WCS instance, typically read from the Roman image metadata.
    :param origin:  Image position corresponding to GWCS pixel (0, 0).
                    [default: galsim.PositionD(1, 1)]
    """

    def __init__(self, wcs, origin=None):
        self.wcs = wcs
        self._color = None
        self._set_origin(galsim.PositionD(1, 1) if origin is None else origin)

    @property
    def origin(self):
        """Image position corresponding to GWCS pixel (0, 0)."""
        return self._origin

    def _radec(self, x, y, color=None):
        # Stamp rendering may evaluate positions just outside the detector bounding box.
        ra, dec = self.wcs(x, y, with_bounding_box=False)
        return np.deg2rad(ra), np.deg2rad(dec)

    def _xy(self, ra, dec, color=None):
        return self.wcs.invert(np.rad2deg(ra), np.rad2deg(dec), with_bounding_box=False)

    def _newOrigin(self, origin):
        return RomanGWCS(self.wcs, origin=origin)

    def copy(self):
        return RomanGWCS(self.wcs, origin=self.origin)

    def _writeHeader(self, header, bounds):
        raise NotImplementedError('A full GWCS cannot be written into a FITS image header')


class _RomanGWCSBuilder(galsim.config.WCSBuilder):
    def buildWCS(self, config, base, logger):
        import asdf

        kwargs = galsim.config.GetAllParams(
            config, base, req={'file_name': str},
            opt={'dir': str, 'path': str})[0]
        file_name = kwargs['file_name']
        if 'dir' in kwargs:
            file_name = os.path.join(kwargs['dir'], file_name)
        logger.info('Reading Roman GWCS from %s', file_name)
        with asdf.open(file_name) as af:
            wcs = get_asdf_value(af.tree, kwargs.get('path', 'roman.meta.wcs'))
            return RomanGWCS(wcs)


galsim.config.RegisterWCSType('RomanGWCS', _RomanGWCSBuilder())
