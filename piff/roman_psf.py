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

"""
.. module:: roman_psf
"""

import galsim
import galsim.roman
import numpy as np

from .model import Model
from .mean_interp import Mean
from .simplepsf import SimplePSF
from .star import Star


class Roman(Model):
    """Model a Roman PSF using GalSim's built-in Roman optical model.

    For now, this uses the canonical GalSim Roman model and allows an optional vector of
    ``extra_aberrations`` values to be carried through the fit machinery. The current fit step is
    deliberately conservative: it treats these parameters as fixed and reports only the resulting
    residual statistics. This is enough to use the Roman optical model as a fixed leading component
    in a composite PSF while we develop a true aberration fitter.
    """

    _type_name = 'Roman'
    _method = 'auto'
    _centered = False
    _model_can_be_offset = False

    def __init__(self, filter, chromatic=True, max_zernike=22, logger=None):
        self.logger = logger
        self.filter = filter
        self.chromatic = chromatic
        self.max_zernike = int(max_zernike)
        self.set_num(None)

        if self.max_zernike < 4 or self.max_zernike > 22:
            raise ValueError("max_zernike must be in the range 4..22")

        bandpasses = galsim.roman.getBandpasses()
        if self.filter not in bandpasses:
            raise ValueError("Roman filter %r is not a valid GalSim Roman bandpass" % self.filter)
        self.bandpass = bandpasses[self.filter]
        self.kwargs = {
            'filter': self.filter,
            'chromatic': self.chromatic,
            'max_zernike': self.max_zernike,
        }

    @property
    def param_len(self):
        return self.max_zernike - 3

    def initialize(self, star, logger=None, default_init=None):
        params = np.zeros(self.param_len, dtype=float)
        params_var = np.zeros_like(params)
        fit = star.fit.newParams(params, params_var=params_var, num=self._num)
        return Star(star.data, fit)

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        image = star.image
        weight = star.weight
        model_image = self.draw(star).image
        chisq = np.std(image.array - model_image.array)
        dof = np.count_nonzero(weight.array) - 3

        params = star.fit.get_params(self._num)
        var = np.zeros_like(params)
        return Star(
            star.data,
            star.fit.newParams(params, params_var=var, num=self._num, chisq=chisq, dof=dof),
        )

    def draw(self, star, copy_image=True):
        params = star.fit.get_params(self._num)
        prof = self.getProfile(params, star=star).shift(star.fit.center) * star.fit.flux
        image = star.image.copy() if copy_image else star.image
        if self.chromatic:
            prof.drawImage(
                image,
                bandpass=self.bandpass,
                method=self._method,
                center=star.image_pos,
            )
        else:
            prof.drawImage(image, method=self._method, center=star.image_pos)
        return Star(star.data.withNew(image=image), star.fit)

    def getProfile(self, params=None, star=None):
        if star is None:
            raise ValueError("Roman.getProfile requires the star argument")
        if params is None:
            params = np.zeros(self.param_len, dtype=float)
        extra_aberrations = np.array(params, dtype=float)
        wavelength = None if self.chromatic else self.bandpass.effective_wavelength
        return galsim.roman.getPSF(
            self._get_sca(star),
            self.filter,
            SCA_pos=star.image_pos,
            wcs=star.data.local_wcs,
            extra_aberrations=extra_aberrations,
            wavelength=wavelength,
        )

    @staticmethod
    def _get_sca(star):
        if 'sca' in star.data.properties:
            return int(star.data.properties['sca'])
        if 'chipnum' in star.data.properties:
            return int(star.data.properties['chipnum'])
        raise ValueError("RomanOptics requires an explicit 'sca' property for each star")


class RomanOptics(SimplePSF):
    """A convenience PSF wrapper for using the Roman optical model with constant parameters."""

    _type_name = 'RomanOptics'

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        kwargs = {}
        kwargs.update(config_psf)
        kwargs.pop('type', None)

        model = Roman(logger=logger, **kwargs)
        interp = Mean()

        return {
            'model': model,
            'interp': interp,
        }

    def _getRawProfile(self, star):
        params = star.fit.get_params(self._num)
        return self.model.getProfile(params, star=star), self.model._method
