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

import numpy as np
import warnings
import galsim

# This file include both the SizeMagSelect selection type and the SizeMagStats stat type.

from .stats import Stats

class SizeMagStats(Stats):
    """Statistics class that plots a size magnitude diagram to check the quality of
    the star selection.
    """

    def __init__(self, file_name=None, zeropoint=30, logger=None):
        """
        :param file_name:       Name of the file to output to. [default: None]
        :param zeropoint:       Zeropoint to use = the magnitude of flux=1. [default: 30]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.file_name = file_name
        self.zeropoint = zeropoint

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        # measure everything with hsm
        logger.info("Measuring Star and Model Shapes")
        _, star_shapes, psf_shapes = self.measureShapes(psf, stars, logger=logger)

        if hasattr(psf, 'initial_stars'):
            # remove initial objects that ended up being used.
            init_pos = [s.image_pos for s in psf.initial_stars]
            initial_objects = [s for s in psf.initial_objects if s.image_pos not in init_pos]
            star_pos = [s.image_pos for s in stars]
            initial_stars = [s for s in psf.initial_stars if s.image_pos not in star_pos]
            _, obj_shapes, _ = self.measureShapes(None, initial_objects, logger=logger)
            _, init_shapes, _ = self.measureShapes(None, initial_stars, logger=logger)
        else:
            # if didn't make psf using process, then inital fields will be absent.
            # Just make them empty arrays.
            obj_shapes = np.empty(shape=(0,7))
            init_shapes = np.empty(shape=(0,7))

        # Pull out the sizes and fluxes
        flag_star = star_shapes[:, 6]
        mask = flag_star == 0
        self.f_star = star_shapes[mask, 0]
        self.T_star = star_shapes[mask, 3]
        flag_psf = psf_shapes[:, 6]
        mask = flag_psf == 0
        self.f_psf = psf_shapes[mask, 0]
        self.T_psf = psf_shapes[mask, 3]
        flag_obj = obj_shapes[:, 6]
        mask = flag_obj == 0
        self.f_obj = obj_shapes[mask, 0]
        self.T_obj = obj_shapes[mask, 3]
        flag_init = init_shapes[:, 6]
        mask = flag_init == 0
        self.f_init = init_shapes[mask, 0]
        self.T_init = init_shapes[mask, 3]

        # Calculate the magnitudes
        self.m_star = self.zeropoint - 2.5 * np.log10(self.f_star)
        self.m_psf = self.zeropoint - 2.5 * np.log10(self.f_psf)
        self.m_init = self.zeropoint - 2.5 * np.log10(self.f_init)
        self.m_obj = self.zeropoint - 2.5 * np.log10(self.f_obj)

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:  Any additional kwargs go into the matplotlib scatter() function.

        :returns: fig, ax
        """
        from matplotlib.figure import Figure
        logger = galsim.config.LoggerWrapper(logger)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)

        xmin = np.floor(np.min(self.m_star))
        xmin = np.min(self.m_init, initial=xmin)  # Do it this way in case m_init is empty.
        xmax = np.ceil(np.max(self.m_star))
        xmax = np.max(self.m_obj, initial=xmax)   # Likewise, m_obj might be empty.
        ymax = max(np.median(self.T_star)*2, np.max(self.T_star)*1.01)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel('Magnitude (ZP=%s)'%self.zeropoint, fontsize=15)
        ax.set_ylabel(r'$T = 2\sigma^2$ (arcsec${}^2$)', fontsize=15)

        # This looks better if we plot all the objects a bit at a time, rather than all of one type
        # than all of the next, etc.  This makes it easier to see if there are some galaxies in
        # the star area or vice versa.  We implement this by sorting all of them in magnitude
        # and then plotting the indices mod 20.
        group_obj = np.ones_like(self.m_obj) * 1
        group_init = np.ones_like(self.m_init) * 2
        group_star = np.ones_like(self.m_star) * 3
        group_psf = np.ones_like(self.m_psf) * 4

        g = np.concatenate([group_obj, group_init, group_star, group_psf])
        m = np.concatenate([self.m_obj, self.m_init, self.m_star, self.m_psf])
        T = np.concatenate([self.T_obj, self.T_init, self.T_star, self.T_psf])

        index = np.argsort(m)
        n = np.arange(len(m))

        for i in range(20):
            s = index[n%20==i]

            obj = ax.scatter(m[s][g[s] == 1], T[s][g[s] == 1],
                             color='black', marker='o', s=2., **kwargs)
            init = ax.scatter(m[s][g[s] == 2], T[s][g[s] == 2],
                              color='magenta', marker='o', s=8., **kwargs)
            psf = ax.scatter(m[s][g[s] == 3], T[s][g[s] == 3],
                             marker='o', s=50.,
                             facecolors='none', edgecolors='cyan', **kwargs)
            star = ax.scatter(m[s][g[s] == 4], T[s][g[s] == 4],
                              color='green', marker='*', s=40., **kwargs)

        ax.legend([obj, init, star, psf],
              ['Detected Object', 'Candidate Star', 'PSF Star', 'PSF Model'],
              loc='lower left', frameon=True, fontsize=15)

        return fig, ax
