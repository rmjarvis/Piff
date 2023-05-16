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
.. module:: star_stats

"""

import numpy as np
import galsim

from .stats import Stats
from .star import Star

class StarStats(Stats):
    """This Stats class can take stars and make a set of plots of them as
    well as their models and residuals.

    By default this will draw 5 random stars, make psf stars, and plot the
    residual of the two.

    After a call to :func:`compute`, the following attributes are accessible:

        :stars:         List of stars used for plotting
        :models:        List of models of stars used for plotting
        :indices:       Indices of input stars that the plotting stars correspond to

    :param nplot:           Number of stars we wish to plot. If 0 or nplot > nstars in PSF,
                            then we plot all stars. Otherwise, we draw nplot stars at random
                            (without replacement). [default: 10]
    :param adjust_stars:    Boolean. If true, when computing, will also fit for best
                            starfit center and flux to match observed star. [default: False]
    :param file_name:       Name of the file to output to. [default: None]
    :param logger:          A logger object for logging debug info. [default: None]
    """
    _type_name = 'StarImages'

    def __init__(self, nplot=10, adjust_stars=False, file_name=None, logger=None):
        self.nplot = nplot
        self.file_name = file_name
        self.adjust_stars = adjust_stars

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        if self.nplot == 0 or self.nplot >= len(stars):
            # select all stars
            self.indices = np.arange(len(stars))
        else:
            self.indices = np.random.choice(len(stars), self.nplot, replace=False)

        logger.info("Making {0} Model Stars".format(len(self.indices)))
        self.stars = []
        for index in self.indices:
            star = stars[index]
            if self.adjust_stars:
                # Do 2 passes, since we sometimes start pretty far from the right values.
                star = psf.reflux(star, logger=logger)
                star = psf.reflux(star, logger=logger)
            self.stars.append(star)
        self.models = psf.drawStarList(self.stars)

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \*\*kwargs: Any additional kwargs go into the matplotlib pcolor() function.

        :returns: fig, ax
        """
        from matplotlib.figure import Figure
        if not hasattr(self, 'indices'):
            raise RuntimeError("Must call compute before calling plot or write")

        logger = galsim.config.LoggerWrapper(logger)

        # 6 x nplot/2 images, with each image (3.5 x 3)
        nplot = len(self.indices)
        nrows = (nplot+1)//2
        fig = Figure(figsize = (21,3*nrows))
        axs = fig.subplots(ncols=6, nrows=nrows, squeeze=False)

        logger.info("Creating %d Star plots", self.nplot)

        for i in range(len(self.indices)):
            star = self.stars[i]
            model = self.models[i]

            # get index, u, v coordinates to put in title
            u = star.data.properties['u']
            v = star.data.properties['v']
            index = self.indices[i]

            ii = i // 2
            jj = (i % 2) * 3

            axs[ii][jj+0].set_title('Star {0}'.format(index))
            axs[ii][jj+1].set_title('PSF at (u,v) = \n ({0:+.02e}, {1:+.02e})'.format(u, v))
            axs[ii][jj+2].set_title('Star - PSF')

            star_image = star.image
            model_image = model.image
            # share color range between star and model images
            vmin = np.percentile([star_image.array, model_image.array], q=10)
            vmax = np.percentile([star_image.array, model_image.array], q=90)

            axs[ii][jj+0].imshow(star_image.array, vmin=vmin, vmax=vmax, **kwargs)
            im = axs[ii][jj+1].imshow(model_image.array, vmin=vmin, vmax=vmax, **kwargs)
            fig.colorbar(im, ax=axs[ii][jj+1])  # plot shared colorbar after model

            # plot star - model with separate colorbar
            im = axs[ii][jj+2].imshow(star_image.array - model_image.array, **kwargs)
            fig.colorbar(im, ax=axs[ii][jj+2])

        return fig, axs

class StarStatsDepr(StarStats):
    _type_name = 'Star'

    def __init__(self, *args, logger=None, **kwargs):
        logger = galsim.config.LoggerWrapper(logger)
        logger.error("WARNING: The name Star is deprecated. Use StarImages instead.")
        super().__init__(*args, logger=logger, **kwargs)
