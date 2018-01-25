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

from __future__ import print_function
import numpy as np
import galsim

from .stats import Stats

class StarStats(Stats):
    """This Statistics class can take stars and make a set of plots of them as
    well as their models and residuals.

    By default this will draw 5 random stars, make psf stars, and plot the
    residual of the two.

    After a call to :func:`compute`, the following attributes are accessible:

        :stars:         List of stars used for plotting
        :models:        List of models of stars used for plotting
        :indices:       Indices of input stars that the plotting stars correspond to
    """

    def __init__(self, number_plot=5,
                 file_name=None, logger=None):
        """
        :param number_plot:         Number of stars we wish to plot. If 0 or
                                    number_plot > than stars in PSF, then we
                                    plot all stars. Otherwise, we draw
                                    number_plot stars at random (without
                                    replacement). [default: 5]
        :param file_name:           Name of the file to output to. [default: None]
        :param logger:              A logger object for logging debug info. [default: None]
        """

        self.number_plot = number_plot
        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        if self.number_plot == 0 or self.number_plot >= len(stars):
            # select all stars
            self.indices = np.arange(len(stars))
        else:
            self.indices = np.random.choice(len(stars), self.number_plot, replace=False)

        logger.info("Making {0} Model Stars".format(len(self.indices)))
        self.stars = []
        for index in self.indices:
            star = stars[index]
            self.stars.append(star)
        self.models = psf.drawStarList(self.stars)

    def plot(self, logger=None, **kwargs):
        """Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib pcolor() function.

        :returns: fig, ax
        """

        # make figure
        from matplotlib.figure import Figure
        logger = galsim.config.LoggerWrapper(logger)
        # 3 x number_plot images, with each image (4 x 3)
        fig = Figure(figsize=(12, 3 * self.number_plot))
        # In matplotlib 2.0, this will be
        # axs = fig.subplots(ncols=3, nrows=3)
        axs = []
        for i in range(self.number_plot):
            axs.append([fig.add_subplot(self.number_plot, 3, i * 3 + 1),
                        fig.add_subplot(self.number_plot, 3, i * 3 + 2),
                        fig.add_subplot(self.number_plot, 3, i * 3 + 3)])
        axs = np.array(axs, dtype=object)

        logger.info("Creating Star Plots")

        for i in range(self.number_plot):
            star = self.stars[i]
            model = self.models[i]

            # get index, u, v coordinates to put in title
            u = star.data.properties['u']
            v = star.data.properties['v']
            index = self.indices[i]

            axs[i][0].set_title('Star {0}'.format(index))
            axs[i][1].set_title('PSF at (u,v) = ({0:+.02e}, {1:+.02e})'.format(u, v))
            axs[i][2].set_title('Star - PSF')

            star_image = star.image
            model_image = model.image
            # share color range between star and model images
            vmin = np.percentile([star_image.array, model_image.array], q=10)
            vmax = np.percentile([star_image.array, model_image.array], q=90)

            axs[i][0].imshow(star_image.array, vmin=vmin, vmax=vmax, **kwargs)
            im = axs[i][1].imshow(model_image.array, vmin=vmin, vmax=vmax, **kwargs)
            fig.colorbar(im, ax=axs[i][1])  # plot shared colorbar after model

            # plot star - model with separate colorbar
            im = axs[i][2].imshow(star_image.array - model_image.array, **kwargs)
            fig.colorbar(im, ax=axs[i][2])

        return fig, axs
