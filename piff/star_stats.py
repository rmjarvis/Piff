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
    :param include_reserve: Whether to inlude reserve stars. [default: True]
    :param only_reserve:    Whether to skip plotting non-reserve stars. [default: False]
    :param include_flagged: Whether to include plotting flagged stars. [default: False]
    :param include_ave:     Whether to inlude the average image. [default: True]
    :param file_name:       Name of the file to output to. [default: None]
    :param logger:          A logger object for logging debug info. [default: None]
    """
    _type_name = 'StarImages'

    def __init__(self, nplot=10, adjust_stars=False,
                 include_reserve=True, only_reserve=False, include_flagged=False,
                 include_ave=True, file_name=None, logger=None):
        self.nplot = nplot
        self.file_name = file_name
        self.adjust_stars = adjust_stars
        self.include_reserve = include_reserve
        self.only_reserve = only_reserve
        self.include_flagged = include_flagged
        self.include_ave = include_ave

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # Determine which stars to plot
        possible_indices = []
        if self.include_reserve:
            possible_indices += [i for i,s in enumerate(stars)
                                 if s.is_reserve and (self.include_flagged or not s.is_flagged)]
        if not self.only_reserve:
            possible_indices += [i for i,s in enumerate(stars)
                                 if not s.is_reserve and (self.include_flagged or not s.is_flagged)]
        possible_indices = sorted(possible_indices)

        if self.nplot == 0 or self.nplot >= len(stars):
            # select all viable stars
            self.indices = possible_indices
        else:
            self.indices = np.random.choice(possible_indices, self.nplot, replace=False)

        # If we need to compute the average image, then we need to reflux and  drawStar for all
        # possible_indices.  Otherwise, only do those steps for the stars we will plot.
        if self.include_ave:
            calculate_indices = possible_indices
        else:
            calculate_indices = self.indices

        logger.info("Making {0} model stars".format(len(calculate_indices)))
        calculated_stars = []
        calculated_models = []
        for i, star in enumerate(stars):
            if i in calculate_indices:
                if self.adjust_stars:
                    # Do 2 passes, since we sometimes start pretty far from the right values.
                    star = psf.reflux(star, logger=logger)
                    star = psf.reflux(star, logger=logger)
                calculated_stars.append(star)
                calculated_models.append(psf.drawStar(star))
            else:
                calculated_stars.append(None)
                calculated_models.append(None)

        # if including the average image, put that first.
        logger.info("Making average star and model")
        if self.include_ave:
            ave_star_image = np.mean([s.image.array for s in calculated_stars if s is not None],
                                     axis=0)
            ave_model_image = np.mean([s.image.array for s in calculated_models if s is not None],
                                      axis=0)
            ave_star_image = galsim.Image(ave_star_image)
            ave_model_image = galsim.Image(ave_model_image)
            ave_star = Star(stars[0].data.withNew(image=ave_star_image), None)
            ave_model = Star(stars[0].data.withNew(image=ave_model_image), None)
            self.stars = [ave_star]
            self.models = [ave_model]
            self.stars.extend([calculated_stars[i] for i in self.indices])
            self.models.extend([calculated_models[i] for i in self.indices])
            self.indices = [-1] + self.indices
        else:
            self.stars = [calculated_stars[i] for i in self.indices]
            self.models = [calculated_models[i] for i in self.indices]


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

        for i in range(nplot):
            star = self.stars[i]
            model = self.models[i]

            ii = i // 2
            jj = (i % 2) * 3

            if self.include_ave and i == 0:
                axs[ii][jj+0].set_title('Average Star')
                axs[ii][jj+1].set_title('Average PSF')
            else:
                # get index, u, v coordinates to put in title
                index = self.indices[i]
                u = star.data.properties['u']
                v = star.data.properties['v']

                title = f'Star {index}'
                if star.is_reserve:
                    title = 'Reserve ' + title
                if star.is_flagged:
                    title = 'Flagged ' + title
                axs[ii][jj+0].set_title(title)
                axs[ii][jj+1].set_title(f'PSF at (u,v) = \n ({u:+.02e}, {v:+.02e})')
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
