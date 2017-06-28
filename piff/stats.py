
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
.. module:: stats

"""

from __future__ import print_function
import numpy as np
import os
import warnings
import galsim

class Stats(object):
    """The base class for getting the statistics of a set of stars.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.

    The usual code flow for using a Stats instance is:

        >>> stats = SomeKindofStats(...)
        >>> stats.compute(psf, stars, logger)
        >>> stats.write(file_name=file_name)

    There is also a ``plot`` method if you want to make the matplot lib fig, ax and do something
    else with it besides just write it to a file.
    """
    @classmethod
    def process(cls, config_stats, logger):
        """Parse the stats field of the config dict.

        :param config_stats:    The configuration dict for the stats field.
        :param logger:          A logger object for logging debug info.

        :returns: a Stats instance
        """
        import piff

        # If it's not a list, make it one.
        try:
            config_stats[0]
        except:
            config_stats = [config_stats]

        stats = []
        for cfg in config_stats:

            if 'type' not in cfg:
                raise ValueError("config['stats'] has no type field")

            # Get the class to use for the stats
            stats_class = getattr(piff, cfg.pop('type') + 'Stats')

            # Read any other kwargs in the stats field
            kwargs = stats_class.parseKwargs(cfg, logger)

            stats.append(stats_class(**kwargs))

        return stats

    @classmethod
    def parseKwargs(cls, config_stats, logger=None):
        """Parse the stats field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_stats:    The stats field of the configuration dict, config['stats']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_stats)
        return kwargs

    def compute(self, psf, stars, logger=None):
        """Compute the given statistic for a PSF solution on a set of stars.

        This needs to be done before the statistic is plotted or written to a file.

        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        raise NotImplementedError("Derived classes must define the plot function")

    def plot(self, logger=None, **kwargs):
        """Make the plots for this statistic.

        :param logger:      A logger object for logging debug info. [default: None]
        :param **kwargs:    Optionally, provide extra kwargs for the matplotlib plot command.

        :returns: (fig, ax) The matplotlib figure and axis with the plot(s).
        """
        raise NotImplementedError("Derived classes must define the plot function")

    def write(self, file_name=None, logger=None, **kwargs):
        """Write plots to a file.

        :param file_name:   The name of the file to write to. [default: Use self.file_name,
                            which is typically read from the config field.]
        :param logger:      A logger object for logging debug info. [default: None]
        :param **kwargs:    Optionally, provide extra kwargs for the matplotlib plot command.
        """
        # Note: don't import matplotlib.pyplot, since that can mess around with the user's
        # pyplot state.  Better to do everything with the matplotlib object oriented API.
        # cf. http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        logger = galsim.config.LoggerWrapper(logger)
        logger.info("Creating plot for %s", self.__class__.__name__)
        fig, ax = self.plot(logger=logger, **kwargs)

        if file_name is None:
            file_name = self.file_name
        if file_name is None:
            raise ValueError("No file_name specified for %s"%self.__class__.__name__)

        logger.warning("Writing %s plot to file %s",self.__class__.__name__,file_name)

        canvas = FigureCanvasAgg(fig)
        # Do this after we've set the canvas to use Agg to avoid warning.
        fig.set_tight_layout(True)
        canvas.print_figure(file_name, dpi=100)

    def measureShapes(self, psf, stars, logger=None):
        """Compare PSF and true star shapes with HSM algorithm

        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           positions of stars, shapes of stars, and shapes of
                            models of stars (sigma, g1, g2)
        """
        import piff
        logger = galsim.config.LoggerWrapper(logger)
        # measure moments with Gaussian on image
        logger.debug("Measuring shapes of real stars")
        shapes_truth = np.array([ piff.util.hsm(star) for star in stars ])
        for star, shape in zip(stars, shapes_truth):
            logger.debug("real shape for star at %s is %s",star.image_pos, shape)

        # Pull out the positions to return
        positions = np.array([ (star.data.properties['u'], star.data.properties['v'])
                               for star in stars ])

        # generate the model stars and measure moments
        logger.debug("Generating and Measuring Model Stars")
        shapes_model = np.array([ piff.util.hsm(psf.drawStar(star)) for star in stars ])
        for star, shape in zip(stars, shapes_model):
            logger.debug("model shape for star at %s is %s",star.image_pos, shape)

        return positions, shapes_truth, shapes_model


class ShapeHistogramsStats(Stats):
    """Stats class for calculating histograms of shape residuals

    This will compute the size and shapes of the observed stars and the PSF models and
    make histograms of both the values and the residuals.

    The plot will have 6 axes.  The top row will have histograms of T, g1, g2, with the model
    and data color coded.  The bottom row will have histograms of the differences.

    After a call to :func:`compute`, the following attributes are accessible:

        :u:         The u positions in field coordinates.
        :v:         The v positions in field coordinates.
        :T:         The size (T = Ixx + Iyy) of the observed stars.
        :g1:        The g1 component of the shapes of the observed stars.
        :g2:        The g2 component of the shapes of the observed stars.
        :T_model:   The size of the PSF model at the same locations as the stars.
        :g1_model   The g1 component of the PSF model at these locations.
        :g2_model   The g2 component of the PSF model at these locations.
        :dT:        The size residual, T - T_model
        :dg1:       The g1 residual, g1 - g1_model
        :dg2:       The g2 residual, g2 - g2_model
    """
    def __init__(self, bins_size=10, bins_shape=10, file_name=None, logger=None):
        """
        The `bins_size` and `bins_shape` parameters are typically the number of bins for the
        size and shape histograms respectively.  However, they will be passed to the
        `matplotlib.pyplot.hist` function as the `bins` parameter, which also accepts
        numpy arrays for the size of each bin, so that is allowed here as well.

        :param bins_size:   Number of bins for size histograms. [default: 10]
        :param bins_shape:  Number of bins for shape histograms. [default: 10]
        :param file_name:   Name of the file to output to. [default: None]
        """
        self.bins_size = bins_size
        self.bins_shape = bins_shape
        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        logger.warning("Calculating shape histograms for %d stars",len(stars))
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # Only use stars for which hsm was successful
        flag_truth = shapes_truth[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_truth == 0) & (flag_model == 0)

        # define terms for the catalogs
        self.u = positions[mask, 0]
        self.v = positions[mask, 1]
        self.T = shapes_truth[mask, 3]
        self.g1 = shapes_truth[mask, 4]
        self.g2 = shapes_truth[mask, 5]
        self.T_model = shapes_model[mask, 3]
        self.g1_model = shapes_model[mask, 4]
        self.g2_model = shapes_model[mask, 5]
        self.dT = self.T - self.T_model
        self.dg1 = self.g1 - self.g1_model
        self.dg2 = self.g2 - self.g2_model

    def plot(self, logger=None, **kwargs):
        """Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib hist() function.

        :returns: fig, ax
        """
        if not hasattr(self, 'T'):
            raise RuntimeError("Shape Histogram has not been computed yet.  Cannot plot.")

        from matplotlib.figure import Figure
        fig = Figure(figsize = (15,10))
        # In matplotlib 2.0, this will be
        # axs = fig.subplots(ncols=3, nrows=2)
        axs = [[ fig.add_subplot(2,3,1),
                 fig.add_subplot(2,3,2),
                 fig.add_subplot(2,3,3) ],
               [ fig.add_subplot(2,3,4),
                 fig.add_subplot(2,3,5),
                 fig.add_subplot(2,3,6) ]]
        axs = np.array(axs, dtype=object)

        # some defaults for the kwargs
        if 'histtype' not in kwargs:
            kwargs['histtype'] = 'step'

        # axs[0,0] = size distributions
        ax = axs[0, 0]
        ax.hist([self.T, self.T_model], bins=self.bins_size, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$T$')
        # axs[0,1] = size difference
        ax = axs[1, 0]
        ax.hist(self.dT, bins=self.bins_size, **kwargs)
        ax.set_xlabel(r'$T_{data} - T_{model}$')

        # axs[1,0] = g1 distribution
        ax = axs[0, 1]
        ax.hist([self.g1, self.g1_model], bins=self.bins_shape, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$g_{1}$')
        # axs[1,0] = g1 difference
        ax = axs[1, 1]
        ax.hist(self.dg1, bins=self.bins_shape, **kwargs)
        ax.set_xlabel(r'$g_{1, data} - g_{1, model}$')

        # axs[2,0] = g2 distribution
        ax = axs[0, 2]
        ax.hist([self.g2, self.g2_model], bins=self.bins_shape, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$g_{2}$')
        # axs[2,0] = g2 difference
        ax = axs[1, 2]
        ax.hist(self.dg2, bins=self.bins_shape, **kwargs)
        ax.set_xlabel(r'$g_{2, data} - g_{2, model}$')

        return fig, ax

class RhoStats(Stats):
    """Stats class for calculating rho statistics.

    This will plot the 5 rho statistics described in Jarvis et al, 2015, section 3.4.

    e = e_psf; de = e_psf - e_model
    T is size; dT = T_psf - T_model

    rho1 = < de* de >
    rho2 = < e* de >  (in the rowe paper this is < e* de + de* e >
    rho3 = < (e* dT / T) (e dT / T) >
    rho4 = < de* (e dT / T) >
    rho5 = < e* (e dT / T) >

    The plots for rho1, rho3, and rho4 will all be on the same axis (left), and the plots for
    rho2 and rho5 will be on the other axis (right).

    Furthermore, these are technically complex quantities, but only the real parts are
    plotted, since the imaginary parts are uninteresting.

    After a call to :func:`compute`, the following attributes are accessible:

        :rho1:      A TreeCorr GGCorrelation instance with the rho1 statistic.
        :rho2:      A TreeCorr GGCorrelation instance with the rho2 statistic.
        :rho3:      A TreeCorr GGCorrelation instance with the rho3 statistic.
        :rho4:      A TreeCorr GGCorrelation instance with the rho4 statistic.
        :rho5:      A TreeCorr GGCorrelation instance with the rho5 statistic.

    The value of the canonical rho statistic is in the ``xip`` attribute of each of the above
    TreeCorr GGCorrelation instances.  But there are other quantities that may be of interest
    in some cases, so we provide access to the full object.
    """
    def __init__(self, min_sep=0.5, max_sep=300, bin_size=0.1, file_name=None,
                 logger=None, **kwargs):
        """
        :param min_sep:     Minimum separation (in arcmin) for pairs. [default: 0.5]
        :param max_sep:     Maximum separation (in arcmin) for pairs. [default: 300]
        :param bin_size:    Size of bins in log(sep). [default 0.1]
        :param file_name:   Name of the file to output to. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :param **kwargs:    Any additional kwargs are passed on to TreeCorr.
        """
        self.tckwargs = kwargs
        self.tckwargs['min_sep'] = min_sep
        self.tckwargs['max_sep'] = max_sep
        self.tckwargs['bin_size'] = bin_size
        if 'sep_units' not in self.tckwargs:
            self.tckwargs['sep_units'] = 'arcmin'
        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import treecorr

        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        logger.warning("Calculating rho statistics for %d stars",len(stars))
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # Only use stars for which hsm was successful
        flag_truth = shapes_truth[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_truth == 0) & (flag_model == 0)

        # define terms for the catalogs
        u = positions[mask, 0]
        v = positions[mask, 1]
        T = shapes_truth[mask, 3]
        g1 = shapes_truth[mask, 4]
        g2 = shapes_truth[mask, 5]
        dT = T - shapes_model[mask, 3]
        dg1 = g1 - shapes_model[mask, 4]
        dg2 = g2 - shapes_model[mask, 5]

        # make the treecorr catalogs
        logger.info("Creating Treecorr Catalogs")

        cat_g = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                 g1=g1, g2=g2)
        cat_dg = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                  g1=dg1, g2=dg2)
        cat_gdTT = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                    g1=g1 * dT / T, g2=g2 * dT / T)

        # setup and run the correlations
        logger.info("Processing rho PSF statistics")

        # save the rho objects
        self.rho1 = treecorr.GGCorrelation(self.tckwargs)
        self.rho1.process(cat_dg)
        self.rho2 = treecorr.GGCorrelation(self.tckwargs)
        self.rho2.process(cat_g, cat_dg)
        self.rho3 = treecorr.GGCorrelation(self.tckwargs)
        self.rho3.process(cat_gdTT)
        self.rho4 = treecorr.GGCorrelation(self.tckwargs)
        self.rho4.process(cat_dg, cat_gdTT)
        self.rho5 = treecorr.GGCorrelation(self.tckwargs)
        self.rho5.process(cat_g, cat_gdTT)

    def alt_plot(self, logger=None, **kwargs):  # pragma: no cover
        # Leaving this version here in case useful, but I (MJ) have a new version of this
        # below based on the figures I made for the DES SV shear catalog paper that I think
        # looks a bit nicer.
        from matplotlib.figure import Figure
        fig = Figure(figsize = (10,5))
        # In matplotlib 2.0, this will be
        # axs = fig.subplots(ncols=2)
        axs = [ fig.add_subplot(1,2,1),
                fig.add_subplot(1,2,2) ]
        axs = np.array(axs, dtype=object)


        # axs[0] gets rho1 rho3 and rho4
        # axs[1] gets rho2 and rho5
        for ax, rho_list, color_list, label_list in zip(
                axs,
                [[self.rho1, self.rho3, self.rho4], [self.rho2, self.rho5]],
                [['k', 'r', 'b'], ['g', 'm']],
                [[r'$\rho_{1}$', r'$\rho_{3}$', r'$\rho_{4}$'],
                 [r'$\rho_{2}$', r'$\rho_{5}$']]):

            # put in some labels
            ax.set_xlabel('log $r$ [arcmin]')
            ax.set_ylabel(r'$\rho$')
            # set the scale
            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')
            for rho, color, label in zip(rho_list, color_list, label_list):
                r = np.exp(rho.logr)
                xi = rho.xip
                # now separate the xi into positive and negative components
                pos = xi > 0
                neg = xi < 0

                # catch linestyles, but otherwise default to - and -- for positive and negative
                # values
                if 'linestyle' in kwargs.keys():
                    linestyle_pos = kwargs.pop('linestyle')
                    linestyle_neg = linestyle_pos
                else:
                    linestyle_pos = '-'
                    linestyle_neg = '--'
                # do the plots
                ax.plot(r[pos], xi[pos], linestyle=linestyle_pos, color=color, label=label,
                        **kwargs)
                # no label for the negative values
                ax.plot(r[neg], -xi[neg], linestyle=linestyle_neg, color=color, **kwargs)
            ax.legend(loc='upper right')

        return fig, axs

    def _plot_single(self, ax, rho, color, marker, offset=0.):
        # Add a single rho stat to the plot.
        meanr = rho.meanr * (1. + rho.bin_size * offset)
        xip = rho.xip
        sig = np.sqrt(rho.varxi)
        ax.plot(meanr, xip, color=color)
        ax.plot(meanr, -xip, color=color, ls=':')
        ax.errorbar(meanr[xip>0], xip[xip>0], yerr=sig[xip>0], color=color, ls='', marker=marker)
        ax.errorbar(meanr[xip<0], -xip[xip<0], yerr=sig[xip<0], color=color, ls='', marker=marker)
        return ax.errorbar(-meanr, xip, yerr=sig, color=color, marker=marker)

    def plot(self, logger=None, **kwargs):
        """Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]

        :returns: fig, ax
        """
        # MJ: Based on the code I used for the plot in the DES SV paper:
        from matplotlib.figure import Figure
        fig = Figure(figsize = (10,5))
        # In matplotlib 2.0, this will be
        # axs = fig.subplots(ncols=2)
        axs = [ fig.add_subplot(1,2,1),
                fig.add_subplot(1,2,2) ]
        axs = np.array(axs, dtype=object)


        # Left plot is rho1,3,4
        rho1 = self._plot_single(axs[0], self.rho1, 'blue', 'o')
        rho3 = self._plot_single(axs[0], self.rho3, 'green', 's', 0.1)
        rho4 = self._plot_single(axs[0], self.rho4, 'red', '^', 0.2)

        axs[0].legend([rho1, rho3, rho4],
                     [r'$\rho_1(\theta)$', r'$\rho_3(\theta)$', r'$\rho_4(\theta)$'],
                     loc='upper right', fontsize=12)
        axs[0].set_ylim(1.e-9, None)
        axs[0].set_xlim(self.tckwargs['min_sep'], self.tckwargs['max_sep'])
        axs[0].set_xlabel(r'$\theta$ (arcmin)')
        axs[0].set_ylabel(r'$\rho(\theta)$')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log', nonposy='clip')

        # Right plot is rho2,5
        rho2 = self._plot_single(axs[1], self.rho2, 'blue', 'o')
        rho5 = self._plot_single(axs[1], self.rho5, 'green', 's', 0.1)

        axs[1].legend([rho2, rho5],
                     [r'$\rho_2(\theta)$', r'$\rho_5(\theta)$'],
                     loc='upper right', fontsize=12)
        axs[1].set_ylim(1.e-7, None)
        axs[1].set_xlim(self.tckwargs['min_sep'], self.tckwargs['max_sep'])
        axs[1].set_xlabel(r'$\theta$ (arcmin)')
        axs[1].set_ylabel(r'$\rho(\theta)$')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log', nonposy='clip')

        return fig, axs
