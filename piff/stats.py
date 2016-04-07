
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

def process_stats(config, logger):
    """Parse the stats field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info.

    :returns: an stats instance
    """
    import piff

    if 'stats' not in config:
        raise ValueError("config dict has no stats field")
    config_stats_list = config['stats']

    stats = []
    for config_stats in config_stats_list:

        if 'type' not in config_stats:
            raise ValueError("config['stats'] has no type field")

        # Get the class to use for the stats
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        stats_class = getattr(piff, config_stats.pop('type'))

        if 'output' not in config_stats:
            raise ValueError("config['stats'] has no output field")
        stats.append([piff.process_output(config_stats, logger=logger), stats_class])
        # can go stats[i][0].write(stats[i][1](psf, stars)) to perform tests

    return stats

class Statistics(object):
    """The base class for getting the statistics of a set of stars.

    Takes in a psf and a list of stars and performs an analysis on it that can
    then be plotted or otherwise saved to disk.
    """

    def __init__(self, psf, stars, logger=None, **kwargs):
        """Perform your statistical operation on the stars.

        :param psf:         A PSF Object
        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the statistical operation!")

    def plot(self, logger=None):
        """Make your plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :returns:           fig and ax
        """
        raise NotImplemented("Derived classes must define the plot function")

    def write(self, file_name, logger=None, **kwargs):
        """Write stats plots to file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        :param kwargs:      kwargs go into plot
        """
        if logger:
            logger.info("Creating Plot")
        fig, ax = self.plot(logger=logger, **kwargs)

        if logger:
            logger.info("Writing Statistics Plot to file %s",file_name)
        # save fig to file_name
        fig.savefig(file_name)

    def hsm(self, image, weight=None, image_pos=None, **kwargs):
        """Return HSM Shape Measurements

        :param image:       A postage stamp image that includes the star
        :param image_pos:   The position in image coordinates to use as the
                            "center" of the star.  Note: this does not have to
                            be the centroid or anything specific about the
                            star.  It is merely the image position of the (0,0)
                            coordinate for the model's internal coordinate
                            system. [default: None; use image.trueCenter()]
        :param weight:      The corresponding weight map for that image. [default: None]
        :param kwargs:      Anything to pass to Gaussian()

        :returns sigma, g1, g2: HSM Shape measurements
        """
        import piff

        if image_pos is None:
            # take the center of the image
            image_pos = image.trueCenter()

        # turn into star for piffy Gaussian model to interpret
        star = piff.StarData(image, image_pos, weight)

        return piff.Gaussian(**kwargs).fitStar(star).getParameters()

    def measureShapes(self, psf, stars, logger=None):
        """Compare PSF and true star shapes with HSM algorithm

        :param psf:         A PSF Object
        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           positions of stars, shapes of stars, and shapes of
                            models of stars (sigma, g1, g2)
        """
        # measure moments with Gaussian on image
        if logger:
            logger.info("Measuring Stars")
        shapes_truth = np.array([ self.hsm(*star.getImage()) for star in stars ])
        # from stars get positions
        if logger:
            logger.info("Getting Star Positions")
        positions = np.array([ psf.interp.getStarPosition(star) for star in stars ])
        # generate the model stars measure moments with Gaussian on
        # interpolated model image
        if logger:
            logger.info("Generating and Measuring Model Stars")
        shapes_model = np.array([ self.hsm(psf.drawImage(position)) for position in positions ])

        return positions, shapes_truth, shapes_model

class ShapeStatistics(Statistics):
    """Returns histograms of shape differences using HSM
    """

    def __init__(self, psf, stars, logger=None):
        """

        :param psf:         A PSF Object
        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """

        # get the shapes
        if logger:
            logger.info("Measuring Star and Model Shapes")
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # define terms for the catalogs
        self.u = positions[:, 0]
        self.v = positions[:, 1]
        self.T = shapes_truth[:, 0]
        self.g1 = shapes_truth[:, 1]
        self.g2 = shapes_truth[:, 2]
        self.T_model = shapes_model[:, 0]
        self.g1_model = shapes_model[:, 1]
        self.g2_model = shapes_model[:, 2]
        self.dT = self.T - self.T_model
        self.dg1 = self.g1 - self.g1_model
        self.dg2 = self.g2 - self.g2_model

    def plot(self, bins_size=10, bins_shape=10, logger=None, **kwargs):
        """Make your plots.

        :param bins_size:   Number of bins for histograms of size and size difference [default: 10]
        :param bins_shape:   Number of bins for histograms of shape and shape difference [default: 10]
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     kwargs go into the plotting.

        :returns: fig, ax
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))

        # some defaults for the kwargs
        if 'histtype' not in kwargs:
            kwargs['histtype'] = 'step'

        # axs[0,0] = size distributions
        ax = axs[0, 0]
        ax.hist([self.T, self.T_model], bins=bins_size, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$T$')
        # axs[0,1] = size difference
        ax = axs[1, 0]
        ax.hist(self.dT, bins=bins_size, **kwargs)
        ax.set_xlabel(r'$T_{data} - T_{model}$')

        # axs[1,0] = g1 distribution
        ax = axs[0, 1]
        ax.hist([self.g1, self.g1_model], bins=bins_shape, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$g_{1}$')
        # axs[1,0] = g1 difference
        ax = axs[1, 1]
        ax.hist(self.dg1, bins=bins_shape, **kwargs)
        ax.set_xlabel(r'$g_{1, data} - g_{1, model}$')

        # axs[2,0] = g2 distribution
        ax = axs[0, 2]
        ax.hist([self.g2, self.g2_model], bins=bins_shape, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$g_{2}$')
        # axs[2,0] = g2 difference
        ax = axs[1, 2]
        ax.hist(self.dg2, bins=bins_shape, **kwargs)
        ax.set_xlabel(r'$g_{2, data} - g_{2, model}$')

        return fig, ax

class RhoStatistics(Statistics):
    """Returns rho statistics using TreeCorr
    """

    def __init__(self, psf, stars, min_sep=1, max_sep=150, bin_size=0.1, logger=None):
        """

        :param psf:         A PSF Object
        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :param min_sep:     Minimum separation (in arcmin) for pairs
        :param max_sep:     Maximum separation (in arcmin) for pairs
        :param bin_size:    Logarithmic size of separation bins.


        Notes
        -----
        Assumes first two coordinates of star position are u, v


        From Jarvis:2015 p 10, eqs 3-18 - 3-22.
        e = e_psf ; de = e_psf - e_model
        T is size

        rho1 = < de* de >
        rho2 = < e* de >  (in the rowe paper this is < e* de + de* e >
        rho3 = < (e* dT / T) (e dT / T) >
        rho4 = < de* (e dT / T) >
        rho5 = < e* (e dT / T) >

        We calculate these quantities using treecorr

        also note that for gN = gNr + i gNi = gN1 + i gN2:
        xi.xip[k] += g1rg2r + g1ig2i;       // g1 * conj(g2)
        xi.xip_im[k] += g1ig2r - g1rg2i;
        xi.xim[k] += g1rg2r - g1ig2i;       // g1 * g2
        xi.xim_im[k] += g1ig2r + g1rg2i;

        so since we probably really want, e.g. 0.5 < e* de + de* e >, then we
        can just use xip.

        """
        import treecorr

        # get the shapes
        if logger:
            logger.info("Measuring Star and Model Shapes")
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # define terms for the catalogs
        u = positions[:, 0]
        v = positions[:, 1]
        T = shapes_truth[:, 0]
        g1 = shapes_truth[:, 1]
        g2 = shapes_truth[:, 2]
        dT = T - shapes_model[:, 0]
        dg1 = g1 - shapes_model[:, 1]
        dg2 = g2 - shapes_model[:, 2]

        # make the treecorr catalogs
        corr_dict = {'min_sep': min_sep, 'max_sep': max_sep,
                     'bin_size': bin_size, 'sep_units': 'arcmin',
                     }
        if logger:
            logger.info("Creating Treecorr Catalogs")

        cat_g = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                 g1=g1, g2=g2)
        cat_dg = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                  g1=dg1, g2=dg2)
        cat_gdTT = treecorr.Catalog(x=u, y=v, x_units='arcsec', y_units='arcsec',
                                    g1=g1 * dT / T, g2=g2 * dT / T)

        # setup and run the correlations
        if logger:
            logger.info("Processing rho PSF statistics")

        # save the rho objects
        self.rho1 = treecorr.GGCorrelation(**corr_dict)
        self.rho1.process(cat_dg)
        self.rho2 = treecorr.GGCorrelation(**corr_dict)
        self.rho2.process(cat_g, cat_dg)
        self.rho3 = treecorr.GGCorrelation(**corr_dict)
        self.rho3.process(cat_gdTT)
        self.rho4 = treecorr.GGCorrelation(**corr_dict)
        self.rho4.process(cat_dg, cat_gdTT)
        self.rho5 = treecorr.GGCorrelation(**corr_dict)
        self.rho5.process(cat_g, cat_gdTT)

    def plot(self, logger=None, **kwargs):
        """Make your plots.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: fig, ax
        """

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

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

                # catch linestyles, but otherwise default to - and -- for positive and negative values
                if 'linestyle' in kwargs.keys():
                    linestyle_pos = kwargs.pop('linestyle')
                    linestyle_neg = linestyle_pos
                else:
                    linestyle_pos = '-'
                    linestyle_neg = '--'
                # do the plots
                ax.plot(r[pos], xi[pos], linestyle=linestyle_pos, color=color, label=label, **kwargs)
                # no label for the negative values
                ax.plot(r[neg], xi[neg], linestyle=linestyle_neg, color=color, **kwargs)
            ax.legend(loc='upper right')

        return fig, axs
