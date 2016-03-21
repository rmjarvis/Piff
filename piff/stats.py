
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

from .psf import PSF

def process_stats(config, logger):
    """Parse the stats field of the config dict.

    :param config:      The configuration dict.
    :param logger:      A logger object for logging debug info.

    :returns: an stats instance
    """
    import piff

    if 'stats' not in config:
        raise ValueError("config dict has no stats field")
    config_stats = config['stats']

    if 'type' not in config_stats:
        raise ValueError("config['stats'] has no type field")

    # Get the class to use for the stats
    # Not sure if this is what we'll always want, but it would be simple if we can make it work.
    stats_class = eval('piff.' + config_stats.pop('type'))

    # Read any other kwargs in the stats field
    kwargs = stats_class.parseKwargs(config_stats)

    # Build stats object
    stats = stats_class(**kwargs)

    return stats

class Statistics(PSF):
    """The base class for getting the statistics of a set of stars.

    It seems reasonable to use the PSF class as our base and build on top of that.
    """

    def stats(self, stars, logger=None, **kwargs):
        """Perform your operation on the stars.

        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the stats function")

    def plot(self, fig=None, ax=None, logger=None, **kwargs):
        """Make your plots.

        :param fig:         A Matplotlib Figure object. [default: None]
        :param ax:          A Matplotlib Axis object. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the plot function")

class RhoStatistics(Statistics):
    """Returns rho statistics using TreeCorr
    """

    def stats(self, stars, min_sep=1, max_sep=150, bin_size=0.1,
              sky_coordinates=False, sep_units='rad',
              ra_units='deg', dec_units='deg', logger=None):
        """

        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :param min_sep:     Minimum separation (in arcmin) for pairs
        :param max_sep:     Maximum separation (in arcmin) for pairs
        :param bin_size:    Logarithmic size of separation bins.
        :param sky_coordinates: Are we in sky coordinates or otherwise? [default: False]
        :param sep_units:   I think only matters if in sky_coordinates.
                            Can be 'rad', 'arcmin', 'deg', etc.
                            Sets the units of separation in min_sep, max_sep
        :param ra_units:    I think only matters if in sky_coordinates.
                            Can be 'rad', 'arcmin', 'deg', etc.
                            Says what the units of the ra coordinate is.
        :param dec_units:   I think only matters if in sky_coordinates.
                            Can be 'rad', 'arcmin', 'deg', etc.
                            Says what the units of the dec coordinate is.

        note: assumes coord[:,0] = ra, coord[:,1] = dec or x/y

        :returns: logr, rho1, rho2, rho3, rho4, rho5 correlation functions


        Notes
        -----
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
        from .gaussian_model import Gaussian
        import treecorr

        hsm = Gaussian()

        # measure moments with Gaussian on image
        if logger:
            logger.info("Measuring Stars")
        shapes_truth = np.array([ hsm.fitStar(star).getParameters() for star in stars ])
        # from stars get positions
        if logger:
            logger.info("Getting Star Positions")
        positions = np.array([ self.interp.getStarPosition(star) for star in stars ])
        # generate the model stars
        if logger:
            logger.info("Generating Model Stars")
        # stars_model = []
        # for position in positions:
        #     print(position)
        #     stars_model.append(self.generate_star(position))
        stars_model = np.array([ self.generate_star(position) for position in positions ])
        # measure moments with Gaussian on interpolated model image
        if logger:
            logger.info("Measuring Model Stars")
        shapes_model = np.array([ hsm.fitStar(star).getParameters() for star in stars_model ])

        # define terms for the catalogs
        ra = positions[:, 0]
        dec = positions[:, 1]
        T = shapes_truth[:, 0]
        g1 = shapes_truth[:, 1]
        g2 = shapes_truth[:, 2]
        dT = T - shapes_model[:, 0]
        dg1 = g1 - shapes_model[:, 1]
        dg2 = g2 - shapes_model[:, 2]

        # make the treecorr catalogs
        corr_dict = {'min_sep': min_sep, 'max_sep': max_sep,
                     'bin_size': bin_size, 'sep_units': sep_units,
                     }
        if logger:
            logger.info("Creating Treecorr Catalogs")

        if sky_coordinates:
            cat_g = treecorr.Catalog(ra=ra, dec=dec,
                                     g1=g1, g2=g2,
                                     ra_units=ra_units, dec_units=dec_units)
            cat_dg = treecorr.Catalog(ra=ra, dec=dec,
                                      g1=dg1, g2=dg2,
                                      ra_units=ra_units, dec_units=dec_units)
            cat_gdTT = treecorr.Catalog(ra=ra, dec=dec,
                                        g1=g1 * dT / T, g2=g2 * dT / T,
                                        ra_units=ra_units, dec_units=dec_units)
        else:
            cat_g = treecorr.Catalog(x=ra, y=dec,
                                     g1=g1, g2=g2)
            cat_dg = treecorr.Catalog(x=ra, y=dec,
                                      g1=dg1, g2=dg2)
            cat_gdTT = treecorr.Catalog(x=ra, y=dec,
                                        g1=g1 * dT / T, g2=g2 * dT / T)

        # setup and run the correlations
        if logger:
            logger.info("Processing rho PSF statistics")

        # save the rho objects
        rho1 = treecorr.GGCorrelation(**corr_dict)
        rho1.process(cat_dg)
        rho2 = treecorr.GGCorrelation(**corr_dict)
        rho2.process(cat_g, cat_dg)
        rho3 = treecorr.GGCorrelation(**corr_dict)
        rho3.process(cat_gdTT)
        rho4 = treecorr.GGCorrelation(**corr_dict)
        rho4.process(cat_dg, cat_gdTT)
        rho5 = treecorr.GGCorrelation(**corr_dict)
        rho5.process(cat_g, cat_gdTT)

        # save the rhos for later
        self.rho = {1: rho1,
                    2: rho2,
                    3: rho3,
                    4: rho4,
                    5: rho5}

        # return directly the correlation functions
        return rho1.logr, rho1.xip, rho2.xip, rho3.xip, rho4.xip, rho5.xip

    def plot(self, rho, fig=None, ax=None, logger=None, **kwargs):
        """Make your plots.

        :param rho:         A treecorr GGCorrelation object.
        :param fig:         A Matplotlib Figure object. [default: None]
        :param ax:          A Matplotlib Axis object. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     kwargs go into the plotting.

        :returns: fig, ax
        """

        # make a figure if the fig and ax are not specified
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            # put in some labels
            ax.set_xlabel('log $r$')
            ax.set_ylabel(r'$\rho$')
            # set the scale
            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')

        r = np.exp(rho.logr)
        xi = rho.xip
        # now separate the xi into positive and negative components
        xi_neg = np.ma.masked_where(xi > 0, -xi)
        xi_pos = np.ma.masked_where(xi < 0, xi)

        if 'color' not in kwargs.keys():
            # set color to k
            color = 'k'
        else:
            # let the kwargs specify the color
            color = kwargs.pop('color')
        # do the plots
        ax.plot(r, xi_pos, color=color, linestyle='-', **kwargs)
        ax.plot(r, xi_neg, color=color, linestyle='--', **kwargs)

        return fig, ax
