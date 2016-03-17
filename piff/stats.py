
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

class Stats(object):
    """The base class for getting the statistics of a set of stars.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def parseKwargs(cls, config_stats):
        """Parse the stats field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_stats:   The stats field of the configuration dict, config['stats']

        :returns: a kwargs dict to pass to the initializer
        """
        return config_stats

    def __init__(self, psf):
        self.psf = psf

    @classmethod
    def build(cls, stars, model, interp, logger=None):
        """The main driver function to build a Stats framework from data.

        :param stars:       A list of StarData instances. Builds the PSF
        :param model:       A Model instance that defines how to model the
                            individual PSFs at the location of each star.
        :param interp:      An Interp instance that defines how to do the
                            interpolation of the data vectors (produced by
                            model for each star).
        :param logger:      A logger object for logging debug info.
                            [default: None]

        :returns: a Stats instance
        """
        # build PSF
        psf = PSF.build(stars, model, interp, logger)
        return cls(psf)

    @classmethod
    def read(cls, file_name, logger=None):
        """Read a Stas object from a stats file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        # read psf
        psf = PSF.read(file_name, logger=logger)
        return cls(psf)

    def stats(self, stars, logger=None, **kwargs):
        """Perform your operation on the stars.

        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the stats function")

    def plot(self, stars, logger=None, **kwargs):
        """Perform your operation on the stars.

        :param stars:       A list of StarData instances.
        :param logger:      A logger object for logging debug info. [default: None]
        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the plot function")

class RhoStatistics(Stats):
    """Returns rho statistics using TreeCorr
    """

    def stats(self, stars,
              min_sep=1, max_sep=150, bin_size=0.1):
        """

        :param stars:               Stars

        :param min_sep:             Minimum separation (in arcmin) for pairs
        :param max_sep:             Maximum separation (in arcmin) for pairs
        :param bin_size:            Logarithmic size of separation bins.

        note: assumes coord[:,0] = ra, coord[:,1] = dec

        :returns: rho1, rho2, rho3, rho4, rho5 treecorr correlation functions
            These have properties like rho1.xip, rho1.meanlogr, rho1.xip_im...

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
        shapes_truth = np.array([ hsm.fitStar(star).getParameters() for star in stars ])
        # from stars get positions
        positions = [ self.psf.interp.getStarPosition(star) for star in stars ]
        # generate the model stars
        stars_model = np.array([ self.psf.generate_star(position) for position in positions ])
        # measure moments with Gaussian on interpolated model image
        shapes_model = np.array([ hsm.fitStar(star).getParameters() for star in stars_model ])

        # define terms for the catalogs
        # TODO: This is probably incorrect!!
        ra = positions[:, 0]
        dec = positions[:, 1]
        T = shapes_truth[:, 0]
        g1 = shapes_truth[:, 1]
        g2 = shapes_truth[:, 2]
        dT = T - shapes_model[:, 0]
        dg1 = g1 - shapes_model[:, 1]
        dg2 = g2 - shapes_model[:, 2]

        # make the treecorr catalogs
        cat_g = treecorr.Catalog(ra=ra, dec=dec, g1=g1, g2=g2)
        cat_dg = treecorr.Catalog(ra=ra, dec=dec, g1=dg1, g2=dg2)
        cat_gdTT = treecorr.Catalog(ra=ra, dec=dec, g1=g1 * dT / T, g2=g2 * dT / T)
        corr_dict = {'min_sep': min_sep, 'max_sep': max_sep,
                     'bin_size': bin_size}

        # setup and run the correlations
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

        return rho1, rho2, rho3, rho4, rho5
