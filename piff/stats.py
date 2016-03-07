
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


TODO: Rewrite the Stats object below to use the PSF class as base class. Then,
when you do the statistics, add in an additional Gaussian model that takes the
interpolated images and measures the moments.
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

    def stats(self, **kwargs):
        """Perform your operation on the stars.

        :params kwargs:     Potential other parameters we might need to input. Images, coordinates, et cetera.

        :returns:           Some kind of data vector.
        """
        raise NotImplemented("Derived classes must define the stats function")


class HSM(Stats):
    """Returns galsim hsm shapes and sizes
    """

    @staticmethod
    def _hsm(image):
        """Run the hsm algorithm

        :param image:       Galsim Image to analyze

        :returns: sigma, g1, g2 from hsm adaptive moments algorithm
        """
        import galsim
        shape = galsim.hsm.FindAdaptiveMom(image)
        return shape.moments_sigma, shape.observed_shape.g1, shape.observed_shape.g2

    def stats(self, images, coordinates=None):
        """Run the hsm algorithm on all the images

        :param images:      Set of galsim images

        :returns: (N,3) datavector of sigma, g1, g2

        Note: doesn't use the coordinates parameter.
        """
        shapes = np.array([self._hsm(image) for image in images])

        return shapes

class RhoStatistics(HSM):
    """Returns rho statistics using TreeCorr
    """

    def stats(self, images, coordinates, images_truth,
              min_sep=1, max_sep=150, bin_size=0.1):
        """

        :param images:              Set of galsim model images
        :param images_truth:        Set of galsim true psf images
        :param coordinates:         The coordinates used.

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

        or in other words that < e* de > -> xip - i xip_im

        """
        import treecorr

        # construct the shapes from the hsm algorithm
        shapes_model = np.array([self._hsm(image) for image in images])
        shapes_truth = np.array([self._hsm(image) for image in images_truth])

        # define terms for the catalogs
        ra = coordinates[:, 0]
        dec = coordinates[:, 1]
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

class FocalStats(HSM):
    """Returns average shapes across focal plane, binned. Can give back inputs
    for whisker plots.
    """

    @classmethod
    def set_bins_decamutil(cls, number_divisions):
        """Convenience function for defining bin boundaries based on subdivisions of chip.

        :param number_divisions: how many bins per ccd? if number_divions==0,
            then one bin per ccd (rectangle), elif number_divisions==1, 2 bins
            per ccd (square). Each additional number divides each box in half.

        :returns: list of two arrays [x_edges, y_edges]
        """

        # calculate center to center chip distances
        # hard-coded in mm.
        x_step = 33.816
        y_step = 63.89
        x_min = -236.712
        y_min = -223.615
        # shrink our steps appropriately.
        if number_divisions > 0:
            x_step /= 2 ** (number_divisions - 1)
            y_step /= 2 ** (number_divisions)
        # set x_max bigger than -x_min to make sure arange includes -x_min
        x_max = -x_min + x_step
        y_max = -y_min + y_step

        # create the bins
        x_edges = np.arange(x_min, x_max, x_step)
        y_edges = np.arange(y_min, y_max, y_step)

        return [x_edges, y_edges]

    def set_bins(self, number_divisions, per_ccd=False):
        """Define bins across focal plane. Assumes mm!

        :param number_divisions: integer. If per_ccd false, then number of bins
            in x-direction, with twice as many in y direction.
            If per_ccd is true, then sets the number of divisions (vertical +
            horizontal) in each ccd.

        :param per_ccd: bool. Sets whether we want bins according to dividing
            up the ccd, or according to just a raw number of points.

        :returns: a list of two arrays [x_edges, y_edges]
        """

        if per_ccd:
            return self.set_bins_decamutil(number_divisions)
        else:
            x_min = -236.712
            y_min = -223.615
            x_max = -x_min
            y_max = -y_min
            x_edges = np.linspace(x_min, x_max, number_divisions)
            y_edges = np.linspace(y_min, y_max, 2 * number_divisions)
            return [x_edges, y_edges]

    def hist2d(self, x, y, z,
               reducer=np.median, number_divisions=1, per_ccd=True,
               make_2d=False):
        """Use pandas to return array of values which are the average in some
        bins

        :param x, y, z: lists of values (x,y,z) where reducer(z_{i,j}) for z in
            x bin x_i and y bin y_j. Assumes x and y are in focal plane mm

        :param reducer: function to apply on objects in given bin

        :param number_divisions: integer. If per_ccd false, then number of bins
            in x-direction, with twice as many in y direction.
            If per_ccd is true, then sets the number of divisions (vertical +
            horizontal) in each ccd.

        :param per_ccd: bool. Sets whether we want bins according to dividing
            up the ccd, or according to just a raw number of points.

        :param make_2d: bool. If True, return 2d masked arrays for x_av, y_av,
            z_av. If False, return flattened arrays with no masked elements

        :returns: arrays x_av, y_av, z_av for average value of each in arrays
        """
        import pandas as pd

        # create pandas dataframe
        data = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # create the bins
        bins_x, bins_y = self.set_bins(number_divisions, per_ccd)

        # pandas magic
        indx_x = np.arange(len(bins_x) - 1)
        indx_y = np.arange(len(bins_y) - 1)
        groups = data.groupby([pd.cut(x, bins_x, labels=indx_x),
                               pd.cut(y, bins_y, labels=indx_y)])
        field = groups.aggregate(reducer)
        # filter out nanmins on x and y for empty bins
        field = field[field['y'].notnull() & field['y'].notnull()]

        if make_2d:
            # now turn the field into 2d arrays x_av, y_av, z_av
            indx_x_transform = field.index.labels[0].values()
            indx_y_transform = field.index.labels[1].values()

            # make arrays
            x_av = np.ma.zeros((bins_x.size - 1, bins_y.size - 1))
            x_av.mask = np.ones((bins_x.size - 1, bins_y.size - 1))
            np.add.at(x_av, [indx_x_transform, indx_y_transform],
                      field['z'].values)
            np.multiply.at(x_av.mask, [indx_x_transform, indx_y_transform], 0)
            # I got this backwards
            x_av = x_av.T
            y_av = np.ma.zeros((bins_x.size - 1, bins_y.size - 1))
            y_av.mask = np.ones((bins_x.size - 1, bins_y.size - 1))
            np.add.at(y_av, [indx_x_transform, indx_y_transform],
                      field['z'].values)
            np.multiply.at(y_av.mask, [indx_x_transform, indx_y_transform], 0)
            # I got this backwards
            y_av = y_av.T
            z_av = np.ma.zeros((bins_x.size - 1, bins_y.size - 1))
            z_av.mask = np.ones((bins_x.size - 1, bins_y.size - 1))
            np.add.at(z_av, [indx_x_transform, indx_y_transform],
                      field['z'].values)
            np.multiply.at(z_av.mask, [indx_x_transform, indx_y_transform], 0)
            # I got this backwards
            z_av = z_av.T
        else:
            x_av = field['x'].values
            y_av = field['y'].values
            z_av = field['z'].values

        return x_av, y_av, z_av


#TODO: Fill this out using decamutil
def convert_pixel_to_mm(xpix, ypix, ext):
    pass
