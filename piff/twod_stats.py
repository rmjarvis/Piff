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
.. module:: twod_stats
"""

from __future__ import print_function
import numpy as np

from .stats import Stats

class TwoDHistStats(Stats):
    """Statistics class that can make pretty colormaps where each bin has some
    arbitrary function applied to it.
    By default this will make a color map based on u and v coordinates of the
    input stars. The color scale is based on (by default) the median value of
    the objects in particular u-v voxel.
    After a call to :func:`compute`, the following attributes are accessible:
        :twodhists:     A dictionary of two dimensional histograms, with keys
                        ['T', 'g1', 'g2',
                         'T_model', 'g1_model', 'g2_model',
                         'dT', 'dg1', 'dg2']
    These histograms are two dimensional masked arrays where the value of the
    pixel corresponds to reducing_function([objects in u-v voxel])
    """

    def __init__(self, number_bins_u=11, number_bins_v=22, reducing_function='np.median', file_name=None, logger=None):
        """
        :param number_bins_u:       Number of bins in u direction [default: 11]
        :param number_bins_v:       Number of bins in v direction [default: 22]
        :param reducing_function:   Type of function to apply to grouped objects. numpy functions are prefixed by np. [default: 'np.median']
        :param file_name:   Name of the file to output to. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        self.number_bins_u = number_bins_u
        self.number_bins_v = number_bins_v
        self.reducing_function = eval(reducing_function)

        self.file_name = file_name

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """

        # get the shapes
        if logger:
            logger.info("Measuring Star and Model Shapes")
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
        T_model = shapes_model[mask, 3]
        g1_model = shapes_model[mask, 4]
        g2_model = shapes_model[mask, 5]
        dT = T - T_model
        dg1 = g1 - g1_model
        dg2 = g2 - g2_model


        # compute the indices
        if logger:
            logger.info("Computing TwoDHist indices")

        # fudge the bins by multiplying 1.01 so that the max entries are in the bins
        self.bins_u = np.linspace(np.min(u), np.max(u) * 1.01, num=self.number_bins_u)
        self.bins_v = np.linspace(np.min(v), np.max(v) * 1.01, num=self.number_bins_v)

        # digitize u and v. No such thing as entries below their min, so -1 to index
        indx_u = np.digitize(u, self.bins_u) - 1
        indx_v = np.digitize(v, self.bins_v) - 1

        # get unique indices
        unique_indx = np.vstack({tuple(row) for row in np.vstack((indx_u, indx_v)).T})

        # compute the arrays
        if logger:
            logger.info("Computing TwoDHist arrays")
        self.twodhists = {}

        # T
        self.twodhists['T'] = self._array_to_2dhist(T, indx_u, indx_v, unique_indx)

        # g1
        self.twodhists['g1'] = self._array_to_2dhist(g1, indx_u, indx_v, unique_indx)

        # g2
        self.twodhists['g2'] = self._array_to_2dhist(g2, indx_u, indx_v, unique_indx)

        # T_model
        self.twodhists['T_model'] = self._array_to_2dhist(T, indx_u, indx_v, unique_indx)

        # g1_model
        self.twodhists['g1_model'] = self._array_to_2dhist(g1_model, indx_u, indx_v, unique_indx)

        # g2_model
        self.twodhists['g2_model'] = self._array_to_2dhist(g2_model, indx_u, indx_v, unique_indx)

        # dT
        self.twodhists['dT'] = self._array_to_2dhist(dT, indx_u, indx_v, unique_indx)

        # dg1
        self.twodhists['dg1'] = self._array_to_2dhist(dg1, indx_u, indx_v, unique_indx)

        # dg2
        self.twodhists['dg2'] = self._array_to_2dhist(dg2, indx_u, indx_v, unique_indx)

    def plot(self, logger=None, **kwargs):
        """Make the plots.
        :param logger:      A logger object for logging debug info. [default: None]
        :params **kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]
        :returns: fig, ax
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 9))
        # left column gets the Y coordinate label
        axs[0, 0].set_ylabel('v')
        axs[1, 0].set_ylabel('v')
        axs[2, 0].set_ylabel('v')

        # bottom row gets the X coordinate label
        axs[2, 0].set_xlabel('u')
        axs[2, 1].set_xlabel('u')
        axs[2, 2].set_xlabel('u')

        # make the colormaps
        if logger:
            logger.info("Creating TwoDHist colormaps")
        # T and T_model share colorbar
        vmin__T = np.min([self.twodhists['T'], self.twodhists['T_model']])
        vmax__T = np.max([self.twodhists['T'], self.twodhists['T_model']])
        cmap__T = self._shift_cmap(vmin__T, vmax__T)
        # g1, g2, g1_model, g2_model share colorbar
        vmin__g = np.min([self.twodhists['g1'], self.twodhists['g1_model'], self.twodhists['g2'], self.twodhists['g2_model']])
        vmax__g = np.max([self.twodhists['g1'], self.twodhists['g1_model'], self.twodhists['g2'], self.twodhists['g2_model']])
        cmap__g = self._shift_cmap(vmin__g, vmax__g)
        # dT gets own colorbar
        vmin__dT = np.min(self.twodhists['dT'])
        vmax__dT = np.max(self.twodhists['dT'])
        cmap__dT = self._shift_cmap(vmin__dT, vmax__dT)
        # dg1 and dg2 share a colorbar
        vmin__dg = np.min([self.twodhists['dg1'], self.twodhists['dg2']])
        vmax__dg = np.max([self.twodhists['dg1'], self.twodhists['dg2']])
        cmap__dg = self._shift_cmap(vmin__dg, vmax__dg)

        # make the plots
        if logger:
            logger.info("Creating TwoDHist plots")
        ax = axs[0, 0]
        ax.set_title('T')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['T'], cmap=cmap__T, vmin=vmin__T, vmax=vmax__T)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 0]
        ax.set_title('T Model')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['T_model'], cmap=cmap__T, vmin=vmin__T, vmax=vmax__T)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 0]
        ax.set_title('dT')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dT'], cmap=cmap__dT, vmin=vmin__dT, vmax=vmax__dT)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[0, 1]
        ax.set_title('g1')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g1'], cmap=cmap__g, vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 1]
        ax.set_title('g1 Model')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g1_model'], cmap=cmap__g, vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 1]
        ax.set_title('dg1')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dg1'], cmap=cmap__dg, vmin=vmin__dg, vmax=vmax__dg)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[0, 2]
        ax.set_title('g2')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g2'], cmap=cmap__g, vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 2]
        ax.set_title('g2 Model')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g2_model'], cmap=cmap__g, vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 2]
        ax.set_title('dg2')
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dg2'], cmap=cmap__dg, vmin=vmin__dg, vmax=vmax__dg)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        plt.tight_layout()

        return fig, axs

    def _array_to_2dhist(self, z, indx_u, indx_v, unique_indx):
        C = np.ma.zeros((self.number_bins_v - 1, self.number_bins_u - 1))
        C.mask = np.ones((self.number_bins_v - 1, self.number_bins_u - 1))

        for unique in unique_indx:
            ui, vi = unique

            sample = z[(indx_u == ui) & (indx_v == vi)]
            if len(sample) > 0:
                value = self.reducing_function(sample)
                C[vi, ui] = value
                C.mask[vi, ui] = 0

        return C

    def _shift_cmap(self, vmin, vmax):
        import matplotlib.pyplot as plt
        midpoint = (0 - vmin) / (vmax - vmin)

        # if b <= 0, then we want Blues_r
        if vmax <= 0:
            return plt.cm.Blues_r
        # if a >= 0, then we want Reds
        elif vmin >= 0:
            return plt.cm.Reds
        else:
            return self._shiftedColorMap(plt.cm.RdBu_r, midpoint=midpoint)

    def _shiftedColorMap(self, cmap, start=0, midpoint=0.5, stop=1.0,
                         name='shiftedcmap'):
        '''
        Taken from
        https://github.com/olgabot/prettyplotlib/blob/master/prettyplotlib/colors.py
        which makes beautiful plots by the way
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = LinearSegmentedColormap(name, cdict)

        # add some overunders
        newcmap.set_bad(color='g', alpha=0.75)
        newcmap.set_over(color='m', alpha=0.75)
        newcmap.set_under(color='c', alpha=0.75)

        plt.register_cmap(cmap=newcmap)

        return newcmap