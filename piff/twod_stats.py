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

import numpy as np
import warnings
import galsim

from .stats import Stats

class TwoDHistStats(Stats):
    """Statistics class that can make pretty colormaps where each bin has some
    arbitrary function applied to it.

    By default this will make a color map based on u and v coordinates of the
    input stars. The color scale is based on (by default) the median value of
    the objects in particular u-v voxel.

    After a call to :func:`compute`, the following attributes are accessible:

        :twodhists:     A dictionary of two dimensional histograms, with keys
                        'u', 'v', 'T', 'g1', 'g2', 'T_model', 'g1_model', 'g2_model',
                        'dT', 'dg1', 'dg2'

    These histograms are two dimensional masked arrays where the value of the
    pixel corresponds to reducing_function([objects in u-v voxel])

    :param nbins_u:             Number of bins in u direction [default: 20]
    :param nbins_v:             Number of bins in v direction [default: 20]
    :param reducing_function:   Type of function to apply to grouped objects. numpy functions
                                are prefixed by np. [default: 'np.median']
    :param file_name:           Name of the file to output to. [default: None]
    :param logger:              A logger object for logging debug info. [default: None]
    """
    def __init__(self, nbins_u=20, nbins_v=20, reducing_function='np.median',
                 file_name=None, logger=None):
        self.nbins_u = nbins_u
        self.nbins_v = nbins_v
        self.reducing_function = eval(reducing_function)

        self.file_name = file_name
        self.skip = False

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.twodhists = {}

        # get the shapes
        logger.info("Measuring Star and Model Shapes")
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # Only use stars for which hsm was successful
        flag_truth = shapes_truth[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_truth == 0) & (flag_model == 0)
        logger.info("%d/%d measurements were successful",np.sum(mask),len(mask))
        if np.sum(mask) == 0:
            logger.warning("All stars had hsm errors.  TwoDHist plot will be empty.")
            self.skip = True
            return

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
        logger.info("Computing TwoDHist indices")

        self.bins_u = np.linspace(np.min(u), np.max(u), num=self.nbins_u + 1)
        self.bins_v = np.linspace(np.min(v), np.max(v), num=self.nbins_v + 1)

        # digitize u and v. No such thing as entries below their min, so -1 to index
        indx_u = np.digitize(u, self.bins_u) - 1
        indx_v = np.digitize(v, self.bins_v) - 1

        # Make sure no points go one past the end (due to rounding when u=max(u), etc.
        np.putmask(indx_u, indx_u >= self.nbins_u, self.nbins_u-1)
        np.putmask(indx_v, indx_v >= self.nbins_v, self.nbins_v-1)

        # get unique indices
        unique_indx = np.vstack([tuple(row) for row in np.vstack((indx_u, indx_v)).T])

        # compute the arrays
        logger.info("Computing TwoDHist arrays")

        # throw in coordinates for good measure
        self.twodhists['u'] = self._array_to_2dhist(u, indx_u, indx_v, unique_indx)
        self.twodhists['v'] = self._array_to_2dhist(v, indx_u, indx_v, unique_indx)
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
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]

        :returns: fig, ax
        """
        from matplotlib.figure import Figure
        logger = galsim.config.LoggerWrapper(logger)
        fig = Figure(figsize=(12,9))
        # In matplotlib 2.0, this will be
        # axs = fig.subplots(ncols=3, nrows=3)
        axs = [[ fig.add_subplot(3,3,1),
                 fig.add_subplot(3,3,2),
                 fig.add_subplot(3,3,3) ],
               [ fig.add_subplot(3,3,4),
                 fig.add_subplot(3,3,5),
                 fig.add_subplot(3,3,6) ],
               [ fig.add_subplot(3,3,7),
                 fig.add_subplot(3,3,8),
                 fig.add_subplot(3,3,9) ]]
        axs = np.array(axs, dtype=object)

        # left column gets the Y coordinate label
        axs[0, 0].set_ylabel('v')
        axs[1, 0].set_ylabel('v')
        axs[2, 0].set_ylabel('v')

        # bottom row gets the X coordinate label
        axs[2, 0].set_xlabel('u')
        axs[2, 1].set_xlabel('u')
        axs[2, 2].set_xlabel('u')

        axs[0, 0].set_title('T')
        axs[1, 0].set_title('T Model')
        axs[2, 0].set_title('dT')
        axs[0, 1].set_title('g1')
        axs[1, 1].set_title('g1 Model')
        axs[2, 1].set_title('dg1')
        axs[0, 2].set_title('g2')
        axs[1, 2].set_title('g2 Model')
        axs[2, 2].set_title('dg2')
        if self.skip:
            return fig, axs

        if not hasattr(self, 'twodhists'):
            raise RuntimeError("Must call compute before calling plot or write")

        # make the colormaps
        logger.info("Creating TwoDHist colormaps")
        # T and T_model share colorbar
        vmin__T = np.min([self.twodhists['T'], self.twodhists['T_model']])
        vmax__T = np.max([self.twodhists['T'], self.twodhists['T_model']])
        vcent__T = np.median([self.twodhists['T'], self.twodhists['T_model']])
        cmap__T = self._shift_cmap(vmin__T, vmax__T, center=vcent__T)
        # g1, g2, g1_model, g2_model share colorbar
        vmin__g = np.min([self.twodhists['g1'], self.twodhists['g1_model'],
                          self.twodhists['g2'], self.twodhists['g2_model']])
        vmax__g = np.max([self.twodhists['g1'], self.twodhists['g1_model'],
                          self.twodhists['g2'], self.twodhists['g2_model']])
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
        logger.info("Creating TwoDHist plots")
        ax = axs[0, 0]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['T'], cmap=cmap__T,
                       vmin=vmin__T, vmax=vmax__T)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 0]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['T_model'], cmap=cmap__T,
                       vmin=vmin__T, vmax=vmax__T)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 0]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dT'], cmap=cmap__dT,
                       vmin=vmin__dT, vmax=vmax__dT)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[0, 1]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g1'], cmap=cmap__g,
                       vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 1]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g1_model'], cmap=cmap__g,
                       vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 1]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dg1'], cmap=cmap__dg,
                       vmin=vmin__dg, vmax=vmax__dg)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[0, 2]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g2'], cmap=cmap__g,
                       vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[1, 2]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['g2_model'], cmap=cmap__g,
                       vmin=vmin__g, vmax=vmax__g)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        ax = axs[2, 2]
        IM = ax.pcolor(self.bins_u, self.bins_v, self.twodhists['dg2'], cmap=cmap__dg,
                       vmin=vmin__dg, vmax=vmax__dg)
        ax.set_xlim(min(self.bins_u), max(self.bins_u))
        ax.set_ylim(min(self.bins_v), max(self.bins_v))
        fig.colorbar(IM, ax=ax)

        return fig, axs

    def _array_to_2dhist(self, z, indx_u, indx_v, unique_indx):
        C = np.ma.zeros((self.nbins_v, self.nbins_u))
        C.mask = np.ones((self.nbins_v, self.nbins_u))

        for unique in unique_indx:
            ui, vi = unique

            sample = z[(indx_u == ui) & (indx_v == vi)]
            assert len(sample) > 0  # This is ensured by how we calculate unique_indx
            value = self.reducing_function(sample)
            C[vi, ui] = value
            C.mask[vi, ui] = 0

        return C

    @classmethod
    def _shift_cmap(self, vmin, vmax, center=0):
        from matplotlib import cm
        # want midpoint to be a float!
        midpoint = (center - vmin) * 1. / (vmax - vmin)

        # if b <= 0, then we want Blues_r
        if vmax <= center and vmin <= center:
            return cm.Blues_r
        # if a >= 0, then we want Reds
        elif vmin >= center and vmax >= center:
            return cm.Reds
        # catch inverse
        elif vmin >= center and vmax <= center:
            return self._shiftedColorMap(cm.RdBu_r, start=1.0, midpoint=1 - midpoint, stop=0.0)
        else:
            return self._shiftedColorMap(cm.RdBu_r, midpoint=midpoint)

    @classmethod
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

        return newcmap

class WhiskerStats(Stats):
    """Statistics class that can make whiskerplots.

    By default this will make a whisker plot based on u and v coordinates of
    the input stars. The whisker scale is based on (by default) the median
    value of the objects in a particular u-v voxel.

    After a call to :func:`compute`, the following attributes are accessible:

        :twodhists:     A dictionary of two dimensional histograms, with keys
                        'u', 'v', 'w1', 'w2', 'w1_model', 'w2_model', 'dw1', 'dw2'

    These histograms are two dimensional masked arrays where the value of the
    pixel corresponds to reducing_function([objects in u-v voxel])

    .. note::

        There are a couple different ways to define your whiskers. Here we
        have taken the approach that the whisker represents the ellipticity as:

            theta = arctan(e2, e1) / 2
            r = sqrt(e1 ** 2 + e2 ** 2)
            w1 = r cos(theta)
            w2 = r sin(theta)

    Because e1, e2 do not have units, w does not either.

    :param file_name:           Name of the file to output to. [default: None]
    :param nbins_u:             Number of bins in u direction [default: 20]
    :param nbins_v:             Number of bins in v direction [default: 20]
    :param reducing_function:   Type of function to apply to grouped objects. numpy functions
                                are prefixed by np. [default: 'np.median']
    :param scale:               An overal scale factor by which to scale the size of the
                                all whiskers. [default: 1]
    :param resid_scale:         An additional factor for the scale size of the residual
                                whiskers only. [default: 2]
    :param logger:              A logger object for logging debug info. [default: None]
    """
    def __init__(self, file_name=None, nbins_u=20, nbins_v=20, reducing_function='np.median',
                 scale=1, resid_scale=2, logger=None):
        self.file_name = file_name
        self.nbins_u = nbins_u
        self.nbins_v = nbins_v
        self.reducing_function = eval(reducing_function)
        self.scale = scale
        self.resid_scale = resid_scale
        self.skip = False

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        self.twodhists = {}

        # get the shapes
        logger.info("Measuring Star and Model Shapes")
        positions, shapes_truth, shapes_model = self.measureShapes(psf, stars, logger=logger)

        # Only use stars for which hsm was successful
        flag_truth = shapes_truth[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_truth == 0) & (flag_model == 0)
        logger.info("%d/%d measurements were successful",np.sum(mask),len(mask))
        if np.sum(mask) == 0:
            logger.warning("All stars had hsm errors.  Whisker plot will be empty.")
            self.skip = True
            return

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

        mag_w = np.sqrt(np.square(g1) + np.square(g2))
        phi = np.arctan2(g2, g1) / 2.
        w1 = mag_w * np.cos(phi)
        w2 = mag_w * np.sin(phi)
        mag_w_model = np.sqrt(np.square(g1_model) + np.square(g2_model))
        phi_model = np.arctan2(g2_model, g1_model) / 2.
        w1_model = mag_w_model * np.cos(phi_model)
        w2_model = mag_w_model * np.sin(phi_model)
        dmag_w = np.sqrt(np.square(dg1) + np.square(dg2))
        dphi = np.arctan2(dg2, dg1) / 2.
        dw1 = dmag_w * np.cos(dphi)
        dw2 = dmag_w * np.sin(dphi)

        # compute the indices
        logger.info("Computing Whisker indices")

        self.bins_u = np.linspace(np.min(u), np.max(u), num=self.nbins_u + 1)
        self.bins_v = np.linspace(np.min(v), np.max(v), num=self.nbins_v + 1)

        # digitize u and v. No such thing as entries below their min, so -1 to index
        indx_u = np.digitize(u, self.bins_u) - 1
        indx_v = np.digitize(v, self.bins_v) - 1

        # Make sure no points go one past the end (due to rounding when u=max(u), etc.
        np.putmask(indx_u, indx_u >= self.nbins_u, self.nbins_u-1)
        np.putmask(indx_v, indx_v >= self.nbins_v, self.nbins_v-1)

        # get unique indices
        unique_indx = np.vstack([tuple(row) for row in np.vstack((indx_u, indx_v)).T])

        # compute the arrays
        logger.info("Computing Whisker arrays")

        self.twodhists['u'] = self._array_to_2dhist(u, indx_u, indx_v, unique_indx)
        self.twodhists['v'] = self._array_to_2dhist(v, indx_u, indx_v, unique_indx)

        # w1
        self.twodhists['w1'] = self._array_to_2dhist(w1, indx_u, indx_v, unique_indx)

        # w2
        self.twodhists['w2'] = self._array_to_2dhist(w2, indx_u, indx_v, unique_indx)

        # w1_model
        self.twodhists['w1_model'] = self._array_to_2dhist(w1_model, indx_u, indx_v, unique_indx)

        # w2_model
        self.twodhists['w2_model'] = self._array_to_2dhist(w2_model, indx_u, indx_v, unique_indx)

        # dw1
        self.twodhists['dw1'] = self._array_to_2dhist(dw1, indx_u, indx_v, unique_indx)

        # dw2
        self.twodhists['dw2'] = self._array_to_2dhist(dw2, indx_u, indx_v, unique_indx)

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:   Any additional kwargs go into the matplotlib plot() function.
                            [ignored in this function]

        :returns: fig, ax
        """
        from matplotlib.figure import Figure
        logger = galsim.config.LoggerWrapper(logger)
        fig = Figure(figsize=(7.5,4))
        ax0, ax1 = fig.subplots(1, 2, sharey=True, subplot_kw={'aspect' : 'equal'})

        ax0.axis('off')
        ax0.set_title('Raw PSF')
        #ax0.set_xlabel('u')
        #ax0.set_ylabel('v')
        ax1.axis('off')
        ax1.set_title('PSF Residuals')
        #ax1.set_xlabel('u')
        #ax1.set_ylabel('v')
        if self.skip:
            return fig, [ax0, ax1]

        if not hasattr(self, 'twodhists'):
            raise RuntimeError("Must call compute before calling plot or write")

        # make the plots
        logger.info("Creating Whisker plots")

        # configure to taste
        # bigger scale = smaller whiskers
        quiver_dict = dict(headlength=0,
                           headwidth=0,
                           headaxislength=0,
                           minlength=0,
                           pivot='middle',
                           scale_units='xy',
                           width=0.001,
                           color='blue',
                           )

        # raw whiskers
        u = self.twodhists['u']
        v = self.twodhists['v']
        w1 = self.twodhists['w1']
        w2 = self.twodhists['w2']
        dw1 = self.twodhists['dw1']
        dw2 = self.twodhists['dw2']
        use = u.mask == 0
        # Note: the quiver "scale" is such that larger value = smaller whiskers.
        #       This is backwards of how I think of scale, which is why the / here, not *.
        scale = 1.e-4 / self.scale
        qv = ax0.quiver(u[use], v[use], w1[use], w2[use], scale=scale, **quiver_dict)
        # quiverkey
        ref = 0.03
        ref_label = 'e = %s'%ref
        ax0.quiverkey(qv, 0.15, 0.00, ref, ref_label,
                      coordinates='axes', color='darkred', labelcolor='darkred',
                      labelpos='S')

        # residual whiskers
        scale = 1.e-4 / self.scale / self.resid_scale
        qv = ax1.quiver(u[use], v[use], dw1[use], dw2[use], scale=scale, **quiver_dict)
        # quiverkey
        ref = 0.03 / self.resid_scale
        ref_label = 'e = %s'%ref
        ax1.quiverkey(qv, 0.85, 0.00, ref, ref_label,
                      coordinates='axes', color='darkred', labelcolor='darkred',
                      labelpos='S')

        return fig, [ax0, ax1]

    def _array_to_2dhist(self, z, indx_u, indx_v, unique_indx):
        C = np.ma.zeros((self.nbins_v, self.nbins_u))
        C.mask = np.ones((self.nbins_v, self.nbins_u))

        for unique in unique_indx:
            ui, vi = unique

            sample = z[(indx_u == ui) & (indx_v == vi)]
            assert len(sample) > 0
            value = self.reducing_function(sample)
            C[vi, ui] = value
            C.mask[vi, ui] = 0

        return C
