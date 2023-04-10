
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
    def process(cls, config_stats, logger=None):
        """Parse the stats field of the config dict.

        :param config_stats:    The configuration dict for the stats field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a Stats instance
        """
        import piff

        # If it's not a list, make it one.
        try:
            config_stats[0]
        except KeyError:
            config_stats = [config_stats]

        stats = []
        for cfg in config_stats:

            if 'type' not in cfg:
                raise ValueError("config['stats'] has no type field")

            # Get the class to use for the stats
            stats_class = getattr(piff, cfg['type'] + 'Stats')

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
        kwargs.pop('type',None)
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
        r"""Make the plots for this statistic.

        :param logger:      A logger object for logging debug info. [default: None]
        :param \**kwargs:    Optionally, provide extra kwargs for the matplotlib plot command.

        :returns: (fig, ax) The matplotlib figure and axis with the plot(s).
        """
        raise NotImplementedError("Derived classes must define the plot function")

    def write(self, file_name=None, logger=None, **kwargs):
        r"""Write plots to a file.

        :param file_name:   The name of the file to write to. [default: Use self.file_name,
                            which is typically read from the config field.]
        :param logger:      A logger object for logging debug info. [default: None]
        :param \**kwargs:    Optionally, provide extra kwargs for the matplotlib plot command.
        """
        # Note: don't import matplotlib.pyplot, since that can mess around with the user's
        # pyplot state.  Better to do everything with the matplotlib object oriented API.
        # cf. http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
        import matplotlib
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
        if matplotlib.__version__ >= "3.6":
            fig.set_layout_engine('tight')
        else:  # pragma: no cover
            fig.set_tight_layout(True)
        canvas.print_figure(file_name, dpi=100)

    def measureShapes(self, psf, stars, model_properties=None, fourth_order=False,
                      raw_moments=False, logger=None):
        """Compare PSF and true star shapes with HSM algorithm

        :param psf:              A PSF Object
        :param stars:            A list of Star instances.
        :param model_properties: Optionally a dict of properties to use for the model rendering.
                                 The default behavior is to use the properties of the star itself
                                 for any properties that are not overridden by model_properties.
                                 [default: None]
        :param fourth_order:     Whether to include the fourth-order quantities as well in columns
                                 7 through 11 of the output array. [default: False]
        :param raw_moments:      Whether to include the complete set of raw moments as calculated
                                 by `calculate_moments`.  If requested, these come after the
                                 fourth-order quantities if those are being returned or after the
                                 regular quantities otherwise.  Either way they are the last 18
                                 columns in the output array. [default: False]
        :param logger:           A logger object for logging debug info. [default: None]

        :returns:           positions of stars, shapes of stars, and shapes of
                            models of stars (T, g1, g2)
        """
        from .util import calculate_moments
        logger = galsim.config.LoggerWrapper(logger)
        # measure moments with Gaussian on image
        logger.debug("Measuring shapes of real stars")

        shapes_data = [ list(star.hsm) for star in stars ]
        nshapes = 7

        if fourth_order or raw_moments:
            if fourth_order:
                kwargs = dict(fourth_order=True)
                nshapes += 5
            if raw_moments:
                nshapes += 18
                kwargs = dict(third_order=True, fourth_order=True, radial=True)
            for i, star in enumerate(stars):
                d = shapes_data[i]

                try:
                    m = calculate_moments(star, **kwargs)
                except RuntimeError as e:
                    # Make sure the flag is set.
                    if 'HSM failed' in e.args[0]:
                        d[6] = int(e.args[0].split()[-1])
                    else: # pragma: no cover
                        # The HSM failed error is the only one we expect, but just in case...
                        d[6] = 1
                    d.extend([0] * (nshapes - 7))
                else:
                    if fourth_order:
                        d.extend([m['M22']/m['M11'],
                                m['M31']/m['M11']**2, m['M13']/m['M11']**2,
                                m['M40']/m['M11']**2, m['M04']/m['M11']**2])

                        # Subtract off 3e from the 4th order shapes to remove the leading order
                        # term from the overall ellipticity, which is already captured in the
                        # second order shape.  (For a Gaussian, this makes g4 very close to 0.)
                        shape = galsim.Shear(g1=star.hsm[4], g2=star.hsm[5])
                        d[8] -= 3*shape.e1
                        d[9] -= 3*shape.e2
                    if raw_moments:
                        d.extend([m['M00'], m['M10'], m['M01'], m['M11'], m['M20'], m['M02'],
                                m['M21'], m['M12'], m['M30'], m['M03'],
                                m['M22'], m['M31'], m['M13'], m['M40'], m['M04'],
                                m['M22n'], m['M33n'], m['M44n']])

        # Turn it into a proper numpy array.
        shapes_data = np.array(shapes_data)
        # If no stars, then shapes_data is the wrong shape.  This line is normally a no op
        # but makes things work right if len(stars)=0.
        shapes_data = shapes_data.reshape((len(stars),nshapes))
        for star, shape in zip(stars, shapes_data):
            logger.debug("real shape for star at %s is %s",star.image_pos, shape)

        # Convert from sigma to T
        # Note: the hsm sigma is det(M)^1/4, not sqrt(T/2), so need to account for the effect
        #       of the ellipticity.
        #       If M = [ sigma^2 (1+e1)   sigma^2 e2   ]
        #              [   sigma^2 e2   sigma^2 (1-e1) ]
        #       Then:
        #         det(M) = sigma^4 (1-|e|^2) = sigma_hsm^4
        #         T = tr(M) = 2 * sigma^2
        #       So:
        #         T = 2 * sigma_hsm^2 / sqrt(1-|e|^2)
        # Using |e| = 2 |g| / (1+|g|^2), we obtain:
        #         T = 2 * sigma_hsm^2 * (1+|g|^2)/(1-|g|^2)
        gsq = shapes_data[:,4]**2 + shapes_data[:,5]**2
        shapes_data[:,3] = 2*shapes_data[:,3]**2 * (1+gsq)/(1-gsq)

        # Pull out the positions to return
        positions = np.array([ (star.data.properties['u'], star.data.properties['v'])
                               for star in stars ])

        # generate the model stars and measure moments
        if psf is None:
            shapes_model = None
        else:
            logger.debug("Generating and Measuring Model Stars")
            if model_properties is not None:
                stars = [star.withProperties(**model_properties) for star in stars]
                stars = psf.interpolateStarList(stars)
            model_stars = psf.drawStarList(stars)
            shapes_model = [list(star.hsm) for star in model_stars]

            if fourth_order or raw_moments:
                for i, star in enumerate(model_stars):
                    d = shapes_model[i]
                    try:
                        m = calculate_moments(star, **kwargs)
                    except RuntimeError as e:
                        if 'HSM failed' in e.args[0]:
                            d[6] = int(e.args[0].split()[-1])
                        else: # pragma: no cover
                            d[6] = 1
                        d.extend([0] * (nshapes - 7))
                    else:
                        if fourth_order:
                            d.extend([m['M22']/m['M11'],
                                    m['M31']/m['M11']**2, m['M13']/m['M11']**2,
                                    m['M40']/m['M11']**2, m['M04']/m['M11']**2])
                            shape = galsim.Shear(g1=star.hsm[4], g2=star.hsm[5])
                            d[8] -= 3*shape.e1
                            d[9] -= 3*shape.e2
                        if raw_moments:
                            d.extend([m['M00'], m['M10'], m['M01'], m['M11'], m['M20'], m['M02'],
                                    m['M21'], m['M12'], m['M30'], m['M03'],
                                    m['M22'], m['M31'], m['M13'], m['M40'], m['M04'],
                                    m['M22n'], m['M33n'], m['M44n']])

            shapes_model = np.array(shapes_model)
            shapes_model = shapes_model.reshape((len(model_stars),nshapes))

            gsq = shapes_model[:,4]**2 + shapes_model[:,5]**2
            shapes_model[:,3] = 2*shapes_model[:,3]**2 * (1+gsq)/(1-gsq)
            for star, shape in zip(model_stars, shapes_model):
                logger.debug("model shape for star at %s is %s",star.image_pos, shape)

        return positions, shapes_data, shapes_model


class ShapeHistStats(Stats):
    """Stats class for calculating histograms of shape residuals

    This will compute the size and shapes of the observed stars and the PSF models and
    make histograms of both the values and the residuals.

    The plot will have 6 axes.  The top row will have histograms of T, g1, g2, with the model
    and data color coded.  The bottom row will have histograms of the differences.

    After a call to :func:`compute`, the following attributes are accessible:

        :u:         The u positions in field coordinates.
        :v:         The v positions in field coordinates.
        :T:         The size (T = Iuu + Ivv) of the observed stars.
        :g1:        The g1 component of the shapes of the observed stars.
        :g2:        The g2 component of the shapes of the observed stars.
        :T_model:   The size of the PSF model at the same locations as the stars.
        :g1_model:  The g1 component of the PSF model at these locations.
        :g2_model:  The g2 component of the PSF model at these locations.
        :dT:        The size residual, T - T_model
        :dg1:       The g1 residual, g1 - g1_model
        :dg2:       The g2 residual, g2 - g2_model

    :param file_name:   Name of the file to output to. [default: None]
    :param nbins:       Number of bins to use. [default: sqrt(n_stars)]
    :param cut_frac:    Fraction to cut off from histograms at the high and low ends.
                        [default: 0.01]
    :param model_properties: Optionally a dict of properties to use for the model rendering.
                             [default: None]
    """
    def __init__(self, file_name=None, nbins=None, cut_frac=0.01, model_properties=None,
                 logger=None):
        self.file_name = file_name
        self.nbins = nbins
        self.cut_frac = cut_frac
        self.model_properties = model_properties
        self.skip = False

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        logger.warning("Calculating shape histograms for %d stars",len(stars))
        positions, shapes_data, shapes_model = self.measureShapes(
                psf, stars, model_properties=self.model_properties, logger=logger)

        # Only use stars for which hsm was successful
        flag_data = shapes_data[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_data == 0) & (flag_model == 0)
        if np.sum(mask) == 0:
            logger.warning("All stars had hsm errors.  ShapeHist plot will be empty.")
            self.skip = True

        # define terms for the catalogs
        self.u = positions[mask, 0]
        self.v = positions[mask, 1]
        self.T = shapes_data[mask, 3]
        self.g1 = shapes_data[mask, 4]
        self.g2 = shapes_data[mask, 5]
        self.T_model = shapes_model[mask, 3]
        self.g1_model = shapes_model[mask, 4]
        self.g2_model = shapes_model[mask, 5]
        self.dT = self.T - self.T_model
        self.dg1 = self.g1 - self.g1_model
        self.dg2 = self.g2 - self.g2_model

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:   Any additional kwargs go into the matplotlib hist() function.

        :returns: fig, ax
        """
        logger = galsim.config.LoggerWrapper(logger)
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

        axs[0, 0].set_xlabel(r'$T$')
        axs[1, 0].set_xlabel(r'$T_{data} - T_{model}$')
        axs[0, 1].set_xlabel(r'$g_{1}$')
        axs[1, 1].set_xlabel(r'$g_{1, data} - g_{1, model}$')
        axs[0, 2].set_xlabel(r'$g_{2}$')
        axs[1, 2].set_xlabel(r'$g_{2, data} - g_{2, model}$')
        if self.skip:
            return fig, axs

        if not hasattr(self, 'T'):
            raise RuntimeError("Must call compute before calling plot or write")

        # some defaults for the kwargs
        if 'histtype' not in kwargs:
            kwargs['histtype'] = 'step'

        nbins = self.nbins
        if nbins is None:
            nbins = int(np.sqrt(len(self.T))+1)
            logger.info("nstars = %d, using %d bins for Shape Histograms",len(self.T),nbins)

        if np.__version__ >= '1.23':  # pragma: no branch
            lower = dict(method='lower')
            higher = dict(method='higher')
        else:
            lower = dict(interpolation='lower')
            higher = dict(interpolation='higher')

        # axs[0,0] = size distributions
        ax = axs[0, 0]
        all_T = np.concatenate([self.T_model, self.T])
        logger.info("nbins = %s",nbins)
        logger.info("cut_frac = %s",self.cut_frac)
        rng = (np.quantile(all_T, self.cut_frac, **lower),
               np.quantile(all_T, 1-self.cut_frac, **higher))
        logger.info("T_d: Full range = (%f, %f)",np.min(self.T),np.max(self.T))
        logger.info("T_m: Full range = (%f, %f)",np.min(self.T_model),np.max(self.T_model))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist([self.T, self.T_model], bins=nbins, range=rng, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        # axs[0,1] = size difference
        ax = axs[1, 0]
        rng = (np.quantile(self.dT, self.cut_frac, **lower),
               np.quantile(self.dT, 1-self.cut_frac, **higher))
        logger.info("dT: Full range = (%f, %f)",np.min(self.dT),np.max(self.dT))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist(self.dT, bins=nbins, range=rng, **kwargs)

        # axs[1,0] = g1 distribution
        ax = axs[0, 1]
        all_g1 = np.concatenate([self.g1_model, self.g1])
        rng = (np.quantile(all_g1, self.cut_frac, **lower),
               np.quantile(all_g1, 1-self.cut_frac, **higher))
        logger.info("g1_d: Full range = (%f, %f)",np.min(self.g1),np.max(self.g1))
        logger.info("g1_m: Full range = (%f, %f)",np.min(self.g1_model),np.max(self.g1_model))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist([self.g1, self.g1_model], bins=nbins, range=rng, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        # axs[1,0] = g1 difference
        ax = axs[1, 1]
        rng = (np.quantile(self.dg1, self.cut_frac, **lower),
               np.quantile(self.dg1, 1-self.cut_frac, **higher))
        logger.info("dg1: Full range = (%f, %f)",np.min(self.dg1),np.max(self.dg1))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist(self.dg1, bins=nbins, range=rng, **kwargs)

        # axs[2,0] = g2 distribution
        ax = axs[0, 2]
        all_g2 = np.concatenate([self.g2_model, self.g2])
        rng = (np.quantile(all_g2, self.cut_frac, **lower),
               np.quantile(all_g2, 1-self.cut_frac, **higher))
        logger.info("g2_d: Full range = (%f, %f)",np.min(self.g2),np.max(self.g2))
        logger.info("g2_m: Full range = (%f, %f)",np.min(self.g2_model),np.max(self.g2_model))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist([self.g2, self.g2_model], bins=nbins, range=rng, label=['data', 'model'], **kwargs)
        ax.legend(loc='upper right')
        # axs[2,0] = g2 difference
        ax = axs[1, 2]
        rng = (np.quantile(self.dg2, self.cut_frac, **lower),
               np.quantile(self.dg2, 1-self.cut_frac, **higher))
        logger.info("dg2: Full range = (%f, %f)",np.min(self.dg2),np.max(self.dg2))
        logger.info("Display range = (%f, %f)",rng[0],rng[1])
        ax.hist(self.dg2, bins=nbins, range=rng, **kwargs)

        return fig, ax

class RhoStats(Stats):
    r"""Stats class for calculating rho statistics.

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

    :param min_sep:     Minimum separation (in arcmin) for pairs. [default: 0.5]
    :param max_sep:     Maximum separation (in arcmin) for pairs. [default: 300]
    :param bin_size:    Size of bins in log(sep). [default 0.1]
    :param file_name:   Name of the file to output to. [default: None]
    :param model_properties: Optionally a dict of properties to use for the model rendering.
                             [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    :param \**kwargs:    Any additional kwargs are passed on to TreeCorr.
    """
    def __init__(self, min_sep=0.5, max_sep=300, bin_size=0.1, file_name=None,
                 model_properties=None, logger=None, **kwargs):
        self.tckwargs = kwargs
        self.tckwargs['min_sep'] = min_sep
        self.tckwargs['max_sep'] = max_sep
        self.tckwargs['bin_size'] = bin_size
        if 'sep_units' not in self.tckwargs:
            self.tckwargs['sep_units'] = 'arcmin'
        self.file_name = file_name
        self.model_properties = model_properties
        self.skip = False  # Set this to true if there is a problem and we need to skip plots.

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import treecorr
        treecorr.set_max_omp_threads(1)

        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        logger.warning("Calculating rho statistics for %d stars",len(stars))
        positions, shapes_data, shapes_model = self.measureShapes(
                psf, stars, model_properties=self.model_properties, logger=logger)

        # Only use stars for which hsm was successful
        flag_data = shapes_data[:, 6]
        flag_model = shapes_model[:, 6]
        mask = (flag_data == 0) & (flag_model == 0)
        if np.sum(mask) == 0:
            logger.warning("All stars had hsm errors.  Rho plot will be empty.")
            self.skip = True
            return

        # define terms for the catalogs
        u = positions[mask, 0]
        v = positions[mask, 1]
        T = shapes_data[mask, 3]
        g1 = shapes_data[mask, 4]
        g2 = shapes_data[mask, 5]
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
        treecorr.set_max_omp_threads(None)

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
        for ax in axs:
            ax.set_xlabel('log $r$ [arcmin]')
            ax.set_ylabel(r'$\rho$')
        if self.skip:
            return fig,axs

        # axs[0] gets rho1 rho3 and rho4
        # axs[1] gets rho2 and rho5
        for ax, rho_list, color_list, label_list in zip(
                axs,
                [[self.rho1, self.rho3, self.rho4], [self.rho2, self.rho5]],
                [['k', 'r', 'b'], ['g', 'm']],
                [[r'$\rho_{1}$', r'$\rho_{3}$', r'$\rho_{4}$'],
                 [r'$\rho_{2}$', r'$\rho_{5}$']]):

            # put in some labels
            # set the scale
            ax.set_xscale("log", nonpositive='clip')
            ax.set_yscale("log", nonpositive='clip')
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
        sig = np.sqrt(rho.varxip)
        ax.plot(meanr, xip, color=color)
        ax.plot(meanr, -xip, color=color, ls=':')
        ax.errorbar(meanr[xip>0], xip[xip>0], yerr=sig[xip>0], color=color, ls='', marker=marker)
        ax.errorbar(meanr[xip<0], -xip[xip<0], yerr=sig[xip<0], color=color, ls='', marker=marker,
                    fillstyle='none', mfc='white')
        return ax.errorbar(-meanr, xip, yerr=sig, color=color, marker=marker)

    def plot(self, logger=None, **kwargs):
        r"""Make the plots.

        :param logger:      A logger object for logging debug info. [default: None]
        :params \**kwargs:   Any additional kwargs go into the matplotlib plot() function.
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

        axs[0].set_xlim(self.tckwargs['min_sep'], self.tckwargs['max_sep'])
        axs[0].set_xlabel(r'$\theta$ (arcmin)')
        axs[0].set_ylabel(r'$\rho(\theta)$')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log', nonpositive='clip')

        axs[1].set_xlim(self.tckwargs['min_sep'], self.tckwargs['max_sep'])
        axs[1].set_xlabel(r'$\theta$ (arcmin)')
        axs[1].set_ylabel(r'$\rho(\theta)$')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log', nonpositive='clip')

        if self.skip:
            # If we're skipping the plot, the auto ymax doesn't work well (it uses 1.0),
            # so pick something reasonable here.
            # Otherwise, do this at the end to fix just ymin, but still have an auto ymax.
            axs[0].set_ylim(1.e-9, 1.e-4)
            axs[1].set_ylim(1.e-9, 1.e-4)
            return fig,axs

        if not hasattr(self, 'rho1'):
            raise RuntimeError("Must call compute before calling plot or write")

        # Left plot is rho1,3,4
        rho1 = self._plot_single(axs[0], self.rho1, 'blue', 'o')
        rho3 = self._plot_single(axs[0], self.rho3, 'green', 's', 0.1)
        rho4 = self._plot_single(axs[0], self.rho4, 'red', '^', 0.2)

        axs[0].legend([rho1, rho3, rho4],
                     [r'$\rho_1(\theta)$', r'$\rho_3(\theta)$', r'$\rho_4(\theta)$'],
                     loc='upper right', fontsize=12)

        # Right plot is rho2,5
        rho2 = self._plot_single(axs[1], self.rho2, 'blue', 'o')
        rho5 = self._plot_single(axs[1], self.rho5, 'green', 's', 0.1)

        axs[1].legend([rho2, rho5],
                     [r'$\rho_2(\theta)$', r'$\rho_5(\theta)$'],
                     loc='upper right', fontsize=12)

        axs[0].set_ylim(1.e-9, None)
        axs[1].set_ylim(1.e-9, None)
        return fig, axs

class HSMCatalogStats(Stats):
    r"""Stats class for writing the shape information to an output file.

    This will compute the size and shapes of the observed stars and the PSF models
    and write these data to a file.

    The HSM adaptive momemnt measurements sometimes fail in various ways.  When it does,
    we still output the information that we have for a star, but mark the failure with
    a flag: flag_data for errors in the data measurement, flag_model for errors in the
    model measurement.  The meaning of these flags are (treated as a bit mask):

    Flags:

        0 = Success
        1 = HSM returned a non-zero moments_status.
        2 = HSM returned a negative flux.
        4 = HSM's centroid moved by more than 1 pixel from the input position.

    The output file will include the following columns:

        :ra:        The RA of the star in degrees. (or 0 if the wcs is not a CelestialWCS)
        :dec:       The Declination of the star in degrees. (or 0 if the wcs is not a CelestialWCS)
        :u:         The u position in field coordinates.
        :v:         The v position in field coordinates.
        :x:         The x position in chip coordinates.
        :y:         The y position in chip coordinates.
        :T_data:    The size (T = Iuu + Ivv) of the observed star.
        :g1_data:   The g1 component of the shapes of the observed star.
        :g2_data:   The g2 component of the shapes of the observed star.
        :T_model:   The size of the PSF model at the same locations as the star.
        :g1_model:  The g1 component of the PSF model.
        :g2_model:  The g2 component of the PSF model.
        :reserve:   Whether the star was a reserve star.
        :flag_data: 0 where HSM succeeded on the observed star, >0 where it failed (see above).
        :flag_model: 0 where HSM succeeded on the PSF model, >0 where it failed (see above).

    If fourth_order=True, then there are additional quantities calculated as well.
    We define the following notation:

    .. math::

        T = \int I(u,v) (u^2 + v^2) du dv
        T^{(4)} = \int I(u,v) (u^2 + v^2)^2 du dv / T
        g^{(4)} = \int I(u,v) (u^2 + v^2) (u + iv)^2 du dv / T^2 - 3e
        h^{(4)} = \int I(u,v) (u + iv)^4 du dv / T^2

    I.e. :math:`T^{(4)}` is a fourth-order spin-0 quantity, analogous to :math:`T` at second
    order, :math:`g^{(4)}` is a fourth-order spin-2 quantity, analogous to :math:`g`, and
    :math:`h^{(4)}` is a spin-4 quantity.  The denominators ensure that the units of
    :math:`T^{(4)}` is :math:`arcsec^2`, just like :math:`T` and that :math:`g^{(4)}` and
    :math:`h^{(4)}` are dimensionless.  And the :math:`-3e` term for :math:`g^{(4)}` subtracts
    off the dominant contribution to the fourth order quantity from the second order shape.
    For a pure elliptical Gaussian, this makes :math:`g^{(4)}` come out very close to zero.

    The output file contains the following additional columns:

        :T4_data:   The fourth-order "size", :math:`T^{(4)}`, of the observed star.
        :g41_data:  The real component of :math:`g^{4}` of the obserbed star.
        :g42_data:  The imaginary component of :math:`g^{4}` of the obserbed star.
        :h41_data:  The real component of :math:`h^{4}` of the obserbed star.
        :h42_data:  The imaginary component of :math:`h^{4}` of the obserbed star.
        :T4_model:  The fourth-order "size", :math:`T^{(4)}`, of the PSF model.
        :g41_model: The real component of :math:`g^{4}` of the PSF model.
        :g42_model: The imaginary component of :math:`g^{4}` of the PSF model.
        :h41_model: The real component of :math:`h^{4}` of the PSF model.
        :h42_model: The imaginary component of :math:`h^{4}` of the PSF model.

    :param file_name:        Name of the file to output to. [default: None]
    :param model_properties: Optionally a dict of properties to use for the model rendering.
                             The default behavior is to use the properties of the star itself
                             for any properties that are not overridden by model_properties.
                             [default: None]
    :param fourth_order:     Whether to include the additional fourth-order quantities described
                             above. [default: False]
    :param raw_moments:      Whether to include the complete set of raw moments as calculated
                             by piff.util.calculate_moments. [default: False]
    """
    def __init__(self, file_name=None, model_properties=None, fourth_order=False,
                 raw_moments=False, logger=None):
        self.file_name = file_name
        self.model_properties = model_properties
        self.fourth_order = fourth_order
        self.raw_moments = raw_moments

    def compute(self, psf, stars, logger=None):
        """
        :param psf:         A PSF Object
        :param stars:       A list of Star instances.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        # get the shapes
        logger.warning("Calculating shape histograms for %d stars",len(stars))
        positions, shapes_data, shapes_model = self.measureShapes(
                psf, stars, model_properties=self.model_properties,
                fourth_order=self.fourth_order, raw_moments=self.raw_moments,
                logger=logger)

        # Build the columns for the output catalog
        if isinstance(stars[0].image.wcs, galsim.wcs.CelestialWCS):
            ra = np.array([star.image.wcs.toWorld(star.image_pos).ra.deg for star in stars])
            dec = np.array([star.image.wcs.toWorld(star.image_pos).dec.deg for star in stars])
        else:
            ra = np.zeros(len(stars))
            dec = np.zeros(len(stars))
        self.cols = [
            positions[:, 0],  # u
            positions[:, 1],  # v
            np.array([star.image_pos.x for star in stars]),
            np.array([star.image_pos.y for star in stars]),
            ra, dec,
            shapes_data[:, 0],  # flux
            np.array([s.is_reserve for s in stars], dtype=bool),  # reserve
            shapes_data[:, 6],  # flag_data
            shapes_model[:, 6],  # flag_model
            shapes_data[:, 3],  # T_data
            shapes_data[:, 4],  # g1_data
            shapes_data[:, 5],  # g2_data
            shapes_model[:, 3],  # T_model
            shapes_model[:, 4],  # g1_model
            shapes_model[:, 5],  # g2_model
        ]
        self.dtypes = [('u', float), ('v', float),
                       ('x', float), ('y', float),
                       ('ra', float), ('dec', float),
                       ('flux', float), ('reserve', bool),
                       ('flag_data', int), ('flag_model', int),
                       ('T_data', float), ('g1_data', float), ('g2_data', float),
                       ('T_model', float), ('g1_model', float), ('g2_model', float)]

        if self.fourth_order:
            self.cols.extend(list(shapes_data[:,7:12].T))
            self.cols.extend(list(shapes_model[:,7:12].T))
            self.dtypes.extend([
                ('T4_data', float), ('g41_data', float), ('g42_data', float),
                ('h41_data', float), ('h42_data', float),
                ('T4_model', float), ('g41_model', float), ('g42_model', float),
                ('h41_model', float), ('h42_model', float),
            ])
            k = 12
        else:
            k = 7
        if self.raw_moments:
            self.cols.extend(list(shapes_data[:,k:].T))
            self.cols.extend(list(shapes_model[:,k:].T))
            self.dtypes.extend([
                ('M00_data', float), ('M10_data', float), ('M01_data', float),
                ('M11_data', float), ('M20_data', float), ('M02_data', float),
                ('M21_data', float), ('M12_data', float),
                ('M30_data', float), ('M03_data', float),
                ('M22_data', float), ('M31_data', float), ('M13_data', float),
                ('M40_data', float), ('M04_data', float),
                ('M22n_data', float), ('M33n_data', float), ('M44n_data', float),
                ('M00_model', float), ('M10_model', float), ('M01_model', float),
                ('M11_model', float), ('M20_model', float), ('M02_model', float),
                ('M21_model', float), ('M12_model', float),
                ('M30_model', float), ('M03_model', float),
                ('M22_model', float), ('M31_model', float), ('M13_model', float),
                ('M40_model', float), ('M04_model', float),
                ('M22n_model', float), ('M33n_model', float), ('M44n_model', float),
            ])

        # Also write any other properties saved in the stars.
        prop_keys = list(stars[0].data.properties)
        # Remove all the position ones, which are handled above.
        exclude_keys = ['x', 'y', 'u', 'v', 'ra', 'dec', 'is_reserve']
        prop_keys = [key for key in prop_keys if key not in exclude_keys]
        # Add any remaining properties
        prop_types = stars[0].data.property_types
        for key in prop_keys:
            if not np.isscalar(stars[0].data.properties[key]): # pragma: no cover
                # TODO: This will apply to wavefront, but we don't have any tests that do this yet.
                #       Once we have one, remove the no cover.
                continue
            self.cols.append(np.array([ s.data.properties[key] for s in stars ]))
            # Use the saved type if available, otherwise use float.
            self.dtypes.append((key, prop_types.get(key, float)))

    def write(self, file_name=None, logger=None):
        """Write catalog to file.

        :param file_name:   The name of the file to write to. [default: Use self.file_name,
                            which is typically read from the config field.]
        :param logger:      A logger object for logging debug info. [default: None]
        """
        from . import __version__ as piff_version
        logger = galsim.config.LoggerWrapper(logger)
        import fitsio
        if file_name is None:
            file_name = self.file_name
        if file_name is None:
            raise ValueError("No file_name specified for %s"%self.__class__.__name__)
        if not hasattr(self, 'cols'):
            raise RuntimeError("Must call compute before calling write")

        logger.warning("Writing HSM catalog to file %s",file_name)

        data = np.array(list(zip(*self.cols)), dtype=self.dtypes)
        header = {'piff_version': piff_version}
        with fitsio.FITS(file_name, 'rw', clobber=True) as f:
            f.write_table(data, header=header)
