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
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: psf
"""

import numpy as np
import galsim

from .psf import PSF
from .util import write_kwargs, read_kwargs
from .star import Star, StarFit
from .outliers import Outliers

class SumPSF(PSF):
    """A PSF class that is the Sum of two or more other PSF types.

    A SumPSF is built from an ordered list of other PSFs.

    When fitting the Sum, the default pattern is that all components after the first one
    are initialized to zero, and the first component is fit as usual, but just using a
    single iteration of the fit.  Then the residuals of this model are fit using the second
    component, and so on.  Once all components are fit, outliers may be rejected, and then
    the process is iterated.

    This pattern can be tweaked somewhat using the initialization options available to
    PSF models.  If a component should be initialized to something other than a zero model,
    then one should explicitly set it.

    Use type name "Sum" in a config field to use this psf type.

    :param components:  A list of PSF instances defining the components to be summed.
    :param outliers:    Optionally, an Outliers instance used to remove outliers.
                        [default: None]
    :param chisq_thresh: Change in reduced chisq at which iteration will terminate.
                        [default: 0.1]
    :param max_iter:    Maximum number of iterations to try. [default: 30]
    """
    _type_name = 'Sum'

    def __init__(self, components, outliers=None, chisq_thresh=0.1, max_iter=30):
        self.components = components
        self.outliers = outliers
        self.chisq_thresh = chisq_thresh
        self.max_iter = max_iter
        self.kwargs = {
            # If components is a list, mark the number of components here for I/O purposes.
            # But if it's just a number, leave it alone.
            'components': len(components) if isinstance(components, list) else components,
            'outliers': 0,
            'chisq_thresh': self.chisq_thresh,
            'max_iter': self.max_iter,
        }
        self.chisq = 0.
        self.last_delta_chisq = 0.
        self.dof = 0
        self.nremoved = 0
        self.niter = 0
        self.set_num(None)

    def set_num(self, num):
        """If there are multiple components involved in the fit, set the number to use
        for this model.
        """
        if isinstance(self.components, list):
            # Then building components has been completed.

            # This array keeps track of which num to use for each component.
            self._nums = np.empty(len(self.components), dtype=int)
            self._num_components = 0  # It might not be len(self.components) if some of them are
                                      # in turn composite types.  Figure it out below.

            k1 = 0 if num is None else num
            for k, comp in enumerate(self.components):
                self._nums[k] = k1
                comp.set_num(k1)
                k1 += comp.num_components
                self._num_components += comp.num_components
            self._num = self._nums[0]
        else:
            # else components are not yet built. This will be called again when that's done.
            self._num = None

    @property
    def num_components(self):
        return self._num_components

    @property
    def interp_property_names(self):
        names = set()
        for c in self.components:
            names.update(c.interp_property_names)
        return names

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        from .outliers import Outliers

        kwargs = {}
        kwargs.update(config_psf)
        kwargs.pop('type',None)

        if 'components' not in kwargs:
            raise ValueError("components field is required in psf field for type=Sum")

        # make components
        components = kwargs.pop('components')
        kwargs['components'] = []
        for comp in components:
            kwargs['components'].append(PSF.process(comp, logger=logger))

        if 'outliers' in kwargs:
            outliers = Outliers.process(kwargs.pop('outliers'), logger=logger)
            kwargs['outliers'] = outliers

        return kwargs

    def setup_params(self, stars):
        """Make sure the stars have the right shape params object
        """
        new_stars = []
        for star in stars:
            if star.fit.params is None:
                fit = star.fit.withNew(params=[None] * self.num_components,
                                       params_var=[None] * self.num_components)
                star = Star(star.data, fit)
            else:
                assert len(star.fit.params) > self._nums[-1]
            new_stars.append(star)
        return new_stars

    def initialize_params(self, stars, logger, default_init=None):
        nremoved = 0

        logger.debug("Initializing components of SumPSF")

        # First make sure the params are the right shape.
        stars = self.setup_params(stars)

        # Now initialize all the components
        for comp in self.components:
            stars, nremoved1 = comp.initialize_params(stars, logger, default_init=default_init)
            nremoved += nremoved1
            # After the first one, set default_init to 'zero'
            default_init='zero'

        # If any components are degenerate, mark the sum as degenerate.
        self.degenerate_points = any([comp.degenerate_points for comp in self.components])

        return stars, nremoved

    def single_iteration(self, stars, logger, convert_funcs, draw_method):
        nremoved = 0  # For this iteration

        # Draw the current models for each component
        current_models = [comp.drawStarList(stars) for comp in self.components]

        # Fit each component in order
        for k, comp in enumerate(self.components):
            logger.info("Starting work on component %d (%s)", k, comp._type_name)
            # Update stars to subtract current fit from other components.
            modified_stars = []
            for i, star in enumerate(stars):
                new_image = star.image.copy()
                for kk in range(len(self.components)):
                    if kk != k:
                        new_image -= current_models[kk][i].image
                modified_star = Star(star.data.withNew(image=new_image), star.fit)
                modified_stars.append(modified_star)

            # Fit this component
            new_stars, nremoved1 = comp.single_iteration(modified_stars, logger, convert_funcs,
                                                         draw_method)

            # new_stars now has the updated fit components, but the data parts are corrupted by
            # subtracting the other components during fitting.
            # Update the stars by copying the undisturbed data from the original stars and the
            # updated fits from new_stars.
            stars = [Star(s1.data, s2.fit) for s1,s2 in zip(stars, new_stars)]
            nremoved += nremoved1

            # Update the current models for later components
            stars = comp.interpolateStarList(stars)
            current_models[k] = comp.drawStarList(stars)

        return stars, nremoved

    @property
    def fit_center(self):
        """Whether to fit the center of the star in reflux.

        This is generally set in the model specifications.
        If all component models includes a shift, then this is False.
        Otherwise it is True.
        """
        return any([comp.fit_center for comp in self.components])

    @property
    def include_model_centroid(self):
        """Whether a model that we want to center can have a non-zero centroid during iterations.
        """
        return any([comp.include_model_centroid for comp in self.components])

    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        stars = self.setup_params(stars)
        for comp in self.components:
            stars = comp.interpolateStarList(stars)
        return stars

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        star, = self.setup_params([star])
        for comp in self.components:
            star = comp.interpolateStar(star)
        return star

    def _drawStar(self, star):
        # Draw each component
        comp_stars = []
        for comp in self.components:
            comp_star = comp._drawStar(star)
            comp_stars.append(comp_star)

        # Add them up.
        image = star.image.copy()
        image.setZero()
        for comp_star in comp_stars:
            image += comp_star.image
        return Star(star.data.withNew(image=image), star.fit)

    def _getRawProfile(self, star):
        # Get each component profile
        profiles = []
        methods = []
        for comp in self.components:
            prof, method = comp._getRawProfile(star)
            profiles.append(prof)
            methods.append(method)
        if any([m != methods[0] for m in methods]):
            assert all([m == 'no_pixel' or m == 'auto' for m in methods])
            # Then some of these are auto, and others are no_pixel.
            # Call the whole thing no_pixel, and manually convolve the auto components.
            pixel = star.data.local_wcs.toWorld(galsim.Pixel(1.0))
            new_profiles = []
            for p,m in zip(profiles, methods):
                if m == 'auto':
                    new_profiles.append(galsim.Convolve(p, pixel))
                else:
                    new_profiles.append(p)
            profiles = new_profiles
            method = 'no_pixel'
        else:
            method = methods[0]

        # Add them up.
        return galsim.Sum(profiles), method

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        logger = galsim.config.LoggerWrapper(logger)
        chisq_dict = {
            'chisq' : self.chisq,
            'last_delta_chisq' : self.last_delta_chisq,
            'dof' : self.dof,
            'nremoved' : self.nremoved,
        }
        write_kwargs(fits, extname + '_chisq', chisq_dict)
        logger.debug("Wrote the chisq info to extension %s",extname + '_chisq')
        for k, comp in enumerate(self.components):
            comp._write(fits, extname + '_' + str(k), logger=logger)
        if self.outliers:
            self.outliers.write(fits, extname + '_outliers')
            logger.debug("Wrote the PSF outliers to extension %s",extname + '_outliers')

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        chisq_dict = read_kwargs(fits, extname + '_chisq')
        for key in chisq_dict:
            setattr(self, key, chisq_dict[key])

        ncomponents = self.components
        self.components = []
        for k in range(ncomponents):
            self.components.append(PSF._read(fits, extname + '_' + str(k), logger=logger))
        if extname + '_outliers' in fits:
            self.outliers = Outliers.read(fits, extname + '_outliers')
        else:
            self.outliers = None
        # Set up all the num's properly now that everything is constructed.
        self.set_num(None)
