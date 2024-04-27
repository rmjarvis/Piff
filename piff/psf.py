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
import fitsio
import galsim
import sys

from .star import Star, StarData
from .util import write_kwargs, read_kwargs

class PSF(object):
    """The base class for describing a PSF model across a field of view.

    The usual way to create a PSF is through one of the two factory functions::

        >>> psf = piff.process(config, logger)
        >>> psf = piff.read(file_name, logger)

    The first is used to build a PSF model from the data according to a config dict.
    The second is used to read in a PSF model from disk.
    """
    # This class-level dict will store all the valid PSF types.
    # Each subclass should set a cls._type_name, which is the name that should
    # appear in a config dict.  These will be the keys of valid_psf_types.
    # The values in this dict will be the PSF sub-classes.
    valid_psf_types = {}

    @classmethod
    def process(cls, config_psf, logger=None):
        """Process the config dict and return a PSF instance.

        As the PSF class is an abstract base class, the returned type will in fact be some
        subclass of PSF according to the contents of the config dict.

        The provided config dict is typically the 'psf' field in the base config dict in
        a YAML file, although for compound PSF types, it may be the field for just one of
        several components.

        This function merely creates a "blank" PSF object.  It does not actually do any
        part of the solution yet.  Typically this will be followed by fit:

            >>> psf = piff.PSF.process(config['psf'])
            >>> stars, wcs, pointing = piff.Input.process(config['input'])
            >>> psf.fit(stars, wcs, pointing)

        at which point, the ``psf`` instance would have a solution to the PSF model.

        :param config_psf:  A dict specifying what type of PSF to build along with the
                            appropriate kwargs for building it.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance of the appropriate type.
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Parsing PSF based on config dict:")

        # Get the class to use for the PSF
        psf_type = config_psf.get('type', 'Simple')
        if psf_type not in PSF.valid_psf_types:
            raise ValueError("type %s is not a valid psf type. "%psf_type +
                             "Expecting one of %s"%list(PSF.valid_psf_types.keys()))
        logger.debug("PSF type is %s",psf_type)

        psf_cls = PSF.valid_psf_types[psf_type]

        # Read any other kwargs in the psf field
        kwargs = psf_cls.parseKwargs(config_psf, logger)

        # Build PSF object
        logger.info("Building %s",psf_type)
        psf = psf_cls(**kwargs)
        logger.debug("Done building PSF")

        # At top level, the num is always None.
        # Composite PSF types will turn this into a series of integer values for each component.
        psf.set_num(None)

        return psf

    def set_num(self, num):
        """If there are multiple components involved in the fit, set the number to use
        for this model.
        """
        # Normally subclasses will need to propagate this further.
        # But this is the minimum action that all subclasses need to do.
        self._num = num

    @property
    def num_components(self):
        # Subclasses for which this is not true can overwrite this
        return 1

    @classmethod
    def __init_subclass__(cls):
        # Classes that don't want to register a type name can either not define _type_name
        # or set it to None.
        if hasattr(cls, '_type_name') and cls._type_name is not None:
            if cls._type_name in PSF.valid_psf_types:
                raise ValueError('PSF type %s already registered'%cls._type_name +
                                 'Maybe you subclassed and forgot to set _type_name?')
            PSF.valid_psf_types[cls._type_name] = cls

    @classmethod
    def parseKwargs(cls, config_psf, logger=None):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        raise NotImplementedError("Derived classes must define the parseKwargs function")

    def initialize_flux_center(self, stars, logger=None):
        """Initialize the flux and center of the stars.

        The flux is just a simple sum of unmasked pixels.
        The center is a simple centroid relative to the nominal position.  It is only updated
        if the PSF model is centered. (I.e. if self.fit_center is True.)

        :param stars:           The initial list of Star instances that will be used to constrain
                                the PSF.
        :param logger:          A logger object for logging progress. [default: None]

        :returns: the initialized stars
        """
        new_stars = []
        for star in stars:
            data, weight, u, v = star.data.getDataVector()
            # Start with the sum of pixels as initial estimate of flux.
            # (Skip w=0 pixels here.)
            mask = weight!=0
            flux = np.sum(data[mask])
            if self.fit_center and not star.data.properties.get('trust_pos',False):
                flux = flux if flux != 0 else 1  # Don't divide by 0.
                # Initial center is the centroid of the data.
                Ix = np.sum(data[mask] * u[mask]) / flux
                Iy = np.sum(data[mask] * v[mask]) / flux
                center = (Ix,Iy)
            else:
                center = star.fit.center
            new_stars.append(Star(star.data, star.fit.withNew(flux=flux, center=center)))
        return new_stars

    def initialize_params(self, stars, logger=None, default_init=None):
        """Initialize the psf solver to begin an iterative solution.

        :param stars:           The initial list of Star instances that will be used to constrain
                                the PSF.
        :param logger:          A logger object for logging progress. [default: None]
        :param default_init:    The default initilization method if the user doesn't specify one.
                                [default: None]

        :returns: the initialized stars, nremoved
        """
        # Probably most derived classes need to do something here, but if not the default
        # behavior is just to return the input stars list.
        return stars, 0

    def single_iteration(self, stars, logger, convert_funcs, draw_method):
        """Perform a single iteration of the solver.

        Note that some object might fail at some point in the fitting, so some objects can be
        flagged as bad during this step, prior to the outlier rejection step.  This information
        is reported in the return tuple as nremoved.

        :param stars:           The list of stars to use for constraining the PSF.
        :param logger:          A logger object for logging progress.
        :param convert_funcs:   An optional list of function to apply to the profiles being fit
                                before drawing it onto the image.  If not None, it should be the
                                same length as stars.
        :param draw_method:     The method to use with the GalSim drawImage command. If None,
                                use the default method for the PSF model being fit.

        :returns: an updated list of all_stars, nremoved
        """
        raise NotImplementedError("Derived classes must define the single_iteration function")

    @property
    def fit_center(self):
        """Whether to fit the center of the star in reflux.

        This is generally set in the model specifications.
        If all component models includes a shift, then this is False.
        Otherwise it is True.
        """
        raise NotImplementedError("Derived classes must define the fit_center property")

    @property
    def include_model_centroid(self):
        """Whether a model that we want to center can have a non-zero centroid during iterations.
        """
        raise NotImplementedError("Derived classes must define the include_model_centroid property")

    def reflux(self, star, logger=None):
        """Fit the PSF to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.  DOF in the result assume
        only flux (& center) are free parameters.

        :param star:        A Star instance
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           New Star instance, with updated flux, center, chisq, dof
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Reflux for star:")
        logger.debug("    flux = %s",star.fit.flux)
        logger.debug("    center = %s",star.fit.center)
        logger.debug("    props = %s",star.data.properties)
        logger.debug("    image = %s",star.data.image)
        #logger.debug("    image = %s",star.data.image.array)
        #logger.debug("    weight = %s",star.data.weight.array)
        logger.debug("    image center = %s",star.data.image(star.data.image.center))
        logger.debug("    weight center = %s",star.data.weight(star.data.weight.center))

        data, weight, u, v = star.data.getDataVector()
        model_star = self.drawStar(star)
        model = model_star.image.array.ravel()

        # Weight by the current model to avoid being influenced by spurious pixels near the edge
        # of the stamp.
        # Also by the weight map to avoid bad pixels.
        W = weight * model
        WD = W * data
        WM = W * model

        f_data = np.sum(WD)
        f_model = np.sum(WM) or 1  # Don't let f_model = 0
        flux_ratio = f_data / f_model

        new_flux = star.fit.flux * flux_ratio
        logger.debug('    new flux = %s', new_flux)

        model *= flux_ratio
        resid = data - model

        if self.fit_center and not star.data.properties.get('trust_pos',False):
            psf_prof, method = self._getRawProfile(model_star)

            # Use finite different to approximate d(model)/duc, d(model)/dvc
            duv = 1.e-5
            temp = star.image.copy()
            center = star.fit.center
            du_prof = psf_prof.shift(center[0] + duv, center[1]) * new_flux
            du_prof.drawImage(temp, method=method, center=star.image_pos)
            dmduc = (temp.array.ravel() - model) / duv
            dv_prof = psf_prof.shift(center[0], center[1] + duv) * new_flux
            dv_prof.drawImage(temp, method=method, center=star.image_pos)
            dmdvc = (temp.array.ravel() - model) / duv

            # Also dmdflux
            dflux = 1.e-5 * max(abs(new_flux), 1.e-5)  # Guard against division by 0
            df_prof = psf_prof.shift(center[0], center[1]) * (new_flux + dflux)
            df_prof.drawImage(temp, method=method, center=star.image_pos)
            dmdf = (temp.array.ravel() - model) / dflux

            # Now construct the design matrix for this minimization
            #
            #    A x = b
            #
            # where x = [ df, duc, dvc ]^T and b = resid.
            #
            # A[0] = dmdf
            # A[1] = dmduc
            # A[2] = dmdvc
            #
            # Solve: AT A x = AT b

            At = np.vstack((dmdf, dmduc, dmdvc))
            Atw = At * np.abs(W)  # weighted least squares
            AtA = Atw.dot(At.T)
            Atb = Atw.dot(resid)
            x = np.linalg.solve(AtA, Atb)
            logger.debug('    centroid shift = %s,%s', x[1], x[2])

            # Extract the values we want.
            df, duc, dvc = x

            if self.include_model_centroid and psf_prof.centroid != galsim.PositionD(0,0):
                # In addition to shifting to the best fit center location, also shift
                # by the centroid of the model itself, so the next next pass through the
                # fit will be closer to centered.  In practice, this converges pretty quickly.
                model_cenu = np.sum(WM * u) / f_model - star.fit.center[0]
                model_cenv = np.sum(WM * v) / f_model - star.fit.center[1]
                logger.debug('    model centroid = %s,%s', model_cenu, model_cenv)
                duc += model_cenu
                dvc += model_cenv

            new_center = (star.fit.center[0] + duc, star.fit.center[1] + dvc)
            new_flux += df
            logger.debug('    new center = %s', new_center)
            logger.debug('    new flux = %s', new_flux)

            new_chisq = np.sum((resid-At.T.dot(x))**2 * weight)
            new_dof = np.count_nonzero(weight) - 3
        else:
            new_center = star.fit.center
            new_chisq = np.sum(resid**2 * weight)
            new_dof = np.count_nonzero(weight) - 1

        logger.debug("    new_chisq = %s",new_chisq)
        logger.debug("    new_dof = %s",new_dof)

        return Star(star.data, star.fit.withNew(flux=new_flux,
                                                center=new_center,
                                                chisq=new_chisq,
                                                dof=new_dof))

    def reflux_stars(self, stars, logger):
        """Calculate new flux and possibly center for a list of stars
        """
        new_stars = []
        nremoved = 0
        for star in stars:
            try:
                star = self.reflux(star, logger=logger)
            except Exception as e:
                logger.warning("Failed trying to reflux star at %s.  Excluding it.",
                                star.image_pos)
                logger.warning("  -- Caught exception: %s", e)
                nremoved += 1
                star = star.flag_if(True)
            new_stars.append(star)
        return new_stars, nremoved

    def remove_outliers(self, stars, iteration, logger):
        """Look for and flag outliers from the list of stars

        :param stars:           The complete list of stars to consider
        :param iteration:       The number of the iteration that was just completed.
        :param logger:          A logger object for logging progress.

        :returns: new_stars, nremoved
        """
        # Perform outlier rejection, but not on first iteration for degenerate solvers.
        if self.outliers and (iteration > 0 or not self.degenerate_points):
            logger.debug("             Looking for outliers")
            stars, nremoved = self.outliers.removeOutliers(stars, logger=logger)
            if nremoved == 0:
                logger.debug("             No outliers found")
            else:
                logger.info("             Removed %d outliers", nremoved)
        else:
            nremoved = 0
        return stars, nremoved

    def fit(self, stars, wcs, pointing, logger=None, convert_funcs=None, draw_method=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        :param convert_funcs:   An optional list of function to apply to the profiles being fit
                                before drawing it onto the image.  This is used by composite PSFs
                                to isolate the effect of just this model component.  If provided,
                                it should be the same length as stars. [default: None]
        :param draw_method:     The method to use with the GalSim drawImage command. If not given,
                                use the default method for the PSF model being fit. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)

        self.wcs = wcs
        self.pointing = pointing

        # Initialize stars as needed by the PSF modeling class.
        stars = self.initialize_flux_center(stars, logger=logger)
        stars, self.nremoved = self.initialize_params(stars, logger=logger)
        nreserve = np.sum([star.is_reserve for star in stars])

        oldchisq = 0.
        for iteration in range(self.max_iter):

            nstars = np.sum([not star.is_reserve and not star.is_flagged for star in stars])
            logger.warning("Iteration %d: Fitting %d stars", iteration+1, nstars)
            if nreserve != 0:
                logger.warning("             (%d stars are reserved)", nreserve)

            # Run a single iteration of the fitter.
            # Different PSF types do different things here.
            stars, iter_nremoved = self.single_iteration(stars, logger, convert_funcs, draw_method)

            # Update estimated poisson noise
            signals = self.drawStarList(stars)
            stars = [s.addPoisson(signal) for s, signal in zip(stars, signals)]

            # Refit and recenter stars, collect stats
            logger.debug("             Re-fluxing stars")
            stars, reflux_nremoved = self.reflux_stars(stars, logger)
            iter_nremoved += reflux_nremoved

            # Find and flag outliers.
            stars, outlier_nremoved = self.remove_outliers(stars, iteration, logger)
            iter_nremoved += outlier_nremoved

            chisq = np.sum([s.fit.chisq for s in stars if not s.is_reserve and not s.is_flagged])
            dof   = np.sum([s.fit.dof for s in stars if not s.is_reserve and not s.is_flagged])
            logger.warning("             Total chisq = %.2f / %d dof", chisq, dof)

            # Save these so we can write them to the output file.
            self.chisq = chisq
            self.last_delta_chisq = oldchisq-chisq
            self.dof = dof

            # Keep track of the total number removed in all iterations.
            self.nremoved += iter_nremoved
            self.niter = iteration+1

            # Also save the current state of stars for the output file.
            self.stars = stars

            # Very simple convergence test here:
            # Note, the lack of abs here means if chisq increases, we also stop.
            # Also, don't quit if we removed any outliers.
            if (iter_nremoved == 0) and (oldchisq > 0) and (oldchisq-chisq < self.chisq_thresh*dof):
                return
            oldchisq = chisq

        logger.warning("PSF fit did not converge.  Max iterations = %d reached.",self.max_iter)

    def draw(self, x, y, chipnum=None, flux=1.0, center=None, offset=None, stamp_size=48,
             image=None, logger=None, **kwargs):
        r"""Draws an image of the PSF at a given location.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, stamp_size=48)

        However, if the PSF interpolation used extra properties for the interpolation
        (cf. psf.interp_property_names), you need to provide them as additional kwargs.

            >>> print(psf.interp_property_names)
            ('u','v','ri_color')
            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, ri_color=0.23, stamp_size=48)

        Normally, the image is constructed automatically based on stamp_size, in which case
        the WCS will be taken to be the local Jacobian at this location on the original image.
        However, if you provide your own image using the :image: argument, then whatever WCS
        is present in that image will be respected.  E.g. if you want an image of the PSF in
        sky coordinates rather than image coordinates, you can provide an image with just a
        pixel scale for the WCS.

        When drawing the PSF, there are a few options regarding how the profile will be
        centered on the image.

        1. The default behavior (``center==None``) is to draw the profile centered at the same
           (x,y) as you requested for the location of the PSF in the original image coordinates.
           The returned image will not (normally) be as large as the full image -- it will just be
           a postage stamp centered roughly around (x,y).  The image.bounds give the bounding box
           of this stamp, and within this, the PSF will be centered at position (x,y).
        2. If you want to center the profile at some other arbitrary position, you may provide
           a ``center`` parameter, which should be a tuple (xc,yc) giving the location at which
           you want the PSF to be centered.  The bounding box will still be around the nominal
           (x,y) position, so this should only be used for small adjustments to the (x,y) value
           if you want it centered at a slightly different location.
        3. If you provide your own image with the ``image`` parameter, then you may set the
           ``center`` to any location in this box (or technically off it -- it doesn't check that
           the center is actually inside the bounding box).  This may be useful if you want to draw
           on an image with origin at (0,0) or (1,1) and just put the PSF at the location you want.
        4. If you want the PSf centered exactly in the center of the image, then you can use
           ``center=True``.  This will work for either an automatically built image or one
           that you provide.
        5. With any of the above options you may additionally supply an ``offset`` parameter, which
           will apply a slight offset to the calculated center.  This is probably only useful in
           conjunction with the default ``center=None`` or ``center=True``.

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [required if the psf model
                            covers more than a single chip]
        :param flux:        Flux of PSF to be drawn [default: 1.0]
        :param center:      (xc,yc) tuple giving the location on the image where you want the
                            nominal center of the profile to be drawn.  Also allowed is the
                            value center=True to place in the center of the image.
                            [default: None, which means draw at the position (x,y) of the PSF.]
        :param offset:      Optional (dx,dy) tuple giving an additional offset relative to the
                            center. [default: None]
        :param stamp_size:  The size of the image to construct if no image is provided.
                            [default: 48]
        :param image:       An existing image on which to draw, if desired. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           A GalSim Image of the PSF
        """
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        prof, method = self.get_profile(x,y,chipnum=chipnum, flux=flux, logger=logger, **kwargs)

        logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        # Make the image if necessary
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)
            # Make the center of the image (close to) the image_pos
            xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
            ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))
            image.setCenter(xcen, ycen)

        # If no wcs is given, use the original wcs
        if image.wcs is None:
            image.wcs = self.wcs[chipnum]

        # Handle the input center
        if center is None:
            center = (x, y)
        elif center is True:
            center = image.true_center
            center = (center.x, center.y)
        elif not isinstance(center, tuple):
            raise ValueError("Invalid center parameter: %r. Must be tuple or None or True"%(
                             center))

        # Handle offset if given
        if offset is not None:
            center = (center[0] + offset[0], center[1] + offset[1])

        prof.drawImage(image, method=method, center=center)

        return image

    def get_profile(self, x, y, chipnum=None, flux=1.0, logger=None, **kwargs):
        r"""Get the PSF profile at the given position as a GalSim GSObject.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0)

        The first return value, prof, is the GSObject describing the PSF profile.
        The second one, method, is the method parameter that should be used when drawing the
        profile using ``prof.drawImage(..., method=method)``.  This may be either 'no_pixel'
        or 'auto' depending on whether the PSF model already includes the pixel response or not.
        Some underlying models includ the pixel response, and some don't, so this difference needs
        to be accounted for properly when drawing.  This method is also appropriate if you first
        convolve the PSF by some other (e.g. galaxy) profile and then draw that.

        If the PSF interpolation used extra properties for the interpolation (cf.
        psf.interp_property_names), you need to provide them as additional kwargs.

            >>> print(psf.interp_property_names)
            ('u','v','ri_color')
            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0, ri_color=0.23)

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [required if the psf model
                            covers more than a single chip]
        :param flux:        Flux of PSF model [default: 1.0]
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           (profile, method)
                            profile = A GalSim GSObject of the PSF
                            method = either 'no_pixel' or 'auto' indicating which method to use
                            when drawing the profile on an image.
        """
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        properties = {'chipnum' : chipnum}
        for key in self.interp_property_names:
            if key in ['x','y','u','v']: continue
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required"%key)
            properties[key] = kwargs.pop(key)
        if len(kwargs) != 0:
            raise TypeError("Unexpected keyword argument(s) %r"%list(kwargs.keys())[0])

        image_pos = galsim.PositionD(x,y)
        wcs = self.wcs[chipnum]
        field_pos = StarData.calculateFieldPos(image_pos, wcs, self.pointing, properties)
        u,v = field_pos.x, field_pos.y

        star = Star.makeTarget(x=x, y=y, u=u, v=v, wcs=wcs, properties=properties,
                               pointing=self.pointing)
        logger.debug("Getting PSF profile at (%s,%s) on chip %s", x, y, chipnum)

        # Interpolate and adjust the flux of the star.
        star = self.interpolateStar(star).withFlux(flux)

        # The last step is implementd in the derived classes.
        prof, method = self._getProfile(star)
        return prof, method

    def _check_chipnum(self, chipnum):
        chipnums = list(self.wcs.keys())
        if chipnum is None:
            if len(chipnums) == 1:
                chipnum = chipnums[0]
            else:
                raise ValueError("chipnum is required.  Must be one of %s", str(chipnums))
        elif chipnum not in chipnums:
            raise ValueError("Invalid chipnum.  Must be one of %s", str(chipnums))
        return chipnum

    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        return [self.interpolateStar(star) for star in stars]

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        raise NotImplementedError("Derived classes must define the interpolateStar function")

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars. Takes advantage of
        interpolateList for significant speedup with some interpolators.

        .. note::

            If the stars already have the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> stars = psf.interpolateList(stars)

            before running `drawStarList`.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.

        :returns:           List of Star instances with its image filled with
                            rendered PSF
        """
        if copy_image is False:
            import warnings
            warnings.warn("The copy_image=False option has been removed from drawStarList",
                          DeprecationWarning)
        if any(star.fit is None or star.fit.get_params(self._num) is None for star in stars):
            stars = self.interpolateStarList(stars)
        return [self._drawStar(star) for star in stars]

    def drawStar(self, star, copy_image=True):
        """Generate PSF image for a given star.

        .. note::

            If the star already has the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> star = psf.interpolateList(star)

            before running `drawStar`.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]

        :returns:           Star instance with its image filled with rendered PSF
        """
        if copy_image is False:
            import warnings
            warnings.warn("The copy_image=False option has been removed from drawStar",
                          DeprecationWarning)
        # Interpolate parameters to this position/properties (if not already done):
        if star.fit is None or star.fit.get_params(self._num) is None:
            star = self.interpolateStar(star)
        # Render the image
        return self._drawStar(star)

    def _drawStar(self, star):
        # Derived classes may choose to override any of the above functions
        # But they have to at least override this one and interpolateStar to implement
        # their actual PSF model.
        raise NotImplementedError("Derived classes must define the _drawStar function")

    def _getProfile(self, star):
        prof, method = self._getRawProfile(star)
        prof = prof.shift(star.fit.center) * star.fit.flux
        return prof, method

    def _getRawProfile(self, star):
        raise NotImplementedError("Derived classes must define the _getRawProfile function")

    def write(self, file_name, logger=None):
        """Write a PSF object to a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Writing PSF to file %s",file_name)

        with fitsio.FITS(file_name,'rw',clobber=True) as f:
            self._write(f, 'psf', logger)

    def _write(self, fits, extname, logger=None):
        """This is the function that actually does the work for the write function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        from . import __version__ as piff_version
        if len(fits) == 1:
            header = {'piff_version': piff_version}
            fits.write(data=None, header=header)
        psf_type = self._type_name
        write_kwargs(fits, extname, dict(self.kwargs, type=psf_type, piff_version=piff_version))
        logger.info("Wrote the basic PSF information to extname %s", extname)
        if hasattr(self, 'stars'):
            Star.write(self.stars, fits, extname=extname + '_stars')
            logger.info("Wrote the PSF stars to extname %s", extname + '_stars')
        if hasattr(self, 'wcs'):
            self.writeWCS(fits, extname=extname + '_wcs', logger=logger)
            logger.info("Wrote the PSF WCS to extname %s", extname + '_wcs')
        self._finish_write(fits, extname=extname, logger=logger)

    @classmethod
    def read(cls, file_name, logger=None):
        """Read a PSF object from a file.

        :param file_name:   The name of the file to read.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Reading PSF from file %s",file_name)

        with fitsio.FITS(file_name,'r') as f:
            logger.debug('opened FITS file')
            return cls._read(f, 'psf', logger)

    @classmethod
    def _read(cls, fits, extname, logger):
        """This is the function that actually does the work for the read function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        # Read the type and kwargs from the base extension
        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        kwargs = read_kwargs(fits, extname)
        psf_type = kwargs.pop('type')

        # Old output files had the full class name.  Fix it if necessary.
        if psf_type.endswith('PSF') and psf_type not in PSF.valid_psf_types:
            psf_type = psf_type[:-len('PSF')]

        # If piff_version is not in the file, then it was written prior to version 1.3.
        # Since we don't know what version it was, we just use None.
        piff_version = kwargs.pop('piff_version',None)

        # Check that this is a valid PSF type
        if psf_type not in PSF.valid_psf_types:
            raise ValueError("psf type %s is not a valid Piff PSF"%psf_type)
        psf_cls = PSF.valid_psf_types[psf_type]

        # Make the PSF instance
        psf = psf_cls(**kwargs)

        # Read the stars, wcs, pointing values
        if extname + '_stars' in fits:
            stars = Star.read(fits, extname + '_stars')
            logger.debug("stars = %s",stars)
            psf.stars = stars
        if extname + '_wcs' in fits:
            wcs, pointing = cls.readWCS(fits, extname + '_wcs', logger=logger)
            logger.debug("wcs = %s, pointing = %s",wcs,pointing)
            psf.wcs = wcs
            psf.pointing = pointing

        # Just in case the class needs to do something else at the end.
        psf._finish_read(fits, extname, logger)

        # Save the piff version as an attibute.
        psf.piff_version = piff_version

        return psf

    def writeWCS(self, fits, extname, logger):
        """Write the WCS information to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write to
        :param logger:      A logger object for logging debug info.
        """
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        logger = galsim.config.LoggerWrapper(logger)

        # Start with the chipnums
        chipnums = list(self.wcs.keys())
        cols = [ chipnums ]
        dtypes = [ ('chipnums', int) ]

        # GalSim WCS objects can be serialized via pickle
        wcs_str = [ base64.b64encode(pickle.dumps(w)) for w in self.wcs.values() ]
        max_len = np.max([ len(s) for s in wcs_str ])
        # Some GalSim WCS serializations are rather long.  In particular, the Pixmappy one
        # is longer than the maximum length allowed for a column in a fits table (28799).
        # So split it into chunks of size 2**14 (mildly less than this maximum).
        chunk_size = 2**14
        nchunks = max_len // chunk_size + 1
        cols.append( [nchunks]*len(chipnums) )
        dtypes.append( ('nchunks', int) )

        # Update to size of chunk we actually need.
        chunk_size = (max_len + nchunks - 1) // nchunks

        chunks = [ [ s[i:i+chunk_size] for i in range(0, max_len, chunk_size) ] for s in wcs_str ]
        cols.extend(zip(*chunks))
        dtypes.extend( ('wcs_str_%04d'%i, bytes, chunk_size) for i in range(nchunks) )

        if self.pointing is not None:
            # Currently, there is only one pointing for all the chips, but write it out
            # for each row anyway.
            dtypes.extend( (('ra', float), ('dec', float)) )
            ra = [self.pointing.ra / galsim.hours] * len(chipnums)
            dec = [self.pointing.dec / galsim.degrees] * len(chipnums)
            cols.extend( (ra, dec) )

        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname)

    @classmethod
    def readWCS(cls, fits, extname, logger):
        """Read the WCS information from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to read from
        :param logger:      A logger object for logging debug info.

        :returns: wcs, pointing where wcs is a dict of galsim.BaseWCS instances and
                                      pointing is a galsim.CelestialCoord instance
        """
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        assert extname in fits
        assert 'chipnums' in fits[extname].get_colnames()
        assert 'nchunks' in fits[extname].get_colnames()

        data = fits[extname].read()

        chipnums = data['chipnums']
        nchunks = data['nchunks']
        nchunks = nchunks[0]  # These are all equal, so just take first one.

        wcs_keys = [ 'wcs_str_%04d'%i for i in range(nchunks) ]
        wcs_str = [ data[key] for key in wcs_keys ] # Get all wcs_str columns
        try:
            wcs_str = [ b''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each
        except TypeError:  # pragma: no cover
            # fitsio 1.0 returns strings
            wcs_str = [ ''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each

        wcs_str = [ base64.b64decode(s) for s in wcs_str ] # Convert back from b64 encoding
        # Convert back into wcs objects
        try:
            wcs_list = [ pickle.loads(s, encoding='bytes') for s in wcs_str ]
        except Exception:
            # If the file was written by py2, the bytes encoding might raise here,
            # or it might not until we try to use it.
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]

        wcs = dict(zip(chipnums, wcs_list))

        try:
            # If this doesn't work, then the file was probably written by py2, not py3
            repr(wcs)
        except Exception:
            logger.info('Failed to decode wcs with bytes encoding.')
            logger.info('Retry with encoding="latin1" in case file written with python 2.')
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]
            wcs = dict(zip(chipnums, wcs_list))
            repr(wcs)

        # Work-around for a change in the GalSim API with 2.0
        # If the piff file was written with pre-2.0 GalSim, this fixes it.
        for key in wcs:
            w = wcs[key]
            if hasattr(w, '_origin') and  isinstance(w._origin, galsim._galsim.PositionD):
                w._origin = galsim.PositionD(w._origin)

        if 'ra' in fits[extname].get_colnames():
            ra = data['ra']
            dec = data['dec']
            pointing = galsim.CelestialCoord(ra[0] * galsim.hours, dec[0] * galsim.degrees)
        else:
            pointing = None

        return wcs, pointing

# Make a global function, piff.read, as an alias for piff.PSF.read, since that's the main thing
# users will want to do as their starting point for using a piff file.
def read(file_name, logger=None):
    """Read a Piff PSF object from a file.

    .. note::

        The returned PSF instance will have an attribute piff_version, which
        indicates the version of Piff that was used to create the file.  (If it was written with
        Piff version >= 1.3.0.)

    :param file_name:   The name of the file to read.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a piff.PSF instance
    """
    return PSF.read(file_name, logger=logger)
