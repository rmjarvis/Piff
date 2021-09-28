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
.. module:: model
"""

import numpy as np
import galsim

from .util import write_kwargs, read_kwargs
from .star import Star, StarData, StarFit


class Model(object):
    """The base class for modeling a single PSF (i.e. no interpolation yet)

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def process(cls, config_model, logger=None):
        """Parse the model field of the config dict.

        :param config_model:    The configuration dict for the model field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a Model instance
        """
        import piff

        if 'type' not in config_model:
            raise ValueError("config['model'] has no type field")

        # Get the class to use for the model
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        model_class = getattr(piff, config_model['type'])

        # Read any other kwargs in the model field
        kwargs = model_class.parseKwargs(config_model, logger)

        # Build model object
        model = model_class(**kwargs)

        return model

    @classmethod
    def parseKwargs(cls, config_model, logger=None):
        """Parse the model field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_model:    The model field of the configuration dict, config['model']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_model)
        kwargs.pop('type', None)
        return kwargs

    def initialize(self, star, logger=None):
        """Initialize a star to work with the current model.

        :param star:    A Star instance with the raw data.
        :param logger:  A logger object for logging debug info. [default: None]

        :returns:       Star instance with the appropriate initial fit values
        """
        raise NotImplementedError("Derived classes must define the initialize function")

    def normalize(self, star):
        """Make sure star.fit.params are normalized properly.

        Note: This modifies the input star in place.
        """
        # This is by default a no op.  Some models may need to do something to noramlize the
        # parameter values in star.fit.
        pass

    def fit(self, star, convert_func=None):
        """Fit the Model to the star's data to yield iterative improvement on
        its PSF parameters, their uncertainties, and flux (and center, if free).
        The returned star.fit.alpha will be inverse covariance of solution if
        it is estimated, else is None.

        :param star:            A Star instance
        :param convert_func:    An optional function to apply to the profile being fit before
                                drawing it onto the image.  This is used by composite PSFs to
                                isolate the effect of just this model component. [default: None]

        :returns:      New Star instance with updated fit information
        """
        raise NotImplementedError("Derived classes must define the fit function")

    def reflux(self, star, fit_center=True, logger=None):
        """Fit the Model to the star's data, varying only the flux (and
        center, if it is free).  Flux and center are updated in the Star's
        attributes.  This is a single-step solution if only solving for flux,
        otherwise an iterative operation.  DOF in the result assume
        only flux (& center) are free parameters.

        :param star:        A Star instance
        :param fit_center:  If False, disable any motion of center
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

        # Make sure input is properly normalized
        self.normalize(star)

        data, weight, u, v = star.data.getDataVector()
        psf_prof = self.getProfile(star.fit.params)
        star_prof = psf_prof.shift(star.fit.center) * star.fit.flux

        model_image = star.image.copy()
        star_prof.drawImage(model_image, method=self._method, center=star.image_pos)
        model = model_image.array.ravel()

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

        if fit_center and self._centered:
            # Use finite different to approximate d(model)/duc, d(model)/dvc
            duv = 1.e-5
            temp = star.image.copy()
            center = star.fit.center
            du_prof = psf_prof.shift(center[0] + duv, center[1]) * new_flux
            du_prof.drawImage(temp, method=self._method, center=star.image_pos)
            dmduc = (temp.array.ravel() - model) / duv
            dv_prof = psf_prof.shift(center[0], center[1] + duv) * new_flux
            dv_prof.drawImage(temp, method=self._method, center=star.image_pos)
            dmdvc = (temp.array.ravel() - model) / duv

            # Now construct the design matrix for this minimization
            #
            #    A x = b
            #
            # where x = [ duc, dvc ]^T and b = resid.
            #
            # A[0] = dmduc
            # A[1] = dmdvc
            #
            # Solve: AT A x = AT b

            At = np.vstack((dmduc, dmdvc))
            Atw = At * np.abs(W)  # weighted least squares
            AtA = Atw.dot(At.T)
            Atb = Atw.dot(resid)
            x = np.linalg.solve(AtA, Atb)
            logger.debug('    centroid shift = %s,%s', x[0], x[1])
            duc = x[0]
            dvc = x[1]

            if self._model_can_be_offset:
                # In addition to shifting to the best fit center location, also shift
                # by the centroid of the model itself, so the next next pass through the
                # fit will be closer to centered.  In practice, this converges pretty quickly.
                model_cenu = np.sum(WM * u) / f_model - star.fit.center[0]
                model_cenv = np.sum(WM * v) / f_model - star.fit.center[1]
                logger.debug('    model centroid = %s,%s', model_cenu, model_cenv)
                duc += model_cenu
                dvc += model_cenv

            new_center = (star.fit.center[0] + duc, star.fit.center[1] + dvc)
            logger.debug('    new center = %s', new_center)

            new_chisq = np.sum((resid-At.T.dot(x))**2 * weight)
            new_dof = np.count_nonzero(weight) - 3
        else:
            new_center = star.fit.center
            new_chisq = np.sum(resid**2 * weight)
            new_dof = np.count_nonzero(weight) - 1

        logger.debug("    new_chisq = %s",new_chisq)
        logger.debug("    new_dof = %s",new_dof)

        return Star(star.data, StarFit(star.fit.params,
                                       flux = new_flux,
                                       center = new_center,
                                       chisq = new_chisq,
                                       dof = new_dof,
                                       params_var = star.fit.params_var))

    def draw(self, star, copy_image=True, center=None):
        """Draw the model on the given image.

        :param star:        A Star instance with the fitted parameters to use for drawing and a
                            data field that acts as a template image for the drawn model.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]
        :param center:      An optional tuple (x,y) location for where to center the drawn profile
                            in the image. [default: None, which draws at the star's location.]

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        prof = self.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        if copy_image:
            image = star.image.copy()
        else:
            image = star.image
        if center is None:
            center = star.image_pos
        else:
            center = galsim.PositionD(*center)
        prof.drawImage(image, method=self._method, center=center)
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, star.fit)

    def write(self, fits, extname):
        """Write a Model to a FITS file.

        Note: this only writes the initialization kwargs to the fits extension, not the parameters.

        The base class implemenation works if the class has a self.kwargs attribute and these
        are all simple values (str, float, or int)

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the model information.
        """
        # First write the basic kwargs that works for all Model classes
        model_type = self.__class__.__name__
        write_kwargs(fits, extname, dict(self.kwargs, type=model_type))

        # Now do any class-specific steps.
        self._finish_write(fits, extname)

    def _finish_write(self, fits, extname):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Model classes need to write extra information to the
        fits file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension
        """
        pass

    @classmethod
    def read(cls, fits, extname):
        """Read a Model from a FITS file.

        Note: the returned Model will not have its parameters set.  This just initializes a fresh
        model that can be used to interpret interpolated vectors.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the model information.

        :returns: a model built with a information in the FITS file.
        """
        import piff

        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        model_type = fits[extname].read()['type']
        assert len(model_type) == 1
        try:
            model_type = str(model_type[0].decode())
        except AttributeError:
            # fitsio 1.0 returns strings
            model_type = model_type[0]

        # Check that model_type is a valid Model type.
        model_classes = piff.util.get_all_subclasses(piff.Model)
        valid_model_types = dict([ (c.__name__, c) for c in model_classes ])
        if model_type not in valid_model_types:
            raise ValueError("model type %s is not a valid Piff Model"%model_type)
        model_cls = valid_model_types[model_type]

        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type',None)
        if 'force_model_center' in kwargs: # pragma: no cover
            # old version of this parameter name.
            kwargs['centered'] = kwargs.pop('force_model_center')
        model = model_cls(**kwargs)
        model._finish_read(fits, extname)
        return model

    def _finish_read(self, fits, extname):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Model classes need to read extra information from the
        fits file.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        pass
