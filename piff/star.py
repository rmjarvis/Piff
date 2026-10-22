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
.. module:: stardata
"""

import numpy as np
import copy
import os
import bisect
import galsim

from .util import make_flat

_shape_method = 'hsm'
_shape_weight_fwhm = None

def set_shape_method(method, weight_fwhm=None):
    """Set the process-wide measurement method: ``hsm``, ``gauss``, ``am``, or ``wmom``.

    The Gaussian, am, and wmom backends require the optional ngmix package. Existing Star caches
    remain valid: measurements are cached separately for each method and weight size.

    :param method:      Measurement backend. HSM is the initial default.
    :param weight_fwhm: Required for wmom: circular Gaussian weight FWHM in local world
                        units (normally arcsec), shared by all objects. Ignored otherwise.
    """
    if method not in ('hsm', 'gauss', 'am', 'wmom'):
        raise ValueError("shape_method must be 'hsm', 'gauss', 'am', or 'wmom'")
    if method == 'wmom' and (weight_fwhm is None or
                             not np.isfinite(weight_fwhm) or weight_fwhm <= 0):
        raise ValueError('wmom requires a finite positive shape_weight_fwhm')
    if method != 'hsm':
        import ngmix  # Import only when explicitly requested.
    global _shape_method, _shape_weight_fwhm
    _shape_method = method
    _shape_weight_fwhm = weight_fwhm if method == 'wmom' else None

def get_shape_method():
    """Return the active size/shape measurement method."""
    return _shape_method

def get_shape_weight_fwhm():
    """Return the fixed wmom weight FWHM, or None for the other backends."""
    return _shape_weight_fwhm

class Star(object):
    """Information about a "star", which may be either a real star or an interpolated star
    at some target location.

    The Star object is the fundamental way that Piff keeps track of information connected to
    a particular observation location.  This includes both the input data (postage stamp image,
    position on the detector, position in the field of view, weight image, etc.) and derived
    information connected to whatever Model is being used


    Stars are not normally constructed directly by the user.  They are built by various
    Piff functions such as:

        stars = input_handler.makeStars(logger)
        stars, wcs, pointing, bandpass = piff.Input.process(config['input'])
        target_star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, ...)

    However, a star can be constructed directly from a StarData instance and a StarFit instance.
    The former keeps track of the data about the star (either as observed or the model at a
    location) and the latter keeps track of fitted parameters related to fitting the data to
    a model.

        star = piff.Star(star_data, star_fit)

    Stars have an immutable design, so any functions that change either the data or the fitted
    parameters return a new object and  don't change the original.  e.g.

        star = psf.drawStar(star)
        star = star.withFlux(14)
        star = star.addPoisson(gain=gain)

    Stars have the following attributes:

        star.data       The component StarData object
        star.fit        The component StarFit object

    and the following read-only properties:

        star.image      The image of the star
        star.weight     The weight map connected to the image data
        star.image_pos  The position of the star in image coordinates, aka (x,y)
        star.field_pos  The position of the star in field coordinates, aka (u,v)
        star.x          The x position of the star in image coordinates
        star.y          The y position of the star in image coordinates
        star.u          The u position of the star in field coordinates
        star.v          The v position of the star in field coordinates
        star.chipnum    The chip number where this star was observed (or would be observed)
        star.flux       The flux of the object
        star.center     The nominal center of the object (not necessarily the centroid)
        star.is_reserve Whether the star is reserved from being used to fit the PSF
        star.is_flagged Whether the star was flagged as an outlier or had some error in fitting
        star.hsm        HSM measurements for this star as a tuple: (flux, cenu, cenv, sigma, g1, g2, flag)

    :param data: A StarData instance (invariant)
    :param fit:  A StarFit instance (invariant)
    """
    def __init__(self, data, fit):
        self.data = data
        if fit is None:
            fit = StarFit(None, flux=1.0, center=(0.,0.))
        self.fit = fit

    def withFlux(self, flux=None, center=None):
        """Update the flux and/or center values

        :param flux:    The new flux.  [default: None, which means keep the existing value.]
        :param center:  The new center.  [default: None, which means keep the existing value.]
        """
        fit = self.fit.copy()
        if flux is not None:
            fit.flux = flux
        if center is not None:
            fit.center = center
        return Star(self.data, fit)

    def withProperties(self, **kwargs):
        r"""Set or change any properties in the star.

        :param \*\*kwargs:  Each named kwarg is taken to be a property to set in the returned star.

        :returns: a new Star with the given properties, but is otherwise the same as self.
        """
        data = self.data.copy()
        props = data.properties.copy()
        props.update(kwargs)
        data.properties = props
        return Star(data, self.fit)

    def __getitem__(self, key):
        """Get a property of the star.

        This may be one of the values in the properties dict that was given when the data object
        was initialized, or one of 'x', 'y', 'u', 'v', where x,y are the position in image
        coordinates and u,v are the position in field coordinates.

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        # This is a StarData method.  Just pass the request on to it.
        return self.data[key]

    # Some properties that pass through to the data attribute to make using them easier.
    @property
    def image(self):
        return self.data.image

    @property
    def weight(self):
        return self.data.weight

    @property
    def image_pos(self):
        return self.data.image_pos

    @property
    def field_pos(self):
        return self.data.field_pos

    @property
    def x(self):
        return self.data.image_pos.x

    @property
    def y(self):
        return self.data.image_pos.y

    @property
    def u(self):
        return self.data.field_pos.x

    @property
    def v(self):
        return self.data.field_pos.y

    @property
    def chipnum(self):
        return self.data.properties.get('chipnum',0)

    @property
    def flux(self):
        return self.fit.flux

    @property
    def center(self):
        return self.fit.center

    @property
    def is_reserve(self):
        return self.data.properties.get('is_reserve',False)

    @property
    def is_flagged(self):
        return self.data.properties.get('is_flagged',False)

    def flag_if(self, flag):
        if flag and not self.is_flagged:
            return self.withProperties(is_flagged=True)
        else:
            return self

    def run_hsm(self):
        """Use HSM (as implemented by GalSim) to measure moments of star image.

        This usually isn't called directly.  The results are accessible as star.hsm,
        which caches the results, so repeated access is efficient.

        When the global measurement algorithm is set to 'hsm' (which is the default),
        this function is responsible for all size/shape measurements used in Piff.
        cf. `set_shape_method`.

        :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
        """
        image, weight, image_pos = self.data.getImage()
        # Note that FindAdaptiveMom only respects the weight function in a binary sense.
        # I.e., pixels with non-zero weight will be included in the moment measurement, those
        # with weight=0.0 will be excluded.
        mom = image.FindAdaptiveMom(weight=weight, strict=False)

        sigma = mom.moments_sigma
        shape = mom.observed_shape
        # These are in pixel coordinates.  Need to convert to world coords.
        jac = image.wcs.jacobian(image_pos=image_pos)
        scale, shear, theta, flip = jac.getDecomposition()
        # Fix sigma
        sigma *= scale
        # Fix shear.  First the flip, if any.
        if flip:
            shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
        # Next the rotation
        shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
        # Finally the shear
        shape = shear + shape

        flux = mom.moments_amp

        localwcs = image.wcs.local(image_pos)
        center = localwcs.toWorld(mom.moments_centroid) - localwcs.toWorld(image_pos)

        # Do a few sanity checks and flag likely bad fits.
        flag = mom.moments_status
        if flag != 0:
            flag = 1
        if flux < 0:
            flag |= 2
        if center.x**2 + center.y**2 > 1:
            flag |= 4

        return flux, center.x, center.y, sigma, shape.g1, shape.g2, flag

    def _make_ngmix_observation(self):
        """Build an ngmix observation, or return None when no usable pixels remain."""
        import ngmix

        image, weight, image_pos = self.data.getImage()
        valid = np.isfinite(image.array) & np.isfinite(weight.array) & (weight.array > 0)
        if not np.any(valid):
            return None
        data = np.where(valid, image.array, 0.).astype(float)
        weights = np.where(valid, weight.array, 0.).astype(float)
        jac = image.wcs.jacobian(image_pos=image_pos)
        # ngmix uses zero-based array coordinates, not GalSim image bounds. Anchor
        # the Jacobian at the catalog position so fitted centroids are offsets from it.
        ngjac = ngmix.Jacobian(row=image_pos.y - image.ymin,
                            col=image_pos.x - image.xmin, wcs=jac)
        return ngmix.Observation(data, weight=weights, jacobian=ngjac)

    def measure_gauss(self):
        """Fit an elliptical Gaussian to the star image.

        This usually isn't called directly.  The results are accessible as star.gauss,
        which caches the results, so repeated access is efficient.

        When the global measurement algorithm is set to 'gauss', this function is responsible
        for all size/shape measurements used in Piff. cf. `set_shape_method`.

        :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
        """
        import ngmix

        obs = self._make_ngmix_observation()
        if obs is None:
            return (0., 0., 0., 0., 0., 0., 1)
        # Start with a circular one-pixel Gaussian, without using adaptive moments.
        guess = np.array([0., 0., 0., 0., 2 * obs.jacobian.area, obs.image.sum()])
        result = ngmix.fitting.Fitter(model='gauss').go(obs, guess)
        if result['flags'] != 0:
            return (0., 0., 0., 0., 0., 0., 1)
        v, u, g1, g2, T, flux = result['pars']
        # Match HSM's determinant-radius convention: sigma = det(covariance)**(1/4).
        gsq = g1*g1 + g2*g2
        sigma = np.sqrt(T / 2 * (1 - gsq) / (1 + gsq))
        flag = 0
        if flux < 0:
            flag |= 2
        if u*u + v*v > 1:
            flag |= 4
        return flux, u, v, sigma, g1, g2, flag

    def measure_am(self):
        """Measure adaptive moments using ngmix.

        This usually isn't called directly.  The results are accessible as star.am,
        which caches the results, so repeated access is efficient.

        When the global measurement algorithm is set to 'am', this function is responsible
        for all size/shape measurements used in Piff. cf. `set_shape_method`.

        :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
        """
        import ngmix

        obs = self._make_ngmix_observation()
        if obs is None:
            return (0., 0., 0., 0., 0., 0., 1)
        # A full Gaussian guess avoids ngmix's randomized guesses from a scalar size.
        guess = ngmix.GMixModel([0., 0., 0., 0., 2 * obs.jacobian.area, 1.], 'gauss')
        result = ngmix.admom.AdmomFitter().go(obs, guess)
        if result['flags'] != 0:
            return (0., 0., 0., 0., 0., 0., 1)
        v, u = result['pars'][:2]
        shape = galsim.Shear(e1=result['e'][0], e2=result['e'][1])
        sigma = np.sqrt(result['T'] / 2 * np.sqrt(1 - shape.e**2))
        # ngmix returns Gaussian-weighted flux. Double it for HSM's adaptive-flux convention.
        flux = 2 * result['flux']
        flag = 4 if u*u + v*v > 1 else 0
        return flux, u, v, sigma, shape.g1, shape.g2, flag

    def measure_wmom(self, weight_fwhm):
        """Measure fixed, circular Gaussian-weighted moments about the catalog position.

        This usually isn't called directly.  The results are accessible as star.wmom(fwhm),
        which caches the results for each fwhm value, so repeated access is efficient.

        When the global measurement algorithm is set to 'wmom', this function is responsible
        for all size/shape measurements used in Piff. cf. `set_shape_method`.

        :returns: (flux, cenu, cenv, sigma, g1, g2, flag)
        """
        import ngmix

        obs = self._make_ngmix_observation()
        if obs is None:
            return (0., 0., 0., 0., 0., 0., 1)
        # ngmix sometimes emits a warning for divide by zero if the centroid is exactly 0,0
        # from some moment error estimates. They are harmless, so we suppres them here.
        with np.errstate(divide='ignore', invalid='ignore'):
            result = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm).go(obs)
        if result['flags'] != 0:
            return (0., 0., 0., 0., 0., 0., 1)
        e1, e2 = result['e']
        esq = e1*e1 + e2*e2
        # Noise can yield an unphysical moment matrix even when ngmix reports success.
        if esq >= 1:
            return (0., 0., 0., 0., 0., 0., 1)
        denom = 1 + np.sqrt(1 - esq)
        g1, g2 = e1 / denom, e2 / denom
        sigma = np.sqrt(result['T'] / 2 * np.sqrt(1 - esq))
        v, u = result['sums'][:2] / result['sums'][5]
        flag = 4 if u*u + v*v > 1 else 0
        return result['flux'], u, v, sigma, g1, g2, flag

    @property
    def hsm(self):
        if not hasattr(self, '_hsm'):
            # flux, cenu, cenv, size, g1, g2, flag
            self._hsm = self.run_hsm()
        return self._hsm

    @property
    def gauss(self):
        if not hasattr(self, '_gauss'):
            # flux, cenu, cenv, size, g1, g2, flag
            self._gauss = self.measure_gauss()
        return self._gauss

    @property
    def am(self):
        if not hasattr(self, '_am'):
            self._am = self.measure_am()
        return self._am

    def wmom(self, fwhm):
        if not hasattr(self, '_wmom'):
            # flux, cenu, cenv, size, g1, g2, flag
            self._wmom = {}
        if fwhm not in self._wmom:
            self._wmom[fwhm] = self.measure_wmom(fwhm)
        return self._wmom[fwhm]

    def measure_size_shape(self):
        """Measure flux, centroid, size and shape using whatever algorithm is currently
        selected for size/shape measurements. (Default is HSM, but see
        `Size and shape measurements` for details.)

        Returns ``(flux, cenu, cenv, sigma, g1, g2, flag)`` in local world coordinates.
        ``sigma`` is the determinant radius, ``g1/g2`` are reduced shear, and a nonzero
        flag marks a failed or suspect measurement.

        The **wmom** algorithm returns weighted flux and raw second moments about the
        catalog position, without deweighting.
        """
        method = get_shape_method()
        if method == 'hsm':
            return self.hsm
        elif method == 'gauss':
            return self.gauss
        elif method == 'am':
            return self.am
        else:
            return self.wmom(get_shape_weight_fwhm())

    @classmethod
    def makeTarget(cls, x=None, y=None, u=None, v=None, properties={}, wcs=None, scale=None,
                   stamp_size=48, image=None, weight=None, pointing=None, flux=1.0,
                   bandpass=None, **kwargs):
        r"""
        Make a target Star object with the requested properties.

        The image will be blank (all zeros), and the properties field will match the given
        input properties.

        The input properties must have either 'x' and 'y' or 'u' and 'v'.  The other pair will
        be calculated from the wcs if one is provided, or you may pass in both sets of coordinates
        and leave out the wcs

        :param x:           The image x position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param y:           The image y position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param u:           The image u position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param v:           The image v position. [optional, see above; may also be given as part
                            of the :properties: dict.]
        :param properties:  The requested properties for the target star, including any other
                            requested properties besides x,y,u,v.  You may also provide x,y,u,v
                            in this dict rather than explicitly as kwargs. [default: None]
        :param wcs:         The requested WCS.  [optional]
        :param scale:       If wcs is None, you may instead provide a pixel scale. [default: None]
        :param stamp_size:  The size in each direction of the (blank) image. [default: 48]
        :param image:       An existing image to use instead of making a new one, if desired.
                            [default: None; this overrides stamp_size]
        :param weight:      An existing image to use for the weight. [default: None]
        :param pointing:    The pointing direction to use. [default: None]
        :param flux:        The flux of the target star. [default: 1]
        :param bandpass:    The observing bandpass for loading an SED on demand. [default: None]
        :param \*\*kwargs:  Additional properties can also be given as keyword arguments if that
                            is more convenient than populating the properties dict.

        :returns:   A Star instance
        """
        # Check that input parameters are valid
        for param, name in [(x,'x'), (y,'y'), (u,'u'), (v,'v')]:
            if param is not None and name in properties:
                raise TypeError("%s may not be given both as a kwarg and in properties"%name)

        properties = properties.copy()  # So we can modify it and not mess up the caller.
        x = properties.pop('x', x)
        y = properties.pop('y', y)
        u = properties.pop('u', u)
        v = properties.pop('v', v)
        properties.update(kwargs)  # Add any extra kwargs into properties
        if (x is None) != (y is None):
            raise TypeError("Either x and y must both be given, or neither.")
        if (u is None) != (v is None):
            raise TypeError("Either u and v must both be given, or neither.")
        if x is None and u is None:
            raise TypeError("Some kind of position must be given.")
        if wcs is not None and scale is not None:
            raise TypeError("Scale is invalid when also providing wcs.")

        # Figure out what the wcs should be if not provided
        if wcs is None:
            if scale is None:
                scale = 1.
            wcs = galsim.PixelScale(scale)

        # Make field_pos if we have u,v
        if u is not None:
            field_pos = galsim.PositionD(float(u),float(v))
        else:
            field_pos = None

        # Figure out the image_pos
        if x is None:
            image_pos = wcs.toImage(field_pos)
            x = image_pos.x
            y = image_pos.y
        else:
            image_pos = galsim.PositionD(float(x),float(y))

        # Make the blank image
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)
            # Make the center of the image (close to) the image_pos
            xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
            ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))

            #image.setCenter(xcen, ycen)
            # This next line is equivalent, but more efficient.
            image._shift(galsim.PositionI(xcen,ycen) - image.center)
        if image.wcs is None:
            image.wcs = wcs
        if weight is not None:
            weight = galsim.Image(weight.array, wcs=image.wcs, copy=True, bounds=image.bounds)

        # Build the StarData instance
        data = StarData(image, image_pos, field_pos=field_pos, properties=properties,
                        pointing=pointing, weight=weight, bandpass=bandpass)
        fit = StarFit(None, flux=flux, center=(0.,0.))
        return cls(data, fit)

    @classmethod
    def write(cls, stars, writer, name):
        """Write a list of stars to a writer object.

        :param stars:       A list of stars to write
        :param writer:      A writer object that encapsulates the serialization format.
        :param name:        A name to associate with these stars in the serialized output.
        """
        # TODO This doesn't write everything out.  Probably want image as an optional I/O.

        cols = []
        dtypes = []

        # Start with the data properties
        prop_keys = list(stars[0].data.properties)
        # Do the position ones first
        for key in [ 'x', 'y', 'u', 'v' ]:
            dtypes.append( (key, float) )
            cols.append( [ s.data.properties[key] for s in stars ] )
            prop_keys.remove(key)
        # Add reserve, flag_psf
        dtypes.extend([('reserve', bool), ('flag_psf', int)])
        cols.extend([ [s.is_reserve for s in stars],
                      [s.is_flagged for s in stars] ])
        prop_keys = [key for key in prop_keys if key not in ['is_reserved', 'is_flagged']]
        # Add any remaining properties
        for key in prop_keys:
            dtypes.append( (key, stars[0].data.property_types.get(key, float)) )
            cols.append( [ s.data.properties[key] for s in stars ] )

        # Add the local WCS values
        dtypes.extend( [('dudx', float), ('dudy', float), ('dvdx', float), ('dvdy', float) ] )
        cols.append( [s.data.local_wcs.jacobian().dudx for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dudy for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dvdx for s in stars] )
        cols.append( [s.data.local_wcs.jacobian().dvdy for s in stars] )

        # Add the bounds
        dtypes.extend( [('xmin', int), ('xmax', int), ('ymin', int), ('ymax', int) ] )
        cols.append( [s.data.image.bounds.xmin for s in stars] )
        cols.append( [s.data.image.bounds.xmax for s in stars] )
        cols.append( [s.data.image.bounds.ymin for s in stars] )
        cols.append( [s.data.image.bounds.ymax for s in stars] )

        # Now the easy parts of fit:
        dtypes.extend( [ ('flux', float), ('center', float, 2), ('chisq', float), ('dof', float), ] )
        cols.append( [s.fit.flux for s in stars] )
        cols.append( [s.fit.center for s in stars] )
        cols.append( [s.fit.chisq for s in stars] )
        cols.append( [s.fit.dof for s in stars] )

        # params might not be set, so check if it is None
        header = None
        if stars[0].fit.params is not None:
            params = [s.fit.params for s in stars]

            # params might need to be flattened
            if stars[0].fit.params_lens is not None:
                header = {'PARAMS_LENS' : str(stars[0].fit.params_lens)}
                params = [ StarFit.flatten_params(p) for p in params ]

            dtypes.append( ('params', float, len(params[0])) )
            cols.append(params)

        # params_var might not be set, so check if it is None
        if stars[0].fit.params_var is not None:
            params_var = [s.fit.params_var for s in stars]
            if stars[0].fit.params_lens is not None:
                params_var = [ StarFit.flatten_params(p) for p in params_var ]
            dtypes.append( ('params_var', float, len(params_var[0])) )
            cols.append(params_var)

        # If pointing is set, write that
        if stars[0].data.pointing is not None:
            dtypes.extend( [('point_ra', float), ('point_dec', float)] )
            cols.append( [s.data.pointing.ra / galsim.degrees for s in stars ] )
            cols.append( [s.data.pointing.dec / galsim.degrees for s in stars ] )

        data = np.array(list(zip(*cols)), dtype=dtypes)
        writer.write_table(name, data, metadata=header)

    @classmethod
    def read_coords_params(cls, fits, extname):
        """Read only star fit parameters and coordinates from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to read from

        :returns: the arrays coords and params
        """
        if extname not in fits:
            raise IOError('{0} not found in FITS object'.format(extname))
        colnames = fits[extname].get_colnames()

        columns = ['u', 'v', 'params']
        for key in columns:
            if key not in colnames:
                raise IOError('{0} not found in table'.format(key))

        data = fits[extname].read(columns=columns)
        coords = np.array([data['u'], data['v']]).T
        params = data['params']
        return coords, params

    @classmethod
    def read(cls, reader, name, bandpass=None):
        """Read a list of stars via an open reader object.

        :param reader:      A reader object that encapsulates the serialization format.
        :param name:        Name associated with the stars in the serialized output.
        :param bandpass:    galsim.Bandpass to use when reconstructing stored SEDs.

        :returns: a list of Star instances, or None if there aren't any
        """
        metadata = {}
        data = reader.read_table(name, metadata)
        if data is None:
            return None

        colnames = list(data.dtype.names)
        if 'PARAMS_LENS' in metadata:
            params_lens = np.atleast_1d(eval(metadata['PARAMS_LENS']))
        else:
            params_lens = None

        for key in ['x', 'y', 'u', 'v',
                    'dudx', 'dudy', 'dvdx', 'dvdy',
                    'xmin', 'xmax', 'ymin', 'ymax',
                    'flux', 'center', 'chisq']:
            assert key in colnames
            colnames.remove(key)
        # These two might not be there, but if they are, remove them.
        colnames = [key for key in colnames if key not in ['reserve', 'flag_psf']]

        x_list = data['x']
        y_list = data['y']
        u_list = data['u']
        v_list = data['v']
        dudx = data['dudx']
        dudy = data['dudy']
        dvdx = data['dvdx']
        dvdy = data['dvdy']
        xmin = data['xmin']
        xmax = data['xmax']
        ymin = data['ymin']
        ymax = data['ymax']
        flux = data['flux']
        center = data['center']
        chisq = data['chisq']

        # We didn't used to write dof.  Be graceful if it's not there.
        if 'dof' in colnames:
            dof = data['dof']
            colnames.remove('dof')
        else:
            dof = [ None ] * len(data)

        if 'params' in colnames:
            params = data['params']
            colnames.remove('params')
            if params_lens is not None:
                params = [ StarFit.reshape_params(p, params_lens) for p in params ]
        else:
            params = [ None ] * len(data)

        if 'params_var' in colnames:
            params_var = data['params_var']
            colnames.remove('params_var')
            if params_lens is not None:
                params_var = [ StarFit.reshape_params(p, params_lens) for p in params_var ]
        else:
            params_var = [ None ] * len(data)

        if 'point_ra' in colnames:
            pointing_list = [ galsim.CelestialCoord(row['point_ra'] * galsim.degrees,
                                                    row['point_dec'] * galsim.degrees)
                              for row in data ]
            colnames.remove('point_ra')
            colnames.remove('point_dec')
        else:
            pointing_list = [ None ] * len(data)

        fit_list = [ StarFit(p, flux=f, center=c, chisq=x, dof=d, params_var=pv)
                     for (p,f,c,x,d,pv) in zip(params, flux, center, chisq, dof, params_var) ]

        # The rest of the columns are the data properties
        prop_list = [ { c : row[c] for c in colnames } for row in data ]
        property_types = {c: data.dtype[c] for c in colnames}
        for i, row in enumerate(data):
            if 'reserve' in data.dtype.names and row['reserve']:
                prop_list[i]['is_reserve'] = True
            if 'flag_psf' in data.dtype.names and row['flag_psf']:
                prop_list[i]['is_flagged'] = True

        wcs_list = [ galsim.JacobianWCS(*jac) for jac in zip(dudx,dudy,dvdx,dvdy) ]
        pos_list = [ galsim.PositionD(*pos) for pos in zip(x_list,y_list) ]
        wpos_list = [ galsim.PositionD(*pos) for pos in zip(u_list,v_list) ]
        wcs_list = [ w.withOrigin(p, wp) for w,p,wp in zip(wcs_list, pos_list, wpos_list) ]
        bounds_list = [ galsim.BoundsI(*b) for b in zip(xmin,xmax,ymin,ymax) ]
        image_list = [ galsim.Image(bounds=b, wcs=w) for b,w in zip(bounds_list, wcs_list) ]
        weight_list = [ galsim.Image(init_value=1.0, bounds=b, wcs=w)
                        for b,w in zip(bounds_list, wcs_list) ]
        data_list = [ StarData(im, pos, weight=w, properties=prop, pointing=point,
                               bandpass=bandpass, property_types=property_types)
                      for im,pos,w,prop,point in zip(image_list, pos_list, weight_list,
                                                     prop_list, pointing_list) ]

        stars = [ Star(d,f) for (d,f) in zip(data_list, fit_list) ]
        return stars

    @staticmethod
    def load_images(stars, file_name, pointing=None,
                    image_hdu=None, weight_hdu=None, badpix_hdu=None, noise=None, sky=None,
                    logger=None):
        """Load the image data into a list of Stars.

        We don't store the image data for Stars when we write them to a file, since that
        would take up a lot of space and is usually not desired.  However, we do store the
        bounds in the original image where the star was cutout, so if you want to load back in
        the original data from the image file, you can do so with this function.

        :param stars:           A list of Star instances.
        :param file_name:       The file with the image data for these stars.
        :param pointing:        The pointing direction to use. [default: None]
        :param image_hdu:       The hdu to use for the main image. [default: None, which means
                                either 0 or 1 as appropriate according to the compression.]
        :param weight_hdu:      The hdu to use for the weight image. [default: None]
        :param badpix_hdu:      The hdu to use for the bad pixel mask. [default: None]
        :param noise:           A constant noise value to use in lieu of a weight map
                                [default: None]
        :param sky:             Optional sky image or float value to subtract from the main
                                image. [default: None]
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a new list of Stars with the images information loaded.
        """
        from .input import InputFiles
        from .config import LoggerWrapper

        logger = LoggerWrapper(logger)

        logger.error("WARNING: The Star.load_images function is deprecated."
                     "Use InputFiles.load_images instead.")

        logger.verbose("Loading image information from file %s",file_name)
        config = {
            'image_file_name': file_name,
            'cat_file_name': None,  # We don't need this, but it needs to be present.
            'image_hdu': image_hdu,
            'weight_hdu': weight_hdu,
            'badpix_hdu': badpix_hdu,
            'noise': noise,
        }

        if sky is not None:
            # This should have been set in the star.data.properties already, but in the old
            # API, it was allowed to set a value for sky that was not already in the stars.
            # So set the sky property in each star here.
            stars = [Star(StarData(image=s.data.image,
                                   image_pos=s.data.image_pos,
                                   weight=s.data.weight,
                                   orig_weight=s.data.orig_weight,
                                   pointing=s.data.pointing,
                                   bandpass=s.data.bandpass,
                                   sed_eff=s.data._sed_eff,
                                   properties=dict(s.data.properties, sky=sky),
                                   property_types=s.data.property_types,
                                   _xyuv_set = True),
                          s.fit) for s in stars]

        input_files = InputFiles(config, logger=logger)
        return input_files.load_images(stars)

    def offset_to_center(self, offset):
        """A utility routine to convert from an offset in image coordinates to the corresponding
        center position in focal plane coordinates on the postage stamp image.

        :param offset:      A tuple (dx,dy) in image coordinates

        :returns:           The corresponding (du,dv) in focal plane coordinates.
        """
        # The model is in sky coordinates, so figure out what (u,v) corresponds to this offset.
        jac = self.data.local_wcs.jacobian()
        dx, dy = offset
        du = jac.dudx * dx + jac.dudy * dy
        dv = jac.dvdx * dx + jac.dvdy * dy
        return (du,dv)

    def center_to_offset(self, center):
        """A utility routine to convert from a center position in focal plane coordinates to the
        corresponding offset in image coordinates on the postage stamp image.

        :param center:      A tuple (u,v) in focal plane coordinates

        :returns:           The corresponding (dx,dy) in image coordinates.
        """
        jac = self.data.local_wcs.jacobian().inverse()
        du, dv = center
        # The names (u,v) and (x,y) are reversed for jac, since we've taken its inverse,
        # so this looks a little confusing.  e.g. jac.dudx is really (dx/du), etc.
        dx = jac.dudx * du + jac.dudy * dv
        dy = jac.dvdx * du + jac.dvdy * dv
        return (dx,dy)

    def addPoisson(self, signal=None, gain=None):
        """Return new Star with the weight values altered to reflect
        Poisson shot noise from a signal source, e.g. when the weight
        only contains variance from background and read noise.

        :param signal:  The signal (as a Star instance) from which the Poisson variance is
                        extracted.  If None, the data image is used.  All signals are
                        clipped from below at zero.
        :param gain:    The gain, in e per ADU, assumed in calculating new weights.  If None
                        is given, then the 'gain' property is used, else defaults to not
                        adding any variance.

        :returns: a new Star instance with updated weight array.
        """
        # The functionality is implemented as a StarData method.
        # So just pass this task on to that and recast the return value as a Star instance.
        return Star(self.data.addPoisson(signal=signal, gain=gain), self.fit)


class StarData(object):
    """A class that encapsulates all the relevant information about an observed star.

    **Class intended to be immutable once returned from the method that creates it.**

    This includes:
      - a postage stamp of the imaging data
      - the weight map for these pixels (zero weight means the pixel is masked or otherwise bad)
      - whether the pixel values represent flux or surface brightness
      - the pixel area on the sky
      - the position of the star on the image
      - the position of the star in the full field-of-view (local tangent plane projection)
      - possibly extra information, such as colors

    A StarData object can render the pixel data in one of two ways:

      - getImage() returns the pixels as a galsim.Image.
      - getDataVector() returns the pixels as a numpy array.

    Different use cases may prefer the data in one of these forms or the other.

    A StarData object also has this property:
      :property pixel_area:    Solid angle on sky subtended by each pixel,
                               in units of the uv system.

    The other information is stored in a dict.  This dict will include at least the following
    items:

      - x
      - y
      - u
      - v

    And anything else you want to store.  e.g.

      - chipnum
      - ra
      - dec
      - color_ri
      - color_iz
      - gain

    Any of these values may be used for interpolation.  The most typical choices would be either
    (x,y) or (u,v).  The choice of what values to use is made by the interpolator.  All such
    values should be either int or float values.

    Note that the position given for the star does not have to be a proper centroid.  It is
    rather just some position in image coordinates to use as the origin of the model.

    :param image:       A postage stamp image that includes the star
    :param image_pos:   The position in image coordinates to use as the "center" of the star.
                        Note: this does not have to be the centroid or anything specific about the
                        star.  It is merely the image position of the (0,0) coordinate for the
                        model's internal coordinate system.
    :param weight:      The corresponding weight map for that image. [default: None]
    :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                        exposure.  It does not have to be the exact center of the field of view.
                        But it should be the same for all StarData objects in the exposure.
                        This is required if image.wcs is a CelestialWCS, but should be None
                        if image.wcs is a EuclideanWCS. [default: None]
    :param field_pos:   Optionally provide the field_pos directly, rather than calculating it from
                        the wcs and a pointing. [default: None]
    :param properties:  A dict containing other properties about the star that might be of
                        interest. [default: None]
    :param property_types: A dict containing the types to use for properties if we are eventually
                           going to output them to a file.  (Only really necessary for any that
                           aren't float.) [default: None]
    :param orig_weight: The original weight map prior to any additional Poisson variance being
                        added.  [default: None, which means use orig_weight=weight]
    :param logger:      A logger object for logging debug info. [default: None]
    :param bandpass:    Observing bandpass used to reconstruct a serialized SED on access.
                        [default: None]
    :param sed_eff:     The unit-flux SED multiplied by the observing bandpass.
                        [default: None]
    """
    _sed_cache = {}

    def __init__(self, image, image_pos, weight=None, pointing=None, field_pos=None,
                 properties=None, property_types=None, orig_weight=None, logger=None,
                 _xyuv_set=False, bandpass=None, sed_eff=None):
        # Save all of these as attributes.
        self.image = image
        self.image_pos = image_pos
        # Make sure we have a local wcs in case the provided image is more complex.
        self.local_wcs = image.wcs.local(image_pos)
        self.pixel_area = self.local_wcs.pixelArea()

        if weight is None:
            self.weight = galsim.Image(image.bounds, init_value=1, wcs=image.wcs, dtype=float)
        elif type(weight) in [int, float]:
            self.weight = galsim.Image(image.bounds, init_value=weight, wcs=image.wcs, dtype=float)
        else:
            self.weight = galsim.Image(weight, dtype=float, wcs=weight.wcs)

        if orig_weight is None:
            self.orig_weight = self.weight
        else:
            self.orig_weight = orig_weight

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        if property_types is None:
            self.property_types = {}
        else:
            self.property_types = property_types

        self.pointing = pointing
        self.bandpass = bandpass
        self._sed_eff = sed_eff
        if field_pos is None:
            self.field_pos = self.calculateFieldPos(image_pos, image.wcs, pointing, self.properties)
        else:
            self.field_pos = field_pos
        self.pixel_area = self.local_wcs.pixelArea()

        # Make sure the user didn't provide their own x,y,u,v in properties.
        for key in ['x', 'y', 'u', 'v']:
            if properties is not None and key in properties and not _xyuv_set:
                raise TypeError("Cannot provide property %s in properties dict."%key)

        self.properties['x'] = self.image_pos.x
        self.properties['y'] = self.image_pos.y
        self.properties['u'] = self.field_pos.x
        self.properties['v'] = self.field_pos.y

    def copy(self):
        return copy.deepcopy(self)

    def withNew(self, **kwargs):
        r"""Return new StarData that has new values of some attributes.

        :param \*\*kwargs:  Any properties for the star.data you want to change.

        :returns:  New StarData object with altered parameters.
        """
        ret = self.copy()
        for key, value in kwargs.items():
            setattr(ret, key, value)
        return ret

    @staticmethod
    def calculateFieldPos(image_pos, wcs, pointing, properties=None):
        """
        Convert from image coordinates to field coordinates.

        :param image_pos:   The position in image coordinates, as a galsim.PositionD instance.
        :param wcs:         The wcs to use to connect image coordinates with sky coordinates:
        :param pointing:    A galsim.CelestialCoord representing the pointing coordinate of the
                            exposure.  This is required if image.wcs is a CelestialWCS, but should
                            be None if image.wcs is a EuclideanWCS. [default: None]
        :param properties:  If properties is provided, and the wcs is a CelestialWCS, then add
                            'ra', 'dec' properties (in hours, degrees respectively) based on the
                            sky coordinates. [default: None]

        :returns: a galsim.PositionD instance representing the position in field coordinates.
        """
        # Calculate the field_pos, the position in the fov coordinates
        if wcs.isCelestial():
            if pointing is None:
                raise TypeError("If the image uses a CelestialWCS then pointing is required.")
            sky_pos = wcs.toWorld(image_pos)
            if properties is not None:
                if 'ra' not in properties:
                    properties['ra'] = sky_pos.ra / galsim.hours
                if 'dec' not in properties:
                    properties['dec'] = sky_pos.dec / galsim.degrees
            u, v = pointing.project(sky_pos)
            return galsim.PositionD(u/galsim.arcsec, v/galsim.arcsec)
        else:
            return wcs.toWorld(image_pos)

    def __getitem__(self, key):
        """Get a property of the star.

        This may be one of the values in the properties dict that was given when the object
        was initialized, or one of 'x', 'y', 'u', 'v', where x,y are the position in image
        coordinates and u,v are the position in field coordinates.

        :param key:     The name of the property to return

        :returns: the value of the given property.
        """
        return self.properties[key]

    @property
    def sed_eff(self):
        """The effective SED given the observing bandpass.

        We only need the SED information over the range of the observing bandpass, and
        only in the combination sed * bandpass. So this combined quantity is built once
        and saved as sed_eff. It includes thinning the combined result used sed_tol
        if given.

        This is both a lazy property and a cached quantity. SEDs are only built once per
        sed file, so multiple stars that use the same file have the same SED instance.

        Returns None when no SED was supplied. Missing SED files or bandpass metadata
        raise an error when sed_file_name is given.
        """
        prop = self.properties
        if self._sed_eff is None and 'sed_file_name' in prop:
            if self.bandpass is None:
                raise ValueError("bandpass is required to restore stars with serialized SEDs")
            self._sed_eff = StarData._read_sed_file(
                prop['sed_file_name'],
                sed_wave_type=prop.get('sed_wave_type'),
                sed_flux_type=prop.get('sed_flux_type'),
                sed_wave_key=prop.get('sed_wave_key', 'WAVE'),
                sed_flux_key=prop.get('sed_flux_key', 'FLUX'),
                sed_tol=prop.get('sed_tol', 0.0),
                sed_max_samples=prop.get('sed_max_samples', 0),
                bandpass=self.bandpass)
        return self._sed_eff

    @staticmethod
    def _read_sed_file(sed_file_name, sed_wave_type, sed_flux_type,
                       sed_wave_key, sed_flux_key, sed_tol, sed_max_samples, bandpass):
        cache_key = (sed_file_name, sed_wave_type, sed_flux_type, sed_wave_key, sed_flux_key)
        if cache_key in StarData._sed_cache:
            return StarData._sed_cache[cache_key]

        wave_type = StarData._parse_wave_type(sed_wave_type)
        flux_type = StarData._parse_flux_type(sed_flux_type)
        try:
            # If GalSim can read the file directly, let it do so.
            sed = galsim.SED(sed_file_name, wave_type=wave_type, flux_type=flux_type)
        except Exception as e:
            # If GalSim failed because the file doesn't exist, just raise this error.
            if not os.path.isfile(sed_file_name):
                raise
            # Otherwise, try to read it ourselves.  In particular, handle FITS tables,
            # which GalSim currently cannot read directly.
            import fitsio
            try:
                with fitsio.FITS(sed_file_name) as f:
                    data = f[1].read()
            except Exception:
                raise ValueError("Could not read SED file %r as a FITS file"%(
                                 sed_file_name,)) from None
            if data.dtype.names is None:
                raise ValueError("Could not parse SED file %r as a FITS table"%(
                                 sed_file_name,)) from None

            if sed_wave_key in data.dtype.names:
                wave = np.array(data[sed_wave_key], dtype=float)
            else:
                raise ValueError("sed_wave_key = %r not found in the FITS table"%(
                                 sed_wave_key,)) from None
            if sed_flux_key in data.dtype.names:
                flux = np.array(data[sed_flux_key], dtype=float)
            else:
                raise ValueError("sed_flux_key = %r not found in the FITS table"%(
                                 sed_flux_key,)) from None

            table = galsim.LookupTable(wave, flux, interpolant='linear')
            sed = galsim.SED(table, wave_type=wave_type, flux_type=flux_type)

        # We will only ever use the sed in combination with the bandpass, so we make
        # that combination here and then possibly thin the result.
        sed = sed * bandpass

        # This means that the bandpass for drawing needs to be a flat bandpass with the same
        # limits as the original, which we call flat_bandpass.
        flat_bandpass = make_flat(bandpass)

        wave = np.array(sed.wave_list, dtype=float)
        if len(wave) == 0:
            # If the sed doesn't have a wave_list, we can't thin it.  (This is very rare!)
            sed = sed.withFlux(1.0, flat_bandpass)
            StarData._sed_cache[cache_key] = sed
            return sed

        if sed_tol > 0 or sed_max_samples > 0:
            flux = np.array([sed(w) for w in wave], dtype=float)
            sample_wave = wave
            if sed_tol > 0:
                # sed_tol means use GalSim's thin function.
                sample_wave = np.array(sed.thin(rel_err=sed_tol).wave_list, dtype=float)
            if sed_max_samples > 0:
                # If requested, reduce the number of samples down to the given maximum.
                if sed_max_samples == 1:
                    raise ValueError("sed_max_samples must be >= 2")
                sample_wave = StarData._select_greedy_samples(
                    wave, flux, sample_wave, sed_max_samples
                )

            # Either version of thinning ends with adjusting the flux values to optimize the
            # piecewise-linear approximation (the LookupTable).
            sample_flux = StarData._fit_piecewise_linear_values(wave, flux, sample_wave)
            table = galsim.LookupTable(sample_wave, sample_flux, interpolant='linear')
            sed = galsim.SED(table, wave_type='nm', flux_type='fphotons')

        # Normalize to unit flux.
        sed = sed.withFlux(1.0, flat_bandpass)

        StarData._sed_cache[cache_key] = sed
        return sed

    @staticmethod
    def _parse_wave_type(wave_type):
        if wave_type is None:
            raise ValueError("wave_type is required for reading sed files.")
        wt = wave_type.lower().strip()
        try:
            from astropy import units as u
            return u.Unit(wt)
        except Exception:
            raise ValueError("Invalid sed_wave_type %r" % (wave_type,))

    @staticmethod
    def _parse_flux_type(flux_type):
        if flux_type is None:
            raise ValueError("flux_type is required for reading sed files.")
        ft = flux_type.lower().strip()
        if ft in ('flambda', 'fnu', 'fphotons'):
            return ft
        else:
            try:
                from astropy import units as u
                return u.Unit(ft)
            except Exception:
                raise ValueError("Invalid sed_flux_type %r" % (flux_type,))

    @staticmethod
    def _trapezoid_weights(wave):
        weights = np.empty_like(wave)
        weights[0] = 0.5 * (wave[1] - wave[0])
        weights[-1] = 0.5 * (wave[-1] - wave[-2])
        weights[1:-1] = 0.5 * (wave[2:] - wave[:-2])
        return weights

    @staticmethod
    def _fit_piecewise_linear_values(wave, flux, sample_wave):
        # Given an original piecewise linear function defined by (wave, flux) pairs and a
        # (smaller) set of target sample locations, sample_wave, this function solves for flux
        # values at those sample locations which give a good approximation to the original.
        #
        # More specifically, it solves for the sample values whose linearly interpolated
        # LookupTable is the best weighted least-squares approximation to the original SED.
        # This is a standard linear least-squares projection onto the space of piecewise-linear
        # functions spanned by the hat-function basis on sample_wave.

        # Make the design matrix A which gives the new piecewise linear approximation at the
        # location of each original flux value.
        # If (w,f) are the original pairs, and (s,g) are the samples, then there is a row in A
        # for each w, and a column for each s.
        #    A @ g = f'
        # gives the approximate flux values, which we want to be close to the original f.
        n = len(wave)
        m = len(sample_wave)
        A = np.zeros((n, m), dtype=float)
        idx = np.searchsorted(sample_wave, wave, side='right') - 1
        idx = np.clip(idx, 0, m-2)
        s0 = sample_wave[idx]
        s1 = sample_wave[idx+1]
        # Each dense sample depends only on the two neighboring retained samples.
        # The coefficient is the barycentric weight of the linear interpolation on the interval.
        # i.e. the fraction of the interval to each neighboring sample.
        t = (wave - s0) / (s1 - s0)
        A[np.arange(n), idx] = 1.0 - t
        A[np.arange(n), idx+1] = t
        A[wave == sample_wave[0], :] = 0.0    # These 4 should already be computed correctly,
        A[wave == sample_wave[0], 0] = 1.0    # but this makes them exact without any possible
        A[wave == sample_wave[-1], :] = 0.0   # numerical rounding errors.
        A[wave == sample_wave[-1], -1] = 1.0
        # Weight by the local wavelength interval so the fit roughly respects the integral
        # over wavelength rather than just pointwise errors on a nonuniform grid.
        sqrt_w = np.sqrt(StarData._trapezoid_weights(wave))
        Aw = A * sqrt_w[:, np.newaxis]
        fw = flux * sqrt_w
        g = np.linalg.lstsq(Aw, fw, rcond=None)[0]

        # g is now the vector that gives the best least squares approximation f'.
        return g

    @staticmethod
    def _select_greedy_samples(wave, flux, candidate_wave, sed_max_samples):
        # This function selects a subset of candidate_wave locations that would be most
        # effective for making a piecewise-linear approximation to the SED defined by
        # (wave, flux).

        if len(candidate_wave) <= sed_max_samples:
            return candidate_wave

        # Greedy knot placement for piecewise-linear approximation:
        # keep the endpoints, repeatedly refit the current piecewise-linear model, then add the
        # candidate wavelength with the largest weighted residual.  This is the same general idea
        # as orthogonal matching pursuit / greedy basis selection, specialized to a linear spline
        # basis with fixed candidate knot locations.

        # Start with just the two end points.  We always want these.
        selected = [0, len(candidate_wave)-1]

        # Use the original dense SED definition for the target values of our fit.
        # This is piecewise interpolated values using all the candidate waves.
        target = np.array([np.interp(w, wave, flux) for w in candidate_wave], dtype=float)

        # Get the weights that would be used in the approximation using all candidates.
        # These will stay fixed throughout.  The selection process will use the combination
        # of the residual at each point and the weight.
        cand_weights = StarData._trapezoid_weights(candidate_wave)

        # Add candidate waves to our selection list, refitting each time.
        while len(selected) < sed_max_samples:
            sample_wave = candidate_wave[selected]
            sample_flux = StarData._fit_piecewise_linear_values(wave, flux, sample_wave)
            approx = np.interp(candidate_wave, sample_wave, sample_flux)

            # Use weighted residuals so the chosen samples target the integral-relevant
            # approximation error, not just the largest pointwise deviation on the candidate grid.
            resid = np.abs(target - approx) * np.sqrt(cand_weights)

            # Add the item with the highest weighted residual
            resid[selected] = -1.0  # Don't reselect something we already have.
            bisect.insort(selected, np.argmax(resid))

        return candidate_wave[selected]

    def getImage(self):
        """Get the pixel data as a galsim.Image.

        Also returns the weight image and the origin position (the position in x,y coordinates
        to use as the origin of the PSF model).

        :returns: image, weight, image_pos
        """
        return self.image, self.weight, self.image_pos

    def getDataVector(self):
        """Get the pixel data as a numpy array.

        Also returns the weight values and the local u,v coordinates of the pixels.
        Any pixels with zero weight (e.g. from masking in the original image) will not be
        included in the returned arrays.

        :returns: data_vector, weight_vector, u_vector, v_vector
        """
        # Remember that numpy arrays use y,x indexing.
        ny, nx = self.image.array.shape

        # Image coordinates of pixels relative to nominal center
        xvals = np.arange(self.image.bounds.xmin, self.image.bounds.xmax+1, dtype=float)
        yvals = np.arange(self.image.bounds.ymin, self.image.bounds.ymax+1, dtype=float)
        x,y = np.meshgrid(xvals, yvals)
        x -= self.image_pos.x
        y -= self.image_pos.y

        # Convert to u,v coords
        u = self.local_wcs._u(x,y)
        v = self.local_wcs._v(x,y)

        # Get flat versions of everything
        u = u.flatten()
        v = v.flatten()
        pix = self.image.array.flatten()
        wt = self.weight.array.flatten()
        return pix, wt, u, v

    def addPoisson(self, signal=None, gain=None):
        """Return new StarData with the weight values altered to reflect
        Poisson shot noise from a signal source, e.g. when the weight
        only contains variance from background and read noise.

        :param signal:  The signal (as a Star instance) from which the Poisson variance is
                        extracted.  If None, the data image is used.  All signals are
                        clipped from below at zero.
        :param gain:    The gain, in e per ADU, assumed in calculating new weights.  If None
                        is given, then the 'gain' property is used, else defaults to not
                        adding any variance.

        :returns: a new StarData instance with updated weight array.
        """
        # Get the gain.  None both here and in properties, means don't add any variance.
        if gain is None:
            gain = self.properties.get('gain',None)
        if gain is None:
            return self

        # Mark the pixels that are not already worthless
        use = self.orig_weight.array!=0.

        # Get the signal data
        if signal is None:
            signal = self
        variance = signal.image.array
        if variance.shape != self.orig_weight.array.shape:
            raise ValueError('In addPoisson, signal has wrong shape: %s != %s'%(
                    variance.shape, self.orig_weight.array.shape))

        # clip variance
        use &= (variance >= 0)

        # Add to weight
        # Note: use the original weight here, not the current weight, since this may have already
        # had some Poisson noise added to the weight.
        newweight = self.orig_weight.copy()
        newweight.array[use] = 1. / (1./self.orig_weight.array[use] + variance[use] / gain)

        # Return new object
        return StarData(image=self.image,
                        image_pos=self.image_pos,
                        weight=newweight,
                        orig_weight=self.orig_weight,
                        pointing=self.pointing,
                        bandpass=self.bandpass,
                        sed_eff=self._sed_eff,
                        properties=dict(self.properties, gain=gain),
                        property_types=self.property_types,
                        _xyuv_set = True)


class StarFit(object):
    """Class to hold the results of fitting a Model to some StarData, or specify
    the PSF interpolated to an unmeasured location.

    **Class is intended to be invariant once created.**

    This class can be extended
    to carry information of use to a given Model instance (such as intermediate
    results), but interpolators will be looking for some subset of these properties:

    :params:      numpy array of parameters of the PSF that apply to this star
    :params_var:  numpy array of variance error parameters of the PSF
    :flux:        flux of the star
    :center:      (u,v) tuple giving position of stellar center (relative
                  to data.image_pos)
    :chisq:       Chi-squared of  fit to the data (if any) with current params
    :dof:         Degrees of freedom in the fit (will depend on whether last fit had
                  parameters free or just the flux/center).
    :A, b:        matrix, vector, giving design matrix equation for the Taylor expansion of chisq
                  in the model's fitting coordinates about their current values. These are
                  normally params, but models may use a reduced parameterization. The alpha
                  matrix, AT A, is the inverse covariance matrix in fitting coordinates.

    :param params: A 1d numpy array holding estimated PSF parameters
    :param params_var: A 1d numpy array holding estimates PSF variance error parameters
    :param flux:   Estimated flux for this star
    :param center: Estimated or fixed center position (u,v) of this star relative to
                   the StarData.image_pos reference point.
    :param A:      Design matrix for the quadratic dependence of chi-squared on fitting
                   coordinates about current values. Quadratic term is dpT AT A dp.
    :param b:      Vector portion of design equation. Linear term of chi-squared dependence
                   on fitting coordinates about current values is -2 AT b.
    :param chisq:  chi-squared value at current parameters.
    :param dof:    The number of degrees of freedom for the given chisq.
    """
    def __init__(self, params, flux=1., center=(0.,0.), params_var=None, A=None, b=None,
                 chisq=None, dof=None):
        # center might be a galsim.PositionD.  That's fine, but we'll convert to a tuple here.
        try:
            center = (center.x, center.y)
        except AttributeError:
            pass

        self.params = params
        self.params_var = params_var
        self.flux = flux
        self.center = center
        self.A = A
        self.b = b
        self.chisq = chisq
        self.dof = dof

    @property
    def alpha(self):
        return self.A.T.dot(self.A)

    @property
    def beta(self):
        return self.A.T.dot(self.b)

    @classmethod
    def reshape_params(cls, params, params_lens):
        new_params = []
        i1 = 0
        for plen in params_lens:
            i2 = i1 + plen
            new_params.append(params[i1:i2])
            i1 = i2
        return new_params

    @classmethod
    def flatten_params(cls, params):
        return np.concatenate(params)

    @property
    def params_lens(self):
        if isinstance(self.params, list):
            return [len(p) for p in self.params]
        else:
            return None

    def newParams(self, params, params_var=None, num=None, **kwargs):
        r"""Return new StarFit that has the array params installed as new parameters.

        :param params:      A 1d array holding new parameters; must match size of current ones
        :param params_var:  A 1d array holding new parameter variances; if given, must match size
                            of current ones.  [default: None]
        :param num:         If there are multiple sets of model params being stored, then
                            this is the number to update. [default: None]
        :param \*\*kwargs:  Any other additional properties for the star.  Keeps current values
                            of anything not given a new value.

        :returns: New StarFit object with altered parameters.  All chisq-related parameters
                  are set to None since they are no longer valid.
        """
        old_params = self.get_params(num)
        old_params_var = self.get_params_var(num)
        if old_params is not None and np.asarray(params).shape != old_params.shape:
            raise ValueError('new StarFit parameters do not match dimensions of old ones')

        if num is not None:
            new_params = self.params.copy()
            new_params[num] = np.asarray(params)
            kwargs['params'] = new_params
            if params_var is not None:
                new_params_var = self.params_var.copy()
                new_params_var[num] = np.asarray(params_var)
                kwargs['params_var'] = new_params_var
            elif self.params_var is not None and self.params_var[num] is None:
                new_params_var = self.params_var.copy()
                new_params_var[num] = np.zeros_like(params)
                kwargs['params_var'] = new_params_var
        else:
            kwargs['params'] = np.asarray(params)
            if params_var is not None:
                kwargs['params_var'] = np.asarray(params_var)

        return self.withNew(**kwargs)

    def updateParams(self, params, num=None, **kwargs):
        """Update the parameters to the given params in place, rather than make a new star.

        Note: the new params must be the same size/shape as the old params.

        Equivalent to star = star.newParams(params, ...) but when it is safe to do so,
        setting the parameters in place is a bit more efficient than making a new object.
        """
        if num is not None:
            self.params[num] = np.asarray(params)
        else:
            self.params = np.asarray(params)

    def get_params(self, num):
        if num is not None and self.params is not None:
            return self.params[num]
        else:
            return self.params

    def get_params_var(self, num):
        if num is not None and self.params_var is not None:
            return self.params_var[num]
        else:
            return self.params_var

    def copy(self):
        return copy.deepcopy(self)

    def withNew(self, **kwargs):
        r"""Return new StarFit that has new values of some attributes.

        :param \*\*kwargs:  Any properties for the star.fit you want to change.

        :returns:  New StarFit object with altered parameters.
        """
        ret = self.copy()
        for key, value in kwargs.items():
            setattr(ret, key, value)
        return ret
