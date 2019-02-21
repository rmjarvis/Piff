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
.. module:: util
"""

from __future__ import print_function
import numpy as np
import os
import galsim

# Courtesy of
# http://stackoverflow.com/questions/3862310/how-can-i-find-all-subclasses-of-a-given-class-in-python
def get_all_subclasses(cls):
    """Get all subclasses of an existing class.
    """
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

def ensure_dir(target):
    """Ensure that the directory for a target output file exists.

    :param target:      The file that you want to write to.
    """
    d = os.path.dirname(target)
    if d != '' and  not os.path.exists(d):
        os.makedirs(d)

def make_dtype(key, value):
    """A helper function that makes a dtype appropriate for a given value

    :param key:     The key to use for the column name in the dtype.
    :param value:   The input value (just one item if using a column of multiple values)

    :returns: a numpy.dtype instance
    """
    def make_dt_tuple(key, t, size):
        # If size == 0, then it's not an array, so return a 2 element tuple.
        # Otherwise, the size is the third item in the tuple.
        if size == 0:
            return (key, t)
        else:
            return (key, t, size)

    try:
        # Note: this works for either arrays or strings
        size = len(value)
        t = type(value[0])
    except TypeError:
        size = 0
        t = type(value)
    dt = np.dtype(t) # just used to categorize the type into int, float, str

    if dt.kind in np.typecodes['AllInteger']:
        t = int
    elif dt.kind in np.typecodes['AllFloat']:
        t = float
    elif dt.kind in ['S','U'] and not isinstance(value, str):
        # catch lists of strings
        t = np.array(value).dtype.str
        t = t.replace('U','S')
    elif dt.kind in ['S','U']:
        t = bytes
    else:
        # Other objects should be manually serialized by the initializer or the finish_read and
        # finish_write functions.
        raise ValueError("Cannot serialize object of type %s"%t)
    dt = make_dt_tuple(key, t, size)

    return dt

def adjust_value(value, dtype):
    """Possibly adjust a value to match the type expected for the given dtype.

    e.g. change np.int16 -> int if dtype expects int.  Or vice versa.

    :param value:   The input value to possible adjust.

    :returns: the adjusted value to use for writing in a FITS table.
    """
    t = dtype[1]
    if len(dtype) == 2 or dtype[2] == 0:
        # dtype is either (key, t) or (key, t, size)
        # if no size or size == 0, then just use t as the type.
        return t(value)
    elif t == bytes:
        # Strings may need to be encoded.
        try:
            return value.encode()
        except AttributeError:
            return value
    else:
        try:
            # Arrays of strings may need to be encoded.
            return np.array([v.encode() for v in value])
        except AttributeError:
            # For other numpy arrays, we can use astype instead.
            return np.array(value).astype(t)

def write_kwargs(fits, extname, kwargs):
    """A helper function for writing a single row table into a fits file with the values
    and column names given by a kwargs dict.

    :param fits:        An open fitsio.FITS instance
    :param extname:     The extension to write to
    :param kwargs:      A kwargs dict to be written as a FITS binary table.
    """
    cols = []
    dtypes = []
    for key, value in kwargs.items():
        # Don't add values that are None to the table.
        if value is None:
            continue
        dt = make_dtype(key, value)
        value = adjust_value(value,dt)
        cols.append([value])
        dtypes.append(dt)
    data = np.array(list(zip(*cols)), dtype=dtypes)
    fits.write_table(data, extname=extname)

def read_kwargs(fits, extname):
    """A helper function for reading a single row table from a fits file returning the values
    and column names as a kwargs dict.

    :param fits:        An open fitsio.FITS instance
    :param extname:     The extension to read.

    :returns: A dict of the kwargs that were read from the file.
    """
    cols = fits[extname].get_colnames()
    data = fits[extname].read()
    assert len(data) == 1
    kwargs = dict([ (col, data[col][0]) for col in cols ])
    for key,value in kwargs.items():
        # Convert any byte strings to a regular str
        try:
            value = str(value.decode())
            kwargs[key] = value
        except AttributeError:
            # Also convert arrays of bytes into arrays of strings.
            try:
                value = np.array([str(v.decode()) for v in value])
                kwargs[key] = value
            except (AttributeError, TypeError):
                pass
    return kwargs

def measure_snr(star):
    """Calculate the signal-to-noise of a given star.

    :param star:    Input star, with stamp, weight

    :returns:       the SNR value.
    """
    # The S/N value that we use will be the weighted total flux where the
    # weight function is the star's profile itself.  This is the maximum
    # S/N value that any flux measurement can possibly produce, which will
    # be closer to an in-practice S/N than using all the pixels equally.
    #
    # F = Sum_i w_i I_i^2
    # var(F) = Sum_i w_i^2 I_i^2 var(I_i)
    #        = Sum_i w_i I_i^2             <--- Assumes var(I_i) = 1/w_i
    #
    # S/N = F / sqrt(var(F))
    image, weight, image_pos = star.data.getImage()
    I = image.array
    w = weight.array
    flux = (w*I*I).sum(dtype=float)
    snr = flux**0.5
    return snr

def hsm(star):
    """ Use HSM to measure moments of star image. Does not go beyond second moments.

    :param star:    Input star, with stamp, weight

    :returns:       The shape. Does not go beyond second moments. Also returns a flag.
    """
    image, weight, image_pos = star.data.getImage()
    # Note that FindAdaptiveMom only respects the weight function in a binary sense.  I.e., pixels
    # with non-zero weight will be included in the moment measurement, those with weight=0.0 will be
    # excluded.
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
    flag = mom.moments_status

    return flux, center.x, center.y, sigma, shape.g1, shape.g2, flag




def hsm_error(star, logger=None, return_debug=False, return_error=True):
    """ Use python implementation of HSM to measure moments of star image to get errors. Does not go beyond second moments.

    Slow since it's python, not C, but we should only have to do this once per star.

    calculate the error on our e0,e1,e2

    $ e_0= \sum \left[ (x-x_0)^2 + (y-y_0)^2  \right] K(x,y) I(x,y) $

    where K(x,y) is the HSM kernel, I(x,y) is the image, s(x,y) is the shot noise per pixel
    so

    $ \sigma^2(e_0) = \sum \left\{ \left[ (x-x_0)^2 + (y-y_0)^2  \right] K(x,y) \right\}^2 s^2(x,y) $


    TODO: might be a factor of 2 missing still?
    TODO: what do the _i subscripts indicate? can I cut that and keep clarity?


    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]
    :param return_debug:    Boolean. If true, will return debug.
                            [default: False]
    :param return_error:    Boolean. If true, will also return the shape error.
                            [default: True]

    :returns:               The shape (and error if return_error). Does not go beyond second moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

    star = star.copy()

    # get vectors for data, weight and u, v
    data_i, weight_i, u_i, v_i = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    flux, cenu, cenv, size, g1, g2, flag = hsm(star)
    profile = galsim.Gaussian(sigma=1.0).dilate(size).shear(g1=g1, g2=g2).shift(cenu, cenv) * flux
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', offset=(star.image_pos-image.true_center))
    # convert image into kernel
    kernel_i = image.array.flatten()

    # now apply mask
    mask = weight_i != 0.
    data_i = data_i[mask]
    weight_i = weight_i[mask]
    kernel_i = kernel_i[mask]
    u_i = u_i[mask]
    v_i = v_i[mask]

    # with HSM as our starting guess, and kernel, let's use the weights for a final step. This makes everything a lot simpler, conceptually. We place all these results here, and then work through the errors later
    flux_calc = np.sum(weight_i * data_i * kernel_i)
    normalization = flux_calc

    u0_calc = np.sum(data_i * weight_i * kernel_i * u_i) / normalization
    v0_calc = np.sum(data_i * weight_i * kernel_i * v_i) / normalization
    # calculate moments
    du_i = u_i - u0_calc
    dv_i = v_i - v0_calc
    Muu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i) / normalization
    Mvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i) / normalization
    Muv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i) / normalization

    """

    Muu_true = 2 * Muu
    e0_true = Muu_true + Mvv_true
    e1_true = Muu_true - Mvv_true
    e0_calc = Muu + Mvv = 0.5 e0_true
    e1_calc = (Muu - Mvv) / 2 = 0.25 e1_true


    Q: if kernel above is actually K^2, not K, how does above change?
    """
    # now e0,e1,e2
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = Muu - Mvv
    e2_calc = 2 * Muv
    if not return_error:
        return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc

    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum(np.power(weight_i ** 2 * kernel_i ** 2, 2) * sigma2_data_i)
    sigma_normalization = np.sqrt(sigma2_normalization)
    # flux is 2x normalization in hsm.cpp, so probably a factor of 2 here
    sigma_flux = 2 * sigma_normalization

    # flux fudge factors?
    flux_fudge_factor = 1.
    sigma_flux = sigma_flux * np.sqrt(flux_fudge_factor)
    sigma_normalization = 1. * sigma_normalization

    #####
    # u0, v0
    #####

    sigma2_u0_data = np.sum(np.power(weight_i * kernel_i * u_i / normalization, 2) * sigma2_data_i)
    sigma2_v0_data = np.sum(np.power(weight_i * kernel_i * v_i / normalization, 2) * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = np.power(u0_calc * sigma_normalization / normalization, 2)
    sigma2_v0_flux = np.power(v0_calc * sigma_normalization / normalization, 2)

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i + dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i - dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e2_data = np.sum(np.power(2 * weight_i * kernel_i * 2 * du_i * dv_i / normalization, 2) * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e0_v0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e2_v0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))

    # add sigma_normalization
    sigma2_e0_flux = np.power(e0_calc * sigma_normalization / normalization, 2)
    sigma2_e1_flux = np.power(e1_calc * sigma_normalization / normalization, 2)
    sigma2_e2_flux = np.power(e2_calc * sigma_normalization / normalization, 2)

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly correlated, so I'm probably missing a negative covariance term from the kernel that would bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    if return_debug:
        if logger:
            logger.debug('Star hsm_error. Value of flux, u0, v0, e0, e1, e2 are:')
            logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc))
            # logger.debug('Star hsm_error. Value of Gaussian model flux, u0, v0, e0, e1, e2 are:')
            # logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(flux, u0, v0, e0, e1, e2))
            logger.debug('Star hsm_error. Value of errors for flux, u0, v0, e0, e1, e2 are:')
            logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2))
            logger.debug('Star hsm_error. Relative un-fudged contributions for u0 from data and flux are (in sigma2):')
            logger.debug('{0:.2e} {1:.2e}'.format(sigma2_u0_data, sigma2_u0_flux))
            logger.debug('Star hsm_error. Relative un-fudged contributions for v0 from data and flux are (in sigma2):')
            logger.debug('{0:.2e} {1:.2e}'.format(sigma2_v0_data, sigma2_v0_flux))
            logger.debug('Star hsm_error. Relative un-fudged contributions for e0 from data, flux, u0, and v0 are (in sigma2):')
            logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e}'.format(sigma2_e0_data, sigma2_e0_flux, sigma2_e0_u0, sigma2_e0_v0))
            logger.debug('Star hsm_error. Relative un-fudged contributions for e1 from data, flux, u0, and v0 are (in sigma2):')
            logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e}'.format(sigma2_e1_data, sigma2_e1_flux, sigma2_e1_u0, sigma2_e1_v0))
            logger.debug('Star hsm_error. Relative un-fudged contributions for e2 from data, flux, u0, and v0 are (in sigma2):')
            logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e}'.format(sigma2_e2_data, sigma2_e2_flux, sigma2_e2_u0, sigma2_e2_v0))
        return sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, \
               flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, \
               sigma2_e0_data, sigma2_e0_u0, sigma2_e0_v0, sigma2_e0_flux, \
               sigma2_e1_data, sigma2_e1_u0, sigma2_e1_v0, sigma2_e1_flux, \
               sigma2_e2_data, sigma2_e2_u0, sigma2_e2_v0, sigma2_e2_flux
    else:
        return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2


def hsm_third_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to third moments of star image.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape. Goes up to third moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

    star = star.copy()

    # get vectors for data, weight and u, v
    data_i, weight_i, u_i, v_i = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    flux, cenu, cenv, size, g1, g2, flag = hsm(star)
    profile = galsim.Gaussian(sigma=1.0).dilate(size).shear(g1=g1, g2=g2).shift(cenu, cenv) * flux
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', offset=(star.image_pos-image.true_center))
    # convert image into kernel
    kernel_i = image.array.flatten()

    # now apply mask
    mask = weight_i != 0.
    data_i = data_i[mask]
    weight_i = weight_i[mask]
    kernel_i = kernel_i[mask]
    u_i = u_i[mask]
    v_i = v_i[mask]

    # with HSM as our starting guess, and kernel, let's use the weights for a final step. This makes everything a lot simpler, conceptually. We place all these results here, and then work through the errors later
    flux_calc = np.sum(weight_i * data_i * kernel_i)
    normalization = flux_calc

    u0_calc = np.sum(data_i * weight_i * kernel_i * u_i) / normalization
    v0_calc = np.sum(data_i * weight_i * kernel_i * v_i) / normalization
    # calculate moments
    du_i = u_i - u0_calc
    dv_i = v_i - v0_calc
    Muu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i) / normalization
    Mvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i) / normalization
    Muv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i) / normalization

    # now e0,e1,e2
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = Muu - Mvv
    e2_calc = 2 * Muv

    # calculate third moments
    Muuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i) / normalization
    Muuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i) / normalization
    Muvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i) / normalization
    Mvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i) / normalization

    zeta1_calc = Muuu + Muvv
    zeta2_calc = Mvvv + Muuv
    delta1_calc = Muuu - 3 * Muvv
    delta2_calc = -(Mvvv - 3 * Muuv)

    return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, zeta1_calc, zeta2_calc, delta1_calc, delta2_calc






def hsm_error_third_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to third moments of star image to get errors.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape error. Goes up to third moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

    star = star.copy()

    # get vectors for data, weight and u, v
    data_i, weight_i, u_i, v_i = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    flux, cenu, cenv, size, g1, g2, flag = hsm(star)
    profile = galsim.Gaussian(sigma=1.0).dilate(size).shear(g1=g1, g2=g2).shift(cenu, cenv) * flux
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', offset=(star.image_pos-image.true_center))
    # convert image into kernel
    kernel_i = image.array.flatten()

    # now apply mask
    mask = weight_i != 0.
    data_i = data_i[mask]
    weight_i = weight_i[mask]
    kernel_i = kernel_i[mask]
    u_i = u_i[mask]
    v_i = v_i[mask]

    # with HSM as our starting guess, and kernel, let's use the weights for a final step. This makes everything a lot simpler, conceptually. We place all these results here, and then work through the errors later
    flux_calc = np.sum(weight_i * data_i * kernel_i)
    normalization = flux_calc

    u0_calc = np.sum(data_i * weight_i * kernel_i * u_i) / normalization
    v0_calc = np.sum(data_i * weight_i * kernel_i * v_i) / normalization
    # calculate moments
    du_i = u_i - u0_calc
    dv_i = v_i - v0_calc
    Muu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i) / normalization
    Mvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i) / normalization
    Muv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i) / normalization

    # now e0,e1,e2
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = Muu - Mvv
    e2_calc = 2 * Muv

    # calculate third moments
    Muuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i) / normalization
    Muuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i) / normalization
    Muvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i) / normalization
    Mvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i) / normalization

    zeta1_calc = Muuu + Muvv
    zeta2_calc = Mvvv + Muuv
    delta1_calc = Muuu - 3 * Muvv
    delta2_calc = -(Mvvv - 3 * Muuv)

    #return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, zeta1_calc, zeta2_calc, delta1_calc, delta2_calc


    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum(np.power(weight_i ** 2 * kernel_i ** 2, 2) * sigma2_data_i)
    sigma_normalization = np.sqrt(sigma2_normalization)
    # flux is 2x normalization in hsm.cpp, so probably a factor of 2 here
    sigma_flux = 2 * sigma_normalization

    # flux fudge factors?
    flux_fudge_factor = 1.
    sigma_flux = sigma_flux * np.sqrt(flux_fudge_factor)
    sigma_normalization = 1. * sigma_normalization

    #####
    # u0, v0
    #####

    sigma2_u0_data = np.sum(np.power(weight_i * kernel_i * u_i / normalization, 2) * sigma2_data_i)
    sigma2_v0_data = np.sum(np.power(weight_i * kernel_i * v_i / normalization, 2) * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = np.power(u0_calc * sigma_normalization / normalization, 2)
    sigma2_v0_flux = np.power(v0_calc * sigma_normalization / normalization, 2)

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i + dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i - dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e2_data = np.sum(np.power(2 * weight_i * kernel_i * 2 * du_i * dv_i / normalization, 2) * sigma2_data_i)

    sigma2_zeta1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i + du_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_zeta2_data = np.sum(np.power(2 * weight_i * kernel_i * (dv_i * dv_i * dv_i + du_i * du_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_delta1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i - 3 * du_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_delta2_data = np.sum(np.power(2 * weight_i * kernel_i * -(dv_i * dv_i * dv_i - 3 * du_i * du_i * dv_i) / normalization, 2) * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e0_v0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e2_v0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))

    sigma2_zeta1_u0 = np.sum(np.power(2 * (3 * du_i * du_i + dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_zeta1_v0 = np.sum(np.power(2 * 2 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_zeta2_u0 = np.sum(np.power(2 * 2 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_zeta2_v0 = np.sum(np.power(2 * (du_i * du_i + 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_delta1_u0 = np.sum(np.power(2 * (3 * du_i * du_i - 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_delta1_v0 = np.sum(np.power(2 * -(6 * du_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_delta2_u0 = np.sum(np.power(2 * 6 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_delta2_v0 = np.sum(np.power(2 * (3 * du_i * du_i - 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))

    # add sigma_normalization
    sigma2_e0_flux = np.power(e0_calc * sigma_normalization / normalization, 2)
    sigma2_e1_flux = np.power(e1_calc * sigma_normalization / normalization, 2)
    sigma2_e2_flux = np.power(e2_calc * sigma_normalization / normalization, 2)

    sigma2_zeta1_flux = np.power(zeta1_calc * sigma_normalization / normalization, 2)
    sigma2_zeta2_flux = np.power(zeta2_calc * sigma_normalization / normalization, 2)
    sigma2_delta1_flux = np.power(delta1_calc * sigma_normalization / normalization, 2)
    sigma2_delta2_flux = np.power(delta2_calc * sigma_normalization / normalization, 2)

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly correlated, so I'm probably missing a negative covariance term from the kernel that would bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    sigma_zeta1 = np.sqrt(sigma2_zeta1_data + sigma2_zeta1_u0 + sigma2_zeta1_v0 + sigma2_zeta1_flux)
    sigma_zeta2 = np.sqrt(sigma2_zeta2_data + sigma2_zeta2_u0 + sigma2_zeta2_v0 + sigma2_zeta2_flux)
    sigma_delta1 = np.sqrt(sigma2_delta1_data + sigma2_delta1_u0 + sigma2_delta1_v0 + sigma2_delta1_flux)
    sigma_delta2 = np.sqrt(sigma2_delta2_data + sigma2_delta2_u0 + sigma2_delta2_v0 + sigma2_delta2_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    #sigma_zeta1 = sigma_zeta1 * 0.523
    #sigma_zeta2 = sigma_zeta2 * 0.545
    sigma_zeta1 = sigma_zeta1 * 0.52
    sigma_zeta2 = sigma_zeta2 * 0.55

#below are the numbers gleaned when looking for fudge factors using simulated stars of different snr and location
#snr 90, location (500, 500, 25), 1000 runs
#[ 1.12804519  1.07293111  1.07145898  1.13775826  1.10059309  1.10546455  1.05268536  0.99594863  0.9396108   0.93940038]
#snr 90, location (500, 500, 25), 100 runs
#[ 1.13037156  1.07224726  1.07309272  1.14044123  1.09669306  1.10661247  1.04602584  0.99712779  0.94301216  0.93644381]
#snr 70, location (500, 500, 25), 100 runs
#[ 1.12649224  1.07171045  1.07870498  1.14094039  1.09996217  1.10760745  1.04477721  0.99593876  0.93987085  0.94086416]
#snr 50, location (500, 500, 25), 100 runs
#[ 1.12264633  1.07698594  1.06969354  1.13584159  1.09865871  1.09636084  1.0537384   0.99585817  0.92885245  0.93264428]
#snr 90, location (100, 100, 55), 100 runs
#[ 1.13083505  1.08115786  1.06882666  1.14116332  1.0969551   1.10358841  1.04876238  0.99314536  0.94111276  0.94455409]
#snr 90, location (300, 200, 5), 100 runs
#[ 1.13755535  1.07547336  1.07685751  1.13319619  1.09941079  1.10964125  1.0472514   0.99847778  0.94133408  0.94094472]

    return sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2







def hsm_fourth_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to fourth moments of star image.


    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape. Goes up to fourth moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

    star = star.copy()

    # get vectors for data, weight and u, v
    data_i, weight_i, u_i, v_i = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    flux, cenu, cenv, size, g1, g2, flag = hsm(star)
    profile = galsim.Gaussian(sigma=1.0).dilate(size).shear(g1=g1, g2=g2).shift(cenu, cenv) * flux
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', offset=(star.image_pos-image.true_center))
    # convert image into kernel
    kernel_i = image.array.flatten()

    # now apply mask
    mask = weight_i != 0.
    data_i = data_i[mask]
    weight_i = weight_i[mask]
    kernel_i = kernel_i[mask]
    u_i = u_i[mask]
    v_i = v_i[mask]

    # with HSM as our starting guess, and kernel, let's use the weights for a final step. This makes everything a lot simpler, conceptually. We place all these results here, and then work through the errors later
    flux_calc = np.sum(weight_i * data_i * kernel_i)
    normalization = flux_calc

    u0_calc = np.sum(data_i * weight_i * kernel_i * u_i) / normalization
    v0_calc = np.sum(data_i * weight_i * kernel_i * v_i) / normalization
    # calculate moments
    du_i = u_i - u0_calc
    dv_i = v_i - v0_calc
    Muu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i) / normalization
    Mvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i) / normalization
    Muv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i) / normalization

    # now e0,e1,e2
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = Muu - Mvv
    e2_calc = 2 * Muv

    # calculate third moments
    Muuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i) / normalization
    Muuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i) / normalization
    Muvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i) / normalization
    Mvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i) / normalization

    zeta1_calc = Muuu + Muvv
    zeta2_calc = Mvvv + Muuv
    delta1_calc = Muuu - 3 * Muvv
    delta2_calc = -(Mvvv - 3 * Muuv)

    # calculate fourth moments
    Muuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i * du_i) / normalization
    Muuuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i * dv_i) / normalization
    Muuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i * dv_i) / normalization
    Muvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i * dv_i) / normalization
    Mvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i * dv_i) / normalization

    xi_calc = Muuuu + 2 * Muuvv + Mvvvv
    eta1_calc = Muuuu - Mvvvv
    eta2_calc = 2 * Muuuv + 2 * Muvvv
    lambda1_calc = Muuuu - 6 * Muuvv + Mvvvv
    lambda2_calc = 4 * Muuuv - 4 * Muvvv 

    return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, zeta1_calc, zeta2_calc, delta1_calc, delta2_calc, xi_calc, eta1_calc, eta2_calc, lambda1_calc, lambda2_calc







def hsm_error_fourth_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to fourth moments of star image to get errors.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape error. Goes up to fourth moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

    star = star.copy()

    # get vectors for data, weight and u, v
    data_i, weight_i, u_i, v_i = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    flux, cenu, cenv, size, g1, g2, flag = hsm(star)
    profile = galsim.Gaussian(sigma=1.0).dilate(size).shear(g1=g1, g2=g2).shift(cenu, cenv) * flux
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', offset=(star.image_pos-image.true_center))
    # convert image into kernel
    kernel_i = image.array.flatten()

    # now apply mask
    mask = weight_i != 0.
    data_i = data_i[mask]
    weight_i = weight_i[mask]
    kernel_i = kernel_i[mask]
    u_i = u_i[mask]
    v_i = v_i[mask]

    # with HSM as our starting guess, and kernel, let's use the weights for a final step. This makes everything a lot simpler, conceptually. We place all these results here, and then work through the errors later
    flux_calc = np.sum(weight_i * data_i * kernel_i)
    normalization = flux_calc

    u0_calc = np.sum(data_i * weight_i * kernel_i * u_i) / normalization
    v0_calc = np.sum(data_i * weight_i * kernel_i * v_i) / normalization
    # calculate moments
    du_i = u_i - u0_calc
    dv_i = v_i - v0_calc
    Muu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i) / normalization
    Mvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i) / normalization
    Muv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i) / normalization

    # now e0,e1,e2
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = Muu - Mvv
    e2_calc = 2 * Muv

    # calculate third moments
    Muuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i) / normalization
    Muuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i) / normalization
    Muvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i) / normalization
    Mvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i) / normalization

    zeta1_calc = Muuu + Muvv
    zeta2_calc = Mvvv + Muuv
    delta1_calc = Muuu - 3 * Muvv
    delta2_calc = -(Mvvv - 3 * Muuv)

    # calculate fourth moments
    Muuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i * du_i) / normalization
    Muuuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * du_i * dv_i) / normalization
    Muuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * du_i * dv_i * dv_i) / normalization
    Muvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i * dv_i * dv_i) / normalization
    Mvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i * dv_i * dv_i * dv_i) / normalization

    xi_calc = Muuuu + 2 * Muuvv + Mvvvv
    eta1_calc = Muuuu - Mvvvv
    eta2_calc = 2 * Muuuv + 2 * Muvvv
    lambda1_calc = Muuuu - 6 * Muuvv + Mvvvv
    lambda2_calc = 4 * Muuuv - 4 * Muvvv 

    #return flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc, zeta1_calc, zeta2_calc, delta1_calc, delta2_calc, xi_calc, eta1_calc, eta2_calc, lambda1_calc, lambda2_calc


    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum(np.power(weight_i ** 2 * kernel_i ** 2, 2) * sigma2_data_i)
    sigma_normalization = np.sqrt(sigma2_normalization)
    # flux is 2x normalization in hsm.cpp, so probably a factor of 2 here
    sigma_flux = 2 * sigma_normalization

    # flux fudge factors?
    flux_fudge_factor = 1.
    sigma_flux = sigma_flux * np.sqrt(flux_fudge_factor)
    sigma_normalization = 1. * sigma_normalization

    #####
    # u0, v0
    #####

    sigma2_u0_data = np.sum(np.power(weight_i * kernel_i * u_i / normalization, 2) * sigma2_data_i)
    sigma2_v0_data = np.sum(np.power(weight_i * kernel_i * v_i / normalization, 2) * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = np.power(u0_calc * sigma_normalization / normalization, 2)
    sigma2_v0_flux = np.power(v0_calc * sigma_normalization / normalization, 2)

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i + dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i - dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_e2_data = np.sum(np.power(2 * weight_i * kernel_i * 2 * du_i * dv_i / normalization, 2) * sigma2_data_i)

    sigma2_zeta1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i + du_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_zeta2_data = np.sum(np.power(2 * weight_i * kernel_i * (dv_i * dv_i * dv_i + du_i * du_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_delta1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i - 3 * du_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_delta2_data = np.sum(np.power(2 * weight_i * kernel_i * -(dv_i * dv_i * dv_i - 3 * du_i * du_i * dv_i) / normalization, 2) * sigma2_data_i)
    
    sigma2_xi_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i * du_i + 2 * du_i * du_i * dv_i * dv_i + dv_i * dv_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_eta1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i * du_i - dv_i * dv_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_eta2_data = np.sum(np.power(2 * weight_i * kernel_i * (2 * du_i * du_i * du_i * dv_i + 2 * du_i * dv_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_lambda1_data = np.sum(np.power(2 * weight_i * kernel_i * (du_i * du_i * du_i * du_i - 6 * du_i * du_i * dv_i * dv_i + dv_i * dv_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)
    sigma2_lambda2_data = np.sum(np.power(2 * weight_i * kernel_i * (4 * du_i * du_i * du_i * dv_i - 4 * du_i * dv_i * dv_i * dv_i) / normalization, 2) * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e0_v0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(np.power(2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_e2_v0 = np.sum(np.power(2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))

    sigma2_zeta1_u0 = np.sum(np.power(2 * (3 * du_i * du_i + dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_zeta1_v0 = np.sum(np.power(2 * 2 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_zeta2_u0 = np.sum(np.power(2 * 2 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_zeta2_v0 = np.sum(np.power(2 * (du_i * du_i + 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_delta1_u0 = np.sum(np.power(2 * (3 * du_i * du_i - 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_delta1_v0 = np.sum(np.power(2 * -(6 * du_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_delta2_u0 = np.sum(np.power(2 * 6 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_delta2_v0 = np.sum(np.power(2 * (3 * du_i * du_i - 3 * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    
    sigma2_xi_u0 = np.sum(np.power(2 * (4 * du_i * du_i * du_i + 4 * du_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_xi_v0 = np.sum(np.power(2 * (4 * du_i * du_i * dv_i + 4 * dv_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_eta1_u0 = np.sum(np.power(2 * 4 * du_i * du_i * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_eta1_v0 = np.sum(np.power(2 * -(4 * dv_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_eta2_u0 = np.sum(np.power(2 * (6 * du_i * du_i * dv_i + 2 * dv_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_eta2_v0 = np.sum(np.power(2 * (2 * du_i * du_i * du_i + 6 * du_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_lambda1_u0 = np.sum(np.power(2 * (4 * du_i * du_i * du_i - 12 * du_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_lambda1_v0 = np.sum(np.power(2 * (4 * dv_i * dv_i * dv_i - 12 * du_i * du_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))
    sigma2_lambda2_u0 = np.sum(np.power(2 * (12 * du_i * du_i * dv_i - 4 * dv_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_u0, 2))
    sigma2_lambda2_v0 = np.sum(np.power(2 * (4 * du_i * du_i * du_i - 12 * du_i * dv_i * dv_i) * weight_i * kernel_i * data_i / normalization * sigma_v0, 2))    

    # add sigma_normalization
    sigma2_e0_flux = np.power(e0_calc * sigma_normalization / normalization, 2)
    sigma2_e1_flux = np.power(e1_calc * sigma_normalization / normalization, 2)
    sigma2_e2_flux = np.power(e2_calc * sigma_normalization / normalization, 2)

    sigma2_zeta1_flux = np.power(zeta1_calc * sigma_normalization / normalization, 2)
    sigma2_zeta2_flux = np.power(zeta2_calc * sigma_normalization / normalization, 2)
    sigma2_delta1_flux = np.power(delta1_calc * sigma_normalization / normalization, 2)
    sigma2_delta2_flux = np.power(delta2_calc * sigma_normalization / normalization, 2)
    
    sigma2_xi_flux = np.power(xi_calc * sigma_normalization / normalization, 2)
    sigma2_eta1_flux = np.power(eta1_calc * sigma_normalization / normalization, 2)
    sigma2_eta2_flux = np.power(eta2_calc * sigma_normalization / normalization, 2)
    sigma2_lambda1_flux = np.power(lambda1_calc * sigma_normalization / normalization, 2)
    sigma2_lambda2_flux = np.power(lambda2_calc * sigma_normalization / normalization, 2)

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly correlated, so I'm probably missing a negative covariance term from the kernel that would bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    sigma_zeta1 = np.sqrt(sigma2_zeta1_data + sigma2_zeta1_u0 + sigma2_zeta1_v0 + sigma2_zeta1_flux)
    sigma_zeta2 = np.sqrt(sigma2_zeta2_data + sigma2_zeta2_u0 + sigma2_zeta2_v0 + sigma2_zeta2_flux)
    sigma_delta1 = np.sqrt(sigma2_delta1_data + sigma2_delta1_u0 + sigma2_delta1_v0 + sigma2_delta1_flux)
    sigma_delta2 = np.sqrt(sigma2_delta2_data + sigma2_delta2_u0 + sigma2_delta2_v0 + sigma2_delta2_flux)
    
    sigma_xi = np.sqrt(sigma2_xi_data + sigma2_xi_u0 + sigma2_xi_v0 + sigma2_xi_flux)
    sigma_eta1 = np.sqrt(sigma2_eta1_data + sigma2_eta1_u0 + sigma2_eta1_v0 + sigma2_eta1_flux)
    sigma_eta2 = np.sqrt(sigma2_eta2_data + sigma2_eta2_u0 + sigma2_eta2_v0 + sigma2_eta2_flux)
    sigma_lambda1 = np.sqrt(sigma2_lambda1_data + sigma2_lambda1_u0 + sigma2_lambda1_v0 + sigma2_lambda1_flux)
    sigma_lambda2 = np.sqrt(sigma2_lambda2_data + sigma2_lambda2_u0 + sigma2_lambda2_v0 + sigma2_lambda2_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    #sigma_zeta1 = sigma_zeta1 * 0.523
    #sigma_zeta2 = sigma_zeta2 * 0.545
    sigma_zeta1 = sigma_zeta1 * 0.52
    sigma_zeta2 = sigma_zeta2 * 0.55
#below are the numbers gleaned when looking for fudge factors using simulated stars of different snr and location
#snr 90, location (500, 500, 25), 1000 runs
#[ 1.12804519  1.07293111  1.07145898  1.13775826  1.10059309  1.10546455  1.05268536  0.99594863  0.9396108   0.93940038]
#snr 90, location (500, 500, 25), 100 runs
#[ 1.13037156  1.07224726  1.07309272  1.14044123  1.09669306  1.10661247  1.04602584  0.99712779  0.94301216  0.93644381]
#snr 70, location (500, 500, 25), 100 runs
#[ 1.12649224  1.07171045  1.07870498  1.14094039  1.09996217  1.10760745  1.04477721  0.99593876  0.93987085  0.94086416]
#snr 50, location (500, 500, 25), 100 runs
#[ 1.12264633  1.07698594  1.06969354  1.13584159  1.09865871  1.09636084  1.0537384   0.99585817  0.92885245  0.93264428]
#snr 90, location (100, 100, 55), 100 runs
#[ 1.13083505  1.08115786  1.06882666  1.14116332  1.0969551   1.10358841  1.04876238  0.99314536  0.94111276  0.94455409]
#snr 90, location (300, 200, 5), 100 runs
#[ 1.13755535  1.07547336  1.07685751  1.13319619  1.09941079  1.10964125  1.0472514   0.99847778  0.94133408  0.94094472]

    sigma_xi = sigma_xi * 2.4
    sigma_eta1 = sigma_eta1 * 2.6
    sigma_eta2 = sigma_eta2 * 2.6
    sigma_lambda1 = sigma_lambda1 * 1.1
    sigma_lambda2 = sigma_lambda2 * 1.1
#below are the numbers gleaned when looking for fudge factors using simulated stars of different snr and location
#snr 90, location (500, 500, 25), 100 runs
#[ 1.12781393  1.07440408  1.07547636  1.13551989  1.09799606  1.106043 1.04924972  0.99272887  0.93874354  0.94343509  0.99765259  0.9923932 1.0064688   1.03260727  1.03864627]
#snr 70, location (500, 500, 25), 100 runs
#[ 1.12131667  1.07808312  1.07338833  1.14472137  1.10117143  1.10211635 1.05297964  0.99333467  0.94037372  0.938079    1.00528673  0.9960066 1.00318868  1.04309171  1.04164243]
#snr 50, location (500, 500, 25), 100 runs
#[ 1.11280269  1.07289472  1.07523513  1.13778037  1.09686648  1.10188804 1.05146489  0.99525581  0.93859345  0.93938414  0.99820261  0.99002694 1.00116629  1.03686934  1.04992096]
#snr 90, location (100, 100, 55), 100 runs
#[ 1.12708727  1.07834415  1.07967473  1.13646662  1.0983406   1.10894742 1.05298616  0.99330377  0.94068533  0.942428    0.99922724  0.99681427 1.00924441  1.04122951  1.03503203]
#snr 90, location (300, 200, 5), 100 runs
#[ 1.12930729  1.07552532  1.07740932  1.14049254  1.10198351  1.10707529 1.047431    0.99691908  0.94090114  0.93752018  1.00435191  0.99778555 1.00900593  1.03330217  1.0272572 ]

    return sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_xi, sigma_eta1, sigma_eta2, sigma_lambda1, sigma_lambda2









