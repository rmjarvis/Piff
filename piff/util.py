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

def estimate_cov_from_jac(jac):
    """Estimate a covariance matrix from a jacobian as returned by scipy.optimize.least_squares

    .. math::

        C = (J^T J)^{-1}

    This is computed using Moore-Penrose inversion to discard singular values.

    :param jac:     The Jacobian as a 2d numpy array

    :returns: cov, a numpy array giving the estimated covariance.
    """
    import scipy
    # Cribbed from implemenation in scipy.optimize.curve_fit
    # https://github.com/scipy/scipy/blob/maintenance/1.3.x/scipy/optimize/minpack.py#L771

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = scipy.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    cov = np.dot(VT.T / s**2, VT)
    return cov


def hsm_error(star, logger=None, return_error=True):
    r"""Use python implementation of HSM to measure moments of star image to get errors.

    Does not go beyond second moments.

    Slow since it's python, not C, but we should only have to do this once per star.

    calculate the error on our e0,e1,e2

    .. math::

        e_0 = \sum \left[ (x-x_0)^2 + (y-y_0)^2  \right] K(x,y) I(x,y)

    where K(x,y) is the HSM kernel, I(x,y) is the image, s(x,y) is the shot noise per pixel
    so

    .. math::

        \sigma^2(e_0) = \sum \left\( \left[ (x-x_0)^2 + (y-y_0)^2 \right] K(x,y) \right\)^2 s^2(x,y)

    TODO: might be a factor of 2 missing still?
    TODO: what do the _i subscripts indicate? can I cut that and keep clarity?

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]
    :param return_error:    Boolean. If true, will also return the shape error.
                            [default: True]

    :returns:               The shape (and error if return_error).
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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
    sigma2_normalization = np.sum((weight_i ** 2 * kernel_i ** 2)**2 * sigma2_data_i)
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

    sigma2_u0_data = np.sum((weight_i * kernel_i * u_i / normalization)**2 * sigma2_data_i)
    sigma2_v0_data = np.sum((weight_i * kernel_i * v_i / normalization)**2 * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = (u0_calc * sigma_normalization / normalization)**2
    sigma2_v0_flux = (v0_calc * sigma_normalization / normalization)**2

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e2_data = np.sum(
        (2 * weight_i * kernel_i * 2*du_i*dv_i / normalization)**2 * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(
        (2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e0_v0 = np.sum(
        (2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(
        (2 * 2 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e2_v0 = np.sum(
        (2 * 2 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    # add sigma_normalization
    sigma2_e0_flux = (e0_calc * sigma_normalization / normalization)**2
    sigma2_e1_flux = (e1_calc * sigma_normalization / normalization)**2
    sigma2_e2_flux = (e2_calc * sigma_normalization / normalization)**2

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly
    # correlated, so I'm probably missing a negative covariance term from the kernel that would
    # bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
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

    return (flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc,
            sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2)


def hsm_third_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to third moments of star image.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape. Goes up to third moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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

    return (flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc,
            zeta1_calc, zeta2_calc, delta1_calc, delta2_calc)

def hsm_error_third_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to third moments of star image to get errors.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape error. Goes up to third moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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


    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum((weight_i**2 * kernel_i**2)**2 * sigma2_data_i)
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

    sigma2_u0_data = np.sum((weight_i * kernel_i * u_i / normalization)**2 * sigma2_data_i)
    sigma2_v0_data = np.sum((weight_i * kernel_i * v_i / normalization)**2 * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = (u0_calc * sigma_normalization / normalization)**2
    sigma2_v0_flux = (v0_calc * sigma_normalization / normalization)**2

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e2_data = np.sum(
        (2 * weight_i * kernel_i * 2*du_i*dv_i / normalization)**2 * sigma2_data_i)

    sigma2_zeta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_zeta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 - 3*dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (3*du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e0_v0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e2_v0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_zeta1_u0 = np.sum(
        (2 * (3*du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta1_v0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_zeta2_u0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta2_v0 = np.sum(
        (2 * (du_i**2 + 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_delta1_u0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta1_v0 = np.sum(
        (-12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_delta2_u0 = np.sum(
        (12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta2_v0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    # add sigma_normalization
    sigma2_e0_flux = (e0_calc * sigma_normalization / normalization)**2
    sigma2_e1_flux = (e1_calc * sigma_normalization / normalization)**2
    sigma2_e2_flux = (e2_calc * sigma_normalization / normalization)**2

    sigma2_zeta1_flux = (zeta1_calc * sigma_normalization / normalization)**2
    sigma2_zeta2_flux = (zeta2_calc * sigma_normalization / normalization)**2
    sigma2_delta1_flux = (delta1_calc * sigma_normalization / normalization)**2
    sigma2_delta2_flux = (delta2_calc * sigma_normalization / normalization)**2

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly
    # correlated, so I'm probably missing a negative covariance term from the kernel that would
    # bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    sigma_zeta1 = np.sqrt(sigma2_zeta1_data + sigma2_zeta1_u0 + sigma2_zeta1_v0 + sigma2_zeta1_flux)
    sigma_zeta2 = np.sqrt(sigma2_zeta2_data + sigma2_zeta2_u0 + sigma2_zeta2_v0 + sigma2_zeta2_flux)
    sigma_delta1 = np.sqrt(
        sigma2_delta1_data + sigma2_delta1_u0 + sigma2_delta1_v0 + sigma2_delta1_flux)
    sigma_delta2 = np.sqrt(
        sigma2_delta2_data + sigma2_delta2_u0 + sigma2_delta2_v0 + sigma2_delta2_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    sigma_zeta1 = sigma_zeta1 * 0.52
    sigma_zeta2 = sigma_zeta2 * 0.55

    return (sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
            sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2)

def hsm_fourth_moments(star, logger=None):
    """ Use python implementation of HSM to measure up to fourth moments of star image.


    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape. Goes up to fourth moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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

    return (flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc,
            zeta1_calc, zeta2_calc, delta1_calc, delta2_calc, xi_calc,
            eta1_calc, eta2_calc, lambda1_calc, lambda2_calc)

def hsm_error_fourth_moments(star, logger=None):
    """Use python implementation of HSM to measure up to fourth moments of star image to get errors.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:               The shape error. Goes up to fourth moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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

    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum((weight_i ** 2 * kernel_i ** 2)**2 * sigma2_data_i)
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

    sigma2_u0_data = np.sum((weight_i * kernel_i * u_i / normalization)**2 * sigma2_data_i)
    sigma2_v0_data = np.sum((weight_i * kernel_i * v_i / normalization)**2 * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = (u0_calc * sigma_normalization / normalization)**2
    sigma2_v0_flux = (v0_calc * sigma_normalization / normalization)**2

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e2_data = np.sum(
        (2 * weight_i * kernel_i * 2*du_i*dv_i / normalization)**2 * sigma2_data_i)

    sigma2_zeta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_zeta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (dv_i**2 + du_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 - 3*dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (3*du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)

    sigma2_xi_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 + dv_i**2)**2 / normalization)**2 * sigma2_data_i)
    sigma2_eta1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**4 - dv_i**4) / normalization)**2 * sigma2_data_i)
    sigma2_eta2_data = np.sum(
        (2 * weight_i * kernel_i * 2*du_i*dv_i * (du_i**2 + dv_i**2) / normalization)**2
        * sigma2_data_i)

    sigma2_lambda1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**4 - 6 * du_i**2 * dv_i**2 + dv_i**4) / normalization)**2
        * sigma2_data_i)
    sigma2_lambda2_data = np.sum(
        (2 * weight_i * kernel_i * 4 * du_i * dv_i * (du_i**2 - dv_i**2) / normalization)**2
        * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e0_v0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e2_v0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_zeta1_u0 = np.sum(
        (2 * (3*du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta1_v0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_zeta2_u0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta2_v0 = np.sum(
        (2 * (du_i**2 + 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_delta1_u0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta1_v0 = np.sum(
        (-12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_delta2_u0 = np.sum(
        (12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta2_v0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_xi_u0 = np.sum(
        (8 * du_i * (du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_xi_v0 = np.sum(
        (8 * dv_i * (du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_eta1_u0 = np.sum(
        (8 * du_i**3 * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_eta1_v0 = np.sum(
        (-8 * dv_i**3 * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_eta2_u0 = np.sum(
        (4 * dv_i * (3*du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_eta2_v0 = np.sum(
        (4 * du_i * (du_i**2 + 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_lambda1_u0 = np.sum(
        (8 * du_i * (du_i**2 - 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_lambda1_v0 = np.sum(
        (8 * dv_i * (dv_i**2 - 3*du_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_lambda2_u0 = np.sum(
        (8 * dv_i * (3*du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_lambda2_v0 = np.sum(
        (8 * du_i * (du_i**2 - 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    # add sigma_normalization
    sigma2_e0_flux = (e0_calc * sigma_normalization / normalization)**2
    sigma2_e1_flux = (e1_calc * sigma_normalization / normalization)**2
    sigma2_e2_flux = (e2_calc * sigma_normalization / normalization)**2

    sigma2_zeta1_flux = (zeta1_calc * sigma_normalization / normalization)**2
    sigma2_zeta2_flux = (zeta2_calc * sigma_normalization / normalization)**2
    sigma2_delta1_flux = (delta1_calc * sigma_normalization / normalization)**2
    sigma2_delta2_flux = (delta2_calc * sigma_normalization / normalization)**2

    sigma2_xi_flux = (xi_calc * sigma_normalization / normalization)**2
    sigma2_eta1_flux = (eta1_calc * sigma_normalization / normalization)**2
    sigma2_eta2_flux = (eta2_calc * sigma_normalization / normalization)**2
    sigma2_lambda1_flux = (lambda1_calc * sigma_normalization / normalization)**2
    sigma2_lambda2_flux = (lambda2_calc * sigma_normalization / normalization)**2

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly
    # correlated, so I'm probably missing a negative covariance term from the kernel that would
    # bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    sigma_zeta1 = np.sqrt(sigma2_zeta1_data + sigma2_zeta1_u0 + sigma2_zeta1_v0 + sigma2_zeta1_flux)
    sigma_zeta2 = np.sqrt(sigma2_zeta2_data + sigma2_zeta2_u0 + sigma2_zeta2_v0 + sigma2_zeta2_flux)
    sigma_delta1 = np.sqrt(
        sigma2_delta1_data + sigma2_delta1_u0 + sigma2_delta1_v0 + sigma2_delta1_flux)
    sigma_delta2 = np.sqrt(
        sigma2_delta2_data + sigma2_delta2_u0 + sigma2_delta2_v0 + sigma2_delta2_flux)

    sigma_xi = np.sqrt(sigma2_xi_data + sigma2_xi_u0 + sigma2_xi_v0 + sigma2_xi_flux)
    sigma_eta1 = np.sqrt(sigma2_eta1_data + sigma2_eta1_u0 + sigma2_eta1_v0 + sigma2_eta1_flux)
    sigma_eta2 = np.sqrt(sigma2_eta2_data + sigma2_eta2_u0 + sigma2_eta2_v0 + sigma2_eta2_flux)
    sigma_lambda1 = np.sqrt(
        sigma2_lambda1_data + sigma2_lambda1_u0 + sigma2_lambda1_v0 + sigma2_lambda1_flux)
    sigma_lambda2 = np.sqrt(
        sigma2_lambda2_data + sigma2_lambda2_u0 + sigma2_lambda2_v0 + sigma2_lambda2_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    sigma_zeta1 = sigma_zeta1 * 0.52
    sigma_zeta2 = sigma_zeta2 * 0.55

    sigma_xi = sigma_xi * 2.4
    sigma_eta1 = sigma_eta1 * 2.6
    sigma_eta2 = sigma_eta2 * 2.6
    sigma_lambda1 = sigma_lambda1 * 1.1
    sigma_lambda2 = sigma_lambda2 * 1.1

    return (sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
            sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2, sigma_xi,
            sigma_eta1, sigma_eta2, sigma_lambda1, sigma_lambda2)


def hsm_orthogonal(star, logger=None):
    """ Use python implementation of HSM to measure up to third moments plus orthogonal radial moments up to eighth moments of star image.


    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:   The shape. Goes up to third moments plus orthogonal radial moments up to eighth
                moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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

    orth4_calc = Muuuu + 2 * Muuvv + Mvvvv - 3 * Muu - 3 * Mvv

    # calculate sixth moments
    Muuuuuu = 2 * np.sum(
        data_i * weight_i * kernel_i * du_i * du_i * du_i * du_i * du_i * du_i) / normalization
    Muuuuvv = 2 * np.sum(
        data_i * weight_i * kernel_i * du_i * du_i * du_i * du_i * dv_i * dv_i) / normalization
    Muuvvvv = 2 * np.sum(
        data_i * weight_i * kernel_i * du_i * du_i * dv_i * dv_i * dv_i * dv_i) / normalization
    Mvvvvvv = 2 * np.sum(
        data_i * weight_i * kernel_i * dv_i * dv_i * dv_i * dv_i * dv_i * dv_i) / normalization

    orth6_calc = (Muuuuuu + 3 * Muuuuvv + 3 * Muuvvvv + Mvvvvvv
                  - 8 * Muuuu - 16 * Muuvv - 8 * Mvvvv + 12 * Muu + 12 * Mvv)

    # calculate eighth moments
    Muuuuuuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i**8) / normalization
    Muuuuuuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**6 * dv_i**2) / normalization
    Muuuuvvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**4 * dv_i**4) / normalization
    Muuvvvvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**2 * dv_i**6) / normalization
    Mvvvvvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i**8) / normalization

    orth8_calc = (Muuuuuuuu + 4 * Muuuuuuvv + 6 * Muuuuvvvv + 4 * Muuvvvvvv + Mvvvvvvvv
                  - 15 * Muuuuuu - 45 * Muuuuvv - 45 * Muuvvvv - 15 * Mvvvvvv
                  + 60 * Muuuu + 120 * Muuvv + 60 * Mvvvv - 60 * Muu - 60 * Mvv)

    return (flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc,
            zeta1_calc, zeta2_calc, delta1_calc, delta2_calc,
            orth4_calc, orth6_calc, orth8_calc)


def hsm_error_orthogonal(star, logger=None):
    """Use python implementation of HSM to measure up to fourth moments plus orthogonal radial
    moments up to eighth moments of star image to get errors.

    :param star:            Input star, with stamp, weight
    :param logger:          A logger object for logging debug info.
                            [default: None]

    :returns:   The shape error.
                Goes up to third moments plus orthogonal radial moments up to eighth moments.
    """
    from .gsobject_model import Gaussian
    from .star import Star

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

    # with HSM as our starting guess, and kernel, let's use the weights for a final step.
    # This makes everything a lot simpler, conceptually. We place all these results here,
    # and then work through the errors later
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
    Muuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i**3) / normalization
    Muuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**2 * dv_i) / normalization
    Muvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i**2) / normalization
    Mvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i**3) / normalization

    zeta1_calc = Muuu + Muvv
    zeta2_calc = Mvvv + Muuv
    delta1_calc = Muuu - 3 * Muvv
    delta2_calc = -(Mvvv - 3 * Muuv)

    # calculate fourth moments
    Muuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i**4) / normalization
    Muuuv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**3 * dv_i) / normalization
    Muuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**2 * dv_i**2) / normalization
    Muvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i * dv_i**3) / normalization
    Mvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i**4) / normalization

    orth4_calc = Muuuu + 2 * Muuvv + Mvvvv - 3 * Muu - 3 * Mvv

    # calculate sixth moments
    Muuuuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i**6) / normalization
    Muuuuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**4 * dv_i**2) / normalization
    Muuvvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**2 * dv_i**4) / normalization
    Mvvvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i**6) / normalization

    orth6_calc = (Muuuuuu + 3 * Muuuuvv + 3 * Muuvvvv + Mvvvvvv
                  - 8 * Muuuu - 16 * Muuvv - 8 * Mvvvv + 12 * Muu + 12 * Mvv)

    # calculate eighth moments
    Muuuuuuuu = 2 * np.sum(data_i * weight_i * kernel_i * du_i**8) / normalization
    Muuuuuuvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**6 * dv_i**2) / normalization
    Muuuuvvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**4 * dv_i**4) / normalization
    Muuvvvvvv = 2 * np.sum(data_i * weight_i * kernel_i * du_i**2 * dv_i**6) / normalization
    Mvvvvvvvv = 2 * np.sum(data_i * weight_i * kernel_i * dv_i**8) / normalization

    orth8_calc = (Muuuuuuuu + 4 * Muuuuuuvv + 6 * Muuuuvvvv + 4 * Muuvvvvvv + Mvvvvvvvv
                  - 15 * Muuuuuu - 45 * Muuuuvv - 45 * Muuvvvv - 15 * Mvvvvvv
                  + 60 * Muuuu + 120 * Muuvv + 60 * Mvvvv - 60 * Muu - 60 * Mvv)

    # normalization for the various sums over pixels
    normalization2 = normalization * normalization

    sigma2_data_i = 1. / weight_i
    sigma2_normalization = np.sum((weight_i ** 2 * kernel_i ** 2)**2 * sigma2_data_i)
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

    sigma2_u0_data = np.sum((weight_i * kernel_i * u_i / normalization)**2 * sigma2_data_i)
    sigma2_v0_data = np.sum((weight_i * kernel_i * v_i / normalization)**2 * sigma2_data_i)

    # add sigma_normalization
    sigma2_u0_flux = (u0_calc * sigma_normalization / normalization)**2
    sigma2_v0_flux = (v0_calc * sigma_normalization / normalization)**2

    # technically we also need the contribution to the kernel!

    sigma_u0 = np.sqrt(sigma2_u0_data + sigma2_u0_flux)
    sigma_v0 = np.sqrt(sigma2_v0_data + sigma2_v0_flux)

    # u0, v0 fudge factors
    sigma_u0 = sigma_u0 * 2.1
    sigma_v0 = sigma_v0 * 2.1

    # now calculate errors: ie. shot and read noise per pixel

    # three terms: those proportional to: sdata_i, sigma_u0 and sigma_v0, and sigma_normalization
    sigma2_e0_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e1_data = np.sum(
        (2 * weight_i * kernel_i * (du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_e2_data = np.sum(
        (2 * weight_i * kernel_i * 2 * du_i * dv_i / normalization)**2 * sigma2_data_i)

    sigma2_zeta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_zeta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (du_i**2 + dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta1_data = np.sum(
        (2 * weight_i * kernel_i * du_i * (du_i**2 - 3*dv_i**2) / normalization)**2 * sigma2_data_i)
    sigma2_delta2_data = np.sum(
        (2 * weight_i * kernel_i * dv_i * (3*du_i**2 - dv_i**2) / normalization)**2 * sigma2_data_i)

    sigma2_orth4_data = np.sum(
        (2 * weight_i * kernel_i * ((du_i**2 + dv_i**2)**2 - 3*(du_i**2 - dv_i**2)) / normalization)**2 * sigma2_data_i)
    sigma2_orth6_data = np.sum(
        (2 * weight_i * kernel_i * ((du_i**2 + dv_i**2)**3 - 8 * (du_i**2 + dv_i**2)*2 + 12 * (du_i**2 + dv_i**2)) / normalization)**2 * sigma2_data_i)
    sigma2_orth8_data = np.sum(
        (2 * weight_i * kernel_i * ((du_i**2 + dv_i**2)**4 - 15 * (du_i**2 + dv_i**2)**3 + 60 * (du_i**2 + dv_i**2)**2 - 60 * (du_i**2 + dv_i**2)) / normalization)**2 * sigma2_data_i)

    # add sigma_u0, sigma_v0. This is ignoring the kernel!
    sigma2_e0_u0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e0_v0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_e1_u0 = sigma2_e0_u0
    sigma2_e1_v0 = sigma2_e0_v0
    sigma2_e2_u0 = np.sum(
        (4 * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_e2_v0 = np.sum(
        (4 * du_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_zeta1_u0 = np.sum(
        (2 * (3*du_i**2 + dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta1_v0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_zeta2_u0 = np.sum(
        (4 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_zeta2_v0 = np.sum(
        (2 * (du_i**2 + 3*dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_delta1_u0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta1_v0 = np.sum(
        (-12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_delta2_u0 = np.sum(
        (12 * du_i * dv_i * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_delta2_v0 = np.sum(
        (6 * (du_i**2 - dv_i**2) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    sigma2_orth4_u0 = np.sum(
        (4 * du_i * (2 * du_i**2 + 2 * dv_i**2 - 3) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_orth4_v0 = np.sum(
        (4 * dv_i * (2 * du_i**2 + 2 * dv_i**2 - 3) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_orth6_u0 = np.sum(
        (4 * du_i * (3 * (du_i**2 + dv_i**2)**2 - 16 * (du_i**2 + dv_i**2) + 12) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_orth6_v0 = np.sum(
        (4 * dv_i * (3 * (du_i**2 + dv_i**2)**2 - 16 * (du_i**2 + dv_i**2) + 12) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)
    sigma2_orth8_u0 = np.sum(
        (4 * du_i * (4 * (du_i**2 + dv_i**2)**3 - 45 * (du_i**2 + dv_i**2)**2 + 120 * (du_i**2 + dv_i**2) - 60) * weight_i * kernel_i * data_i / normalization * sigma_u0)**2)
    sigma2_orth8_v0 = np.sum(
        (4 * dv_i * (4 * (du_i**2 + dv_i**2)**3 - 45 * (du_i**2 + dv_i**2)**2 + 120 * (du_i**2 + dv_i**2) - 60) * weight_i * kernel_i * data_i / normalization * sigma_v0)**2)

    # add sigma_normalization
    sigma2_e0_flux = (e0_calc * sigma_normalization / normalization)**2
    sigma2_e1_flux = (e1_calc * sigma_normalization / normalization)**2
    sigma2_e2_flux = (e2_calc * sigma_normalization / normalization)**2

    sigma2_zeta1_flux = (zeta1_calc * sigma_normalization / normalization)**2
    sigma2_zeta2_flux = (zeta2_calc * sigma_normalization / normalization)**2
    sigma2_delta1_flux = (delta1_calc * sigma_normalization / normalization)**2
    sigma2_delta2_flux = (delta2_calc * sigma_normalization / normalization)**2

    sigma2_orth4_flux = (orth4_calc * sigma_normalization / normalization)**2
    sigma2_orth6_flux = (orth6_calc * sigma_normalization / normalization)**2
    sigma2_orth8_flux = (orth8_calc * sigma_normalization / normalization)**2

    # taking out the flux - e0 errors for now. lmfit finds that these two variables are highly
    # correlated, so I'm probably missing a negative covariance term from the kernel that would
    # bring this back in line. As it is, including sigma2_e0_flux leads to overestimated errors
    sigma_e0 = np.sqrt(sigma2_e0_data + sigma2_e0_u0 + sigma2_e0_v0)# + sigma2_e0_flux)
    sigma_e1 = np.sqrt(sigma2_e1_data + sigma2_e1_u0 + sigma2_e1_v0 + sigma2_e1_flux)
    sigma_e2 = np.sqrt(sigma2_e2_data + sigma2_e2_u0 + sigma2_e2_v0 + sigma2_e2_flux)

    sigma_zeta1 = np.sqrt(sigma2_zeta1_data + sigma2_zeta1_u0 + sigma2_zeta1_v0 + sigma2_zeta1_flux)
    sigma_zeta2 = np.sqrt(sigma2_zeta2_data + sigma2_zeta2_u0 + sigma2_zeta2_v0 + sigma2_zeta2_flux)
    sigma_delta1 = np.sqrt(
        sigma2_delta1_data + sigma2_delta1_u0 + sigma2_delta1_v0 + sigma2_delta1_flux)
    sigma_delta2 = np.sqrt(
        sigma2_delta2_data + sigma2_delta2_u0 + sigma2_delta2_v0 + sigma2_delta2_flux)

    sigma_orth4 = np.sqrt(sigma2_orth4_data + sigma2_orth4_u0 + sigma2_orth4_v0 + sigma2_orth4_flux)
    sigma_orth6 = np.sqrt(sigma2_orth6_data + sigma2_orth6_u0 + sigma2_orth6_v0 + sigma2_orth6_flux)
    sigma_orth8 = np.sqrt(sigma2_orth8_data + sigma2_orth8_u0 + sigma2_orth8_v0 + sigma2_orth8_flux)

    #####
    # FUDGE VALUES
    # in my experience (based on creating these for fixed noise level and measuring variance)
    # the errors need these fudge factors.
    #####

    sigma_e0 = sigma_e0 * 1.8
    sigma_e1 = sigma_e1 * 2.3
    sigma_e2 = sigma_e2 * 2.3

    sigma_zeta1 = sigma_zeta1 * 0.52
    sigma_zeta2 = sigma_zeta2 * 0.55

    sigma_orth4 = sigma_orth4 * 0.81
    sigma_orth6 = sigma_orth6 * 0.34
    sigma_orth8 = sigma_orth8 * 0.51

    return (sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2,
            sigma_zeta1, sigma_zeta2, sigma_delta1, sigma_delta2,
            sigma_orth4, sigma_orth6, sigma_orth8)
