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
import sys
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

def hsm(star):
    """ Use HSM to measure moments of star image.
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
    import scipy.linalg
    # Cribbed from implemenation in scipy.optimize.curve_fit
    # https://github.com/scipy/scipy/blob/maintenance/1.3.x/scipy/optimize/minpack.py#L771

    # Do Moore-Penrose inverse discarding zero singular values.
    try:
        _, s, VT = scipy.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s**2, VT)
    except np.linalg.LinAlgError as e:   # pragma: no cover
        # If we get an error, set the variance to "infinity".
        # MJ: I'm not sure if this can happen.  It shouldn't happen for singular matrices
        #     or other kinds of normal ill conditions.  But better safe than sorry.
        var = np.ones(jac.shape[1]) * 1.e100
        cov = np.diag(var)
    return cov

def _run_multi_helper(func, i, args, kwargs, log_level):
    if sys.version_info < (3,0):
        from io import BytesIO as StringIO
    else:
        from io import StringIO

    import logging
    import traceback

    # In multiprocessing, we cannot pass in the logger, so log to a string and then
    # return that back at the end to be logged by the parent process.
    logger = logging.getLogger('logtostring_%d'%i)
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    logger.setLevel(log_level) # Input logger in this case is the level to use.

    try:
        out = func(*args, logger=logger, **kwargs)
    except Exception as e:
        # Exceptions don't propagate through multiprocessing.  So best alternative
        # is to catch it and return it.  We can deal with it somehow on the other end.
        # Also add more details here with verbose>=2 to help with debugging.
        tr = traceback.format_exc()
        logger.info("Caught exception:\n%s",tr)
        out = e

    handler.flush()
    buf.flush()
    return i, out, buf.getvalue()


def run_multi(func, nproc, args, logger, kwargs=None):
    """Run a function possibly in multiprocessing mode.

    This is basically just doing a Pool.map, but it handles the logger properly (which cannot
    be pickled, so it cannot be passed to the function being run by the workers).

    :param func:    The function to run.  Signature should be:
                        func(*args, logger=logger, **kwargs)
    :param nproc:   How many processes to run.  If nproc=1, no multiprocessing is done.
                    nproc <= 0 means use all the cores.
    :param args:    a list of args for func for each job to run.
    :param logger:  The logger you would pass to func in single-processor mode.
    :param kwargs:  a list of kwargs for func for each job to run.  May also be a single dict
                    to use for all jobs. [default: None]

    :returns:   The output of func(*args[i], **kwargs[i]) for each item in the args, kwargs lists.
    """
    from multiprocessing import Pool
    njobs = len(args)
    nproc = galsim.config.util.UpdateNProc(nproc, len(args), {}, logger)

    output_list = [None] * njobs

    def log_output(result):
        i, out, log = result
        logger.info(log)
        if isinstance(out, Exception):
            logger.warning("Caught exception in multiprocessing job: %r",out)
        else:
            output_list[i] = out

    if nproc == 1:
        for i in range(njobs):
            if isinstance(kwargs, dict):
                k = kwargs
            elif kwargs is None:
                k = {}
            else:  # pragma: no cover  (We don't use this option currently)
                k = kwargs[i]
            out = func(*args[i], logger=logger, **k)
            output_list[i] = out
    else:
        pool = Pool(nproc)
        results = []
        for i in range(njobs):
            if isinstance(kwargs, dict):
                k = kwargs
            elif kwargs is None:
                k = {}
            else:  # pragma: no cover  (We don't use this option currently)
                k = kwargs[i]
            result = pool.apply_async(_run_multi_helper,
                                        args=(func, i, args[i], k, logger.logger.level),
                                        callback=log_output)
            results.append(result)
        # Make sure we get all the results.  Without this, it works fine on success, but
        # errors seems to be swallowed.
        [result.get() for result in results]
        # These are always necessary to close out the pool.
        pool.close()
        pool.join()
        pool.terminate()

    return output_list


def calculateSNR(image, weight):
    """Calculate the signal-to-noise of a given image.

    :param image:       The stamp image for a star
    :param weight:      The weight image for a star
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: the SNR value.
    """
    # The S/N value that we use will be the weighted total flux where the weight function
    # is the star's profile itself.  This is the maximum S/N value that any flux measurement
    # can possibly produce, which will be closer to an in-practice S/N than using all the
    # pixels equally.
    #
    # F = Sum_i w_i I_i^2
    # var(F) = Sum_i w_i^2 I_i^2 var(I_i)
    #        = Sum_i w_i I_i^2             <--- Assumes var(I_i) = 1/w_i
    #
    # S/N = F / sqrt(var(F))
    #
    # Note that if the image is pure noise, this will produce a "signal" of
    #
    # F_noise = Sum_i w_i 1/w_i = Npix
    #
    # So for a more accurate estimate of the S/N of the actual star itself, one should
    # subtract off Npix from the measured F.
    #
    # The final formula then is:
    #
    # F = Sum_i w_i I_i^2
    # S/N = (F-Npix) / sqrt(F)

    I = image.array
    w = weight.array
    mask = np.isfinite(I) & np.isfinite(w)
    F = (w[mask]*I[mask]**2).sum(dtype=float)
    Npix = np.sum(mask)
    if F < Npix:
        return 0.
    else:
        return (F - Npix) / np.sqrt(F)

