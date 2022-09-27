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

import numpy as np
import os
import sys
import traceback
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
    else:  # pragma: no cover
        # Other objects should be manually serialized by the initializer or the finish_read and
        # finish_write functions.
        # (We don't hit this in tests, so don't cover it, but if it happens in development,
        #  this helps produce a more sensible error.)
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
        # bytes need to be encoded.
        return value.encode()
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
    return kwargs

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

def _run_multi_helper(func, i, args, kwargs, log_level): # pragma: no cover
    # Note: This is covered by test_wcs.py:test_parallel, but for some reason it's not
    # showing up in codecov.  It's supposed to get captured by the combination of
    # concurrency=multiprocessing and calling coverage combine before uploading.
    # We're doing both of those things, but it's still not showing up.
    from io import StringIO
    import logging

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


def run_multi(func, nproc, raise_except, args, logger, kwargs=None):
    """Run a function possibly in multiprocessing mode.

    This is basically just doing a Pool.map, but it handles the logger properly (which cannot
    be pickled, so it cannot be passed to the function being run by the workers).

    :param func:        The function to run.  Signature should be:
                            func(\*args, logger=logger, \*\*kwargs)
    :param nproc:       How many processes to run.  If nproc=1, no multiprocessing is done.
                        nproc <= 0 means use all the cores.
    :param raise_except: Whether to raise any exceptions that happen in individual jobs.
    :param args:        a list of args for func for each job to run.
    :param logger:      The logger you would pass to func in single-processor mode.
    :param kwargs:      a list of kwargs for func for each job to run.  May also be a single dict
                        to use for all jobs. [default: None]

    :returns:   The output of func(\*args[i], \*\*kwargs[i]) for each item in the args, kwargs lists.
    """
    from multiprocessing import Pool
    if galsim.__version_info__ >= (2,4):
        from galsim.utilities import single_threaded
    else:
        try:
            from contextlib import nullcontext as single_threaded
        except ImportError:
            # python 3.6 equivalent of nullcontext (when we don't need the `as` part)
            from contextlib import suppress as single_threaded

    njobs = len(args)
    nproc = galsim.config.util.UpdateNProc(nproc, len(args), {}, logger)

    output_list = [None] * njobs
    err_list = [None] * njobs

    def log_output(result):
        i, out, log = result
        logger.info(log)
        if isinstance(out, Exception):
            logger.warning("Caught exception in multiprocessing job: %r",out)
            err_list[i] = out
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
            try:
                out = func(*args[i], logger=logger, **k)
            except Exception as e:
                if raise_except:
                    raise
                else:
                    tr = traceback.format_exc()
                    logger.warning("Caught exception:\n%s",tr)
                    logger.warning("Ignoring this failure and continuing on.")
            else:
                output_list[i] = out
    else:
        with single_threaded():
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

        # Now we can raise an error if there was one.
        if raise_except:
            errs = [e for e in err_list if e is not None]
            if len(errs) > 0:
                raise errs[0]

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


def calculate_moments(star, third_order=False, fourth_order=False, radial=False, errors=False, logger=None):
    r"""Calculate a bunch of moments using HSM for the weight function.

    The flux, 1st, and 2nd order moments are always calculated:

    .. math::

        M_00 &= \sum W(u,v) I(u,v) \\
        M_10 &= \sum W(u,v) I(u,v) du \\
        M_01 &= \sum W(u,v) I(u,v) dv \\
        M_11 &= \sum W(u,v) I(u,v) (du^2 + dv^2) \\
        M_20 &= \sum W(u,v) I(u,v) (du^2 - dv^2) \\
        M_02 &= \sum W(u,v) I(u,v) (2 du dv)

    where W(u,v) is the weight from the HSM fit and du,dv are the positions relative to the
    HSM measured centroid.

    If ``third_order`` is set to True, then 3rd order moments are also calculated and returned:

    .. math::

        M_21 &= \sum W(u,v) I(u,v) du (du^2 + dv^2) \\
        M_12 &= \sum W(u,v) I(u,v) dv (du^2 + dv^2) \\
        M_30 &= \sum W(u,v) I(u,v) du (du^2 - 3 dv^2) \\
        M_03 &= \sum W(u,v) I(u,v) dv (3 du^2 - dv^2)

    If ``fourth_order`` is set to True, then 4th order moments are also calculated and returned:

    .. math::

        M_22 &= \sum W(u,v) I(u,v) (du^2 + dv^2)^2 \\
        M_31 &= \sum W(u,v) I(u,v) (du^2 + dv^2) (du^2 - dv^2) \\
        M_13 &= \sum W(u,v) I(u,v) (du^2 + dv^2) (2 du dv) \\
        M_40 &= \sum W(u,v) I(u,v) (du^4 - 6 du^2 dv^2 + dv^4) \\
        M_04 &= \sum W(u,v) I(u,v) (du^2 - dv^2) (4 du dv)

    Higher order normalized radial moments (4th through 8th, even) are calculated if ``radial``
    is set to True:

    .. math::

        r^2 &\equiv du^2 + dv^2 \\
        M_22 &= \sum W(u,v) I(u,v) r^4 \\
        M_33 &= \sum W(u,v) I(u,v) r^6 \\
        M_44 &= \sum W(u,v) I(u,v) r^8  \\
        M_22n &= M_22/M_11^2 \\
        M_33n &= M_33/M_11^3 \\
        M_44n &= M_44/M_11^4

    For all of these, one can also have error estimates returned if ``errors`` is set to True.

    :param star:            Input star, with stamp, weight
    :param third_order:     Return the 3rd order moments? [default: False]
    :param fourth_order:    Return the 4th order moments? [default: False]
    :param radial:          Return the higher order radial moments? [default: False]
    :param errors:          Return the variance estimates of other returned values? [default: False]
    :param logger:          A logger object for logging debug info.  [default: None]

    :returns: A dict of the calculated moments, with the following keys/values:

        * M_00, M_10, M_01, M_11, M_20, M_02
        * M_21, M_12, M_30, M_03                          if ``third_order`` = True
        * M_22, M_31, M_13, M_40, M_04                    if ``fourth_order`` = True
        * M_22n, M_33n, M_44n                             if ``radial`` = True

    If ``errors`` = True, then also a second dict (with the same keys) giving the variances.
    """
    # get vectors for data, weight and u, v
    data, weight, u, v = star.data.getDataVector()
    # also get the values for the HSM kernel, which is just the fitted hsm model
    f, u0, v0, sigma, g1, g2, flag = star.hsm

    if flag:
        raise RuntimeError("HSM failed with flag %s" % flag)

    # build the HSM weight, writing into image
    profile = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0)
    image = profile.drawImage(star.image.copy(), method='sb', center=star.image_pos)

    # convert image into kernel
    kernel = image.array.flatten()

    # Anywhere the data is masked, fill in with the hsm profile.
    mask = weight == 0.
    if np.any(mask):
        data[mask] = kernel[mask] * np.sum(data[~mask])/np.sum(kernel[~mask])

    # Notation:
    #   W = kernel
    #   I = data
    #   V = var(data) -- used below.
    WI = kernel * data

    M00 = np.sum(WI)
    WI /= M00   # M00 is the normalization for all other moments.  So just divide once.

    # subtract off centroid
    u -= u0
    v -= v0

    # Store some quantities that we will use repeatedly below.
    # Note: This could still be sped up more by caching more combinations.
    usq = u*u
    vsq = v*v
    uv = u*v
    rsq = usq + vsq
    usqmvsq = usq - vsq

    WIu = WI * u
    WIv = WI * v
    WIrsq = WI * rsq
    WIuv = WI * uv

    # 1st moments, including the centroids
    M10 = np.sum(WIu) + u0
    M01 = np.sum(WIv) + v0

    # 2nd moments
    M11 = np.sum(WIrsq)
    M20 = np.sum(WI * usqmvsq)
    M02 = 2 * np.sum(WIuv)

    # Keep track of the tuple to return.  We may add more.
    ret = dict(M00=M00, M10=M10, M01=M01, M11=M11, M20=M20, M02=M02)

    # 3rd moments
    if third_order:
        M21 = np.sum(WIu * rsq)
        M12 = np.sum(WIv * rsq)
        M30 = np.sum(WIu * (usq-3*vsq))
        M03 = np.sum(WIv * (3*usq-vsq))
        ret.update(M21=M21, M12=M12, M30=M30, M03=M03)

    # 4th moments
    if fourth_order or radial or errors:
        rsq2 = rsq * rsq
        M22 = np.sum(WI * rsq2)
    if fourth_order:
        M31 = np.sum(WIrsq * usqmvsq)
        M13 = 2 * np.sum(WIrsq * uv)
        M40 = np.sum(WI * (usqmvsq**2 - 4*uv**2))
        M04 = 4 * np.sum(WIuv * usqmvsq)
        ret.update(M22=M22, M31=M31, M13=M13, M40=M40, M04=M04)

    # normalized radial moments
    if radial or (fourth_order and errors):
        rsq3 = rsq2 * rsq
        M33 = np.sum(WI * rsq3)
    if radial:
        rsq4 = rsq3 * rsq
        M44 = np.sum(WI * rsq4)
        M22n = M22/(M11**2)
        M33n = M33/(M11**3)
        M44n = M44/(M11**4)
        ret.update(M22n=M22n, M33n=M33n, M44n=M44n)

    if errors:

        #
        # Consider M00 first.
        #

        # If we take W, w to be fixed and assume that var(I) = 1/w, then
        #
        # var(M00) = var(sum_k W_k I_k)
        #          = sum_k W_k^2 var(I_k)
        #          = sum W_k^2 (1/w_k)
        #
        # However, W is not actually noiseless, since it is estimated from the data.
        # In particular it is set to null 5 constraint equations, which are:
        #
        # sum_k W_k I_k (u-u0) = 0
        # sum_k W_k I_k (v-v0) = 0
        # sum_k W_k I_k (rsq - ssq) = 0
        # sum_k W_k I_k (usqmvsq - e1 ssq) = 0
        # sum_k W_k I_k (2 uv - e2 ssq) = 0
        #
        # This means dW_i/dI_k != 0.  So,
        #
        # var(M00) = sum_k (dM00/dI_k)^2 (1/w_k)
        #          = sum_k (W_k + sum_i I_i dW_i/dI_k)^2 (1/w_k)
        #
        # dW_i/dI_k = (dW_i/du0) (du0/dI_k)
        #             + (dW_i/dv0) (dv0/dI_k)
        #             + (dW_i/dssq) (dssq/dI_k)
        #             + (dW_i/de1) (de1/dI_k)
        #             + (dW_i/de2) (de2/dI_k)
        #
        # The first factor in each term comes directly from the Gaussian form of W:
        #
        # dW_i/du0 = W_i (u_i - u0) / ssq
        # dW_i/du0 = W_i (v_i - v0) / ssq
        # dW_i/dssq = W_i (rsq_i - 2ssq) / (2ssq^2)
        # dW_i/de1 = W_i usqmvsq_i / (2ssq)
        # dW_i/de2 = W_i 2 uv_i / (2ssq)
        #
        # The second factors we can get from the three constraint equations by taking the
        # derivative of the whole equation with respect to I_k and then solving for the relevant
        # derivative (du0/dI_k, etc.) in each case.  We show the work to derive these below,
        # but here are the results:
        #
        # du0/dI_k = 2 W_k (u_k - u0) / M00
        # dv0/dI_k = 2 W_k (v_k - v0) / M00
        # dssq/dI_k = 2 W_k (rsq_k - ssq) / M00 / (3 - M22/ssq^2)
        # de1/dI_k = 2 W_k (usqmvsq_k / ssq) / M00 / (2 - M22/2ssq^2)
        # de2/dI_k = 2 W_k (2 uv_k / ssq) / M00 / (2 - M22/2ssq^2)
        #
        # Note: those final factors in the last 3 eqns are 1 for Gaussian profiles, but are not
        # in general.  Including them makes a difference for highly non-Gaussian profiles, such as
        # low-beta Moffat profiles.
        #
        # To simplify the subsequent notation, we define:
        #
        # A = 1/(3-M22/ssq^2)
        # B = 1/(2-M22/2ssq^2)
        #
        # So,
        #
        # dssq/dI_k = 2A W_k (rsq_k - ssq) / M00
        # de1/dI_k = 2B W_k (usqmvsq_k / ssq) / M00
        # de2/dI_k = 2B W_k (2 uv_k / ssq) / M00
        #
        # Putting these together:
        #
        # dW_i/dI_k = (2 W_i W_k / ssq M00) [ (u_i - u0) (u_k - u0)
        #                                     + (v_i - v0) (v_k - v0)
        #                                     + A (rsq_i - 2ssq) (rsq_k - ssq) / 2 ssq
        #                                     + B (usqmvsq_i / 2) (usqmvsq_k / ssq)
        #                                     + 2B (uv_i) (uv_k / ssq) ]
        #
        # Note: For most of these calculations we'll ignore terms that are first order in e1 or e2,
        #       since stars usually have fairly small ellitpicities, and including those factors
        #       properly adds a lot of complication with little impact on accuracy.
        #
        # dM00/dI_k = W_k + A ([sum_i I_i W_i (rsq_i/ssq - 2)/M00] W_k (rsq_k/ssq - 1))
        #                   Note: [..] = -1  (And the corresponding u0,v0,e1,e2 sums are all 0.)
        #           = W_k (1 - A (rsq_k/ssq - 1))
        #
        # So, finally, we have
        #
        # var(M00) = sum_k W_k^2/w_k (1 - A (rsq_k/ssq - 1))^2

        # WV = W^2 1/w
        WV = kernel**2
        WV[~mask] /= weight[~mask]  # Only use 1/w where w != 0
        WV[mask] /= np.mean(weight[~mask])

        A = 1/(3-M22/M11**2)
        B = 2/(4-M22/M11**2)
        dM00 = 1 - A*(rsq/M11-1)  # We'll need this combination a lot below, so save it.
        varM00 = np.sum(WV * dM00**2)

        # Set WV = W^2 1/w / M00^2 to incorporate the normalization into the weight for the rest.
        WV /= M00**2

        #
        # First order moments:
        #

        # For these, we add on the hsm centroid estimate
        #
        # M10 = sum(WIu) / M00 + u0
        # M01 = sum(WIv) / M00 + v0
        #
        # So var(u0) and var(v0) need to be included in the variance of M10, M01.
        # But also, the first term is something that is designed to be essentially 0 in the
        # HSM solution.  So the variances are really just var(u0), var(v0).
        #
        # var(u0) = sum_k (du0/dI_k)^2 var(I_k)
        # var(v0) = sum_k (dv0/dI_k)^2 var(I_k)
        #
        # We already gave these derivatives above, but we didn't really derive them.
        # We can do that here for du0/dI_k to show how that goes.
        #
        # We will need the relation:
        #
        # dW_i/du0 = W_i (u_i - u0) / ssq
        #
        # Start with
        #
        # sum_i W_i I_i (u_i-u0) = 0
        #
        # and differentiate with respect to I_k (for some particular, but arbitrary, k):
        #
        # d/dI_k (sum_i W_i I_i (u_i-u0)) = 0
        #
        # sum_i W_i I_i (-1) du0/dI_k + sum_i (dW_i/du0 du0/dI_k) I_i (u_i-u0) + W_k (u_k-u0) = 0
        #
        # (du0/dI_k) [sum_i W_i I_i ((u_i-u0)^2/ssq - 1)] = -W_k (u_k-u0)
        #
        # The sum in bracets is basically (<(u-u0)^2>/ssq - 1) M00.
        # By symmetry considerations and the fact that <(u-u0)^2 + (v-v0)^2> = ssq, we know that
        # <(u-u0)^2>/ssq = 1/2.  (This trick will show up a few times below too.)  So,
        #
        # (du0/dI_k) (-1/2 M00) = -W_k (u_k-u0)
        #
        # du0/dI_k = 2 W_k (u_k-u0) / M00
        #
        # var(u0) = sum (2 W (u-u0) / M00)^2 1/w
        #         = 4 sum WV (u-u0)^2
        #
        # Likewise,
        #
        # var(v0) = 4 sum WV (v-v0)^2

        varM10 = 4 * np.sum(WV * usq)
        varM01 = 4 * np.sum(WV * vsq)

        #
        # Second order moments:
        #

        # M11 = ssq, so we already have what we need to calculate the variance from what we
        # derived above for varM00.
        #
        # dssq/dI_k = 2A W_k (rsq_k - ssq) / M00
        #
        # Again, we quoted this result above, but let's derive it here.
        # Note: sigma never appears unsquared, so we treat the squared value ssq = sigma^2
        #       as the relevant variable.
        #
        # d/dI_k (sum_i W_i I_i (rsq_i - ssq)) = 0
        #
        # sum_i W_i I_i (-1) dssq/dI_k
        #       + sum_i (dW_i/dssq dssq/dI_k) I_i (rsq_i-ssq)
        #       + W_k (rsq_k-ssq) = 0
        #
        # (dssq/dI_k) [sum_i W_i I_i ((rsq_i-2ssq)(rsq_i-ssq)/2ssq^2 - 1)] = -W_k (rsq_k - ssq)
        # (dssq/dI_k) [sum_i W_i I_i ((1/2 rsq_i^4/ssq^2 - 3/2 rsq_i/ssq)] = -W_k (rsq_k - ssq)
        # (dssq/dI_k) M00 (1/2 M22/M11^2 - 3/2) = -W_k (rsq_k - ssq)
        #
        # dssq/dI_k = 2 W_k (rsq_k - ssq) / M00 / (3 - M22/M11^2)
        #           = 2A W_k (rsq_k - ssq) / M00
        #
        # var(M11) = sum_k (dM11/dI_k)^2 var(I_k)
        #          = sum_k (dssq/dI_k)^2 1/w
        #          = sum_k (2A W_k/M00 (rsq-ssq))^2 1/w
        #          = 4A^2 sum_k WV (rsq-ssq)**2

        varM11 = 4 * A**2 * np.sum(WV * (rsq - M11)**2)

        # M20 = ssq e1
        # M02 = ssq e2
        #
        # de1/dI_k = 2B W_k (usqmvsq_k / ssq) / M00
        #
        # Derivation:
        #
        # d/dI_k (sum_i W_i I_i (usq_i - vsq_i - e1 ssq)) = 0
        #
        # sum_i W_i I_i (-ssq) de1/dI_k
        #       + sum_i (dW_i/de1 de1/dI_k) I_i (usq_i - vsq_i - e1 ssq)
        #       + W_k (usq_k - vsq_k - e1 ssq) = 0
        #
        # (de1/dI_k) [sum_i W_i I_i ((usq_i - vsq_i)(usq_i - vsq_i - e1 ssq)/2ssq - ssq)]
        #                   = -W_k (usq_k - vsq_k - e1 ssq)
        # (de1/dI_k) [ sum_i W_i I_i (usq_i-vsq_i)^2/2ssq
        #              - e1/2 sum_i W_i I_i (usq_i-vsq_i)
        #              - ssq sum_i W_i I_i ] = -W_k (usq_k - vsq_k - e1 ssq)
        # Discard the terms linear in e1.
        # (de1/dI_k) M00 ssq (M22/4M11^2 - 1) = -W_k (usq_k - vsq_k)
        #
        # de1/dI_k = 2 W_k (usqmvsq_k / ssq) / M00 / (2 - M22/2M11^2)
        #          = 2B W_k (rsq_k - ssq) / M00
        #
        # dM20/dI_k = e1 dssq/dI_k + ssq de1/dI_k
        #           = 2 W_k / M00 (e1 A (rsq_k-ssq) + B usqmvsq_k)
        #
        # var(M20) = sum_k (dM20/dI_k)^2 var(I_k)
        #          = sum_k (2 W_k/M00 (B usqmvsq_k + e1 A (rsq_k - ssq)))^2 1/w
        #          = 4 sum_k WV (B usqmvsq_k + A e1 (rsq_k - ssq))^2
        #
        # Likewise,
        #
        # var(M02) = 4 sum_k WV (2B uv_k + A e2 (rsq_k - ssq))^2

        varM20 = 4 * np.sum(WV * (B*usqmvsq + A*M20 * (rsq/M11 - 1))**2)
        varM02 = 4 * np.sum(WV * (2*B*uv + A*M02 * (rsq/M11 - 1))**2)

        # Add these to the return tuple
        ret_var = dict(M00=varM00, M10=varM10, M01=varM01, M11=varM11, M20=varM20, M02=varM02)

        #
        # Third order moments:
        #

        # The third order moments M21 and M12 are strongly affected by the u0,v0 constraints,
        # so their variances are quite a bit lower than what you get from assuming W is constant.
        #
        # M21 = sum_k W_k I_k ((u-u0)^2 + (v-v0)^2) (u-u0) / M00
        #
        # From above:
        #
        # du0/dI_k = 2 W_k (u_k - u0) / M00
        # dM00/dI_k = W_k dM00
        # dW_i/dI_k = (2 W_i W_k / ssq M00) [ (u_i - u0) (u_k - u0)
        #                                     + (v_i - v0) (v_k - v0)
        #                                     + A (rsq_i - 2ssq) (rsq_k - ssq) / 2ssq
        #                                     + B (usqmvsq_i / 2) (usqmvsq_k / ssq)
        #                                     + 2B (uv_i) (uv_k / ssq) ]
        #
        # (For dW_i/dI_k, only the u-u0 term is important.)
        #
        # dM21/dI_k = W_k rsq_k (u_k-u0) / M00
        #             + sum_i W_i I_i (-3(u-u0)^2 - (v-v0)^2)) du0/dI_k / M00
        #             + sum_i dW_i/dI_k I_i rsq_i (u_i-u0) / M00
        #             - M21/M00 dM00/dI_k
        #           = W_k rsq_k (u_k-u0) / M00
        #             - 2 W_k (u_k-u0) / M00 [sum_i W_i I_i (rsq + 2(u-u0)^2)] / M00
        #             + 2 W_k (u_k-u0) / M00 [sum_i W_i I_i rsq_i/ssq (u_i-u0)^2] / M00
        #             - W_k M21/M00 dM00
        #           = W_k/M00 [(rsq - 4*M11 + M22/M11) (u-u0) - M21 dM00]
        # (where, as usual, we ignore terms related to e1,e2.)
        #
        # Similarly,
        #
        # dM12/dI_k = W_k/M00 [(rsq - 4*M11 + M22/M11) (v-v0) - M12 dM00]
        #
        # It turns out that M30 and M03 are not significantly affected by any of the constraint
        # equations.  The result is what you would get by keeping W constant:
        #
        # M30 = sum_k W_k I_k ((u-u0)^2 - 3(v-v0)^2) (u-u0) / M00
        #
        # dM30/dI_k = W_k (u_k-u0)^2 - 3(v_k-v0)^2) (u_k-u0) / M00
        #             + sum_i W_i I_i (-3(u-u0)^2 + 3(v-v0)^2)) du0/dI_k / M00
        #             + sum_i W_i I_i (6(v-v0)^2 (u-u0)) dv0/dI_k / M00
        #             + sum_i dW_i/dI_k I_i ((u_i-u0)^2 - 3(v-v0)^2) (u_i-u0) / M00
        #             - M30/M00 dM00/dI_k
        #           = W_k/M00 [(u^2-3v^2) (u-u0) - M30 dM00]
        #
        # dM03/dI_k = W_k/M00 [(3u^2-v^2) (v-v0) - M13 dM00]

        if third_order:
            varM21 = np.sum(WV * (u*(rsq - 4*M11 + M22/M11) - M21 * dM00)**2)
            varM12 = np.sum(WV * (v*(rsq - 4*M11 + M22/M11) - M12 * dM00)**2)
            varM30 = np.sum(WV * (u*(usq-3*vsq) - M30 * dM00)**2)
            varM03 = np.sum(WV * (v*(3*usq-vsq) - M03 * dM00)**2)
            ret_var.update(M21=varM21, M12=varM12, M30=varM30, M03=varM03)

        #
        # Fourth order moments:
        #

        # M22 is primarily affected by the ssq constraint perturbing W:
        #
        # dW_i/dI_k = (2A W_i W_k / ssq M00) (rsq_i - 2ssq) (rsq_k - ssq) / 2ssq
        #
        # M22 = sum_k W_k I_k rsq^2 / M00
        #
        # dM22/dI_k = W_k rsq_k^2 / M00
        #             + W_k A (rsq_k/ssq-1)/(ssq M00) sum_i W_i I_i rsq_i^2 (rsq_i-2ssq) / M00
        #             - M22/M00 dM00/dI_k
        #           = W_k rsq_k^2 / M00
        #             + W_k A (rsq_k/ssq-1)/M00 (M33/ssq - 2M22)
        #             - W_k M22/M00 dM00
        #           = W_k/M00 [rsq^2 + A*(rsq/ssq-1)*(M33/ssq - 2M22) - M22 dM00]
        #
        # M31 is affected by the e1 constraint equation:
        #
        # dW_i/dI_k = B (W_i W_k / ssq M00) usqmvsq_i usqmvsq_k / ssq
        #
        # M31 = sum_k W_k I_k rsq * ((u-u0)^2 - (v-v0)^2) / M00
        #
        # dM31/dI_k = W_k rsq_k usqmvsq_k / M00
        #             + B W_k usqmvsq_k/(ssq^2 M00) sum_i W_i I_i rsq_i usqmvsq_i^2 / M00
        #             - M31/M00 dM00/dI_k
        #           = W_k rsq_k usqmvsq_k / M00
        #             + B W_k usqmvsq_k/(ssq^2 M00) (M33/2)
        #             - W_k M31/M00 dM00
        #           = W_k/M00 [usqmvsq (rsq + B M33/(2ssq^2)) - M31 dM00]
        #
        # Similarly (using the e2 constraint),
        #
        # dM13/dI_k = W_k/M00 [2*uv (rsq + M33/(2ssq^2)) - M13 dM00]
        #
        # M40 and M04 are essentialy unaffected by the constraints, so the naive calculation
        # is pretty accurate (just like M30 and M03).
        #
        # dM40/dI_k = W_k/M00 [(usq^2 - 6uv^2 + vsq^2) - M40 dM00]
        # dM04/dI_k = W_k/M00 [(usq - vsq)(4uv) - M40 dM00]

        if fourth_order or radial:
            varM22 = np.sum(WV * (rsq2 + A*(rsq/M11-1)*(M33/M11-2*M22) - M22 * dM00)**2)
        if fourth_order:
            varM31 = np.sum(WV * (usqmvsq * (rsq + B*M33/(2*M11**2)) - M31 * dM00)**2)
            varM13 = np.sum(WV * (2*uv * (rsq + B*M33/(2*M11**2)) - M13 * dM00)**2)
            varM40 = np.sum(WV * (usqmvsq**2 - 4*uv**2 - M40 * dM00)**2)
            varM04 = np.sum(WV * (4*usqmvsq*uv - M04 * dM00)**2)
            ret_var.update(M22=varM22, M31=varM31, M13=varM13, M40=varM40, M04=varM04)

        #
        # Normalized radial moments
        #

        # The normalized radial moment of degree d is
        #
        # Mddn = Mdd M11^-d
        #      = sum_k W_k I_k (rsq/ssq)^d / M00
        #
        # dMddn/dI_k = W_k (rsq_k/ssq)^d / M00
        #              + (1/M00) sum_i W_i I_i rsq_i^d (-d ssq^-(d+1)) dssq/dI_k
        #              + W_k A (rsq/ssq-1)/(ssq M00) sum_i W_i I_i (rsq_i/ssq)^d (rsq_i-2ssq) / M00
        #              - W_k Mddn/M00 dM00
        #            = W_k (rsq_k/ssq)^d / M00
        #              - W_k/M00 A (rsq/ssq-1) (2d Mddn)
        #              + W_k/M00 A (rsq/ssq-1) (Md+1,d+1n - 2 Mddn)
        #              - W_k/M00 Mddn dM00
        #            = W_k/M00 [(rsq/ssq)^d + A (rsq/ssq - 1) (Md+1,d+1n - (2d+2) Mddn) - Mddn dM00]

        if radial:
            M55n = np.sum(WI * rsq4 * rsq) / M11**5
            varM22n = np.sum(WV * (rsq2/M11**2 + A*(rsq/M11-1)*(M33n - 6*M22n) - M22n*dM00)**2)
            varM33n = np.sum(WV * (rsq3/M11**3 + A*(rsq/M11-1)*(M44n - 8*M33n) - M33n*dM00)**2)
            varM44n = np.sum(WV * (rsq4/M11**4 + A*(rsq/M11-1)*(M55n - 10*M44n) - M44n*dM00)**2)
            ret_var.update(M22n=varM22n, M33n=varM33n, M44n=varM44n)

        return ret, ret_var

    else:
        return ret
