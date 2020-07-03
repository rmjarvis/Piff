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
        # Py2.7 strings need to be encoded.
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

    Higher order radial moments both unnormalized and normalized (4th through 8th, even) are calculated if ``radial``
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

    :returns: A tuple of the calculated moments:

        * M_00, M_10, M_01, M_11, M_20, M_02
        * M_21, M_12, M_30, M_03                          if ``third_order`` = True
        * M_22, M_31, M_13, M_40, M_04                    if ``fourth_order`` = True
        * M_22, M_33, M_44, M_22n, M_33n, M_44n           if ``radial`` = True
        * variance of all previous values (in same order) if ``errors`` = True
    """
    # get vectors for data, weight and u, v
    data, weight, u, v = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    f, u0, v0, sigma_m, g1, g2, flag = star.hsm
    # flux, u0,v0 (centroid in sky coord), sigma_m (detM^1/4 in sky coord), e1,e2, flag

    # convert g1,g2 to e1,e2
    s = galsim.Shear(g1=g1, g2=g2)
    e1 = s.e1
    e2 = s.e2

    if flag: #pragma: no cover 
        raise RuntimeError("HSM failed with flag %s" % flag)

    # build the HSM weight, writing into image
    profile = galsim.Gaussian(sigma=sigma_m, flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0)

    image = galsim.Image(star.image, dtype=float)
    profile.drawImage(image, method='sb', center=star.image_pos)

    # convert image into kernel
    kernel = image.array.flatten()

    # Anywhere the data is masked, fill in with the hsm profile.
    mask = weight == 0.
    if np.any(mask):
        data[mask] = kernel[mask] * np.sum(data[~mask])/np.sum(kernel[~mask])

    # These are masked image values, which we use in all the sums below.
    # The mask is defined in the line above, taking all pixels with non-zero weight
    # Notation:
    #   W = kernel    
    #   I = data
    #   V = var(data) -- used below.
    WI = kernel * data

    M00 = np.sum(WI)
    norm = M00            # This is the normalization for all other moments.
    WI /= norm

    # now subtract off centroid
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
    WIrsq = WI*rsq
    WIusqmvsq = WI*usqmvsq
    WIuv = WI*uv

    rsq2 = rsq * rsq
    rsq3 = rsq2 * rsq
    rsq4 = rsq3 * rsq

    # centroids
    M10 = np.sum(WIu) + u0
    M01 = np.sum(WIv) + v0
    
    # 2nd moments
    M11 = np.sum(WIrsq)
    M20 = np.sum(WIusqmvsq)
    M02 = 2 * np.sum(WIuv)

    # Keep track of the tuple to return.  We may add more.
    ret = (M00, M10, M01, M11, M20, M02)

    # 3rd moments
    if third_order:
        M21 = np.sum(WIu * rsq)
        M12 = np.sum(WIv * rsq)
        M30 = np.sum(WIu * (usq-3*vsq))
        M03 = np.sum(WIv * (3*usq-vsq))
        ret += (M21, M12, M30, M03)

    # 4th moments
    #M22 = np.sum(WI * rsq2)
    if fourth_order:
        M22 = np.sum(WI * rsq2)
        M31 = np.sum(WIrsq * usqmvsq)
        M13 = 2 * np.sum(WIrsq * uv)
        M40 = np.sum(WI * (usq*usq - 6*usq*vsq + vsq*vsq))
        M04 = 4 * np.sum(WIusqmvsq * uv)
        ret += (M22, M31, M13, M40, M04)

    # radial moments, return normalized moments
    if radial:
        M22 = np.sum(WI * rsq2)
        M33 = np.sum(WI * rsq3)
        M44 = np.sum(WI * rsq4)

        # normalized radial moments
        M22n = M22/(M11**2)
        M33n = M33/(M11**3)
        M44n = M44/(M11**4)

        ret += (M22n,M33n,M44n)

    if errors:
        # If we take W, w to be fixed and assume that var(I) = 1/w, then just considering the error in the numerator gives:

        # var(M00) = sum W^2 1/w
        # var(M10) = sum W^2 1/w u^2 / M00^2
        # var(M01) = sum W^2 1/w v^2 / M00^2
        # var(M11) = sum W^2 1/w (u^2 + v^2)^2 / M00^2
        # var(M20) = sum W^2 1/w (u^2 - v^2)^2 / M00^2
        # var(M02) = sum W^2 1/w (2uv)^2 / M00^2

        # WV = W^2 1/w
        WV = kernel**2
        WV[~mask] /= weight[~mask]  # Only use 1/w where w != 0
        WV[mask] /= np.mean(weight[~mask])

        # varM00
        varM00 = np.sum(WV)

        # now set WV = W^2 1/w / M00^2
        WV /= norm**2

        # varMnm for 1st and 2nd moments
        varM10 = np.sum( WV * (u)**2 )         #  really varu0 u = u-u0 so this also includes error on M00 denominator
        varM01 = np.sum( WV * (v)**2 )
        varM11 = np.sum( WV * (rsq - M11)**2 )      #  -M11 term includes error on M00 denominator
        varM20 = np.sum( WV * (usqmvsq - M20)**2 )  #  -M20 term includes error on M00 denominator
        varM02 = np.sum( WV * (2.*uv - M02)**2 )    #  -M02 term includes error on M00 denominator

        # scale variances
        varM10 *= (2.00**2)
        varM01 *= (2.00**2)
        varM11 *= (2.26**2)
        varM20 *= (2.13**2)
        varM02 *= (2.13**2)

        ret_err = (varM00, varM10, varM01, varM11, varM20, varM02)

        # variance for 3rd moments
        if third_order:
            varM21 = np.sum( WV * (u*rsq - M21)**2 )
            varM12 = np.sum( WV * (v*rsq - M12)**2 )
            varM30 = np.sum( WV * (u*(usq-3*vsq) - M30)**2 )
            varM03 = np.sum( WV * (v*(3*usq-vsq) - M03)**2 )

            varM21 *= (0.66**2)
            varM12 *= (0.66**2)
            varM30 *= (1.00**2)
            varM03 *= (1.00**2)

            ret_err += (varM21, varM12, varM30, varM03)

        # variance for r4th moments
        if fourth_order:
            varM22 = np.sum( WV * (rsq2 - M22)**2 )
            varM31 = np.sum( WV * (rsq*usqmvsq - M31)**2 )
            varM13 = np.sum( WV * (2*rsq*uv - M13)**2 )
            varM40 = np.sum( WV * (usq*usq - 6*usq*vsq + vsq*vsq - M40)**2)
            varM04 = np.sum( WV * (4*usqmvsq*uv - M04)**2 )

            varM22 *= (2.62**2)
            varM31 *= (2.38**2)
            varM13 *= (2.38**2)
            varM40 *= (1.05**2)
            varM04 *= (1.05**2)

            ret_err += (varM22, varM31, varM13, varM40, varM04)

        # variance for radial moments
        if radial:
            varM22 = np.sum( WV * (rsq2 - M22)**2 )
            varM33 = np.sum( WV * (rsq3 - M33)**2 )
            varM44 = np.sum( WV * (rsq4 - M44)**2 )

            # variance for normalized radial moments
            varM22n = np.sum(WV *( rsq2 - 2*M22*rsq/M11 + M22 )**2) / (M11**4)
            varM33n = np.sum(WV *( rsq3 - 3*M33*rsq/M11 + 2*M33 )**2) / (M11**6)
            varM44n = np.sum(WV *( rsq4 - 4*M44*rsq/M11 + 3*M44 )**2) / (M11**8)

            varM22 *= (2.62**2)
            varM33 *= (2.76**2)
            varM44 *= (2.72**2)
            varM22n *= (0.91**2)
            varM33n *= (0.88**2)
            varM44n *= (0.86**2)

            #ret_err += (varM22, varM33, varM44)
            ret_err += (varM22n, varM33n, varM44n)

    if errors:
        return ret + ret_err
    else:
        return ret
