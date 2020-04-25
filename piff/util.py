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

from .star import Star

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
    scale, shear, theta, flip = star.data.local_wcs._toJacobian().getDecomposition()
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

    localwcs = star.data.local_wcs
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

def calculate_moments(star, third_order=False, fourth_order=False, radial=False, errors=False,
                      logger=None):
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
        M_04 &= \sum W(u,v) I(u,v) (du^2 - dv^2) (4 du dv) \\

    Higher order "orthogonal" radial moments (4th through 8th, even) are calculated if ``radial``
    is set to True:

    .. math::

        r^2 &\equiv du^2 + dv^2 \\
        M*_22 &= \sum W(u,v) I(u,v) (r^4 - 3r^2)
        M*_33 &= \sum W(u,v) I(u,v) (r^6 - 8r^4 + 12r^2)
        M*_44 &= \sum W(u,v) I(u,v) (r^8 - 15r^6 + 60r^4 - 60r^2)

    For all of these, one can also have error estimates returned if ``errors`` is set to True.

    :param star:            Input star, with stamp, weight
    :param third_order:     Return the 3rd order moments? [default: False]
    :param fourth_order:    Return the 4th order moments? [default: False]
    :param raidal:          Return the higher order radial moments? [default: False]
    :param errors:          Return the variance estimates of other returned values? [default: False]
    :param logger:          A logger object for logging debug info.  [default: None]

    :returns: A tuple of the calculated moments:

        * M_00, M_10, M_01, M_11, M_20, M_02
        * M_21, M_12, M_30, M_03                          if ``third_order`` = True
        * M_22, M_31, M_13, M_40, M_04                    if ``fourth_order`` = True
        * M*_22, M*_33, M*_44                             if ``radial`` = True
        * variance of all previous values (in same order) if ``errors`` = True
    """
    # get vectors for data, weight and u, v
    data, weight, u, v = star.data.getDataVector(include_zero_weight=True)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    f, u0, v0, sigma, g1, g2, flag = hsm(star)
    if flag:
        raise RuntimeError("flag = %d from hsm"%flag)
    profile = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0)
    image = galsim.Image(star.image.copy(), dtype=float)
    profile.drawImage(image, method='sb', center=star.image_pos)
    # convert image into kernel
    kernel = image.array.flatten()

    # Anywhere the data is masked, fill in with the hsm profile.
    mask = weight == 0.
    if np.any(mask):
        data[mask] = kernel[mask] * np.sum(data[~mask])/np.sum(kernel[~mask])

    # This is the weighted image values, which we use in all the sums below.
    # Notation:
    #   W = weight * kernel
    #   I = data
    #   V = var(data) -- used below.
    WI = kernel * data

    M00 = np.sum(WI)
    norm = M00            # This is the normalization for all other moments.

    u -= u0
    v -= v0
    WI /= norm
    M10 = np.sum(WI * u)
    M01 = np.sum(WI * v)

    # Subtract off the measured first moments
    # XXX: The definitions given in the doc string imply that we don't do the following two lines.
    #      So I'd prefer to remove them.  However, they are required for now at least to maintain
    #      compatibility with the old definitions.
    u -= M10
    v -= M01

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
    M11 = np.sum(WIrsq)
    M20 = np.sum(WIusqmvsq)
    M02 = 2 * np.sum(WIuv)

    # Keep track of the tuple to return.  We may add more.
    ret = (M00, M10, M01, M11, M20, M02)

    if errors:
        # It we take W, w to be fixed and assume that var(I) = 1/w, then

        # var(M00) = sum W^2 1/w
        # var(M10) = sum W^2 1/w u^2 / M00^2
        # var(M01) = sum W^2 1/w v^2 / M00^2
        # var(M11) = sum W^2 1/w (u^2 + v^2)^2 / M00^2
        # var(M20) = sum W^2 1/w (u^2 - v^2)^2 / M00^2
        # var(M02) = sum W^2 1/w (2uv)^2 / M00^2

        WV = kernel**2
        WV[~mask] /= weight[~mask]  # Only use 1/w where w != 0
        WV[mask] /= np.mean(weight[~mask])

        # XXX: This equation doesn't make any sense.  But it matches the old code.
        #      We do ignore the flux (and centroid) when we actually make the chisq, so in
        #      itself, it doesn't actually matter for anything.  However, it is used as part
        #      of varnorm, which does show up in all the other values.  For bright stars, this
        #      means that the term using varnorm is highly overweighted.  So terms proportional
        #      to M_ij**2 are too large, which downweights them in the chisq.  Maybe this is a
        #      clue to why it seems to work better this way?  Maybe it's acting as a kind of
        #      outlier suppression.
        pix_area = star.data.pixel_area
        varM00 = np.sum(WV**2 * weight**3) * f**2 * pix_area**2

        WV /= norm**2
        # XXX: I (MJ) don't think the +u0 here is justified, but it matches what Aaron did.
        #      It also don't really matter, since we ignore the centroid values in the chisq.
        varM10 = np.sum(WV * (u+u0+M10)**2)
        varM01 = np.sum(WV * (v+v0+M01)**2)
        varM11 = np.sum(WV * rsq**2)
        varM20 = np.sum(WV * usqmvsq**2)
        varM02 = 4 * np.sum(WV * uv**2)

        # However, these variance estimates are too low because there is uncertainty in the
        # correct kernel as well.  The above formulae assume that there is no uncertainty in W.
        # For now, we have some slightly ad hoc corrections derived by Aaron and Ares.
        # They generally start with reasonably well-motivated corrections trying to propagate
        # the uncertainties in u0, v0, norm into the variances.  But they don't work very well,
        # so then there are some order unity fudge factors that get applied.

        varnorm = varM00 / M00**2  # Save this for use below.

        # Variance in weighted flux is ~double this due to uncertainty in kernel.
        varM00 *= 4

        # Add variance due to u0,v0 uncertainties
        varM10 += varnorm * (M10+u0)**2
        varM01 += varnorm * (M01+v0)**2
        # Fudge factors.  See comments below.
        varM10 *= 2.1**2
        varM01 *= 2.1**2

        # Add variance due to u0,v0 uncertainties
        # XXX: Again, the WI**2 doesn't make any sense, but it's what the old code used to do, and
        # if we do this, then test_optics_and_test_fit_model passes.
        # These formula are known to be wrong anyway, so this is probably a sign that we need
        # to work more on getting these to be correct.
        varM11 += varM10 * np.sum(WI**2 * usq) * 4
        varM11 += varM01 * np.sum(WI**2 * vsq) * 4
        varM20 += varM10 * np.sum(WI**2 * usq) * 4
        varM20 += varM01 * np.sum(WI**2 * vsq) * 4
        varM02 += varM10 * np.sum(WI**2 * vsq) * 4
        varM02 += varM01 * np.sum(WI**2 * usq) * 4

        # Add variance due to normalization uncertainties
        #varM11 += varnorm * M11**2  # This is disabled in Aaron's code
        varM20 += varnorm * M20**2
        varM02 += varnorm * M02**2

        # Fudge factors, since the above semi-motivated calculation doesn't work all that well.
        # XXX: These fudge factors are not adequately justified.  They at least need better
        # justification if a priori dervations are not possible (as seems likely).
        # Also, the resulting values are not particularly accurate across a range of profiles,
        # indicating that there might need to be additional terms, e.g. ones related to
        # the uncertainties in sigma, g1, g2.
        # Recommend a script in devel/ the runs through a range of profiles and fits for the
        # appropriate coefficients of the various terms.  I.e. probably separate coefficients
        # for each of the above expected terms plus g1,g2 terms, rather than just one overall
        # fudge factor. Then this code can reference that script as justification.
        varM11 *= 1.8**2
        varM20 *= 2.3**2
        varM02 *= 2.3**2

        ret_err = (varM00, varM10, varM01, varM11, varM20, varM02)

    if third_order:
        M21 = np.sum(WIu * rsq)
        M12 = np.sum(WIv * rsq)
        M30 = np.sum(WIu * (usq-3*vsq))
        M03 = np.sum(WIv * (3*usq-vsq))
        ret += (M21, M12, M30, M03)

        if errors:
            WVusq = WV * usq
            WVvsq = WV * vsq
            varM21 = np.sum(WVusq * rsq**2)
            varM12 = np.sum(WVvsq * rsq**2)
            varM30 = np.sum(WVusq * (usq-3*vsq)**2)
            varM03 = np.sum(WVvsq * (3*usq-vsq)**2)

            # Add variance due to u0,v0 uncertainties
            varM21 += varM10 * np.sum(WI**2 * (3*u**2 + v**2)**2)
            varM21 += varM01 * np.sum(WI**2 * (2*u*v)**2)
            varM12 += varM10 * np.sum(WI**2 * (2*u*v)**2)
            varM12 += varM01 * np.sum(WI**2 * (u**2 + 3*v**2)**2)
            varM30 += varM10 * np.sum(WI**2 * (3*(u**2 - v**2))**2)
            varM30 += varM01 * np.sum(WI**2 * (6*u*v)**2)
            varM03 += varM10 * np.sum(WI**2 * (6*u*v)**2)
            varM03 += varM01 * np.sum(WI**2 * (3*(u**2 - v**2))**2)

            # Add variance due to normalization uncertainties
            varM21 += varnorm * M21**2
            varM12 += varnorm * M12**2
            varM30 += varnorm * M30**2
            varM03 += varnorm * M03**2

            # XXX: Again, these fudge factors are not adequately justified, nor are they
            # particularly accurate for varied profiles.
            varM21 *= 0.52**2
            varM12 *= 0.55**2

            ret_err += (varM21, varM12, varM30, varM03)

    if fourth_order:
        M22 = np.sum(WIrsq * rsq)
        M31 = np.sum(WIusqmvsq * rsq)
        M13 = 2. * np.sum(WIuv * rsq)
        M40 = np.sum(WI * (usqmvsq**2 - 4.*uv**2))
        M04 = 4. * np.sum(WIuv * usqmvsq)
        ret += (M22, M31, M13, M40, M04)

        if errors:
            varM22 = np.sum(WV * rsq**4)
            varM31 = np.sum(WV * usqmvsq**2 * rsq**2)
            varM13 = 4 * np.sum(WV * uv**2 * rsq**2)
            varM40 = np.sum(WV * (usqmvsq**2-4.*uv**2)**2)
            varM04 = 16. * np.sum(WV * uv**2 * usqmvsq**2)

            # Add variance due to u0,v0 uncertainties
            varM22 += varM10 * np.sum(WI**2 * (4*u*rsq)**2)
            varM22 += varM01 * np.sum(WI**2 * (4*v*rsq)**2)
            varM31 += varM10 * np.sum(WI**2 * (4*u**3)**2)
            varM31 += varM01 * np.sum(WI**2 * (4*v**3)**2)
            varM13 += varM10 * np.sum(WI**2 * (2*v*(3*u**2 + v**2))**2)
            varM13 += varM01 * np.sum(WI**2 * (2*u*(u**2 + 3*v**2))**2)
            varM40 += varM10 * np.sum(WI**2 * (4*u*(u**2 - 3*v**2))**2)
            varM40 += varM01 * np.sum(WI**2 * (4*v*(3*u**2 - v**2))**2)
            varM04 += varM10 * np.sum(WI**2 * (4*v*(3*u**2 - v**2))**2)
            varM04 += varM01 * np.sum(WI**2 * (4*u*(u**2 - 3*v**2))**2)

            # Add variance due to normalization uncertainties
            varM22 += varnorm * M22**2
            varM31 += varnorm * M31**2
            varM13 += varnorm * M13**2
            varM40 += varnorm * M40**2
            varM04 += varnorm * M04**2

            # XXX: Ditto re fudge factors
            varM22 *= 2.4**2
            varM31 *= 2.6**2
            varM13 *= 2.6**2
            varM40 *= 1.1**2
            varM04 *= 1.1**2

            ret_err += (varM22, varM31, varM13, varM40, varM04)

    if radial:
        M22 = np.sum(WIrsq * rsq)
        M33 = np.sum(WIrsq * rsq**2)
        M44 = np.sum(WIrsq * rsq**3)

        # XXX: Note that I am preserving the definitions you had in the old code, but these
        #      "orthogonal" radial moments don't make any sense.  You are mixing terms with
        #      different units.  The u and v values have units here (arcsec), so you are
        #      adding and subtracting terms with units of arcsec**2, arcsec**4, arcsec**6, etc.
        #      Maybe you meant to normalize these by powers of M11 or sigma?
        #      But probably better to just not bother with these and simply return the above
        #      M22, M33, M44.  Surely the random forest can do something equally useful with them.
        #      Maybe better, since it wouldn't have to disentangle your mixed unit values.
        #      (The use case where you just use these for the chisq calculation in lieu of the
        #      pixel-based chisq would be fine, since the only real benefit of that is that you
        #      let hsm marginalize over the centroid rather than have to do it manually.)
        altM22 = M22 - 3*M11
        altM33 = M33 - 8*M22 + 12*M11
        altM44 = M44 - 15*M33 + 60*M22 - 60*M11
        ret += (altM22, altM33, altM44)

        if errors:
            varM22 = np.sum(WV * (rsq**2 - 3*rsq)**2)
            varM33 = np.sum(WV * (rsq**3 - 8*rsq**2 + 12*rsq)**2)
            varM44 = np.sum(WV * (rsq**4 - 15*rsq**3 + 60*rsq**2 - 60*rsq)**2)

            # Add variance due to u0,v0 uncertainties
            varM22 += varM10 * np.sum(WI**2 * (2*u*(2*rsq-3))**2)
            varM22 += varM01 * np.sum(WI**2 * (2*v*(2*rsq-3))**2)
            varM33 += varM10 * np.sum(WI**2 * (2*u*(3*rsq**2-16*rsq+12))**2)
            varM33 += varM01 * np.sum(WI**2 * (2*v*(3*rsq**2-16*rsq+12))**2)
            varM44 += varM10 * np.sum(WI**2 * (2*u*(4*rsq**3-45*rsq**2+120*rsq-60))**2)
            varM44 += varM01 * np.sum(WI**2 * (2*v*(4*rsq**3-45*rsq**2+120*rsq-60))**2)

            # Add variance due to normalization uncertainties
            varM22 += varnorm * altM22**2
            varM33 += varnorm * altM33**2
            varM44 += varnorm * altM44**2

            # XXX: Ditto re fudge factors
            varM22 *= 0.81**2
            varM33 *= 0.34**2
            varM44 *= 0.51**2

            ret_err += (varM22, varM33, varM44)

    if errors:
        return ret + ret_err
    else:
        return ret

# Make this also available as a method of Star
Star.calculate_moments = calculate_moments
