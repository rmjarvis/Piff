# Copyright (c) 2021 by Mike Jarvis and the other collaborators on GitHub at
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

import numpy as np
import os
import galsim
import piff
import copy

# For reference, as of commit 1f4ad4425b80ade590addb2, the output of this script is:
#
# M00: mean ratio = 0.996 +- 0.034
# M10: mean ratio = 1.006 +- 0.038
# M01: mean ratio = 1.003 +- 0.031
# M11: mean ratio = 0.994 +- 0.037
# M20: mean ratio = 0.997 +- 0.052
# M02: mean ratio = 0.982 +- 0.035
# M21: mean ratio = 1.021 +- 0.073
# M12: mean ratio = 1.005 +- 0.034
# M30: mean ratio = 1.003 +- 0.040
# M03: mean ratio = 0.997 +- 0.032
# M22: mean ratio = 0.995 +- 0.038
# M31: mean ratio = 1.007 +- 0.060
# M13: mean ratio = 0.990 +- 0.037
# M40: mean ratio = 1.049 +- 0.064
# M04: mean ratio = 1.051 +- 0.063
# M22n: mean ratio = 1.018 +- 0.040
# M33n: mean ratio = 1.023 +- 0.043
# M44n: mean ratio = 1.026 +- 0.047
#
# As a result, we no longer apply any fudge factors to the error estimates in calculate_moments.


def calculate_moments(star):
    """This is based on piff.util.calculate_moments, except it always computes all moments
    and errors.

    It used to also not apply calibration fudge factors.  Indeed, this was the script that we
    had used to calculate those factors.  But we now have analytic formulae that are quite accurate
    (at least when e1,e2 are small), so we don't need those factors anymore.
    """

    # get vectors for data, weight and u, v
    data, weight, u, v = star.data.getDataVector()
    # also get the values for the HSM kernel, which is just the fitted hsm model
    f, u0, v0, sigma, g1, g2, flag = star.hsm
    if flag:
        star.image.write('failed_image.fits')
        star.weight.write('failed_weight.fits')
        raise RuntimeError("flag = %d from hsm"%flag)

    profile = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0)
    image = star.image.copy()
    profile.drawImage(image, method='no_pixel', center=star.image_pos)
    W = image.array.flatten()

    WI = W * data
    M00 = np.sum(WI)
    WI /= M00
    u -= u0
    v -= v0
    usq = u*u
    vsq = v*v
    uv = u*v
    rsq = usq + vsq
    usqmvsq = usq - vsq

    if 0:
        # Verify the formulae for dW/d*
        ssq = np.sum(WI * rsq)
        e1 = np.sum(WI * usqmvsq) / ssq
        e2 = 2 * np.sum(WI * uv) / ssq
        big = np.where(W > 0.8 * np.max(W))

        # u0
        p2 = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2).shift(u0+1.e-3, v0)
        im2 = p2.drawImage(image.copy(), method='no_pixel', center=star.image_pos)
        W2 = im2.array.flatten()
        print('empirical dW/du0 = ',(W2-W)[big]/1.e-3)
        print('predicted dW/du0 = ',W[big] * u[big]/ssq)

        # v0
        p2 = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0+1.e-3)
        im2 = p2.drawImage(image.copy(), method='no_pixel', center=star.image_pos)
        W2 = im2.array.flatten()
        print('empirical dW/dv0 = ',(W2-W)[big]/1.e-3)
        print('predicted dW/dv0 = ',W[big] * v[big]/ssq)

        # ssq
        p2 = galsim.Gaussian(sigma=sigma*(1.+1.e-3), flux=1.0).shear(g1=g1, g2=g2).shift(u0, v0)
        im2 = p2.drawImage(image.copy(), method='no_pixel', center=star.image_pos)
        W2 = im2.array.flatten()
        dssq = ssq * ((1+1.e-3)**2-1)
        print('empirical dW/dssq = ',(W2-W)[big]/dssq)
        print('predicted dW/dssq = ',W[big] * (rsq[big]-2*ssq)/(2*ssq**2))

        # e1
        p2 = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1+1.e-3, g2=g2).shift(u0, v0)
        im2 = p2.drawImage(image.copy(), method='no_pixel', center=star.image_pos)
        W2 = im2.array.flatten()
        de1 = galsim.Shear(g1=g1+1.e-3,g2=g2).e1 - e1
        print('empirical dW/de1 = ',(W2-W)[big]/de1)
        print('predicted dW/de1 = ',W[big] * usqmvsq[big]/(2*ssq))

        # e2
        p2 = galsim.Gaussian(sigma=sigma, flux=1.0).shear(g1=g1, g2=g2+1.e-3).shift(u0, v0)
        im2 = p2.drawImage(image.copy(), method='no_pixel', center=star.image_pos)
        W2 = im2.array.flatten()
        de2 = galsim.Shear(g1=g1, g2=g2+1.e-3).e2 - e2
        print('empirical dW/de2 = ',(W2-W)[big]/de2)
        print('predicted dW/de2 = ',W[big] * 2* uv[big]/(2*ssq))
        quit()

    WIu = WI * u
    WIv = WI * v
    WIrsq = WI*rsq
    WIusqmvsq = WI*usqmvsq
    WIuv = WI*uv

    M10 = np.sum(WIu) + u0
    M01 = np.sum(WIv) + v0

    M11 = np.sum(WIrsq)
    M20 = np.sum(WIusqmvsq)
    M02 = 2 * np.sum(WIuv)

    M21 = np.sum(WIu * rsq)
    M12 = np.sum(WIv * rsq)
    M30 = np.sum(WIu * (usq-3*vsq))
    M03 = np.sum(WIv * (3*usq-vsq))

    rsq2 = rsq * rsq
    rsq3 = rsq2 * rsq
    rsq4 = rsq3 * rsq
    M22 = np.sum(WI * rsq2)
    M31 = np.sum(WIrsq * usqmvsq)
    M13 = 2 * np.sum(WIrsq * uv)
    M40 = np.sum(WI * (usqmvsq**2 - 4*uv**2))
    M04 = 4 * np.sum(WIusqmvsq * uv)

    M33 = np.sum(WI * rsq3)
    M44 = np.sum(WI * rsq4)
    M22n = M22/(M11**2)
    M33n = M33/(M11**3)
    M44n = M44/(M11**4)

    mom = np.array([M00, M10, M01, M11, M20, M02, M21, M12, M30, M03,
                    M22, M31, M13, M40, M04, M22n, M33n, M44n])

    A = 1/(3-M22/M11**2)
    B = 2/(4-M22/M11**2)
    dM00 = 1 - A*(rsq/M11-1)
    WV = W**2 / weight
    varM00 = np.sum(WV * dM00**2)
    WV /= M00**2

    if 0:
        # Verify the d/dI_k formulae
        ssq = M11
        e1 = M20/M11
        e2 = M02/M11

        big = np.where(W > 0.3 * np.max(W))
        print('big = ',big)
        k = big[0][0]
        print('k = ',k)
        print('Wk = ',W[k])
        print('Ik = ',data[k])
        dIk = data[k] * 3.e-2
        print('dIk = ',dIk)
        _star = copy.deepcopy(star)
        _star.image.array.ravel()[k] += dIk
        print('Ik = ',_star.image.array.ravel()[k])

        delattr(_star,'_hsm')
        _f, _u0, _v0, _sigma, _g1, _g2, _flag = _star.hsm
        _profile = galsim.Gaussian(sigma=_sigma, flux=1.0).shear(g1=_g1, g2=_g2).shift(_u0, _v0)
        _image = _star.image.copy()
        _profile.drawImage(_image, method='no_pixel', center=star.image_pos)
        _W = _image.array.flatten()
        _WI = _W * _star.image.array.flatten()
        _M00 = np.sum(_WI)
        _WI /= _M00
        _u = u + u0 - _u0
        _v = v + v0 - _v0
        _usq = _u*u
        _vsq = _v*v
        _uv = _u*v
        _rsq = _usq + vsq
        _usqmvsq = _usq - _vsq

        _WIu = _WI * _u
        _WIv = _WI * _v
        _WIrsq = _WI * _rsq
        _WIusqmvsq = _WI * _usqmvsq
        _WIuv = _WI * _uv

        _M10 = np.sum(_WIu) + _u0
        _M01 = np.sum(_WIv) + _v0
        _M11 = np.sum(_WIrsq)
        _M20 = np.sum(_WIusqmvsq)
        _M02 = 2 * np.sum(_WIuv)
        _ssq = _M11
        _e1 = _M20/M11
        _e2 = _M02/M11

        _M21 = np.sum(_WIu * _rsq)
        _M12 = np.sum(_WIv * _rsq)
        _M30 = np.sum(_WIu * (_usq-3*_vsq))
        _M03 = np.sum(_WIv * (3*_usq-_vsq))

        _rsq2 = _rsq * _rsq
        _rsq3 = _rsq2 * _rsq
        _rsq4 = _rsq3 * _rsq
        _M22 = np.sum(_WI * _rsq2)
        _M31 = np.sum(_WIrsq * _usqmvsq)
        _M13 = 2 * np.sum(_WIrsq * _uv)
        _M40 = np.sum(_WI * (_usqmvsq**2 - 4*_uv**2))
        _M04 = 4 * np.sum(_WIusqmvsq * _uv)


        print('empirical du0/dIk = ',(_u0 - u0) / dIk)
        print('predicted du0/dIk = ',2*W[k]*u[k]/M00)

        print('empirical dv0/dIk = ',(_v0 - v0) / dIk)
        print('predicted dv0/dIk = ',2*W[k]*v[k]/M00)

        print('empirical dssq/dIk = ',(_ssq - ssq) / dIk)
        print('predicted dssq/dIk = ',2*W[k]*(rsq[k] - ssq) / M00 / (3-M22/ssq**2))
        print('predicted dssq/dIk = ',2*A*W[k]*(rsq[k] - ssq) / M00)

        print('empirical de1/dIk = ',(_e1 - e1) / dIk)
        print('predicted de1/dIk = ',W[k]*usqmvsq[k]/ssq/M00 / (1-M22/(4*ssq**2)))
        print('predicted de1/dIk = ',2*B*W[k]*usqmvsq[k]/ssq/M00)

        print('empirical de2/dIk = ',(_e2 - e2) / dIk)
        print('predicted de2/dIk = ',W[k]*2*uv[k]/ssq/M00 / (1-M22/(4*ssq**2)))
        print('predicted de2/dIk = ',2*B*W[k]*2*uv[k]/ssq/M00)
        print()

        print('empirical dM00/dIk = ',(_M00-M00)/dIk)
        print('predicted dM00/dIk = ',W[k] * (4 - M22/ssq**2 - rsq[k]/ssq)/(3-M22/ssq**2))
        print('predicted dM00/dIk = ',W[k] * dM00[k])
        print()

        print('empirical dM10/dIk = ',(_M10-M10)/dIk)
        print('predicted dM10/dIk = ',2*W[k] * u[k] / M00)

        print('empirical dM01/dIk = ',(_M01-M01)/dIk)
        print('predicted dM01/dIk = ',2*W[k] * v[k] / M00)
        print()

        print('empirical dM11/dIk = ',(_M11-M11)/dIk)
        print('predicted dM11/dIk = ',2*A*W[k] * (rsq[k]-ssq) / M00)

        print('empirical dM20/dIk = ',(_M20-M20)/dIk)
        print('predicted dM20/dIk = ',2*W[k] * (A*e1*(rsq[k]-ssq) + B*usqmvsq[k]) / M00)

        print('empirical dM02/dIk = ',(_M02-M02)/dIk)
        print('predicted dM02/dIk = ',2*W[k] * (A*e2*(rsq[k]-ssq) + 2*B*uv[k]) / M00)
        print()

        print('empirical dM21/dIk = ',(_M21-M21)/dIk)
        print('predicted dM21/dIk = ',W[k] * (u[k]*(rsq[k]**2 - 4*M11 + M22/M11) - M21*dM00[k])/M00)
        print('empirical dM12/dIk = ',(_M12-M12)/dIk)
        print('predicted dM12/dIk = ',W[k] * (v[k]*(rsq[k]**2 - 4*M11 + M22/M11) - M12*dM00[k])/M00)
        print('empirical dM30/dIk = ',(_M30-M30)/dIk)
        print('predicted dM30/dIk = ',W[k] * ((3*usq[k] - vsq[k])*u[k] - M30*dM00[k])/M00)
        print('empirical dM03/dIk = ',(_M03-M03)/dIk)
        print('predicted dM03/dIk = ',W[k] * ((usq[k] - 3*vsq[k])*v[k] - M03*dM00[k])/M00)
        print()

        print('empirical dM22/dIk = ',(_M22-M22)/dIk)
        print('predicted dM22/dIk = ',W[k] * (rsq[k]**2 + (1-dM00[k])*(M33/M11-2*M22) - M22 * dM00[k])/M00)
        print('predicted dM22/dIk = ',W[k] * (rsq[k]**2 + (M33/ssq-2*M22) - (M33/ssq - M22)*dM00[k])/M00)
        print('predicted dM22/dIk = ',W[k] * (rsq[k]**2 + A*(rsq[k]/ssq-1)*(M33/ssq-2*M22) - M22*dM00[k])/M00)
        quit()


    varM10 = 4 * np.sum(WV * usq)
    varM01 = 4 * np.sum(WV * vsq)

    varM11 = 4 * A**2 * np.sum(WV * (rsq - M11)**2)
    varM20 = 4 * np.sum(WV * (B*usqmvsq + A*M20 * (rsq/M11 - 1))**2)
    varM02 = 4 * np.sum(WV * (2*B*uv + A*M02 * (rsq/M11 - 1))**2)

    varM21 = np.sum(WV * (u*(rsq-4*M11+M22/M11) - M21 * dM00)**2)
    varM12 = np.sum(WV * (v*(rsq-4*M11+M22/M11) - M12 * dM00)**2)
    varM30 = np.sum(WV * (u*(usq-3*vsq) - M30 * dM00)**2)
    varM03 = np.sum(WV * (v*(3*usq-vsq) - M03 * dM00)**2)

    varM22 = np.sum(WV * (rsq2 + A*(rsq/M11-1)*(M33/M11-2*M22) - M22 * dM00)**2)
    varM31 = np.sum(WV * (usqmvsq * (rsq + B*M33/(2*M11**2)) - M31 * dM00)**2)
    varM13 = np.sum(WV * (2*uv * (rsq + B*M33/(2*M11**2)) - M13 * dM00)**2)
    varM40 = np.sum(WV * (usqmvsq**2 - 4*uv**2 - M40 * dM00)**2)
    varM04 = np.sum(WV * (4*usqmvsq*uv - M04 * dM00)**2)

    M55n = np.sum(WI * rsq4 * rsq) / M11**5
    varM22n = np.sum(WV * (rsq2/M11**2 + A*(rsq/M11-1)*(M33n-6*M22n) - M22n*dM00)**2)
    varM33n = np.sum(WV * (rsq3/M11**3 + A*(rsq/M11-1)*(M44n-8*M33n) - M33n*dM00)**2)
    varM44n = np.sum(WV * (rsq4/M11**4 + A*(rsq/M11-1)*(M55n-10*M44n) - M44n*dM00)**2)

    var = np.array([varM00, varM10, varM01, varM11, varM20, varM02, varM21, varM12, varM30, varM03,
                    varM22, varM31, varM13, varM40, varM04, varM22n, varM33n, varM44n])

    return mom, var

def make_psf(rng, iprof=None, scale=None, g1=None, g2=None, flux=10000):

    np_rng = np.random.default_rng(rng.raw())

    # Pick from one of several plausible PSF profiles.
    psf_profs = [
             galsim.Gaussian(fwhm=1.),
             galsim.Moffat(beta=1.5, fwhm=1.),
             galsim.Moffat(beta=4.5, fwhm=1.),
             galsim.Kolmogorov(fwhm=1.),
             galsim.Airy(lam=700, diam=4),
             galsim.Airy(lam=1200, diam=6.5, obscuration=0.6),
             galsim.OpticalPSF(lam=700, diam=4, obscuration=0.6,
                               defocus=0.2, coma1=0.2, coma2=-0.2, astig1=-0.1, astig2=0.1),
             galsim.OpticalPSF(lam=900, diam=2.1, obscuration=0.2,
                               aberrations=[0,0,0,0,-0.1,0.2,-0.15,-0.1,0.15,0.1,0.15,-0.2]),
             galsim.OpticalPSF(lam=1200, diam=6.5, obscuration=0.3,
                               aberrations=[0,0,0,0,0.2,0.1,0.15,-0.1,-0.15,-0.2,0.1,0.15]),
            ]
    # The last 5 need an atmospheric part, or else they don't much resemble the kinds of
    # PSF profiles we actually care about.
    psf_profs[-5:] = [galsim.Convolve(galsim.Kolmogorov(fwhm=0.6), p) for p in psf_profs[-5:]]

    if iprof is None:
        psf = np_rng.choice(psf_profs)
    else:
        psf = psf_profs[iprof]

    # Choose a random size and shape within reasonable ranges.
    if g1 is None:
        g1 = np_rng.uniform(-0.03, 0.03)
    if g2 is None:
        g2 = np_rng.uniform(-0.03, 0.03)
    if scale is None:
        # Note: Don't go too small, since hsm fails more often for size close to pixel_scale.
        scale = np.exp(np_rng.uniform(-0.3, 1.0))

    psf = psf.dilate(scale).shear(g1=g1,g2=g2).withFlux(flux)

    return psf

def make_star(psf, rng, pixel_scale=0.2):

    image = galsim.Image(scale=pixel_scale, ncol=128, nrow=128)
    weight = image.copy()
    weight.fill(1.)

    # offset shouldn't matter, but make it random
    offset = (rng() * 2 - 1, rng() * 2 - 1)
    psf.drawImage(image, offset=offset, method='no_pixel')
    star = piff.Star.makeTarget(x=0, y=0, image=image, weight=weight)

    return star

def add_noise(star, rng, noise_var=0.1):

    noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var), rng=rng)
    noisy_image = star.image.copy()
    noisy_image.addNoise(noise)
    weight = star.weight.copy()
    weight.fill(1./noise_var)
    noisy_star = piff.Star.makeTarget(x=0, y=0, image=noisy_image, weight=weight)
    return noisy_star

def estimate_moment_errors(star, rng, num=2000):

    moments = []
    errors = []
    for i in range(num):
        noisy_star = add_noise(star, rng)
        try:
            m, e = calculate_moments(noisy_star)
            moments.append(m)
            errors.append(e)
        except RuntimeError:
            print(f'HSM failed for star number {i}')
            # if hsm fails, just skip this noise realization.
            pass
    mean_moments = np.mean(moments, axis=0)
    mean_errors = np.mean(errors, axis=0)
    var_moments = np.var(moments, axis=0)
    return mean_moments, mean_errors, var_moments

def check_moment_errors(nstars=500):

    rng = galsim.UniformDeviate(1234)
    all_mean_moments = []
    all_mean_errors = []
    all_var_moments = []
    for i in range(nstars):
        print(f'Star {i}/{nstars}')
        psf = make_psf(rng)
        star = make_star(psf, rng)
        mean_moments, mean_errors, var_moments = estimate_moment_errors(star, rng)

        print('mean moments = ',mean_moments)
        #print('mean errors = ',mean_errors)
        #print('var_moments = ',var_moments)
        #print('ratio = ',var_moments / mean_errors)

        all_mean_moments.append(mean_moments)
        all_mean_errors.append(mean_errors)
        all_var_moments.append(var_moments)

    all_mean_moments = np.column_stack(all_mean_moments)
    all_mean_errors = np.column_stack(all_mean_errors)
    all_var_moments = np.column_stack(all_var_moments)

    names = ['M00',
             'M10', 'M01',
             'M11', 'M20', 'M02',
             'M21', 'M12', 'M30', 'M03',
             'M22', 'M31', 'M13', 'M40', 'M04',
             'M22n', 'M33n', 'M44n']

    for k, name in enumerate(names):
        print('{}: mean ratio = {:.3f} +- {:.3f}'.format(
                name,
                np.mean(all_var_moments[k] / all_mean_errors[k]),
                np.std(all_var_moments[k] / all_mean_errors[k])))


if __name__ == '__main__':
    check_moment_errors()
