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

import numpy as np
import os
import galsim
import piff
import sklearn

def calculate_moments(star):
    """This is based on piff.util.calculate_moments, except it always computes all moments
    and errors.

    It also just computes the naive estimates of the errors.  Indeed this script is the source
    of the correction that is in the piff version to get better error estimates.
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
    profile.drawImage(image, method='sb', center=star.image_pos)
    # convert image into kernel
    kernel = image.array.flatten()
    # Anywhere the data is masked, fill in with the hsm profile.
    mask = weight == 0.
    if np.any(mask):
        data[mask] = kernel[mask] * np.sum(data[~mask])/np.sum(kernel[~mask])

    WI = kernel * data
    M00 = np.sum(WI)
    WI /= M00
    u -= u0
    v -= v0
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
    M13 = 2. * np.sum(WIrsq * uv)
    M40 = np.sum(WI * (usqmvsq**2 - 4.*uv**2))
    M04 = 4. * np.sum(WIusqmvsq * uv)

    M33 = np.sum(WI * rsq3)
    M44 = np.sum(WI * rsq4)
    M22n = M22/(M11**2)
    M33n = M33/(M11**3)
    M44n = M44/(M11**4)

    mom = np.array([M00, M10, M01, M11, M20, M02, M21, M12, M30, M03,
                    M22, M31, M13, M40, M04, M22n, M33n, M44n])

    # Calculate naive estimates of errors:
    WV = kernel**2
    WV[~mask] /= weight[~mask]
    WV[mask] /= np.mean(weight[~mask])

    varM00 = np.sum(WV)
    WV /= M00**2

    varM10 = np.sum(WV * usq)
    varM01 = np.sum(WV * vsq)
    varM11 = np.sum(WV * (rsq-M11)**2)
    varM20 = np.sum(WV * (usqmvsq-M20)**2)
    varM02 = np.sum(WV * (2*uv-M02)**2)

    varM21 = np.sum(WV * (u*rsq - M21)**2)
    varM12 = np.sum(WV * (v*rsq - M12)**2)
    varM30 = np.sum(WV * (u*(usq-3*vsq) - M30)**2)
    varM03 = np.sum(WV * (v*(3*usq-vsq) - M03)**2)

    varM22 = np.sum(WV * (rsq2 - M22)**2)
    varM31 = np.sum(WV * (rsq*usqmvsq - M31)**2)
    varM13 = np.sum(WV * (2*rsq*uv - M13)**2)
    varM40 = np.sum(WV * (usqmvsq**2-4.*uv**2 - M40)**2)
    varM04 = np.sum(WV * (4*usqmvsq*uv - M04)**2)

    varM22n = np.sum(WV * (rsq2 - 2*M22*rsq/M11 + M22)**2) / (M11**4)
    varM33n = np.sum(WV * (rsq3 - 3*M33*rsq/M11 + 2*M33)**2) / (M11**6)
    varM44n = np.sum(WV * (rsq4 - 4*M44*rsq/M11 + 3*M44)**2) / (M11**8)

    var = np.array([varM00, varM10, varM01, varM11, varM20, varM02, varM21, varM12, varM30, varM03,
                    varM22, varM31, varM13, varM40, varM04, varM22n, varM33n, varM44n])

    return mom, var

def make_psf(rng, iprof=None, scale=None, g1=None, g2=None, flux=1000):

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
        g1 = np_rng.uniform(-0.2, 0.2)
    if g2 is None:
        g2 = np_rng.uniform(-0.2, 0.2)
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

    rng = galsim.UniformDeviate(12345)
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
