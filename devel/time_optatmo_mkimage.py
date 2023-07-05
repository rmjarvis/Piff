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
import pickle
import galsim
import argparse
import time

import piff
from piff import Star, StarData, StarFit
from piff.util import calculateSNR
from sklearn.gaussian_process.kernels import RBF
from atmosim import make_atmosphere

decaminfo = piff.des.DECamInfo()

def make_blank_star(x, y, chipnum, properties=(), stamp_size=19, **kwargs):
    wcs = decaminfo.get_nominal_wcs(chipnum)
    properties = dict(properties, chipnum=chipnum)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=stamp_size,
                                properties=properties, **kwargs)
    return star

def make_stars(nstars, rng, psf, init_params, atmo_type='None', logger=None):

    # some constants
    foreground = 3000.0
    maglo = 15.0
    maghi = 20.0
    pixels = 19
    max_snr = 0. # effectively remove extra noise
    min_snr = 50.
    pixedge = 20

    # Randomly cover the DES footprint and 61/62 CCDs
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = rng.np.choice(chiplist, nstars)
    icen = rng.np.uniform(1+pixedge, 2048-pixedge, nstars)
    jcen = rng.np.uniform(1+pixedge, 4096-pixedge, nstars)

    # Build blank stars at the desired locations
    blank_stars = []
    for i in range(nstars):
        # make the shell of a Star object
        blank_stars.append(make_blank_star(icen[i], jcen[i], chipnum[i], stamp_size=pixels))

    # fill reference wavefront using optatmo_psf
    stars = psf._get_refwavefront(blank_stars,logger)
    u_arr = np.array([star['u'] for star in stars])
    v_arr = np.array([star['v'] for star in stars])

    # apply a different atmospheric kernel for every star
    if atmo_type=='RBF':

        # use a Gaussian Process to set a value of atmo_size, atmo_g1, atmo_g2 for each star
        x = np.array([u_arr, v_arr]).T

        # creating the correlation matrix / kernel
        kernel = 1. * RBF(1.)
        K = kernel.__call__(x)

        # typical values from DES Y1 image study
        size_sigma = 0.021
        g1_sigma = 0.0025
        g2_sigma = 0.0025

        # generating gaussian random field
        atmo_size = size_sigma * rng.np.multivariate_normal(np.zeros(nstars), K)
        atmo_g1 = g1_sigma * rng.np.multivariate_normal(np.zeros(nstars), K)
        atmo_g2 = g2_sigma * rng.np.multivariate_normal(np.zeros(nstars), K)

        # add these to ofit_params
        for i in range(nstars):
            init_params.register('atmo_size_%d' % (i), atmo_size[i])
            init_params.register('atmo_g1_%d' % (i), atmo_g1[i])
            init_params.register('atmo_g2_%d' % (i), atmo_g2[i])

    elif atmo_type=='Galsim':
        
        raw_seeing = 0.80*galsim.arcsec
        airmass = 1.20
        wavelength = 620  #782.
        atm, target_FWHM, r0_500, L0 = make_atmosphere(airmass,raw_seeing,wavelength,rng)

        decam_aper = galsim.Aperture(diam=4.010, lam=wavelength, obscuration=0.4196)  

        moms = []
        for i in range(nstars):            
            thx = u_arr[i] * galsim.arcsec
            thy = v_arr[i] * galsim.arcsec
            staratmo =  atm.makePSF(lam=wavelength,t0=0.0,exptime=90.,
                                    flux=25_000,theta=(thx,thy),aper=decam_aper)

            img = staratmo.drawImage(method='phot', scale=0.263)  
            mom = galsim.hsm.FindAdaptiveMom(img)
            moms.append(mom)

        e1 = np.array([mom.observed_shape.e1 for mom in moms])
        e2 = np.array([mom.observed_shape.e2 for mom in moms])
        atmo_g1 = np.array([mom.observed_shape.g1 for mom in moms])
        atmo_g2 = np.array([mom.observed_shape.g2 for mom in moms])
        sig_m = np.array([mom.moments_sigma*0.263 for mom in moms])  # convert to arcsec
        atmo_size = sig_m * 2.355

        # not used but for reference here is transformation from HSM to moments
        #M11 = (sig_m/np.power(1.-e1**2-e2**2,0.25) )**2
        #M02 = M11*e2
        #M20 = M11*e1

        # set constant atmosphere terms to be mean of atmo_
        mean_atmo_size = np.mean(atmo_size)
        mean_atmo_g1 = np.mean(atmo_g1)
        mean_atmo_g2 = np.mean(atmo_g2)
        init_params.setValue('opt_size',mean_atmo_size)
        init_params.setValue('opt_g1',mean_atmo_g1)
        init_params.setValue('opt_g2',mean_atmo_g2)

        # add spatially varying atmosphere terms to ofit_params
        for i in range(nstars):
            init_params.register('atmo_size_%d' % (i),atmo_size[i]-mean_atmo_size)
            init_params.register('atmo_g1_%d' % (i),atmo_g1[i]-mean_atmo_g1)
            init_params.register('atmo_g2_%d' % (i),atmo_g2[i]-mean_atmo_g2)
        
    # have the OptAtmo PSF make the model stars
    noiseless_stars = psf.make_modelstars(init_params, stars, psf.model, logger=logger)

    # now add shot noise to the stars and scale to desired flux
    noisy_stars = []
    for i,star in enumerate(noiseless_stars):

        # use to adjust the star's properties
        properties = star.data.properties

        # calculate the flux from a randomly selected magnitude
        mag = rng.np.uniform(maglo,maghi)  # uniform distribution
        flux = 10.**((30.0-mag)/2.5)           # using a zero point of 30th mag

        # scale the image's pixel_sum, work with a copy
        im = star.image * flux
        properties['flux'] = flux

        # Generate a Poisson noise model, with some foreground (assumes that this foreground
        # was already subtracted)
        poisson_noise = galsim.PoissonNoise(rng, sky_level=foreground)
        im.addNoise(poisson_noise)  # adds in place

        # get new weight in photo-electrons (im is a Galsim image)
        inverse_weight = im + foreground
        weight = 1.0/inverse_weight

        # check minimum snr
        snr = calculateSNR(im, weight)
        if snr < min_snr:
            continue

        # set the maximum SNR for this star, by scaling up the weight
        if max_snr > 0 and snr > max_snr:
            factor = (max_snr / snr)**2
            weight *= factor

        # make new noisy star
        star = star.withProperties(snr=snr)
        data = star.data.withNew(image=im, weight=weight)
        noisy_star = Star(data, star.fit)
        noisy_stars.append(noisy_star)

    # return the list of Stars
    return noisy_stars



def make_image(config_file, variables='', seed=12345, nstars=8000, optics_type='Fast',
               atmo_type='None', verbose_level=1):
    """
    This test makes an image's worth of stars using optatmo_psf

    :param config_file:   Configuration file for optatmo_psf
    :param variables:     String with additional configuration variables [default: '']
    :param seed:          Random number seed [default: 12345]
    :param nstars:        Number of stars to make [default: 8000]
    :param optics_type:   Type of optical wavefront to generate, 'Fast' or 'Nominal' [default: Fast]
    :param atmo_type:     Type of atmosphere to generate, 'None', 'RBF', 'Galsim' [default: None]
    :param verbose_level: Verbose level for logger [default: 1]
    """

    # random number generator
    rng = galsim.BaseDeviate(seed)

    # read the yaml
    config = piff.read_config(config_file)
    logger = piff.setup_logger(verbose=verbose_level)

    # modify the config from the command line..
    piff.config.parse_variables(config, variables, logger)

    # build the PSF
    psf = piff.PSF.process(config['psf'], logger=logger)

    # get params object from psf
    init_params = psf._setup_ofit_params(psf.ofit_initvalues, psf.ofit_bounds,
                                         psf.ofit_initerrors, psf.ofit_double_zernike_terms,
                                         psf.ofit_fix)

    # fill with random values
    init_params.setValue('opt_size', rng.np.uniform(0.8,1.2,1)[0])
    init_params.setValue('opt_L0', rng.np.uniform(3.,10.,1)[0])
    init_params.setValue('z4f1', rng.np.uniform(-0.3,0.3,1)[0])
    init_params.setValue('z5f1', rng.np.uniform(-0.2,0.2,1)[0])
    init_params.setValue('z6f1', rng.np.uniform(-0.2,0.2,1)[0])
    init_params.setValue('z11f1', rng.np.uniform(-0.2,0.2,1)[0])

    if optics_type=='Nominal':
        init_params.setValue('opt_g1', rng.np.uniform(-0.05,0.05,1)[0])
        init_params.setValue('opt_g2', rng.np.uniform(-0.05,0.05,1)[0])
        for iz in range(7,10+1):
            init_params.setValue('z%df1' % (iz), rng.np.uniform(-0.2,0.2,1)[0])
        init_params.setValue('z5f2', rng.np.uniform(-0.3,0.3,1)[0])
        init_params.setValue('z5f3', rng.np.uniform(-0.3,0.3,1)[0])
        init_params.setValue('z6f2', rng.np.uniform(-0.3,0.3,1)[0])
        init_params.setValue('z6f3', rng.np.uniform(-0.3,0.3,1)[0])

    # make an image of fake stars
    stars = make_stars(nstars, rng, psf, init_params, atmo_type, logger=logger)

    # return
    return stars, init_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file', type=str, help="Configuration Filename")
    parser.add_argument('variables', type=str, nargs='*',
                        help="add options to configuration",
                        default='')
    parser.add_argument('-f', '--output_file', dest='output_file', type=str,
                        help="Output Filename", default='mkimage.pkl')
    parser.add_argument('-s', '--seed', dest='seed', type=int, help="seed", default=12345)
    parser.add_argument('-n', '--nstars', dest='nstars', type=int, help="nstars", default=800)
    parser.add_argument('-o', '--optics_type', dest='optics_type', type=str,
                        help="optics_type Fast,Nominal", default='Fast')
    parser.add_argument('-a', '--atmo_type', dest='atmo_type', type=str,
                        help="atmo_type None,RBF,Galsim", default='None')

    options = parser.parse_args()
    kwargs = vars(options)
    print(options)

    t0 = time.time()
    stars, init_params = make_image(options.config_file, options.variables, options.seed,
                                    options.nstars, options.optics_type, options.atmo_type)
    t1 = time.time()
    print('Time for make_image = ',t1-t0)

    init_params.print()
    outdict = {"init_params":init_params, "stars":stars}
    pickle.dump(outdict, open(options.output_file, 'wb'))
