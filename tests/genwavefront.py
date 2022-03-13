

import numpy as np
import copy

import galsim
import piff
decaminfo = piff.des.DECamInfo()
from piff import Star, StarData, StarFit
from piff.util import calculateSNR
import treegp


def makeBlankStar(x, y, chipnum, properties={}, stamp_size=19, **kwargs):
    wcs = decaminfo.get_nominal_wcs(chipnum)
    properties_in = {'chipnum': chipnum}
    properties_in.update(properties)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=stamp_size, properties=properties_in, **kwargs)
    return star

def makeStarsGP(nstars,rng,psf,init_params,kernel=None,logger=None):

    # some constants
    foreground = 3000.0
    maglo = 15.0
    maghi = 20.0
    pixels = 19
    max_snr = 1000000000. # effectively remove extra noise
    min_snr = 50.

    fudge = 1.
    size_sigma = fudge * 0.007 * 3.  # systematic in e0 from Taylor * factor of 3 scaling from atmo_size to e0
    g1_sigma = fudge * 0.005 / 2.    # systematic in g1 from Taylor * factor of 0.5 scalling atmo_g1 to e2 (M01)
    g2_sigma = fudge * 0.005 / 2.

    # chipnum,icen,jcen
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = np.random.choice(chiplist,nstars)
    pixedge = 20
    icen = np.random.uniform(1+pixedge,2048-pixedge,nstars)   # random pixel position inside CCD
    jcen = np.random.uniform(1+pixedge,4096-pixedge,nstars)

    # blank stars
    blank_stars = []
    for i in range(nstars):
        # make the shell of a Star object
        blank_stars.append(makeBlankStar(icen[i],jcen[i],chipnum[i],stamp_size=pixels))

    # fill reference wavefront
    stars = psf._get_refwavefront(blank_stars,logger)

    # if we want a Turbulence contribution, add it here:
    if kernel:

        # get value of atmo_size, atmo_g1, atmo_g2 for each star
        u_arr = np.array([star['u'] for star in stars])
        v_arr = np.array([star['v'] for star in stars])

        x = np.array([u_arr, v_arr]).T
        # creating the correlation matrix / kernel
        K = kernel.__call__(x)
        # generating gaussian random field
        atmo_size = size_sigma * np.random.multivariate_normal(np.zeros(nstars), K)
        atmo_g1 = g1_sigma * np.random.multivariate_normal(np.zeros(nstars), K)
        atmo_g2 = g2_sigma * np.random.multivariate_normal(np.zeros(nstars), K)

        # add these to ofit_params
        for i in range(nstars):
            init_params.register('atmo_size_%d' % (i),atmo_size[i])
            init_params.register('atmo_g1_%d' % (i),atmo_g1[i])
            init_params.register('atmo_g2_%d' % (i),atmo_g2[i])

    # have the OptAtmo PSF make the model stars
    noiseless_stars = psf.make_modelstars(init_params,stars,psf.model,logger=logger)

    # now add shot noise to the stars and scale to desired flux
    noisy_stars = []
    for star in noiseless_stars:

        mag = np.random.uniform(maglo,maghi)  # uniform distribution
        flux = 10.**((30.0-mag)/2.5)          # zero point of 30th mag
        # scale the image's pixel_sum, work with a copy
        im = copy.deepcopy(star.image) * flux

        # Generate a Poisson noise model, with some foreground (assumes that this foreground was already subtracted)
        poisson_noise = galsim.PoissonNoise(rng,sky_level=foreground)
        im.addNoise(poisson_noise)  # adds in place

        # get new weight in photo-electrons (im is a Galsim image)
        inverse_weight = im + foreground
        weight = 1.0/inverse_weight

        # set the maximum SNR for this star, by scaling up the weight
        snr = calculateSNR(im, weight)
        if snr > max_snr:
            factor = (max_snr / snr)**2
            weight *= factor

        # check minimum snr
        if snr > min_snr:

            # make new noisy star
            properties = star.data.properties
            properties['snr'] = snr        # store the original SNR here

            for key in ['x', 'y', 'u', 'v']:
                # Get rid of keys that constructor doesn't want to see:
                properties.pop(key, None)

            data = StarData(image=im,
                        image_pos=star.data.image_pos,
                        weight=weight,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        orig_weight=star.data.orig_weight,
                        properties=properties)
            fit = StarFit(None,
                      flux=star.fit.flux,
                      center=star.fit.center)
            noisy_star = Star(data, fit)
            noisy_stars.append(noisy_star)

    # return the list of Stars
    return noisy_stars
