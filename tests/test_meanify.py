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

from __future__ import print_function
import warnings
import galsim
import numpy as np
import piff
import os
import fitsio
import glob

from piff_test_helper import get_script_name, timer

star_type = np.dtype([('u', float),
                      ('v', float),
                      ('hlr', float),
                      ('g1', float),
                      ('g2', float),
                      ('u0', float),
                      ('v0', float),
                      ('flux', float)])


mod0 = piff.GSObjectModel(galsim.Kolmogorov(half_light_radius=1.0),
                          include_pixel=False, fastfit=False)

def make_star(hlr, g1, g2, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
              nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Kolmogorov profile
    :param hlr:         The half_light_radius of the Kolmogorov.
    :param g1, g2:      Shear applied to profile.
    :param u0, v0:      The sub-pixel offset to apply.
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param du:          pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    """
    k = galsim.Kolmogorov(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2).shift(u0,v0)
    if noise == 0.:
        var = 1.e-6
    else:
        var = noise**2
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu*du, v=fpv*du, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    k.drawImage(star.image, method='no_pixel',
                offset=galsim.PositionD(nom_u0/du,nom_v0/du), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


def params_to_stars(params, noise=0.0, rng=None):
    stars = []
    for param in params.ravel():
        u, v, hlr, g1, g2, u0, v0, flux = param
        s = make_star(hlr, g1, g2, u0, v0, flux, noise=noise, du=0.26, fpu=u, fpv=v, rng=rng)
        try:
            s = mod0.initialize(s)
        except:
            print("Failed to initialize star at ",u,v)
        else:
            stars.append(s)
    return stars

def make_average(coord=None, gp=True):
    if coord is None:
        x = np.linspace(0, 2048, 10)
        x, y = np.meshgrid(x,x)
        x = x.reshape(len(x)**2)
        y = y.reshape(len(y)**2)
    else:
        x = coord[:,0]
        y = coord[:,1]

    average = 0.02 + 5e-8*(x-1024)**2 + 5e-8*(y-1024)**2
    params = np.recarray((len(x),), dtype=star_type)

    keys = ['hlr', 'g1', 'g2']

    for key in keys:
        if key == 'hlr':
            params[key] = average + 0.6
        else:
            params[key] = average

    if gp:
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(np.array([x, y]).T))
        cov = 0.03**2 * np.exp(-0.5*dists**2/300.**2)

        # avoids to print warning from numpy when generated uge gaussian random fields.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params['hlr'] += np.random.multivariate_normal([0]*len(x), cov)
            params['g1'] += np.random.multivariate_normal([0]*len(x), cov)
            params['g2'] += np.random.multivariate_normal([0]*len(x), cov)

    params['u'] = x
    params['v'] = y
    params['flux'] = np.ones_like(average)
    params['u0'] = np.zeros_like(average)
    params['v0'] = np.zeros_like(average)

    return params

def setup():
    np.random.seed(42)
    if __name__ == '__main__':
        nimages = 20
        stars_per_image = 500
    else:
        nimages = 2
        stars_per_image = 50

    # Delete any existing image files
    for image_file in glob.glob(os.path.join('output','test_mean_image_*.fits')):
        os.remove(image_file)

    for k in range(nimages):
        print(k)
        image = galsim.Image(2048, 2048, scale=0.26)

        x_list = [np.random.uniform(0, 2048)]
        y_list = [np.random.uniform(0, 2048)]
        i=0
        while i < stars_per_image:
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)
            D = np.sqrt((np.array(x_list)-x)**2 + (np.array(y_list)-y)**2)
            #avoid 2 stars on the same stamp
            if np.all(D > 60):
                x_list.append(x)
                y_list.append(y)
                i+=1

        coord = np.array([x_list,y_list]).T
        params = make_average(coord=coord)
        psfs = []
        for x, y, hlr, g1, g2 in zip(x_list, y_list, params['hlr'], params['g1'], params['g2']):
            psf = galsim.Kolmogorov(half_light_radius=hlr, flux=1.).shear(g1=g1, g2=g2)
            bounds = galsim.BoundsI(int(x-21), int(x+22), int(y-21), int(y+22))
            if not image.bounds.includes(bounds): continue
            offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
            psf.drawImage(image=image[bounds], method='no_pixel', offset=offset)

        image_file = os.path.join('output','test_mean_image_%02i.fits'%k)
        image.write(image_file)

        dtype = [ ('x','f8'), ('y','f8') ]
        data = np.empty(len(x_list), dtype=dtype)
        data['x'] = x_list
        data['y'] = y_list
        cat_file = os.path.join('output','test_mean_cat_%02i.fits'%k)
        fitsio.write(cat_file, data, clobber=True)

        image_file = os.path.join('output','test_mean_image_%02i.fits'%k)
        cat_file = os.path.join('output','test_mean_cat_%02i.fits'%k)
        psf_file = os.path.join('output','test_mean_%02i.piff'%k)

        config = {
            'input' : {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'stamp_size' : 48
            },
            'psf' : {
                'model' : { 'type' : 'Kolmogorov',
                            'fastfit': True,
                            'include_pixel': False },
                'interp' : { 'type' : 'Polynomial',
                             'order' : 2}
            },
            'output' : {
                'file_name' : psf_file
            }
        }
        if __name__ == '__main__':
            config['verbose'] = 2
        else:
            config['verbose'] = 0
        piff.piffify(config)

@timer
def test_meanify():

    if __name__ == '__main__':
        rtol = 4.e-1
        atol = 5.e-2
        bin_spacing = 30  # arcsec
    else:
        rtol = 1.e-1
        atol = 3.e-2
        bin_spacing = 150  # arcsec

    psf_file = 'test_mean_*.piff'
    average_file = 'average.fits'

    psfs_list = sorted(glob.glob(os.path.join('output', 'test_mean_*.piff')))

    config0 = {
        'output' : {
            'file_name' : psfs_list,
        },
        'hyper' : {
            'file_name' : 'output/'+average_file,
        }}

    config1 = {
        'output' : {
            'file_name' : psf_file,
            'dir': 'output',
        },
        'hyper' : {
            'file_name' : average_file,
            'dir': 'output',
            'bin_spacing' : bin_spacing,
            'statistic' : 'mean',
            'params_fitted': [0, 2]
        }}

    config2 = {
        'output' : {
            'file_name' : psf_file,
            'dir': 'output',
        },
        'hyper' : {
            'file_name' : average_file,
            'dir': 'output',
            'bin_spacing' : bin_spacing,
            'statistic' : 'median',
        }}

    for config in [config0, config1, config2]:
        piff.meanify(config)
        ## test if found initial average
        average = fitsio.read(os.path.join('output',average_file))
        params0 = make_average(coord=average['COORDS0'][0] / 0.26, gp=False)
        keys = ['hlr', 'g1', 'g2']
        for i,key in enumerate(keys):
            if config == config1 and i == 1:
                np.testing.assert_allclose(np.zeros(len(average['PARAMS0'][0][:,i])),
                                           average['PARAMS0'][0][:,i], rtol=0, atol=0)
            else:
                np.testing.assert_allclose(params0[key], average['PARAMS0'][0][:,i],
                                           rtol=rtol, atol=atol)

    ## gaussian process testing of meanify
    np.random.seed(68)
    x = np.random.uniform(0, 2048, size=1000)
    y = np.random.uniform(0, 2048, size=1000)
    coord = np.array([x,y]).T
    average = make_average(coord=coord)

    stars = params_to_stars(average, noise=0.0, rng=None)
    stars_training = stars[:900]
    stars_validation = stars[900:]

    fit_hyp = ['none', 'isotropic']

    for fit in fit_hyp:
        gp = piff.GPInterp(kernel="0.009 * RBF(300.*0.26)",
                           optimizer=fit, white_noise=1e-5, average_fits='output/average.fits')
        gp.initialize(stars_training)
        gp.solve(stars_training)
        stars_interp = gp.interpolateList(stars_validation)
        params_interp = np.array([s.fit.params for s in stars_interp])
        params_validation = np.array([s.fit.params for s in stars_validation])
        params_training = np.array([s.fit.params for s in stars_training])
        np.testing.assert_allclose(params_interp, params_validation, rtol=rtol, atol=atol)


@timer
def test_invalid():

    psf_file = 'test_mean_*.piff'
    average_file = 'average.fits'

    psfs_list = os.path.join('output', 'test_mean_*.piff')

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_invalid_config.log')

    config = {
        'output' : {
            'file_name' : psfs_list,
        },
        'hyper' : {
            'file_name' : 'output/'+average_file,
            'bin_spacing' : 30,
            'statistic' : 'mean',
            'params_fitted': [0, 2]
        }}

    # Both output and hyper are required
    with np.testing.assert_raises(ValueError):
        piff.meanify(config={'output':config['output']}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.meanify(config={'hyper':config['hyper']}, logger=logger)
    # Both require file_name
    with np.testing.assert_raises(ValueError):
        piff.meanify(config={'output':config['output'], 'hyper':{}}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.meanify(config={'hyper':config['hyper'], 'output':{}}, logger=logger)
    # Invalid statistic
    config['hyper']['statistic'] = 'invalid'
    with np.testing.assert_raises(ValueError):
        piff.meanify(config=config, logger=logger)
    config['hyper']['statistic'] = 'mean'
    # Invalid params_fitted
    config['hyper']['params_fitted'] = 0
    with np.testing.assert_raises(TypeError):
        piff.meanify(config=config, logger=logger)
    config['hyper']['params_fitted'] = [0,2]
    # Invalid file_name
    config['output']['file_name'] = []
    with np.testing.assert_raises(ValueError):
        piff.meanify(config=config, logger=logger)
    config['output']['file_name'] = os.path.join('output', 'invalid_*.piff')
    with np.testing.assert_raises(ValueError):
        piff.meanify(config=config, logger=logger)
    config['output']['file_name'] = 7
    with np.testing.assert_raises(ValueError):
        piff.meanify(config=config, logger=logger)
    config['output']['file_name'] = psfs_list


if __name__ == '__main__':
    setup()
    test_meanify()
    test_invalid()
