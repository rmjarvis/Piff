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
import subprocess
import yaml
import fitsio

from piff_test_helper import get_script_name, timer

star_type = np.dtype([('u', float),
                      ('v', float),
                      ('hlr', float),
                      ('g1', float),
                      ('g2', float),
                      ('u0', float),
                      ('v0', float),
                      ('flux', float)])


mod0 = piff.GSObjectModel(galsim.Kolmogorov(half_light_radius=1.0), force_model_center=True,
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
        var = 0.1
    else:
        var = noise
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu*du, v=fpv*du, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    k.drawImage(star.image, method='no_pixel',
                offset=galsim.PositionD(nom_u0/du,nom_v0/du), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
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

def make_average(Coord=None, gp=True):
    if Coord is None:
        x = np.linspace(0, 200, 10)
        x, y = np.meshgrid(x,x)
        x = x.reshape(len(x)**2)
        y = y.reshape(len(y)**2)
    else:
        x = Coord[:,0]
        y = Coord[:,1]
    
    average = 0.02 + 5e-8*(x-1024)**2 + 5e-8*(y-1024)**2
    params = np.recarray((len(x),), dtype=star_type)

    keys = ['hlr', 'g1', 'g2']

    for key in keys:
        if key is 'hlr':
            params[key] = average + 0.6
        else:
            params[key] = average

    if gp:
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(np.array([x, y]).T))
        cov = 0.05**2 * np.exp(-0.5*dists**2/30.**2) 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params['hlr'] += np.random.multivariate_normal([0]*len(x), cov)
            params['g1'] += np.random.multivariate_normal([0]*len(x), cov)
            params['g2'] += np.random.multivariate_normal([0]*len(x), cov)

    #import pylab as plt
    #for key  in keys:
    #    plt.scatter(x, y, c=params[key], lw = 0, s=40)
    #    plt.colorbar()
    #    plt.title(key)
    #    plt.show()

    params['u'] = x
    params['v'] = y
    params['flux'] = np.ones_like(average)
    params['u0'] = np.zeros_like(average)
    params['v0'] = np.zeros_like(average)

    return params

def setup():
    np.random.seed(42)
    for im in range(30):
        print(im)
        image = galsim.Image(2048, 2048, scale=0.26)

        x_list = [np.random.uniform(0, 2048)]
        y_list = [np.random.uniform(0, 2048)]
        i=0
        while i < 599:
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2048)
            D = np.sqrt((np.array(x_list)-x)**2 + (np.array(y_list)-y)**2)
            #avoid 2 stars on the same stamp
            if np.sum((D>60)) == len(D):
                x_list.append(x)
                y_list.append(y)
                i+=1

        coord = np.array([x_list,y_list]).T
        params = make_average(Coord=coord)
        psfs = []
        for x, y, hlr, g1, g2 in zip(x_list, y_list, params['hlr'], params['g1'], params['g2']):
            psf = galsim.Kolmogorov(half_light_radius=hlr, flux=1.).shear(g1=g1, g2=g2)
            bounds = galsim.BoundsI(int(x-21), int(x+22), int(y-21), int(y+22))
            offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
            psf.drawImage(image=image[bounds], method='no_pixel', offset=offset)

        image_file = os.path.join('output','test_mean_image_%i.fits'%(im))
        image.write(image_file)

        dtype = [ ('x','f8'), ('y','f8') ]
        data = np.empty(len(x_list), dtype=dtype)
        data['x'] = x_list
        data['y'] = y_list
        cat_file = os.path.join('output','test_mean_cat_%i.fits'%(im))
        fitsio.write(cat_file, data, clobber=True)

        image_file = os.path.join('output','test_mean_image_%i.fits'%(im))
        cat_file = os.path.join('output','test_mean_cat_%i.fits'%(im))
        psf_file = os.path.join('output','test_mean_%i.piff'%(im))

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
        piff.piffify(config)

@timer
def test_meanify():
    
    setup()
    psf_file = os.path.join('output','test_mean_*.piff')
    average_file = os.path.join('output','average.fits')
    config = {
        'input' : {
            'file_name' : psf_file,
            'binning' : 30,
        },
        'output' : {
            'file_name' : average_file,
        }}

    piff.meanify(config)

    average = fitsio.read(average_file)
    params0 = make_average(Coord=average['COORDS0'][0] / 0.26, gp=False)
    keys = ['hlr', 'g1', 'g2']
    import pylab as plt
    for i,key in enumerate(keys):
        #plt.hist(params0[key] - average['PARAMS0'][0][:,i],np.linspace(-0.01,0.01,30))
        np.testing.assert_allclose(params0[key], average['PARAMS0'][0][:,i], rtol=1e-1, atol=1e-3)
        #plt.show()

if __name__ == '__main__':

    #test_meanify()

    x = np.random.uniform(0, 2048, size=700)
    y = np.random.uniform(0, 2048, size=700)
    coord = np.array([x,y]).T
    average = make_average(Coord=coord)
    
    stars = params_to_stars(average, noise=0.0, rng=None)
    stars_training = stars[:600]
    stars_validation = stars[600:]

    gp = piff.GPInterp2pcf(kernel="0.05**2 * RBF(30.*0.26)",
                           optimize=False, white_noise=1e-5, average_fits='output/average.fits')
    gp.initialize(stars_training)
    gp.solve(stars_training)
    stars_interp = gp.interpolateList(stars_validation)

    params_interp = np.array([s.fit.params for s in stars_interp])
    params_validation = np.array([s.fit.params for s in stars_validation])
