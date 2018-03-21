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

def make_average(Coord=None):
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

    params['u'] = x
    params['v'] = y
    params['flux'] = np.ones_like(average)
    params['u0'] = np.zeros_like(average)
    params['v0'] = np.zeros_like(average)

    return params

def setup():
    np.random.seed(42)
    for im in range(10):
        print(im)
        image = galsim.Image(2048, 2048, scale=0.26)

        x_list = [np.random.uniform(0, 2048)]
        y_list = [np.random.uniform(0, 2048)]
        i=0
        while i < 499:
            x = np.random.uniform(0, 2048)
            y = np.random.uniform(0, 2078)
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
            'binning' : 20,
        },
        'output' : {
            'file_name' : average_file,
        }}

    piff.meanify(config)

    average = fitsio.read(average_file)
    params0 = make_average(Coord=average['COORDS0'][0] / 0.26)
    keys = ['hlr', 'g1', 'g2']
    import pylab as plt
    for i,key in enumerate(keys):
        plt.hist(params0[key] - average['PARAMS0'][0][:,i],np.linspace(-0.01,0.01,30))
        np.testing.assert_allclose(params0[key], average['PARAMS0'][0][:,i], rtol=1e-1, atol=1e-3)
        plt.show()
    
if __name__ == '__main__':

    test_meanify()
